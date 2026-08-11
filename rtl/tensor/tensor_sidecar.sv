// Tensor sidecar (M18): the D4-socket coprocessor — INT8 GEMM with local
// tile SRAM (spec §2) + the four frozen LUT special functions (spec §3).
// Bit-true golden: golden/tensor.py. Command interface: latched T_* args
// + doorbell pulse; busy readable while working (the warp polls).
// GEMM flow: DMA A(MxK i8) and B(KxN i8) into tile SRAM -> compute one
// 8-wide K-strip per cycle (8 MACs/cycle) -> DMA C(MxN i32) out.
// LUT flow : stream T_M elements through the selected 256-entry ROM.
//
// Memory face is the v2 edge/credit contract (mem_spec §1b, D-032c
// stage C2): m_req is a one-cycle pulse per beat, completions return
// strictly in order, CRED beats in flight. The streaming masters
// (LD_A/LD_B/LD_C0/ST_C) issue by an ISSUE cursor and place response
// data by a RESPONSE cursor (in-order completion makes it
// well-defined); phase boundaries barrier on a full credit pool, so
// busy still means memory-visible. The LUT stream runs
// one-outstanding on the edge face (serial load->store dependency;
// CRED is a ceiling, not a mandate).
module tensor_sidecar #(
  parameter int unsigned CRED = 4,  // memory-face credits (1..4)
  parameter string LUT_DIR = ""     // path prefix for lut_*.mem
) (
  input  logic        clk,
  input  logic        rst_n,
  // command (from the CSR socket, already uniform-checked by the core)
  input  logic        go,           // doorbell pulse
  input  logic [2:0]  op,           // 0 GEMM8, 1 GELU, 2 EXP, 3 SIGMOID, 4 RSQRT
  input  logic [31:0] addr_a,
  input  logic [31:0] addr_b,
  input  logic [31:0] addr_c,
  input  logic [6:0]  dim_m,        // <= 64 (or element count for LUT ops)
  input  logic [6:0]  dim_n,
  input  logic [6:0]  dim_k,
  input  logic        flag_acc,
  output logic        busy,
  // memory face — v2 edge/credit (mem_spec §1b); dedicated arbiter
  // port, in-order completion pulses
  output logic        m_req,
  output logic        m_we,
  output logic [31:0] m_addr,
  output logic [31:0] m_wdata,
  output logic [3:0]  m_be,
  input  logic        m_rsp_valid,
  input  logic [31:0] m_rdata,
  // observability
  output logic [31:0] cnt_ops       // completed commands
);
  // ---------------- LUT ROMs (golden-generated, G3 diffs byte-exact) ---
  logic [15:0] rom_gelu [256];
  logic [15:0] rom_exp  [256];
  logic [15:0] rom_sig  [256];
  logic [15:0] rom_rsq  [256];
  logic [15:0] rom_geld [256];      // delta-GELU (M19/D-018)
  initial begin
    $readmemh({LUT_DIR, "lut_gelu.mem"},    rom_gelu);
    $readmemh({LUT_DIR, "lut_exp.mem"},     rom_exp);
    $readmemh({LUT_DIR, "lut_sigmoid.mem"}, rom_sig);
    $readmemh({LUT_DIR, "lut_rsqrt.mem"},   rom_rsq);
    $readmemh({LUT_DIR, "lut_gelud.mem"},   rom_geld);
  end

  // ---------------- tile SRAM (D-030: 8 rotate-banked LUTRAMs) -----
  // bank = byte_addr[2:0], word = byte_addr[11:3]. After the n-strip
  // transposition below, every wide access is a CONSECUTIVE-8 window
  // (B row slice, C row slice) or a single byte (A), so each bank
  // needs one write + one state-muxed async read. The k-strip form
  // read tb at a DYNAMIC stride (kk*q_n) and tc at three addresses:
  // 169.6k LUTs of mux trees, placer DRC UTLZ-1 on the K26 (banked
  // ta/tb/tc measured 2026-07-13; xs banking alone changed nothing —
  // synthesis had already decomposed it).
  logic        ta_we [8], tb_we [8], tcw [8];
  logic [8:0]  ta_wa [8], tb_wa [8], tc_wa_b [8], tb_ra [8], tc_ra [8];
  logic [7:0]  ta_wd [8], tb_wd [8];
  logic [31:0] tc_wd_b [8];
  logic [8:0]  ta_ra;
  logic [7:0]  ta_rq [8], tb_rq [8];
  logic [31:0] tc_rq [8];
  for (genvar gb = 0; gb < 8; gb++) begin : g_tile
    (* ram_style = "distributed" *) logic [7:0]  a_m [512];
    (* ram_style = "distributed" *) logic [7:0]  b_m [512];
    (* ram_style = "distributed" *) logic [31:0] c_m [512];
    always_ff @(posedge clk) begin
      if (ta_we[gb]) a_m[ta_wa[gb]] <= ta_wd[gb];
      if (tb_we[gb]) b_m[tb_wa[gb]] <= tb_wd[gb];
      if (tcw[gb])   c_m[tc_wa_b[gb]] <= tc_wd_b[gb];
    end
    assign ta_rq[gb] = a_m[ta_ra];
    assign tb_rq[gb] = b_m[tb_ra[gb]];
    assign tc_rq[gb] = c_m[tc_ra[gb]];
  end

  // latched command
  logic [2:0]  q_op;
  logic [31:0] q_a, q_b, q_c;
  logic [6:0]  q_m, q_n, q_k;
  logic        q_acc;

  typedef enum logic [3:0] {IDLE, LD_A, LD_B, LD_C0, COMP, ST_C,
                            LUT_LD, LUT_ST, DONE} st_e;
  st_e st;
  logic [12:0] idx;                 // ISSUE cursor within a phase
  logic [12:0] rix;                 // RESPONSE cursor (loads place by it)
  logic [12:0] total;               // beats in the current phase
  logic        pend;                // LUT one-outstanding wait flag

  // ------------- v2 memory-face plumbing (D-032c stage C2) ----------
  // Registered one-cycle req pulses; one per-face credit pool. A
  // phase exits only when every beat is issued AND the pool is full
  // again (all completions returned) — the busy contract.
  logic       req_q;
  logic [2:0] cred;
  assign m_req = req_q;
  wire [2:0]  cred_nxt  = cred - 3'(m_req) + 3'(m_rsp_valid);
  wire        pool_full = (cred == 3'(CRED)) && !m_req;
  // compute loop indices
  logic [6:0]  ci_m, ci_n, ci_k;    // D-030: n advances by 8, k by 1
  logic signed [31:0] accv [8];     // per-lane accumulators

  // D-030 window addressing. A: one byte at aa. B: 8 consecutive
  // bytes at bb (row slice — n is now the vector axis). C: 8
  // consecutive words at cc, RMW once per window on the last k beat.
  // Unaligned windows are served by bank rotation: bank b holds
  // byte (base[2:0]+lane) == b at word base[11:3] (+1 on wrap).
  wire [11:0] aa = 12'(32'(ci_m) * 32'(q_k) + 32'(ci_k));
  wire [11:0] bb = 12'(32'(ci_k) * 32'(q_n) + 32'(ci_n));
  wire [11:0] cc = 12'(32'(ci_m) * 32'(q_n) + 32'(ci_n));
  wire [6:0]  n_rem  = q_n - ci_n;   // lanes e < n_rem are in-tile
  // D-031 COMP pipeline registers (docs/FPGA_IMPLEMENTATION.md#tensor-pipeline-optimization):
  // S0 = registered address multiplies, S1 = fetch/rotate, S2 = MAC
  // + C-window commit. II=1; cursor runs ahead; c_done + drain.
  logic        v0, v1, wb0, wb1, c_done;
  logic [11:0] aa0, bb0, cc0;
  logic [6:0]  nrem0, nrem1;
  logic [2:0]  ccr1;
  logic [8:0]  ccw1;
  logic [7:0]  a1;
  logic [7:0]  bl1 [8];
  logic [31:0] cl1 [8];
  (* use_dsp = "yes" *) logic signed [31:0] mulv [8];
  always_comb begin
    ta_ra = aa0[11:3];
    for (int b = 0; b < 8; b++) begin
      tb_ra[b] = bb0[11:3] + 9'((3'(b) < bb0[2:0]) ? 1 : 0);
      // C read port: ST_C reads the issue-cursor word (composed at
      // issue, one beat/cycle); COMP reads the RMW window (same
      // formula as the write address)
      tc_ra[b] = (st == ST_C)
                 ? idx[11:3]
                 : cc0[11:3] + 9'((3'(b) < cc0[2:0]) ? 1 : 0);
    end
    for (int e = 0; e < 8; e++)
      mulv[e] = (7'(e) < nrem1)
                ? 32'($signed(a1)) * 32'($signed(bl1[e]))
                : 32'sd0;
    // write ports: LD_A/LD_B lay 4 bytes/beat (banks {rix[0],j});
    // LD_C0 lays one word/beat; COMP writes the C window on the
    // last k beat, lanes masked to the tile edge. Loads place by the
    // RESPONSE cursor (D-032c: completions are strictly in order, so
    // rsp k is beat k regardless of how far the issue cursor ran)
    for (int b = 0; b < 8; b++) begin
      ta_we[b] = (st == LD_A) && m_rsp_valid && (rix[0] == 1'(b >> 2));
      tb_we[b] = (st == LD_B) && m_rsp_valid && (rix[0] == 1'(b >> 2));
      ta_wa[b] = rix[9:1];
      tb_wa[b] = rix[9:1];
      ta_wd[b] = m_rdata[8 * (b & 3) +: 8];
      tb_wd[b] = m_rdata[8 * (b & 3) +: 8];
      tcw[b] = (st == LD_C0 && m_rsp_valid && rix[2:0] == 3'(b))
               || (v1 && wb1 && 7'(cc_rot(-ccr1, 3'(b))) < nrem1);
      tc_wa_b[b] = (st == LD_C0)
                   ? rix[11:3]
                   : ccw1 + 9'((3'(b) < ccr1) ? 1 : 0);
      tc_wd_b[b] = (st == LD_C0)
                   ? m_rdata
                   : ((q_acc ? cl1[cc_rot(-ccr1, 3'(b))] : 32'd0)
                      + 32'(accv[cc_rot(-ccr1, 3'(b))])
                      + 32'(mulv[cc_rot(-ccr1, 3'(b))]));
    end
  end
  wire [31:0] tc_st0 = tc_rq[idx[2:0]];
  // LUT stream state
  logic [31:0] lut_in_word;
  logic [1:0]  lut_sub;

  // bank/lane rotation helper: (a + b) mod 8 on 3-bit operands
  function automatic logic [2:0] cc_rot(input logic [2:0] a,
                                        input logic [2:0] b);
    cc_rot = 3'(a + b);
  endfunction

  // signed-input tables are domain-ordered in the .mem (x=-128 at row 0):
  // index = byte + 128 = sign-bit flip. rsqrt input is uint8 (raw index).
  wire [7:0] lut_sidx = {~lut_in_word[7], lut_in_word[6:0]};
  wire [15:0] lut_val =
      (q_op == 3'd1) ? rom_gelu[lut_sidx] :
      (q_op == 3'd2) ? rom_exp[lut_sidx] :
      (q_op == 3'd3) ? rom_sig[lut_sidx] :
      (q_op == 3'd5) ? rom_geld[lut_sidx] :
                       rom_rsq[lut_in_word[7:0]];

  assign busy = (st != IDLE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= IDLE; idx <= '0; rix <= '0; total <= '0; pend <= 1'b0;
      q_op <= '0; q_a <= '0; q_b <= '0; q_c <= '0;
      q_m <= '0; q_n <= '0; q_k <= '0; q_acc <= 1'b0;
      ci_m <= '0; ci_n <= '0; ci_k <= '0;
      v0 <= 1'b0; v1 <= 1'b0; c_done <= 1'b0;
      for (int e2 = 0; e2 < 8; e2++) accv[e2] <= '0;
      req_q <= 1'b0; cred <= 3'(CRED);
      m_we <= 1'b0; m_addr <= '0; m_wdata <= '0;
      m_be <= '0; cnt_ops <= '0;
      lut_in_word <= '0; lut_sub <= '0;
    end else begin
      // one-shot strobe + credit bookkeeping
      req_q <= 1'b0;
      cred  <= cred_nxt;
`ifndef SYNTHESIS
      assert (cred <= 3'(CRED))
        else $fatal(1, "tensor_sidecar: credit pool overflow");
      assert (!(m_req && cred == 3'd0 && !m_rsp_valid))
        else $fatal(1, "tensor_sidecar: beat issued without a credit");
`endif
      unique case (st)
        IDLE: if (go) begin
          q_op <= op; q_a <= addr_a; q_b <= addr_b; q_c <= addr_c;
          q_m <= dim_m; q_n <= dim_n; q_k <= dim_k; q_acc <= flag_acc;
          idx <= '0; rix <= '0; pend <= 1'b0;
          if (op == 3'd0) begin
            total <= 13'((32'(dim_m) * 32'(dim_k) + 3) >> 2);
            st <= LD_A;
          end else begin
            total <= 13'((32'(dim_m) + 3) >> 2);
            st <= LUT_LD;
          end
        end
        // ---------- GEMM DMA in: A then B (word beats, packed i8) ------
        // Credit-pipelined: issue one beat per cycle credits
        // permitting; responses land in the banks by rix (comb
        // above); the phase exits on issue-complete + pool refill.
        LD_A: begin
          if (idx != total && cred_nxt != 3'd0) begin
            req_q <= 1'b1; m_we <= 1'b0;
            m_addr <= q_a + {17'd0, idx, 2'b00};
            idx <= idx + 13'd1;
          end
          if (m_rsp_valid) rix <= rix + 13'd1;
          if (idx == total && pool_full) begin
            idx <= '0; rix <= '0;
            total <= 13'((32'(q_k) * 32'(q_n) + 3) >> 2);
            st <= LD_B;
          end
        end
        LD_B: begin
          if (idx != total && cred_nxt != 3'd0) begin
            req_q <= 1'b1; m_we <= 1'b0;
            m_addr <= q_b + {17'd0, idx, 2'b00};
            idx <= idx + 13'd1;
          end
          if (m_rsp_valid) rix <= rix + 13'd1;
          if (idx == total && pool_full) begin
            idx <= '0; rix <= '0;
            ci_m <= '0; ci_n <= '0; ci_k <= '0;
            v0 <= 1'b0; v1 <= 1'b0; c_done <= 1'b0;
            for (int e2 = 0; e2 < 8; e2++) accv[e2] <= '0;
            if (q_acc) begin
              total <= 13'(32'(q_m) * 32'(q_n));
              st <= LD_C0;
            end else st <= COMP;
          end
        end
        LD_C0: begin                    // accumulate mode: prefetch C
          if (idx != total && cred_nxt != 3'd0) begin
            req_q <= 1'b1; m_we <= 1'b0;
            m_addr <= q_c + {17'd0, idx, 2'b00};
            idx <= idx + 13'd1;
          end
          if (m_rsp_valid) rix <= rix + 13'd1;
          if (idx == total && pool_full) begin
            idx <= '0; rix <= '0; st <= COMP;
          end
        end
        // ---------- compute: 3-stage pipeline, II=1 (D-031) ------------
        // S0 issues one beat/cycle; S1 fetches/rotates into registers;
        // S2 MACs and commits the C window on writeback beats (tcw).
        // Bit-exact per D-030; ~2 bubble cycles per COMP session.
        COMP: begin
          if (!c_done) begin
            v0 <= 1'b1;
            aa0 <= aa; bb0 <= bb; cc0 <= cc;
            nrem0 <= n_rem; wb0 <= (ci_k == q_k - 7'd1);
            if (ci_k == q_k - 7'd1) begin
              ci_k <= '0;
              if (ci_n + 7'd8 >= q_n) begin
                ci_n <= '0;
                if (ci_m == q_m - 7'd1) c_done <= 1'b1;
                else ci_m <= ci_m + 7'd1;
              end else ci_n <= ci_n + 7'd8;
            end else ci_k <= ci_k + 7'd1;
          end else v0 <= 1'b0;
          v1 <= v0; wb1 <= wb0; nrem1 <= nrem0;
          ccr1 <= cc0[2:0]; ccw1 <= cc0[11:3];
          a1 <= ta_rq[aa0[2:0]];
          for (int e2 = 0; e2 < 8; e2++) begin
            bl1[e2] <= tb_rq[cc_rot(bb0[2:0], 3'(e2))];
            cl1[e2] <= tc_rq[cc_rot(cc0[2:0], 3'(e2))];
          end
          if (v1) begin
            if (wb1) begin
              for (int e2 = 0; e2 < 8; e2++) accv[e2] <= '0;
            end else begin
              for (int e2 = 0; e2 < 8; e2++)
                accv[e2] <= accv[e2] + mulv[e2];
            end
          end
          if (c_done && !v0 && !v1) begin
            idx <= '0;
            total <= 13'(32'(q_m) * 32'(q_n));
            st <= ST_C;
          end
        end
        ST_C: begin
          // credit-pipelined write stream: compose from the async C
          // bank read at issue (the idx+1 preload trick retires);
          // exit on pool refill = every word memory-visible
          if (idx != total && cred_nxt != 3'd0) begin
            req_q <= 1'b1; m_we <= 1'b1; m_be <= 4'hF;
            m_addr <= q_c + {17'd0, idx, 2'b00};
            m_wdata <= tc_st0;
            idx <= idx + 13'd1;
          end
          if (idx == total && pool_full) st <= DONE;
        end
        // ---------- LUT streaming (in i8 word -> out per spec) ---------
        // ---------- LUT streaming: ONE-OUTSTANDING on the edge face
        // (serial load->store data dependency; mem_spec §1b — CRED is
        // a ceiling, not a mandate). pend = beat in flight.
        LUT_LD: begin
          if (!pend && cred_nxt != 3'd0) begin
            req_q <= 1'b1; m_we <= 1'b0;
            m_addr <= q_a + {17'd0, idx, 2'b00};
            pend <= 1'b1;
          end
          if (m_rsp_valid) begin
            pend <= 1'b0;
            lut_in_word <= m_rdata;
            lut_sub <= 2'd0;
            st <= LUT_ST;
          end
        end
        LUT_ST: begin
          // one element per beat: i8->i8 packs via byte-enables; i16 out
          // writes halfwords (GELU is the only i8-out op)
          if (!pend && cred_nxt != 3'd0) begin
            req_q <= 1'b1; m_we <= 1'b1;
            if (q_op == 3'd1) begin
              m_addr <= q_c + {17'd0, idx, 2'b00};
              m_wdata <= {4{lut_val[7:0]}};
              m_be <= 4'b0001 << lut_sub;
            end else begin
              m_addr <= (q_c + ({17'd0, idx, 2'b00} << 1)
                         + {29'd0, lut_sub, 1'b0}) & 32'hFFFF_FFFC;
              m_wdata <= {2{lut_val}};
              m_be <= lut_sub[0] ? 4'b1100 : 4'b0011;
            end
            pend <= 1'b1;
          end
          if (m_rsp_valid) begin
            pend <= 1'b0;
            if (lut_sub == 2'd3 ||
                (32'(idx) * 4 + 32'(lut_sub) + 1) >= 32'(q_m)) begin
              if ((32'(idx) * 4 + 32'(lut_sub) + 1) >= 32'(q_m))
                st <= DONE;
              else begin
                idx <= idx + 13'd1;
                st <= LUT_LD;
              end
            end else begin
              lut_sub <= lut_sub + 2'd1;
              lut_in_word <= {8'd0, lut_in_word[31:8]};
            end
          end
        end
        DONE: begin
          cnt_ops <= cnt_ops + 32'd1;
          st <= IDLE;
        end
        default: st <= IDLE;
      endcase
    end
  end
endmodule
