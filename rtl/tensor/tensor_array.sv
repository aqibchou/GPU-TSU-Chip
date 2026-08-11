// tensor_array.sv — the SV2 serving engine (docs/SOFTWARE_AND_VALIDATION.md#serving-engine).
// T_OP 6 GEMV_STREAM: y[M] i32 = W[M x K] (i8 | packed i4, zp 8) . x[K] i8.
// T_OP 7 GEMM_VERIFY: Y[M x B] = W[M x K] . X[K x B], B = T_N+1 <= 8 —
// T_OP 6 IS B=1 (spec "T_OP 7 implementation freeze"): one datapath,
// no op decode. One weight beat feeds B accumulate slices; the weight
// operand select and K-guard are shared, the x window / product /
// reduce / accumulate slice is replicated per column.
// Weights STREAM through the wide face (512b/beat, per-row bursts) and
// are used once; X is resident in an internal LUTRAM (column-major,
// columns padded to KPAD = beats*kstep bytes, B*KPAD <= 4096); x loads
// and Y writes ride the narrow v1-style port, with the writeback
// OVERLAPPED with the next row's burst so the wide face stays
// saturated (the frozen SV2.2 bound is 1.10x ideal). Bit-frozen
// goldens: golden/tensor.py gemm8 (W8) and golden/quant4.py
// (W4 + gemm_verify_* + the x_image layout).
// The port-width law (spec v1): one 512b beat feeds 64 W8 / 128 W4
// MACs per slice — sized so the face out-runs the DDR budget at
// 250 MHz; slice capacity beyond the port pays only under batch.
module tensor_array (
  input  logic        clk,
  input  logic        rst_n,
  // command (SoC routes T_OP 6/7 GO here; fields per spec §interfaces)
  input  logic        go,
  input  logic [31:0] t_a,           // W base (64B-aligned)
  input  logic [31:0] t_b,           // X base (word-aligned)
  input  logic [31:0] t_c,           // Y base (word-aligned)
  input  logic [23:0] t_m,           // {k[11:0], m[11:0]}, both nonzero
  input  logic [2:0]  t_n,           // B-1 (0 for T_OP 6)
  input  logic [8:0]  t_flags,       // bit0 W4, bit1 X_RESIDENT
  output logic        busy,
  // wide streaming read face (one outstanding burst, reads only)
  output logic        wr_req,        // 1-cycle pulse: start burst
  output logic [31:0] wr_addr,
  output logic [15:0] wr_beats,
  input  logic        wb_valid,      // one 512b beat (engine always ready)
  input  logic [511:0] wb_data,
  // narrow face (x loads, y writes) — a v1-style one-outstanding port
  output logic        m_valid,
  output logic        m_we,
  output logic [31:0] m_addr,
  output logic [31:0] m_wdata,
  output logic [3:0]  m_be,
  input  logic        m_ack,
  input  logic [31:0] m_rdata,
  // observability
  output logic [31:0] cnt_ops
);
  // x SRAM, D-029 shape: 32 word-position BANKS x 32 super-rows x 32b
  // (RAM32M distributed, the M20 regfile recipe). Every column window
  // xs[kcol .. kcol+kstep-1] is 64-aligned by construction (kstep in
  // {64,128}, KPAD a kstep multiple, kcol advances by kstep), so each
  // slice's readout is one 32:1 super-row select + a W8 half pick,
  // addressed by a REGISTERED per-column cursor — no adder in the read
  // path. The flat-array form hid the alignment from synthesis:
  // 128 lanes x 4096:1 byte muxes = 169.6k LUTs on the sidecar,
  // placer DRC UTLZ-1 (unfittable) — measured 2026-07-12.
  logic [1023:0] x_rows [8];               // per-slice super-row, async
  wire _unused = &{1'b0, t_flags[8:2]};

  typedef enum logic [2:0] {IDLE, LD_X, ROW_REQ, ROW_RUN, FLUSH,
                            DONE} st_e;
  st_e st;

  logic        q_w4;
  logic [3:0]  q_b1;                 // B = t_n + 1
  logic [31:0] q_b;
  logic [11:0] q_m, q_k;
  logic [12:0] xstride;              // KPAD: x column stride, bytes
  logic [12:0] q_xb;                 // B * KPAD (LD_X extent, <= 4096)
  logic [11:0] row;                  // current output row
  logic [31:0] row_base;             // q_a + row*row_bytes, kept
                                     // incrementally (the multiply was
                                     // the 4ns critical path once the
                                     // MAC pipelined: -0.621 measured)
  logic [11:0] kx;                   // W element cursor within the row
  logic [11:0] kcol [8];             // per-slice x byte cursor
  logic [12:0] xidx;                 // LD_X word cursor
  logic        x_pend;
  logic [15:0] row_beats;            // beats per row
  logic [31:0] row_bytes;            // W row stride in bytes
  logic [31:0] ycur;                 // Y write cursor (row-major M x B)
  // y writeback side-channel, DOUBLE-buffered: a slot frees while the
  // next row streams; each slot holds one row's B words
  logic        y_pend0, y_pend1;
  logic [31:0] y_addr0, y_addr1;
  logic signed [31:0] y_data0 [8];
  logic signed [31:0] y_data1 [8];
  logic        y_sel;                // slot being drained (locked)
  logic [3:0]  y_wi;                 // word index within the slot

  // ---------------- the MAC pipeline (D-031 leg 2 shape) -------------
  // 1 beat/cycle throughput, replicated per slice. S0x registers the
  // x window + weight beat (LUTRAM fetch and K-guard leave the
  // multiply cone), S1 registers PURE single products (one DSP each,
  // M/P regs absorbed; two register stages so -retiming cannot rebuild
  // the 2-DSP ALU chain — measured -0.219 with one), then fabric
  // reduce 128 -> 32 -> 8, and 8 -> 1 at the accumulate stage. Flags
  // travel with data; the WEIGHT operand is selected and sign-adjusted
  // at S0x and SHARED by all slices — a mux or subtract inside the
  // multiply expression maps to the DSP preadder + blocks M/P
  // absorption (the module's own line-77 lesson).
  logic signed [8:0]  wsel_q [128];
  logic               v0_q, v1_q, v2_q, v3_q, v4_q, v5_q;
  logic               last0_q, last1_q, last2_q, last3_q, last4_q,
                      last5_q;

  // K-bound guards mask the x OPERAND, never the product — a mux
  // inside the multiply expression blocks DSP inference (measured:
  // 2 DSPs inferred, -3.38ns at 4ns as LUT multipliers). Guard is
  // l < (q_k - kx): one shared subtract, identical to kx+l < q_k;
  // inactive slices (b >= B) mask the same operand to zero.
  // x-store porting: RAM32M carries 3 independent read addresses, so
  // the 8 slice windows are served by THREE explicit copies (2.66
  // windows each) instead of the 8 full replications inference built
  // (measured: 5,120 LUTRAM-LUTs at B=8; ~3/8 of that after this)
  wire xs_we = (st == LD_X) && x_pend && m_ack;
  for (genvar gw = 0; gw < 32; gw++) begin : g_xsb
    for (genvar gc = 0; gc < 3; gc++) begin : g_cp
      logic [31:0] bank [32];
      always_ff @(posedge clk)
        if (xs_we && xidx[4:0] == 5'(gw)) bank[xidx[9:5]] <= m_rdata;
      for (genvar gb = 3 * gc;
           gb < (3 * gc + 3 > 8 ? 8 : 3 * gc + 3); gb++) begin : g_rd
        assign x_rows[gb][32*gw +: 32] = bank[kcol[gb][11:7]];
      end
    end
  end
  wire [12:0] k_rem = 13'(q_k) - 13'(kx);

  wire [11:0] kstep = q_w4 ? 12'd128 : 12'd64;
  wire row_last_beat = 32'(kx) + 32'(kstep) >= 32'(q_k);
  // slots already spoken for: pending writebacks + lasts in the pipe
  wire [3:0] slots_committed = 4'(y_pend0) + 4'(y_pend1)
                             + 4'(last0_q) + 4'(last1_q) + 4'(last2_q)
                             + 4'(last3_q) + 4'(last4_q) + 4'(last5_q);
  wire pipe_busy = v0_q | v1_q | v2_q | v3_q | v4_q | v5_q;

  // per-slice column base = b * KPAD (constant shift-adds per slice)
  wire [12:0] colbase [8];
  for (genvar gb = 0; gb < 8; gb++) begin : g_cb
    assign colbase[gb] =
        ((gb & 1) != 0 ? xstride : 13'd0)
      + ((gb & 2) != 0 ? {xstride[11:0], 1'b0} : 13'd0)
      + ((gb & 4) != 0 ? {xstride[10:0], 2'b00} : 13'd0);
  end
  // B*KPAD at command time (3-term shift-add, no multiplier)
  wire [12:0] xs_w = t_flags[0]
      ? 13'(((32'(t_m[23:12]) + 127) >> 7) << 7)
      : 13'(((32'(t_m[23:12]) + 63) >> 6) << 6);
  wire [15:0] xb_w = (t_n[0] ? 16'(xs_w) : 16'd0)
                   + (t_n[1] ? {2'd0, xs_w, 1'b0} : 16'd0)
                   + (t_n[2] ? {1'd0, xs_w, 2'b00} : 16'd0)
                   + 16'(xs_w);

  // ---------------- per-slice datapath ----------------
  logic signed [31:0] s3_b [8];
  logic signed [31:0] s3r_b [8];       // registered slice sum (the
                                       // p2->acc path was -0.276 @4ns)
  logic signed [31:0] acc_b [8];
  wire  signed [31:0] acc_pl [8];
  for (genvar gs = 0; gs < 8; gs++) begin : g_sl
    wire [511:0] x_half = kcol[gs][6] ? x_rows[gs][1023:512]
                                      : x_rows[gs][511:0];
    logic signed [8:0] x_eff [128];
    always_comb
      for (int l = 0; l < 128; l++) begin
        automatic logic [7:0] xb = q_w4 ? x_rows[gs][8*l +: 8]
                                        : x_half[8*(l & 63) +: 8];
        x_eff[l] = ((32'(l) < 32'(k_rem)) && (4'(gs) < q_b1))
                   ? 9'($signed(xb)) : 9'sd0;
      end
    logic signed [8:0]  xq [128];
    (* use_dsp = "yes" *) logic signed [17:0] prod_q [128];
    (* keep = "true" *) logic signed [17:0] prodr_q [128];
    logic signed [19:0] red1_q [32];
    logic signed [21:0] p2_q [8];
    logic signed [17:0] prod_d [128];
    always_comb
      for (int l = 0; l < 128; l++)
        prod_d[l] = 18'(wsel_q[l] * xq[l]);   // pure reg x reg
    always_comb begin
      s3_b[gs] = 32'sd0;
      for (int q = 0; q < 8; q++) s3_b[gs] += 32'(p2_q[q]);
    end
    assign acc_pl[gs] = acc_b[gs] + s3r_b[gs];
    always_ff @(posedge clk) begin
      if ((st == ROW_RUN) && wb_valid)
        for (int l = 0; l < 128; l++) xq[l] <= x_eff[l];
      if (v0_q)
        for (int l = 0; l < 128; l++) prod_q[l] <= prod_d[l];
      if (v1_q)
        for (int l = 0; l < 128; l++) prodr_q[l] <= prod_q[l];
      if (v2_q)
        for (int r = 0; r < 32; r++) begin
          automatic logic signed [19:0] t4;
          t4 = 20'sd0;
          for (int g = 0; g < 4; g++) t4 += 20'(prodr_q[4 * r + g]);
          red1_q[r] <= t4;
        end
      if (v3_q)
        for (int q = 0; q < 8; q++) begin
          automatic logic signed [21:0] t;
          t = 22'sd0;
          for (int g = 0; g < 4; g++) t += 22'(red1_q[4 * q + g]);
          p2_q[q] <= t;
        end
      if (v4_q) s3r_b[gs] <= s3_b[gs];
      if (v5_q) acc_b[gs] <= last5_q ? 32'sd0 : acc_b[gs] + s3r_b[gs];
    end
  end

  assign busy = (st != IDLE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= IDLE; wr_req <= 1'b0; wr_addr <= '0; wr_beats <= '0;
      m_valid <= 1'b0; m_we <= 1'b0; m_addr <= '0; m_wdata <= '0;
      m_be <= '0; cnt_ops <= '0;
      q_w4 <= 1'b0; q_b1 <= 4'd1; q_b <= '0;
      q_m <= '0; q_k <= '0; row <= '0; kx <= '0; xidx <= '0;
      row_base <= '0; xstride <= '0; q_xb <= '0; ycur <= '0;
      x_pend <= 1'b0; row_beats <= '0; row_bytes <= '0;
      y_pend0 <= 1'b0; y_pend1 <= 1'b0; y_sel <= 1'b0; y_wi <= '0;
      y_addr0 <= '0; y_addr1 <= '0;
      v1_q <= 1'b0; v2_q <= 1'b0; last1_q <= 1'b0; last2_q <= 1'b0;
      for (int b = 0; b < 8; b++) kcol[b] <= '0;
    end else begin
      wr_req <= 1'b0;                // one-shot
      unique case (st)
        IDLE: if (go) begin
          q_w4 <= t_flags[0];
          q_b1 <= 4'(t_n) + 4'd1;
          q_b <= t_b; ycur <= t_c;
          q_m <= t_m[11:0]; q_k <= t_m[23:12];
          row <= '0; kx <= '0; xidx <= '0; x_pend <= 1'b0;
          row_base <= t_a;
          xstride <= xs_w; q_xb <= xb_w[12:0];
          v1_q <= 1'b0; v2_q <= 1'b0; last1_q <= 1'b0; last2_q <= 1'b0;
          // beats/row: ceil(K/64) W8, ceil(K/128) W4; stride = beats*64B
          row_beats <= t_flags[0]
                       ? 16'((32'(t_m[23:12]) + 127) >> 7)
                       : 16'((32'(t_m[23:12]) + 63) >> 6);
          row_bytes <= t_flags[0]
                       ? {10'd0, 16'((32'(t_m[23:12]) + 127) >> 7), 6'd0}
                       : {10'd0, 16'((32'(t_m[23:12]) + 63) >> 6), 6'd0};
          st <= t_flags[1] ? ROW_REQ : LD_X;
        end

        // ---------------- X load (narrow port, 4 bytes/beat) ---------
        // (loads B*KPAD bytes: the frozen column-major x_image layout)
        LD_X: begin
          if (!x_pend && !m_valid) begin
            m_valid <= 1'b1; m_we <= 1'b0; m_be <= '1;
            m_addr <= q_b + {17'd0, xidx, 2'b00};
            x_pend <= 1'b1;
          end
          if (x_pend && m_ack) begin
            // x bytes land in the g_xsb banks (xs_we strobe above)
            if (32'({xidx, 2'b00}) + 4 >= 32'(q_xb)) begin
              m_valid <= 1'b0; x_pend <= 1'b0; xidx <= '0;
              st <= ROW_REQ;
            end else begin
              xidx <= xidx + 13'd1;
              m_addr <= q_b + {17'd0, xidx + 13'd1, 2'b00};
            end
          end
        end

        // ---------------- per-row weight burst ----------------
        // (a burst is requested only when its eventual handoff has a
        // slot: committed = pending writebacks + row-lasts in the pipe)
        ROW_REQ: if (slots_committed < 4'd2) begin
          wr_req <= 1'b1;
          wr_addr <= row_base;
          wr_beats <= row_beats;
          kx <= '0;
          for (int b = 0; b < 8; b++) kcol[b] <= colbase[b][11:0];
          st <= ROW_RUN;
        end
        ROW_RUN: if (wb_valid) begin
          // feed pipeline stage 1; row bookkeeping runs at beat time
          if (row_last_beat) begin
            if (row + 1 == q_m) st <= FLUSH;
            else begin
              row <= row + 12'd1;
              row_base <= row_base + row_bytes;
              kx <= '0;
              for (int b = 0; b < 8; b++) kcol[b] <= colbase[b][11:0];
              // +1: this beat's own handoff is now committed too
              if (slots_committed + 4'd1 < 4'd2) begin
                wr_req <= 1'b1;
                wr_addr <= row_base + row_bytes;
                wr_beats <= row_beats;
              end else st <= ROW_REQ;
            end
          end else begin
            kx <= kx + kstep;
            for (int b = 0; b < 8; b++)
              kcol[b] <= kcol[b] + 12'(kstep);
          end
        end
        FLUSH: if (!pipe_busy && !y_pend0 && !y_pend1) st <= DONE;

        DONE: begin
          cnt_ops <= cnt_ops + 1;
          st <= IDLE;
        end
        default: st <= IDLE;
      endcase

      // ---------------- shared pipeline control advance ---------------
      v0_q <= (st == ROW_RUN) && wb_valid;
      last0_q <= (st == ROW_RUN) && wb_valid && row_last_beat;
      if ((st == ROW_RUN) && wb_valid) begin
        for (int l = 0; l < 128; l++) begin
          if (q_w4)
            wsel_q[l] <= 9'(signed'({1'b0, wb_data[4*l +: 4]}) - 9'sd8);
          else
            wsel_q[l] <= (l < 64) ? 9'($signed(wb_data[8*l +: 8]))
                                  : 9'sd0;
        end
      end
      v1_q <= v0_q; last1_q <= last0_q;
      v2_q <= v1_q; last2_q <= last1_q;
      v3_q <= v2_q; last3_q <= last2_q;
      v4_q <= v3_q; last4_q <= last3_q;
      v5_q <= v4_q; last5_q <= last4_q;
      if (v5_q && last5_q) begin
        // S3 handoff into the free slot (guaranteed by the gates):
        // one row's B words + its Y cursor
        if (!y_pend0) begin
          y_addr0 <= ycur;
          for (int b = 0; b < 8; b++) y_data0[b] <= acc_pl[b];
          y_pend0 <= 1'b1;
        end else begin
          y_addr1 <= ycur;
          for (int b = 0; b < 8; b++) y_data1[b] <= acc_pl[b];
          y_pend1 <= 1'b1;
        end
        ycur <= ycur + {26'd0, q_b1, 2'b00};
      end

      // Y writeback rides the narrow port outside LD_X (never set
      // before the first row completes, so no arbitration with x).
      // The draining slot is LOCKED at its word 0 (y_sel); B words
      // per slot, one narrow beat each.
      if (y_pend0 || y_pend1) begin
        if (!m_valid) begin
          y_sel <= (y_wi != 4'd0) ? y_sel : !y_pend0;
          m_valid <= 1'b1; m_we <= 1'b1; m_be <= 4'hF;
          m_addr <= ((y_wi != 4'd0 ? y_sel : !y_pend0) ? y_addr1
                                                       : y_addr0)
                    + {26'd0, y_wi, 2'b00};
          m_wdata <= (y_wi != 4'd0 ? y_sel : !y_pend0)
                     ? 32'(y_data1[y_wi[2:0]])
                     : 32'(y_data0[y_wi[2:0]]);
        end else if (m_we && m_ack) begin
          m_valid <= 1'b0; m_we <= 1'b0;
          if (y_wi + 4'd1 == q_b1) begin
            y_wi <= '0;
            if (y_sel) y_pend1 <= 1'b0;
            else y_pend0 <= 1'b0;
          end else y_wi <= y_wi + 4'd1;
        end
      end
    end
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk) begin
    // the commit accounting must guarantee a slot at S3 handoff time
    if (v5_q && last5_q)
      assert (!y_pend0 || !y_pend1)
        else $fatal(1, "y writeback overran the pipeline");
    // the frozen v1 capacity bound (golden/gate enforce shapes)
    if (go && st == IDLE)
      assert (xb_w <= 16'd4096)
        else $fatal(1, "B*KPAD exceeds the 4KB x SRAM");
  end
`endif
endmodule
