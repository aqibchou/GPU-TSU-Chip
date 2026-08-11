// SIMT core (M14): the M13 barrel widened to L=8 lanes per warp — shared
// PC/instruction, per-lane regfiles/ALUs, per-warp active mask + max-PC
// reconvergence stack, lane-serial memory unit behind the busy bit.
// Spec: docs/HARDWARE_ARCHITECTURE.md#simt-core §5 (frozen 2026-07-07). Bit-true golden:
// golden/simt_iss.py (the reconvergence loop here MUST stay in sync with
// SimtWarp.step()). Non-memory instructions retire in WB; memory
// instructions leave the pipe (set busy), drain one lane per cycle on the
// single dmem port, then complete via the memory unit (pc update + busy
// clear + mem-commit record) — the two retirement paths are warp-disjoint
// by construction (a draining warp is busy, hence not in the pipe).
module simt_core #(
  parameter int W = 8,
  parameter int L = 8,
  parameter logic [31:0] RESET_PC = 32'h0000_0000,
  parameter int unsigned QDEPTH = 2,   // T-op command queue (§1b)
  parameter int unsigned PROFILE = 0,  // §1c (D-036): 0 union, 1 S, 2 P, 3 T
  localparam int WB_ = $clog2(W),
  localparam int LB_ = $clog2(L)
) (
  input  logic        clk,
  input  logic        rst_n,
  // imem: req this cycle -> rdata next cycle, always ready
  output logic        imem_req,
  output logic [31:0] imem_addr,
  input  logic [31:0] imem_rdata,
  // dmem: owned exclusively by the memory unit (INV-12 v2)
  output logic        dmem_req,
  output logic        dmem_we,
  output logic [3:0]  dmem_be,
  output logic [31:0] dmem_addr,
  output logic [31:0] dmem_wdata,
  input  logic [31:0] dmem_rdata,
  input  logic        dmem_ack,
  // WB commit (non-memory retirements)
  output logic              cmt_valid,
  output logic [WB_-1:0]    cmt_warp,
  output logic [31:0]       cmt_pc,
  output logic [31:0]       cmt_instr,
  output logic [L-1:0]      cmt_mask,
  output logic [4:0]        cmt_rd,
  output logic [L-1:0][31:0] cmt_wdata,
  // MEM commit (loads/stores complete here; stores also observable on the
  // dmem port in lane order during the drain)
  output logic              mcmt_valid,
  output logic [WB_-1:0]    mcmt_warp,
  output logic [31:0]       mcmt_pc,
  output logic [31:0]       mcmt_instr,
  output logic [L-1:0]      mcmt_mask,
  output logic [4:0]        mcmt_rd,
  output logic [L-1:0][31:0] mcmt_wdata,
  // drain observability for the bench (which warp/lane owns each dmem beat)
  output logic [WB_-1:0]    drain_warp,
  output logic [LB_-1:0]    drain_lane,
  // D4 tensor socket (docs/HARDWARE_ARCHITECTURE.md#tensor-sidecar-and-d4-socket §1): global command latches.
  // Widths carry the M21 sampling ISA superset (docs/HARDWARE_ARCHITECTURE.md#sampling-isa
  // §1): T_OP 8-10, T_M image-words/sweeps u24, T_K beta u8, T_FLAGS 9b.
  output logic [3:0]        t_op,
  output logic [31:0]       t_a,
  output logic [31:0]       t_b,
  output logic [31:0]       t_c,
  output logic [23:0]       t_m,
  output logic [6:0]        t_n,
  output logic [7:0]        t_k,
  output logic [8:0]        t_flags,
  output logic              t_go,        // 1-cycle doorbell pulse
  input  logic              t_busy       // tensor_busy | s_busy at the SoC
);
  localparam logic [31:0] MSTATUS_WMASK = 32'h0000_1888;

  // D-032d (tensor_spec §1b): the T-op command queue. The staging
  // CSRs (ts_*) are snapshotted into a QDEPTH-deep record queue at
  // GO; the DISPATCH loads the engine-facing output registers from
  // the HEAD RECORD, never the staging registers (rewrite-after-GO
  // is legal). go_pend covers the dispatch-to-busy-rise window so
  // T_STATUS busy never blinks low mid-command. The EX trap check
  // (tq_full) races a same-cycle WB enqueue only across DISTINCT
  // issuing warps — out of contract (one issuing hart, mk.h); a
  // single issuer's next GO is >= one rotation away, post-commit.
  localparam int unsigned QW = $clog2(QDEPTH) + 1;
  logic [3:0]  ts_op;
  logic [31:0] ts_a, ts_b, ts_c;
  logic [23:0] ts_m;
  logic [6:0]  ts_n;
  logic [7:0]  ts_k;
  logic [8:0]  ts_flags;
  logic [147:0] tq [QDEPTH];
  logic [QW-1:0] tq_wi, tq_ri;        // index + wrap guard bit
  logic          go_pend;
  wire  [QW-1:0] tq_occ  = tq_wi - tq_ri;
  wire           tq_empty = (tq_occ == '0);
  wire           tq_full  = (tq_occ == QW'(QDEPTH));
  localparam int DEPTH = 5;
  localparam int STK = 8;

  // ---------------- scheduler ----------------
  logic          issue_valid;
  logic [WB_-1:0] issue_warp;
  logic [W-1:0]  busy /*verilator public_flat_rd*/;  // leg-A pilot tap
  logic          set_busy;
  logic [WB_-1:0] set_warp;
  logic          clr_busy;
  logic [WB_-1:0] clr_warp;
  // pipe convention (see barrel_core): D=sv[0] EX=sv[1] M=sv[2] WB=sv[3]
  logic [DEPTH-1:0]          sv;
  logic [DEPTH-1:0][WB_-1:0] sw;
  barrel_sched #(.W(W), .DEPTH(DEPTH)) u_sched (
    .clk(clk), .rst_n(rst_n),
    .set_busy(set_busy), .set_warp(set_warp), .hold(walk_pend),
    .clr_busy(clr_busy), .clr_warp(clr_warp),
    .issue_valid(issue_valid), .issue_warp(issue_warp), .busy(busy),
    .stage_valid(sv), .stage_warp(sw)
  );
  wire _unused = &{1'b0, busy, sv[4], sv[1:0], sw[4], sw[2], sw[0]};

  // ---------------- per-warp architectural state ----------------
  logic [31:0] pc      [W] /*verilator public_flat_rd*/;  // leg-A pilot tap
  logic [L-1:0] amask  [W];
  logic [31:0] mstatus [W], mie [W], mtvec [W], mscratch [W];
  logic [31:0] mepc    [W], mcause [W], mtval [W];
  // divergence stack
  logic [L-1:0]  stk_other [W][STK];
  logic [31:0]   stk_rst   [W][STK];
  logic [L-1:0]  stk_join  [W][STK];
  logic [31:0]   stk_rcv   [W][STK];
  logic          stk_pend  [W][STK];
  logic [3:0]    stk_top   [W];       // number of live entries
  // D-037 reconvergence WALKER: the retired STK-chained combinational
  // pop loops (both retire sites) were the chip's critical path — 42-54
  // logic levels, ~19-20ns routed (PR1 2026-07-18). Retire now commits
  // the base npc/mask and raises walk_pend; a single walker does ONE
  // check-or-pop per cycle and the warp is fetch-held (u_sched hold)
  // until its walk clears. Values are bit-identical — the walk replays
  // the exact loop semantics (D-016 pend-arm re-check included) — and
  // the common no-pop case clears inside the barrel rotation slack
  // (retire+1 check, +2 visible, next F slot at +3), so steady-state
  // throughput is unchanged; multi-pop merges cost that warp extra
  // rotations (D-026: cycles are not contract). A pushed (wb_div)
  // retire does NOT walk, matching the original no-check-after-push.
  logic [W-1:0]  walk_pend;

  // ---------------- F ----------------
  assign imem_req  = issue_valid;
  assign imem_addr = pc[issue_warp];

  // ---------------- D: regfile read ----------------
  logic [31:0] d_pc;
  wire  [31:0] d_ir = imem_rdata;
  logic [L-1:0][31:0] rs1_d, rs2_d;
  simt_regfile #(.W(W), .L(L)) u_rf (
    .clk(clk),
    .r_warp(sw[0]), .rs1(d_ir[19:15]), .rs2(d_ir[24:20]),
    .rs1_data(rs1_d), .rs2_data(rs2_d),
    .a_en(wb_rf_we), .a_warp(sw[3]), .a_mask(wb_mask_q),
    .a_rd(wb_ir[11:7]), .a_data(wb_res),
    .b_en(mu_rf_we), .b_warp(mu_warp), .b_lane(mu_lane),
    .b_rd(mu_rd), .b_data(mu_ld_val)
  );

  // ---------------- EX ----------------
  logic [31:0] ex_pc, ex_ir;
  logic [L-1:0][31:0] ex_a, ex_b;
  logic [L-1:0] ex_mask;

  wire [6:0] opc  = ex_ir[6:0];
  wire [2:0] f3   = ex_ir[14:12];
  wire [4:0] rs1f = ex_ir[19:15];
  wire [6:0] f7   = ex_ir[31:25];
  wire [4:0] shamt = ex_ir[24:20];
  wire [11:0] csr_a = ex_ir[31:20];
  wire [WB_-1:0] ex_w = sw[1];

  wire [31:0] imm_i = {{20{ex_ir[31]}}, ex_ir[31:20]};
  wire [31:0] imm_s = {{20{ex_ir[31]}}, ex_ir[31:25], ex_ir[11:7]};
  wire [31:0] imm_b = {{19{ex_ir[31]}}, ex_ir[31], ex_ir[7], ex_ir[30:25], ex_ir[11:8], 1'b0};
  wire [31:0] imm_u = {ex_ir[31:12], 12'b0};
  wire [31:0] imm_j = {{11{ex_ir[31]}}, ex_ir[31], ex_ir[19:12], ex_ir[20], ex_ir[30:21], 1'b0};

  function automatic logic csr_dummy(input logic [11:0] adr);
    return (adr == 12'h310) || (adr == 12'h344) || (adr == 12'h320)
        || (adr == 12'hB00) || (adr == 12'hB02) || (adr == 12'hB80) || (adr == 12'hB82)
        || (adr >= 12'h3A0 && adr <= 12'h3A3)
        || (adr >= 12'h3B0 && adr <= 12'h3BF);
  endfunction
  function automatic logic csr_ro(input logic [11:0] adr);
    return (adr == 12'hF11) || (adr == 12'hF12) || (adr == 12'hF13)
        || (adr == 12'hF14) || (adr == 12'hF15) || (adr == 12'h301)
        || (adr == 12'h8CA);               // T_PROFILE (§1c, read-only)
  endfunction

  // §1c (D-036): per-profile T_OP presence. GO at an absent op traps
  // mcause 2 at the GO site — the same fault family and trap point as
  // the §1b FULL trap. Union (0) = all present: every pre-profile
  // build is the id-0 case bit-for-bit.
  localparam logic [10:0] OP_PRESENT =
      (PROFILE == 1) ? 11'b000_1111_1111 :             // S: 0-7
      (PROFILE == 2 || PROFILE == 3) ? 11'b111_0111_1111 : // P/T: 0-6,8-10
      11'b111_1111_1111;                               // union
  function automatic logic csr_stored(input logic [11:0] adr);
    return (adr == 12'h300) || (adr == 12'h304) || (adr == 12'h305)
        || (adr == 12'h340) || (adr == 12'h341) || (adr == 12'h342) || (adr == 12'h343);
  endfunction
  function automatic logic csr_tensor(input logic [11:0] adr);
    return (adr >= 12'h8C0) && (adr <= 12'h8C9);
  endfunction

  // free-running cycle counter (mcycle/mcycleh; S7.4 kernel timing).
  // Reads are warp-shared (uniform); writes fall in the dummy class.
  logic [63:0] mcycle_q;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) mcycle_q <= '0;
    else        mcycle_q <= mcycle_q + 64'd1;

  logic [31:0] csr_rval_shared;
  always_comb begin
    unique case (csr_a)
      12'h300: csr_rval_shared = mstatus[ex_w];
      12'h304: csr_rval_shared = mie[ex_w];
      12'h305: csr_rval_shared = mtvec[ex_w];
      12'h340: csr_rval_shared = mscratch[ex_w];
      12'h341: csr_rval_shared = mepc[ex_w];
      12'h342: csr_rval_shared = mcause[ex_w];
      12'h343: csr_rval_shared = mtval[ex_w];
      12'hB00: csr_rval_shared = mcycle_q[31:0];
      12'hB80: csr_rval_shared = mcycle_q[63:32];
      12'h301: csr_rval_shared = 32'h4000_0100;
      12'h8C0: csr_rval_shared = {28'd0, ts_op};
      12'h8C1: csr_rval_shared = ts_a;
      12'h8C2: csr_rval_shared = ts_b;
      12'h8C3: csr_rval_shared = ts_c;
      12'h8C4: csr_rval_shared = {8'd0, ts_m};
      12'h8C5: csr_rval_shared = {25'd0, ts_n};
      12'h8C6: csr_rval_shared = {24'd0, ts_k};
      12'h8C7: csr_rval_shared = {23'd0, ts_flags};
      // §1b: bit0 busy = engine | queue | dispatch window; bit1 FULL
      12'h8C9: csr_rval_shared =
          {30'd0, tq_full, t_busy | ~tq_empty | go_pend};
      12'h8CA: csr_rval_shared = 32'(PROFILE);  // §1c T_PROFILE
      default: csr_rval_shared = 32'd0;
    endcase
  end

  // per-lane EX results
  logic [L-1:0][31:0] ex_res;      // wb value per lane
  logic [L-1:0]       ex_tk;       // branch taken per lane
  logic [L-1:0][31:0] ex_addr;     // mem address per lane
  logic        ex_illegal, ex_trap, ex_wb_en;
  logic [31:0] ex_cause, ex_tval, ex_next_pc;
  logic        ex_is_load, ex_is_store, ex_csr_we, ex_is_mret;
  logic [31:0] ex_csr_wval;
  logic [L-1:0] ex_push_other;
  logic         ex_diverge;

  wire [31:0] st_mask32 = (f3[1:0] == 2'd0) ? 32'h0000_00FF
                        : (f3[1:0] == 2'd1) ? 32'h0000_FFFF : 32'hFFFF_FFFF;

  always_comb begin
    logic [31:0] tgt;
    logic [L-1:0] taken, ntaken;
    logic misalign;
    logic [31:0] mis_addr;
    logic jalr_div;
    logic [31:0] jalr_tgt0;
    logic csr_wr, csr_div;
    logic [31:0] csr_src0;

    ex_illegal = 1'b0; ex_trap = 1'b0; ex_cause = 32'd0; ex_tval = 32'd0;
    ex_wb_en = 1'b0; ex_next_pc = ex_pc + 32'd4;
    ex_is_load = 1'b0; ex_is_store = 1'b0;
    ex_csr_we = 1'b0; ex_csr_wval = 32'd0; ex_is_mret = 1'b0;
    ex_diverge = 1'b0; ex_push_other = '0;
    for (int l = 0; l < L; l++) begin
      ex_res[l] = 32'd0; ex_tk[l] = 1'b0; ex_addr[l] = 32'd0;
    end
    taken = '0; ntaken = '0; tgt = 32'd0;
    misalign = 1'b0; mis_addr = 32'd0;
    jalr_div = 1'b0; jalr_tgt0 = 32'd0;
    csr_wr = 1'b0; csr_div = 1'b0; csr_src0 = 32'd0;

    unique case (opc)
      7'b0110111: begin ex_wb_en = 1'b1;
        for (int l = 0; l < L; l++) ex_res[l] = imm_u;
      end
      7'b0010111: begin ex_wb_en = 1'b1;
        for (int l = 0; l < L; l++) ex_res[l] = ex_pc + imm_u;
      end
      7'b1101111: begin                                          // JAL
        tgt = ex_pc + imm_j;
        if (tgt[1:0] != 2'b00) begin
          ex_trap = 1'b1; ex_cause = 32'd0; ex_tval = tgt;
        end else begin
          ex_wb_en = 1'b1; ex_next_pc = tgt;
          for (int l = 0; l < L; l++) ex_res[l] = ex_pc + 32'd4;
        end
      end
      7'b1100111: begin                                          // JALR
        if (f3 != 3'b000) ex_illegal = 1'b1;
        else begin
          jalr_tgt0 = 32'd0;
          begin
            automatic logic found = 1'b0;
            for (int l = 0; l < L; l++) begin
              if (ex_mask[l]) begin
                automatic logic [31:0] t = (ex_a[l] + imm_i) & 32'hFFFF_FFFE;
                if (!found) begin jalr_tgt0 = t; found = 1'b1; end
                else if (t != jalr_tgt0) jalr_div = 1'b1;
              end
            end
          end
          if (jalr_div) ex_illegal = 1'b1;                       // v1 limit
          else if (jalr_tgt0[1:0] != 2'b00) begin
            ex_trap = 1'b1; ex_cause = 32'd0; ex_tval = jalr_tgt0;
          end else begin
            ex_wb_en = 1'b1; ex_next_pc = jalr_tgt0;
            for (int l = 0; l < L; l++) ex_res[l] = ex_pc + 32'd4;
          end
        end
      end
      7'b1100011: begin                                          // branches
        if (f3 == 3'b010 || f3 == 3'b011) ex_illegal = 1'b1;
        else begin
          for (int l = 0; l < L; l++) begin
            unique case (f3)
              3'b000: ex_tk[l] = (ex_a[l] == ex_b[l]);
              3'b001: ex_tk[l] = (ex_a[l] != ex_b[l]);
              3'b100: ex_tk[l] = ($signed(ex_a[l]) < $signed(ex_b[l]));
              3'b101: ex_tk[l] = ($signed(ex_a[l]) >= $signed(ex_b[l]));
              3'b110: ex_tk[l] = (ex_a[l] < ex_b[l]);
              default: ex_tk[l] = (ex_a[l] >= ex_b[l]);
            endcase
          end
          taken  = ex_tk & ex_mask;
          ntaken = ex_mask & ~taken;
          tgt    = ex_pc + imm_b;
          if (taken != '0 && tgt[1:0] != 2'b00) begin
            ex_trap = 1'b1; ex_cause = 32'd0; ex_tval = tgt;
          end else if (taken == '0) ;                            // uniform NT
          else if (ntaken == '0) ex_next_pc = tgt;               // uniform T
          else if (tgt <= ex_pc) ex_illegal = 1'b1;              // v1 limit
          else if (stk_top[ex_w] >= 4'(STK)) ex_illegal = 1'b1;  // overflow
          else begin
            ex_diverge = 1'b1;                                   // push@WB
            ex_push_other = taken;
            ex_next_pc = ex_pc + 32'd4;                          // NT first
          end
        end
      end
      7'b0000011: begin                                          // loads
        if (f3 == 3'b011 || f3 == 3'b110 || f3 == 3'b111) ex_illegal = 1'b1;
        else begin
          for (int l = 0; l < L; l++) begin
            ex_addr[l] = ex_a[l] + imm_i;
            if (ex_mask[l] && !misalign) begin
              if ((f3[1:0] == 2'd1 && ex_addr[l][0]) ||
                  (f3[1:0] == 2'd2 && ex_addr[l][1:0] != 2'b00)) begin
                misalign = 1'b1; mis_addr = ex_addr[l];
              end
            end
          end
          if (misalign) begin
            ex_trap = 1'b1; ex_cause = 32'd4; ex_tval = mis_addr;
          end else ex_is_load = 1'b1;
        end
      end
      7'b0100011: begin                                          // stores
        if (f3 > 3'b010) ex_illegal = 1'b1;
        else begin
          for (int l = 0; l < L; l++) begin
            ex_addr[l] = ex_a[l] + imm_s;
            if (ex_mask[l] && !misalign) begin
              if ((f3[1:0] == 2'd1 && ex_addr[l][0]) ||
                  (f3[1:0] == 2'd2 && ex_addr[l][1:0] != 2'b00)) begin
                misalign = 1'b1; mis_addr = ex_addr[l];
              end
            end
          end
          if (misalign) begin
            ex_trap = 1'b1; ex_cause = 32'd6; ex_tval = mis_addr;
          end else ex_is_store = 1'b1;
        end
      end
      7'b0010011: begin                                          // OP-IMM
        ex_wb_en = 1'b1;
        for (int l = 0; l < L; l++) begin
          unique case (f3)
            3'b000: ex_res[l] = ex_a[l] + imm_i;
            3'b010: ex_res[l] = {31'd0, $signed(ex_a[l]) < $signed(imm_i)};
            3'b011: ex_res[l] = {31'd0, ex_a[l] < imm_i};
            3'b100: ex_res[l] = ex_a[l] ^ imm_i;
            3'b110: ex_res[l] = ex_a[l] | imm_i;
            3'b111: ex_res[l] = ex_a[l] & imm_i;
            3'b001: ex_res[l] = ex_a[l] << shamt;
            default: ex_res[l] = (f7 == 7'h20)
                     ? $unsigned($signed(ex_a[l]) >>> shamt)
                     : ex_a[l] >> shamt;
          endcase
        end
        if (f3 == 3'b001 && f7 != 7'd0) begin
          ex_illegal = 1'b1; ex_wb_en = 1'b0;
        end
        if (f3 == 3'b101 && f7 != 7'd0 && f7 != 7'h20) begin
          ex_illegal = 1'b1; ex_wb_en = 1'b0;
        end
      end
      7'b0110011: begin                                          // OP
        ex_wb_en = 1'b1;
        for (int l = 0; l < L; l++) begin
          unique case ({f7, f3})
            {7'd0,  3'b000}: ex_res[l] = ex_a[l] + ex_b[l];
            {7'h20, 3'b000}: ex_res[l] = ex_a[l] - ex_b[l];
            {7'd0,  3'b001}: ex_res[l] = ex_a[l] << ex_b[l][4:0];
            {7'd0,  3'b010}: ex_res[l] = {31'd0, $signed(ex_a[l]) < $signed(ex_b[l])};
            {7'd0,  3'b011}: ex_res[l] = {31'd0, ex_a[l] < ex_b[l]};
            {7'd0,  3'b100}: ex_res[l] = ex_a[l] ^ ex_b[l];
            {7'd0,  3'b101}: ex_res[l] = ex_a[l] >> ex_b[l][4:0];
            {7'h20, 3'b101}: ex_res[l] = $unsigned($signed(ex_a[l]) >>> ex_b[l][4:0]);
            {7'd0,  3'b110}: ex_res[l] = ex_a[l] | ex_b[l];
            {7'd0,  3'b111}: ex_res[l] = ex_a[l] & ex_b[l];
            default: ex_res[l] = 32'd0;
          endcase
        end
        if (!(f7 inside {7'd0, 7'h20}) ||
            (f7 == 7'h20 && !(f3 inside {3'b000, 3'b101}))) begin
          ex_illegal = 1'b1; ex_wb_en = 1'b0;
        end
      end
      7'b0001111: if (f3 > 3'b001) ex_illegal = 1'b1;            // FENCE
      7'b1110011: begin                                          // SYSTEM
        if (f3 == 3'b000) begin
          unique case (ex_ir)
            32'h0000_0073: begin ex_trap = 1'b1; ex_cause = 32'd11; end
            32'h0010_0073: begin ex_trap = 1'b1; ex_cause = 32'd3; ex_tval = ex_pc; end
            32'h3020_0073: begin ex_is_mret = 1'b1; ex_next_pc = mepc[ex_w]; end
            default: ex_illegal = 1'b1;
          endcase
        end else if (f3 == 3'b100) ex_illegal = 1'b1;
        else begin                                               // Zicsr
          csr_wr = (f3[1:0] == 2'b01) || (rs1f != 5'd0);
          if (!(csr_stored(csr_a) || csr_tensor(csr_a) ||
                csr_ro(csr_a) || csr_dummy(csr_a)))
            ex_illegal = 1'b1;
          else if (csr_wr && csr_ro(csr_a))
            ex_illegal = 1'b1;
          else begin
            ex_wb_en = 1'b1;
            for (int l = 0; l < L; l++)
              ex_res[l] = (csr_a == 12'hF14)
                          ? 32'({ex_w, LB_'(l)})                 // global tid
                          : csr_rval_shared;
            if (csr_wr && (csr_stored(csr_a) || csr_tensor(csr_a))) begin
              // divergent write source -> trap (v1 limit)
              csr_src0 = 32'd0;
              begin
                automatic logic found = 1'b0;
                for (int l = 0; l < L; l++) begin
                  if (ex_mask[l]) begin
                    automatic logic [31:0] src =
                      f3[2] ? {27'd0, rs1f} : ex_a[l];
                    if (!found) begin csr_src0 = src; found = 1'b1; end
                    else if (src != csr_src0) csr_div = 1'b1;
                  end
                end
              end
              if (csr_div) begin ex_illegal = 1'b1; ex_wb_en = 1'b0; end
              else if (csr_a == 12'h8C8 && tq_full) begin
                ex_illegal = 1'b1; ex_wb_en = 1'b0;   // GO at FULL (§1b)
              end
              else if (csr_a == 12'h8C8 && ts_op <= 4'd10
                       && !OP_PRESENT[ts_op]) begin
                ex_illegal = 1'b1; ex_wb_en = 1'b0;   // absent op (§1c)
              end
              else begin
                ex_csr_we = 1'b1;
                unique case (f3[1:0])
                  2'b01:   ex_csr_wval = csr_src0;
                  2'b10:   ex_csr_wval = csr_rval_shared | csr_src0;
                  default: ex_csr_wval = csr_rval_shared & ~csr_src0;
                endcase
              end
            end
          end
        end
      end
      default: ex_illegal = 1'b1;
    endcase

    if (ex_illegal) begin
      ex_trap = 1'b1; ex_cause = 32'd2; ex_tval = ex_ir;
      ex_wb_en = 1'b0; ex_is_load = 1'b0; ex_is_store = 1'b0;
      ex_csr_we = 1'b0; ex_is_mret = 1'b0; ex_diverge = 1'b0;
      ex_next_pc = ex_pc + 32'd4;
    end
  end

  logic [31:0] csr_wval_legal;
  always_comb begin
    csr_wval_legal = ex_csr_wval;
    if (csr_a == 12'h300)      csr_wval_legal = ex_csr_wval & MSTATUS_WMASK;
    else if (csr_a == 12'h305) csr_wval_legal = ex_csr_wval & 32'hFFFF_FFFC;
    else if (csr_a == 12'h341) csr_wval_legal = ex_csr_wval & 32'hFFFF_FFFC;
  end

  // ---------------- M: hand mem ops to the memory unit ----------------
  logic [31:0] m_pc, m_ir, m_next_pc;
  logic [L-1:0][31:0] m_res, m_addr, m_stdat;
  logic [L-1:0] m_mask;
  logic m_trap, m_wb_en, m_is_load, m_is_store, m_csr_we, m_is_mret, m_div;
  logic [L-1:0] m_push_other;
  logic [31:0] m_cause, m_tval, m_csr_wval;
  logic [11:0] m_csr_a;

  wire m_is_mem = sv[2] && (m_is_load || m_is_store) && !m_trap;
  assign set_busy = m_is_mem;
  assign set_warp = sw[2];

  // ---------------- memory unit (M16: coalescing drain) ----------------
  // queue of pending warp memory ops (depth W: <=1 per warp via busy)
  typedef struct packed {
    logic [WB_-1:0]      warp;
    logic                is_store;
    logic [1:0]          size;
    logic [2:0]          f3;
    logic [4:0]          rd;
    logic [L-1:0]        mask;
    logic [31:0]         pc;
    logic [31:0]         ir;
  } muq_t;
  muq_t          muq [W];
  logic [L-1:0][31:0] muq_addr [W];
  logic [L-1:0][31:0] muq_wdat [W];
  logic [WB_:0]  mu_head, mu_tail;
  wire mu_empty = (mu_head == mu_tail);

  typedef enum logic [2:0] {MU_IDLE, MU_PICK, MU_REQ, MU_WLB, MU_DONE} mu_e;
  mu_e mu_st;
  muq_t mu_cur;
  logic [L-1:0][31:0] mu_addrs, mu_wdats;
  logic [L-1:0]       mu_srvcd;          // lanes fully serviced
  logic [L-1:0]       mu_share;          // lanes of the in-flight word
  logic [LB_-1:0]     mu_leader;
  logic [LB_-1:0]     mu_wlb;            // lane being written back
  logic [31:0]        mu_word;           // latched load word
  logic [L-1:0][31:0] mu_ldres;

  // leader = lowest enabled un-serviced lane (docs/HARDWARE_ARCHITECTURE.md#memory-hierarchy §4 order)
  logic [LB_-1:0] pick_leader;
  logic           pick_any;
  always_comb begin
    pick_leader = '0;
    pick_any = 1'b0;
    for (int l = L - 1; l >= 0; l--)
      if (mu_cur.mask[l] && !mu_srvcd[l]) begin
        pick_leader = LB_'(l);
        pick_any = 1'b1;
      end
  end
  wire [31:0] leader_word = mu_addrs[pick_leader] & ~32'd3;
  logic [L-1:0] share_c;
  always_comb
    for (int l = 0; l < L; l++)
      share_c[l] = mu_cur.mask[l] && !mu_srvcd[l] &&
                   ((mu_addrs[l] & ~32'd3) == leader_word);

  // store merge: lane-ascending byte precedence (== golden coalescer)
  logic [31:0] merge_wdata;
  logic [3:0]  merge_be;
  always_comb begin
    merge_wdata = '0;
    merge_be = '0;
    for (int l = 0; l < L; l++) begin
      if (share_c[l]) begin
        for (int b = 0; b < 4; b++) begin
          if (b < (32'd1 << mu_cur.size)) begin
            automatic logic [1:0] pos = mu_addrs[l][1:0] + 2'(b);
            merge_wdata[8*pos +: 8] = mu_wdats[l][8*b +: 8];
            merge_be[pos] = 1'b1;
          end
        end
      end
    end
  end

  // dmem drive: level-valid until ack, only in MU_REQ (INV-12 v2)
  assign dmem_req   = (mu_st == MU_REQ);
  assign dmem_we    = mu_cur.is_store;
  assign dmem_addr  = leader_word;
  assign dmem_wdata = merge_wdata;
  assign dmem_be    = mu_cur.is_store ? merge_be : 4'd0;
  assign drain_warp = mu_cur.warp;
  assign drain_lane = mu_leader;
  wire _unused_mu = &{1'b0, mu_leader};

  // per-lane load extraction from the latched word
  wire [31:0] ld_shift = mu_word >> {mu_addrs[mu_wlb][1:0], 3'b000};
  logic [31:0] mu_ld_val_c;
  always_comb begin
    unique case (mu_cur.f3)
      3'b000:  mu_ld_val_c = {{24{ld_shift[7]}},  ld_shift[7:0]};
      3'b001:  mu_ld_val_c = {{16{ld_shift[15]}}, ld_shift[15:0]};
      3'b100:  mu_ld_val_c = {24'd0, ld_shift[7:0]};
      3'b101:  mu_ld_val_c = {16'd0, ld_shift[15:0]};
      default: mu_ld_val_c = ld_shift;
    endcase
  end
  // next share lane to write back (lowest un-serviced share)
  logic [LB_-1:0] wlb_next;
  logic           wlb_any;
  always_comb begin
    wlb_next = '0;
    wlb_any = 1'b0;
    for (int l = L - 1; l >= 0; l--)
      if (mu_share[l] && !mu_srvcd[l]) begin
        wlb_next = LB_'(l);
        wlb_any = 1'b1;
      end
  end
  // regfile port B
  wire            mu_rf_we  = (mu_st == MU_WLB) && !mu_cur.is_store;
  wire [WB_-1:0]  mu_warp   = mu_cur.warp;
  wire [LB_-1:0]  mu_lane   = mu_wlb;
  wire [4:0]      mu_rd     = mu_cur.rd;
  wire [31:0]     mu_ld_val = mu_ld_val_c;

  // ---------------- WB ----------------
  logic [31:0] wb_pc, wb_ir, wb_next_pc;
  logic [L-1:0][31:0] wb_res;
  logic [L-1:0] wb_mask_q, wb_push_other;
  logic wb_trap, wb_en_q, wb_csr_we, wb_is_mret, wb_div, wb_to_mem;
  logic [31:0] wb_cause, wb_tval, wb_csr_wval;
  logic [11:0] wb_csr_a;

  wire        wb_v     = sv[3] && !wb_to_mem;
  wire [WB_-1:0] wb_w  = sw[3];
  wire        wb_rf_we = wb_v && !wb_trap && wb_en_q;

  // ---------------- state update ----------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      d_pc <= '0; ex_pc <= '0; ex_ir <= '0; ex_mask <= '0;
      for (int l = 0; l < L; l++) begin ex_a[l] <= '0; ex_b[l] <= '0; end
      m_pc <= '0; m_ir <= '0; m_next_pc <= '0; m_mask <= '0;
      m_trap <= 1'b0; m_wb_en <= 1'b0; m_is_load <= 1'b0;
      m_is_store <= 1'b0; m_csr_we <= 1'b0; m_is_mret <= 1'b0;
      m_div <= 1'b0; m_push_other <= '0; m_cause <= '0; m_tval <= '0;
      m_csr_wval <= '0; m_csr_a <= '0;
      wb_pc <= '0; wb_ir <= '0; wb_next_pc <= '0; wb_mask_q <= '0;
      wb_trap <= 1'b0; wb_en_q <= 1'b0; wb_csr_we <= 1'b0;
      wb_is_mret <= 1'b0; wb_div <= 1'b0; wb_to_mem <= 1'b0;
      wb_push_other <= '0; wb_cause <= '0; wb_tval <= '0;
      wb_csr_wval <= '0; wb_csr_a <= '0;
      cmt_valid <= 1'b0; cmt_warp <= '0; cmt_pc <= '0; cmt_instr <= '0;
      cmt_mask <= '0; cmt_rd <= '0;
      mcmt_valid <= 1'b0; mcmt_warp <= '0; mcmt_pc <= '0; mcmt_instr <= '0;
      mcmt_mask <= '0; mcmt_rd <= '0;
      t_op <= '0; t_a <= '0; t_b <= '0; t_c <= '0;
      t_m <= '0; t_n <= '0; t_k <= '0; t_flags <= '0; t_go <= 1'b0;
      ts_op <= '0; ts_a <= '0; ts_b <= '0; ts_c <= '0;
      ts_m <= '0; ts_n <= '0; ts_k <= '0; ts_flags <= '0;
      tq_wi <= '0; tq_ri <= '0; go_pend <= 1'b0;
      mu_head <= '0; mu_tail <= '0; mu_st <= MU_IDLE;
      mu_srvcd <= '0; mu_share <= '0; mu_leader <= '0;
      mu_wlb <= '0; mu_word <= '0;
      clr_busy <= 1'b0; clr_warp <= '0;
      walk_pend <= '0;
      for (int w = 0; w < W; w++) begin
        pc[w] <= RESET_PC; amask[w] <= {L{1'b1}};
        mstatus[w] <= '0; mie[w] <= '0; mtvec[w] <= '0; mscratch[w] <= '0;
        mepc[w] <= '0; mcause[w] <= '0; mtval[w] <= '0;
        stk_top[w] <= '0;
      end
    end else begin
      cmt_valid <= 1'b0;
      mcmt_valid <= 1'b0;
      clr_busy <= 1'b0;
      t_go <= 1'b0;

      // D-032d dispatch: head record -> engine when the slot frees;
      // go_pend holds busy high across the dispatch-to-busy window
      if (t_busy) go_pend <= 1'b0;
      if (!tq_empty && !t_busy && !go_pend) begin
        {t_op, t_a, t_b, t_c, t_m, t_n, t_k, t_flags}
            <= tq[tq_ri[QW-2:0]];
        t_go <= 1'b1;
        go_pend <= 1'b1;
        tq_ri <= tq_ri + QW'(1);
      end

      // F -> D -> EX
      d_pc <= pc[issue_warp];
      ex_pc <= d_pc; ex_ir <= d_ir;
      ex_mask <= amask[sw[0]];
      for (int l = 0; l < L; l++) begin
        ex_a[l] <= rs1_d[l]; ex_b[l] <= rs2_d[l];
      end
      // EX -> M
      m_pc <= ex_pc; m_ir <= ex_ir; m_next_pc <= ex_next_pc;
      m_mask <= ex_mask; m_res <= ex_res; m_addr <= ex_addr;
      for (int l = 0; l < L; l++) m_stdat[l] <= ex_b[l] & st_mask32;
      m_trap <= ex_trap; m_wb_en <= ex_wb_en;
      m_is_load <= ex_is_load; m_is_store <= ex_is_store;
      m_csr_we <= ex_csr_we; m_is_mret <= ex_is_mret;
      m_div <= ex_diverge; m_push_other <= ex_push_other;
      m_cause <= ex_cause; m_tval <= ex_tval;
      m_csr_wval <= csr_wval_legal; m_csr_a <= csr_a;
      // M -> WB (mem ops leave the pipe here)
      wb_pc <= m_pc; wb_ir <= m_ir; wb_next_pc <= m_next_pc;
      wb_res <= m_res; wb_mask_q <= m_mask;
      wb_trap <= m_trap; wb_en_q <= m_wb_en;
      wb_csr_we <= m_csr_we; wb_is_mret <= m_is_mret;
      wb_div <= m_div; wb_push_other <= m_push_other;
      wb_to_mem <= m_is_mem;
      wb_cause <= m_cause; wb_tval <= m_tval;
      wb_csr_wval <= m_csr_wval; wb_csr_a <= m_csr_a;

      // memory-unit enqueue (at M)
      if (m_is_mem) begin
        muq[mu_tail[WB_-1:0]] <= '{warp: sw[2], is_store: m_is_store,
                                   size: m_ir[13:12], f3: m_ir[14:12],
                                   rd: m_ir[11:7], mask: m_mask,
                                   pc: m_pc, ir: m_ir};
        muq_addr[mu_tail[WB_-1:0]] <= m_addr;
        muq_wdat[mu_tail[WB_-1:0]] <= m_stdat;
        mu_tail <= mu_tail + 1'b1;
      end

      // memory-unit FSM (coalescing drain per mem_spec §4)
      unique case (mu_st)
        MU_IDLE: if (!mu_empty) begin
          mu_cur   <= muq[mu_head[WB_-1:0]];
          mu_addrs <= muq_addr[mu_head[WB_-1:0]];
          mu_wdats <= muq_wdat[mu_head[WB_-1:0]];
          mu_head  <= mu_head + 1'b1;
          mu_srvcd <= '0;
          mu_st    <= MU_PICK;
        end
        MU_PICK: begin
          if (!pick_any) mu_st <= MU_DONE;
          else begin
            mu_leader <= pick_leader;
            mu_share  <= share_c;
            mu_st     <= MU_REQ;
          end
        end
        MU_REQ: if (dmem_ack) begin
          if (mu_cur.is_store) begin
            mu_srvcd <= mu_srvcd | mu_share;
            mu_st    <= MU_PICK;
          end else begin
            mu_word <= dmem_rdata;
            mu_st   <= MU_WLB;
          end
        end
        MU_WLB: begin
          // port B writes mu_wlb's lane this cycle (mu_rf_we comb)
          mu_ldres[mu_wlb] <= mu_ld_val_c;
          mu_srvcd[mu_wlb] <= 1'b1;
          if (!wlb_any) mu_st <= MU_PICK;   // this was the last share lane
        end
        MU_DONE: begin
          // D-037: base commit; the walker performs any pops. The
          // single inline check (one compare — the retired loop's
          // first iteration condition, exactly) arms the walk only
          // when a merge is actually due, so the walker carries no
          // steady-state load. (amask[warp] already equals
          // mu_cur.mask — INV-4 — so no mask write here.)
          begin
            automatic logic [3:0] mtop;
            mtop = stk_top[mu_cur.warp];
            if (mtop != 4'd0 &&
                (mu_cur.pc + 32'd4) >= stk_rcv[mu_cur.warp][3'(mtop - 4'd1)])
              walk_pend[mu_cur.warp] <= 1'b1;
          end
          pc[mu_cur.warp] <= mu_cur.pc + 32'd4;
          clr_busy <= 1'b1;
          clr_warp <= mu_cur.warp;
          mcmt_valid <= 1'b1;
          mcmt_warp  <= mu_cur.warp;
          mcmt_pc    <= mu_cur.pc;
          mcmt_instr <= mu_cur.ir;
          mcmt_mask  <= mu_cur.mask;
          mcmt_rd    <= mu_cur.is_store ? 5'd0 : mu_cur.rd;
          mcmt_wdata <= mu_ldres;
          mu_st <= MU_IDLE;
        end
        default: mu_st <= MU_IDLE;
      endcase
      // WLB lane pointer: latch entry into MU_WLB and advance per cycle
      if (mu_st == MU_REQ && dmem_ack && !mu_cur.is_store)
        mu_wlb <= wlb_next;
      else if (mu_st == MU_WLB && wlb_any)
        mu_wlb <= wlb_next;

      // WB: architectural updates for non-mem instructions
      if (wb_v) begin
        if (wb_trap) begin
          mepc[wb_w]   <= wb_pc;
          mcause[wb_w] <= wb_cause;
          mtval[wb_w]  <= wb_tval;
          mstatus[wb_w] <= ({mstatus[wb_w][31:8], 1'b0, mstatus[wb_w][6:4],
                             1'b0, mstatus[wb_w][2:0]} & ~32'h0000_1880)
                          | {28'd0, mstatus[wb_w][3], 3'd0} << 4
                          | 32'h0000_1800;
          pc[wb_w] <= mtvec[wb_w] & 32'hFFFF_FFFC;
        end else begin
          automatic logic [3:0] top;
          if (wb_is_mret)
            mstatus[wb_w] <= ({28'd0, mstatus[wb_w][7], 3'd0}
                              | 32'h0000_0080) & MSTATUS_WMASK;
          if (wb_csr_we) begin
            unique case (wb_csr_a)
              12'h8C0: ts_op <= wb_csr_wval[3:0];
              12'h8C1: ts_a <= wb_csr_wval;
              12'h8C2: ts_b <= wb_csr_wval;
              12'h8C3: ts_c <= wb_csr_wval;
              12'h8C4: ts_m <= wb_csr_wval[23:0];
              12'h8C5: ts_n <= wb_csr_wval[6:0];
              12'h8C6: ts_k <= wb_csr_wval[7:0];
              12'h8C7: ts_flags <= wb_csr_wval[8:0];
              12'h8C8: if (wb_csr_wval[0]) begin   // §1b: enqueue
                tq[tq_wi[QW-2:0]] <= {ts_op, ts_a, ts_b, ts_c,
                                      ts_m, ts_n, ts_k, ts_flags};
                tq_wi <= tq_wi + QW'(1);
              end
              12'h300: mstatus[wb_w]  <= wb_csr_wval;
              12'h304: mie[wb_w]      <= wb_csr_wval;
              12'h305: mtvec[wb_w]    <= wb_csr_wval;
              12'h340: mscratch[wb_w] <= wb_csr_wval;
              12'h341: mepc[wb_w]     <= wb_csr_wval;
              12'h342: mcause[wb_w]   <= wb_csr_wval;
              12'h343: mtval[wb_w]    <= wb_csr_wval;
              default: ;
            endcase
          end
          top = stk_top[wb_w];
          if (wb_div) begin                              // divergence push
            automatic logic [2:0] pi;
            pi = top[2:0];
            stk_other[wb_w][pi] <= wb_push_other;
            stk_rst[wb_w][pi]   <= wb_pc + {{19{wb_ir[31]}}, wb_ir[31],
                                            wb_ir[7], wb_ir[30:25],
                                            wb_ir[11:8], 1'b0};
            stk_join[wb_w][pi]  <= wb_mask_q;
            stk_rcv[wb_w][pi]   <= wb_pc + {{19{wb_ir[31]}}, wb_ir[31],
                                            wb_ir[7], wb_ir[30:25],
                                            wb_ir[11:8], 1'b0};
            stk_pend[wb_w][pi]  <= 1'b1;
            stk_top[wb_w] <= top + 4'd1;
            amask[wb_w] <= wb_mask_q & ~wb_push_other;
            // no walk after a push (original checked only at the
            // NEXT retire — preserves the empty-arm trajectory)
          end else begin
            // D-037: arm the walker only when the retired loop's
            // first-iteration condition holds (one compare; merges
            // are rare, so the walker carries no steady-state load).
            // Walker semantics stay in sync with golden simt_iss.
            if (top != 4'd0 &&
                wb_next_pc >= stk_rcv[wb_w][3'(top - 4'd1)])
              walk_pend[wb_w] <= 1'b1;
          end
          pc[wb_w]      <= wb_next_pc;
          cmt_valid <= 1'b1;
          cmt_warp  <= wb_w;
          cmt_pc    <= wb_pc;
          cmt_instr <= wb_ir;
          cmt_mask  <= wb_mask_q;
          cmt_rd    <= wb_rf_we ? wb_ir[11:7] : 5'd0;
          cmt_wdata <= wb_res;
        end
      end

      // ---- D-037 reconvergence walker: ONE check-or-pop per cycle ----
      // Commits here, AFTER the retire arms — a walking warp is never
      // in the pipe (its fetch is held), and a retiring warp is, so the
      // two never touch the same warp's state in one cycle. The check
      // replays the retired loop exactly: pend arm switches to the
      // other side (rcv := arrival, top unchanged — the D-016 re-check
      // happens next cycle against the updated rcv); join arm merges
      // (mask := join, top decrements); no hit ends the walk.
      begin
        automatic logic [WB_-1:0] p;
        automatic logic [3:0] wtop;
        automatic logic [2:0] wi;
        automatic logic found;
        found = 1'b0; p = '0;
        for (int w = 0; w < W; w++)
          if (!found && walk_pend[w]) begin
            found = 1'b1; p = WB_'(w);
          end
        if (found) begin
          wtop = stk_top[p];
          wi = 3'(wtop - 4'd1);
          if (wtop != 4'd0 && pc[p] >= stk_rcv[p][wi]) begin
            if (stk_pend[p][wi]) begin
              stk_pend[p][wi] <= 1'b0;
              stk_rcv[p][wi]  <= pc[p];       // arrival
              amask[p] <= stk_other[p][wi];
              pc[p]    <= stk_rst[p][wi];
            end else begin
              amask[p] <= stk_join[p][wi];
              stk_top[p] <= wtop - 4'd1;
            end
          end else begin
            walk_pend[p] <= 1'b0;
          end
        end
      end
    end
  end

`ifndef SYNTHESIS
  /* verilator lint_off SYNCASYNCNET */
  always_ff @(posedge clk) if (rst_n) begin
    // INV-15: the in-flight word's lanes are enabled and unserviced
    if (dmem_req) assert ((mu_share & ~mu_cur.mask) == '0 && mu_share != '0)
      else $fatal(1, "INV-15: share set escapes the active mask");
    // INV-16: stack depth bound
    for (int w = 0; w < W; w++)
      assert (stk_top[w] <= 4'(STK))
        else $fatal(1, "INV-16: stack overflow warp %0d", w);
  end
  /* verilator lint_on SYNCASYNCNET */
`endif
endmodule
