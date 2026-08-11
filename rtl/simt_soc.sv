// SoC top (M18): simt_core + tensor sidecar behind the D4 socket, sharing
// one memory face through the round-robin port arbiter. This is the top
// the M17 harness builds; the M21 S-cluster becomes a third arbiter
// requester in this same shape.
module simt_soc #(
  parameter logic [31:0] RESET_PC = 32'h0000_0000,
  parameter int unsigned PROFILE = 0,   // §1c: 0 union, 1 S, 2 P, 3 T
  //                                       (S builds no sampling domain)
  parameter string LUT_DIR = "",
  parameter string SIG_LUT = "sigmoid_lut.mem",
  parameter string Q_LUT = "glut.mem",
  parameter int unsigned SAMP_NB = 13   // sampler site bits: 13 = the
  //  architectural ceiling (sim); device configs pass 10 (D-023/PR1)
) (
  input  logic        clk,
  input  logic        rst_n,
  output logic        imem_req,
  output logic [31:0] imem_addr,
  input  logic [31:0] imem_rdata,
  output logic        dmem_req,
  output logic        dmem_we,
  output logic [31:0] dmem_addr,
  output logic [31:0] dmem_wdata,
  output logic [3:0]  dmem_be,
  input  logic [31:0] dmem_rdata,
  input  logic        dmem_ack,
  // commit buses (passthrough for lockstep/counters)
  output logic        cmt_valid,
  output logic [2:0]  cmt_warp,
  output logic [31:0] cmt_pc,
  output logic [31:0] cmt_instr,
  output logic [7:0]  cmt_mask,
  output logic [4:0]  cmt_rd,
  output logic [7:0][31:0] cmt_wdata,
  output logic        mcmt_valid,
  output logic [2:0]  mcmt_warp,
  output logic [31:0] mcmt_pc,
  output logic [31:0] mcmt_instr,
  output logic [7:0]  mcmt_mask,
  output logic [4:0]  mcmt_rd,
  output logic [7:0][31:0] mcmt_wdata,
  output logic [31:0] tensor_ops,
  output logic [31:0] sampler_ops,
  output logic        tensor_busy
);
  // core <-> arbiter A
  logic        c_valid, c_we, c_ack;
  logic [31:0] c_addr, c_wdata, c_rdata;
  logic [3:0]  c_be;
  // tensor sidecar <-> arbiter B (v2 edge/credit, D-032c stage C2)
  logic        tv_req, tv_we, tv_busy, tv_rsp;
  logic [31:0] tv_addr, tv_wdata, tv_rdata;
  logic [3:0]  tv_be;
  // s_cluster <-> arbiter C (v2 edge/credit, D-032c leg C). The
  // engines are never concurrent (one outstanding command globally),
  // but each now owns its arbiter port — the mux died with the
  // level-valid contract.
  logic        pv_req, pv_we, pv_busy, pv_rsp;
  logic [31:0] pv_addr, pv_wdata, pv_rdata;
  logic [3:0]  pv_be;
  // socket (widths per docs/HARDWARE_ARCHITECTURE.md#sampling-isa §1 superset)
  logic [3:0]  t_op;
  logic [31:0] t_a, t_b, t_c;
  logic [23:0] t_m;
  logic [6:0]  t_n;
  logic [7:0]  t_k;
  logic [8:0]  t_flags;
  logic        t_go, t_busy;
  logic [2:0]  drain_warp_nc;
  logic [2:0]  drain_lane_nc;

  simt_core #(.RESET_PC(RESET_PC), .PROFILE(PROFILE)) u_core (
    .clk(clk), .rst_n(rst_n),
    .imem_req(imem_req), .imem_addr(imem_addr), .imem_rdata(imem_rdata),
    .dmem_req(c_valid), .dmem_we(c_we), .dmem_be(c_be),
    .dmem_addr(c_addr), .dmem_wdata(c_wdata), .dmem_rdata(c_rdata),
    .dmem_ack(c_ack),
    .cmt_valid(cmt_valid), .cmt_warp(cmt_warp), .cmt_pc(cmt_pc),
    .cmt_instr(cmt_instr), .cmt_mask(cmt_mask), .cmt_rd(cmt_rd),
    .cmt_wdata(cmt_wdata),
    .mcmt_valid(mcmt_valid), .mcmt_warp(mcmt_warp), .mcmt_pc(mcmt_pc),
    .mcmt_instr(mcmt_instr), .mcmt_mask(mcmt_mask), .mcmt_rd(mcmt_rd),
    .mcmt_wdata(mcmt_wdata),
    .drain_warp(drain_warp_nc), .drain_lane(drain_lane_nc),
    .t_op(t_op), .t_a(t_a), .t_b(t_b), .t_c(t_c),
    .t_m(t_m), .t_n(t_n), .t_k(t_k), .t_flags(t_flags),
    .t_go(t_go), .t_busy(t_busy)
  );
  assign t_busy = tv_busy | pv_busy;     // T_STATUS.busy (one engine slot)
  assign tensor_busy = t_busy;
  wire _unused = &{1'b0, drain_warp_nc, drain_lane_nc};

  // GO routes by T_OP[3]: 0-7 tensor, 8-10 S-cluster (spec §1)
  tensor_sidecar #(.CRED(4), .LUT_DIR(LUT_DIR)) u_tensor (
    .clk(clk), .rst_n(rst_n),
    .go(t_go & ~t_op[3]), .op(t_op[2:0]),
    .addr_a(t_a), .addr_b(t_b), .addr_c(t_c),
    .dim_m(t_m[6:0]), .dim_n(t_n), .dim_k(t_k[6:0]),
    .flag_acc(t_flags[0]),
    .busy(tv_busy),
    .m_req(tv_req), .m_we(tv_we), .m_addr(tv_addr),
    .m_wdata(tv_wdata), .m_be(tv_be), .m_rsp_valid(tv_rsp),
    .m_rdata(tv_rdata),
    .cnt_ops(tensor_ops)
  );

  generate if (PROFILE != 1) begin : g_samp
    s_cluster #(.NB(SAMP_NB), .CRED(4), .LUT_FILE(SIG_LUT),
                .GLUT_FILE(Q_LUT)) u_sampler (
      .clk(clk), .rst_n(rst_n),
      .go(t_go & t_op[3]), .op(t_op[2:0]),
      .t_a(t_a), .t_c(t_c), .t_m(t_m), .t_k(t_k), .t_flags(t_flags),
      .busy(pv_busy),
      .m_req(pv_req), .m_we(pv_we), .m_addr(pv_addr),
      .m_wdata(pv_wdata), .m_be(pv_be), .m_rsp_valid(pv_rsp),
      .m_rdata(pv_rdata),
      .cnt_sops(sampler_ops)
    );
  end else begin : g_nosamp
    // Profile S (§1c): no sampling domain. GO at T_OP 8-10 already
    // traps in the core; these tie-offs are structural only.
    assign pv_req = 1'b0;
    assign pv_we = 1'b0;
    assign pv_addr = '0;
    assign pv_wdata = '0;
    assign pv_be = '0;
    assign pv_busy = 1'b0;
    assign sampler_ops = 32'd0;
    wire _unused_ns = &{1'b0, pv_rsp, pv_rdata};
  end endgenerate

  logic        x_req, x_we;
  logic [31:0] x_addr, x_wdata;
  logic [3:0]  x_be;
  logic        x_rsp_valid;
  logic [31:0] x_rsp_rdata;

  port_arbiter #(.CRED(4), .CRED_B(4), .CRED_C(4)) u_arb (
    .clk(clk), .rst_n(rst_n),
    .a_valid(c_valid), .a_we(c_we), .a_addr(c_addr), .a_wdata(c_wdata),
    .a_be(c_be), .a_ack(c_ack), .a_rdata(c_rdata),
    .b_req(tv_req), .b_we(tv_we), .b_addr(tv_addr),
    .b_wdata(tv_wdata), .b_be(tv_be), .b_rsp_valid(tv_rsp),
    .b_rsp_rdata(tv_rdata),
    .c_req(pv_req), .c_we(pv_we), .c_addr(pv_addr), .c_wdata(pv_wdata),
    .c_be(pv_be), .c_rsp_valid(pv_rsp), .c_rsp_rdata(pv_rdata),
    .m_req(x_req), .m_we(x_we), .m_addr(x_addr),
    .m_wdata(x_wdata), .m_be(x_be), .m_rsp_valid(x_rsp_valid),
    .m_rsp_rdata(x_rsp_rdata)
  );

  // D-032b (docs/FPGA_IMPLEMENTATION.md#fast-path-optimizations leg B): shared on-chip scratchpad.
  // An 8KB BRAM window at SCRATCH_BASE behind the arbiter funnel —
  // one decoder, every master (core, both engines) gets zero-copy
  // engine<->engine and core<->engine handoffs. To software it is
  // just memory (goldens/ISS see a flat space); only latency differs.
  //
  // D-032c fork: the arbiter's downstream is the v2 credit face
  // (mem_spec §1b); SCRATCH and external memory are two sub-slaves.
  // The fork owns cross-sub-slave ordering: a 4-deep order FIFO
  // records each issued beat's sub-slave; completions fire upstream
  // strictly head-first, with same-cycle bypasses so CRED=1 keeps
  // the v1 latency shape exactly. A younger fast SCRATCH beat's
  // response is HELD (data FIFO) until older external beats retire.
  localparam logic [31:0] SCRATCH_BASE = 32'hF000_0000;
  wire sc_hit = (x_addr[31:13] == SCRATCH_BASE[31:13]);
  (* ram_style = "block" *) logic [31:0] scratch [2048];
  logic [31:0] sc_q;
  logic        sc_done;                 // sc beat completed (pulse)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) sc_done <= 1'b0;
    else        sc_done <= x_req && sc_hit;
  end
  always_ff @(posedge clk) begin
    if (x_req && sc_hit && x_we) begin
      if (x_be[0]) scratch[x_addr[12:2]][7:0]   <= x_wdata[7:0];
      if (x_be[1]) scratch[x_addr[12:2]][15:8]  <= x_wdata[15:8];
      if (x_be[2]) scratch[x_addr[12:2]][23:16] <= x_wdata[23:16];
      if (x_be[3]) scratch[x_addr[12:2]][31:24] <= x_wdata[31:24];
    end
    sc_q <= scratch[x_addr[12:2]];
  end

  // order FIFO (issue order; 1 = scratch beat)
  logic [3:0] ord_q;
  logic [2:0] o_head, o_tail;
  wire  [1:0] o_hidx = o_head[1:0];
  wire        ord_empty = (o_head == o_tail);
  wire        head_is_sc = ord_q[o_hidx];
  // completed-but-unfired response data, per sub-slave (issue order)
  logic [31:0] scf_q [4];
  logic [2:0]  scf_w, scf_r;
  wire         scf_empty = (scf_w == scf_r);
  logic [31:0] exf_q [4];
  logic [2:0]  exf_w, exf_r;
  wire         exf_empty = (exf_w == exf_r);

  // head-first fire, with same-cycle bypass when the head's own
  // completion arrives and nothing older of its kind is buffered
  wire sc_fire  = !ord_empty && head_is_sc &&
                  (!scf_empty || sc_done);
  wire ext_fire = !ord_empty && !head_is_sc &&
                  (!exf_empty || dmem_ack);
  assign x_rsp_valid = sc_fire || ext_fire;
  assign x_rsp_rdata = sc_fire
      ? (scf_empty ? sc_q : scf_q[scf_r[1:0]])
      : (exf_empty ? dmem_rdata : exf_q[exf_r[1:0]]);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      o_head <= '0; o_tail <= '0;
      scf_w <= '0; scf_r <= '0;
      exf_w <= '0; exf_r <= '0;
    end else begin
      if (x_req) begin
        ord_q[o_tail[1:0]] <= sc_hit;
        o_tail <= o_tail + 3'd1;
      end
      // buffer completions not consumed this cycle
      if (sc_done && !(sc_fire && scf_empty)) begin
        scf_q[scf_w[1:0]] <= sc_q;
        scf_w <= scf_w + 3'd1;
      end
      if (dmem_ack && !(ext_fire && exf_empty)) begin
        exf_q[exf_w[1:0]] <= dmem_rdata;
        exf_w <= exf_w + 3'd1;
      end
      if (sc_fire) begin
        o_head <= o_head + 3'd1;
        if (!scf_empty) scf_r <= scf_r + 3'd1;
      end else if (ext_fire) begin
        o_head <= o_head + 3'd1;
        if (!exf_empty) exf_r <= exf_r + 3'd1;
      end
    end
  end

  assign dmem_req   = x_req & ~sc_hit;
  assign dmem_we    = x_we;
  assign dmem_addr  = x_addr;
  assign dmem_wdata = x_wdata;
  assign dmem_be    = x_be;
endmodule
