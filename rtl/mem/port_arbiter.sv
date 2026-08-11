// Three-requester memory-port arbiter, credit-face downstream (D-032c).
// Upstream port A (core) keeps the v1 LEVEL-valid/pulse-ack contract
// (mem_spec §1) — the last unmigrated master, adapted level -> edge
// internally (one outstanding, the infl flag). Ports B (tensor
// sidecar, stage C2) and C (s_cluster, stage C1) are MIGRATED v2
// EDGE-request/credit ports (mem_spec §1b): each req is a one-cycle
// pulse worth exactly one beat; up to CRED_B/CRED_C beats in flight.
// Pulses land in per-port skid FIFOs sized for the port's credit
// bound (the master's own credit counter is the overflow guard); an
// empty-FIFO pulse is granted combinationally so single-beat traffic
// keeps the v1 grant timing. The downstream face is the v2 contract:
// req pulses, strictly in-order responses, CRED beats in flight. The
// D-017 duplicate-beat race is structurally gone — the arbiter mints
// each downstream pulse itself. Responses route by a 2-bit order-tag
// FIFO, so each port sees its own beats complete strictly in its own
// issue order. Engines B and C are never concurrent by construction
// (one outstanding command globally, spec I-2); fairness stays the v1
// two-party alternation: core vs engine-side.
module port_arbiter #(
  parameter int unsigned CRED   = 1,     // downstream credits (1..4)
  parameter int unsigned CRED_B = 4,     // port-B beats in flight (1..4)
  parameter int unsigned CRED_C = 4      // port-C beats in flight (1..4)
) (
  input  logic        clk,
  input  logic        rst_n,
  // requester A (core) — v1 level-valid contract
  input  logic        a_valid,
  input  logic        a_we,
  input  logic [31:0] a_addr,
  input  logic [31:0] a_wdata,
  input  logic [3:0]  a_be,
  output logic        a_ack,
  output logic [31:0] a_rdata,
  // requester B (tensor sidecar) — v2 edge/credit contract
  input  logic        b_req,
  input  logic        b_we,
  input  logic [31:0] b_addr,
  input  logic [31:0] b_wdata,
  input  logic [3:0]  b_be,
  output logic        b_rsp_valid,
  output logic [31:0] b_rsp_rdata,
  // requester C (s_cluster) — v2 edge/credit contract
  input  logic        c_req,
  input  logic        c_we,
  input  logic [31:0] c_addr,
  input  logic [31:0] c_wdata,
  input  logic [3:0]  c_be,
  output logic        c_rsp_valid,
  output logic [31:0] c_rsp_rdata,
  // downstream memory face — v2 credit contract (mem_spec §1b)
  output logic        m_req,
  output logic        m_we,
  output logic [31:0] m_addr,
  output logic [31:0] m_wdata,
  output logic [3:0]  m_be,
  input  logic        m_rsp_valid,
  input  logic [31:0] m_rsp_rdata
);
  localparam logic [1:0] TAG_A = 2'd0, TAG_B = 2'd1, TAG_C = 2'd2;

  // in-flight order FIFO: which upstream port owns each outstanding
  // beat, oldest at head. Depth 4 covers CRED max.
  logic [1:0] tag_q [4];
  logic [2:0] t_head, t_tail;           // circular, 3b guards wrap
  logic [2:0] credits;
  logic       a_infl;                   // legacy port: <=1
  logic       last_e;                   // last grant went engine-side

  // per-port skid FIFOs ({we, be, addr, wdata}); each master's credit
  // bound (<= 4) is the overflow guard
  logic [68:0] bf_q [4];
  logic [2:0]  bf_head, bf_tail;
  wire  [1:0]  bf_hidx = bf_head[1:0];
  wire  [1:0]  bf_tidx = bf_tail[1:0];
  wire         bf_empty = (bf_head == bf_tail);
  logic [68:0] cf_q [4];
  logic [2:0]  cf_head, cf_tail;
  wire  [1:0]  cf_hidx = cf_head[1:0];
  wire  [1:0]  cf_tidx = cf_tail[1:0];
  wire         cf_empty = (cf_head == cf_tail);

  wire [1:0] t_hidx = t_head[1:0];
  wire [1:0] t_tidx = t_tail[1:0];
  wire       have_credit = (credits != 3'd0);

  // grant decision (combinational, issued registered next edge).
  // Engine side picks B over C deterministically (never concurrent
  // in the SoC); a live pulse bypasses only past an empty FIFO so
  // per-port order is preserved.
  wire a_req_ok = a_valid && !a_infl;
  wire b_pend   = !bf_empty || b_req;
  wire c_pend   = !cf_empty || c_req;
  wire e_req_ok = b_pend || c_pend;
  wire grant_a  = have_credit && a_req_ok && (!e_req_ok || last_e);
  wire grant_e  = have_credit && e_req_ok && !grant_a;
  wire grant_b  = grant_e && b_pend;
  wire grant_c  = grant_e && !b_pend;

  // issue sources: FIFO head unless empty (live-pulse bypass)
  wire [68:0] bf_hd    = bf_q[bf_hidx];
  wire        gb_we    = bf_empty ? b_we    : bf_hd[68];
  wire [3:0]  gb_be    = bf_empty ? b_be    : bf_hd[67:64];
  wire [31:0] gb_addr  = bf_empty ? b_addr  : bf_hd[63:32];
  wire [31:0] gb_wdata = bf_empty ? b_wdata : bf_hd[31:0];
  wire [68:0] cf_hd    = cf_q[cf_hidx];
  wire        gc_we    = cf_empty ? c_we    : cf_hd[68];
  wire [3:0]  gc_be    = cf_empty ? c_be    : cf_hd[67:64];
  wire [31:0] gc_addr  = cf_empty ? c_addr  : cf_hd[63:32];
  wire [31:0] gc_wdata = cf_empty ? c_wdata : cf_hd[31:0];

  // response routing: head-of-order tag owns this rsp beat
  wire [1:0] rtag = tag_q[t_hidx];
  assign a_ack       = m_rsp_valid && (rtag == TAG_A);
  assign b_rsp_valid = m_rsp_valid && (rtag == TAG_B);
  assign c_rsp_valid = m_rsp_valid && (rtag == TAG_C);
  assign a_rdata     = m_rsp_rdata;
  assign b_rsp_rdata = m_rsp_rdata;
  assign c_rsp_rdata = m_rsp_rdata;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m_req  <= 1'b0;
      m_we   <= 1'b0;
      m_addr <= '0;
      m_wdata<= '0;
      m_be   <= '0;
      t_head <= '0;
      t_tail <= '0;
      bf_head<= '0;
      bf_tail<= '0;
      cf_head<= '0;
      cf_tail<= '0;
      credits<= 3'(CRED);
      a_infl <= 1'b0;
      last_e <= 1'b1;                   // A first after reset
    end else begin
      // issue (one-cycle pulse per beat)
      m_req <= grant_a || grant_b || grant_c;
      if (grant_a || grant_b || grant_c) begin
        m_we    <= grant_c ? gc_we    : (grant_b ? gb_we    : a_we);
        m_addr  <= grant_c ? gc_addr  : (grant_b ? gb_addr  : a_addr);
        m_wdata <= grant_c ? gc_wdata : (grant_b ? gb_wdata : a_wdata);
        m_be    <= grant_c ? gc_be    : (grant_b ? gb_be    : a_be);
        tag_q[t_tidx] <= grant_c ? TAG_C : (grant_b ? TAG_B : TAG_A);
        t_tail  <= t_tail + 3'd1;
        if (grant_a) a_infl <= 1'b1;
        last_e  <= grant_e;
      end
      // skid maintenance: a live pulse not consumed this cycle
      // queues; a granted head retires
      if (b_req && !(grant_b && bf_empty)) begin
        bf_q[bf_tidx] <= {b_we, b_be, b_addr, b_wdata};
        bf_tail <= bf_tail + 3'd1;
      end
      if (grant_b && !bf_empty)
        bf_head <= bf_head + 3'd1;
      if (c_req && !(grant_c && cf_empty)) begin
        cf_q[cf_tidx] <= {c_we, c_be, c_addr, c_wdata};
        cf_tail <= cf_tail + 3'd1;
      end
      if (grant_c && !cf_empty)
        cf_head <= cf_head + 3'd1;
      // completion (issue+completion same cycle is legal: the order
      // FIFO's head/tail advance independently; the legacy port
      // cannot issue and complete in one cycle at <=1 outstanding)
      if (m_rsp_valid) begin
        t_head <= t_head + 3'd1;
        if (rtag == TAG_A) a_infl <= 1'b0;
        // alternating priority, v1 semantics: after an engine beat
        // completes, prefer A only if A is not waiting (and vice versa)
        if (rtag != TAG_A) last_e <= !a_valid;
        else               last_e <= b_pend || c_pend;
      end
      case ({(grant_a || grant_b || grant_c), m_rsp_valid})
        2'b10:   credits <= credits - 3'd1;
        2'b01:   credits <= credits + 3'd1;
        default: ;                      // both or neither: unchanged
      endcase
`ifndef SYNTHESIS
      // tripwires: the contracts that make the FIFOs un-overflowable
      assert (!(b_req && (bf_tail - bf_head) >= 3'(CRED_B)))
        else $fatal(1, "port_arbiter: port-B pulse beyond CRED_B");
      assert (!(c_req && (cf_tail - cf_head) >= 3'(CRED_C)))
        else $fatal(1, "port_arbiter: port-C pulse beyond CRED_C");
      assert (!((grant_a || grant_b || grant_c)
                && (t_tail - t_head) >= 3'd4))
        else $fatal(1, "port_arbiter: order-tag FIFO overflow");
      assert (!(m_rsp_valid && (t_head == t_tail)))
        else $fatal(1, "port_arbiter: response with no beat in flight");
`endif
    end
  end
endmodule
