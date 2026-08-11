// SG0 bridge (docs/FPGA_IMPLEMENTATION.md#sg0-fpga-bridge, D-registered 2026-07-17/18):
// the ONLY new hardware between the certified simt_soc and real K26
// silicon. Three quarters, all speaking the SoC's existing contracts:
//
//   1. AXI-Lite slave (PS M_AXI_HPM0): {CTRL, STATUS, MCYCLE, IMEM
//      load window}. On silicon the clock free-runs and every kernel
//      mailbox (PARAMS/DONE/ASSERT/LOG) lives in the DRAM carveout
//      the host mmaps — so control is reset/run + instrument reads,
//      nothing more. "RUN n" stepping is a sim-only concept.
//   2. imem BRAM: the core's imem contract is always-ready with
//      rdata the NEXT cycle — incompatible with variable-latency
//      DRAM, so instructions live in a preloaded BRAM (host loads
//      .text through the window before releasing run). The core is
//      untouched.
//   3. dmem -> AXI4 master: the dmem face is the v2 CREDIT face
//      (mem_spec §1b — req PULSE per beat, <=8 outstanding, acks in
//      order). An 8-deep FIFO absorbs the pulses; a one-outstanding
//      AXI FSM drains it in order. Correctness is latency-blind by
//      the face's own contract; bandwidth is the disclosed 32-bit
//      ceiling (SV2's wide face is a later leg).
module sg0_bridge #(
  parameter int unsigned IMEM_KW    = 32768,        // words (128 KB)
  parameter logic [31:0] CARVE_BASE = 32'h4000_0000, // carveout base
  localparam int unsigned IAW = $clog2(IMEM_KW)
)(
  input  logic        clk,
  input  logic        rst_n,          // board/PS reset
  // ---------------- AXI-Lite slave (control) ----------------
  input  logic        s_axil_awvalid,
  output logic        s_axil_awready,
  input  logic [7:0]  s_axil_awaddr,
  input  logic        s_axil_wvalid,
  output logic        s_axil_wready,
  input  logic [31:0] s_axil_wdata,
  input  logic [3:0]  s_axil_wstrb,
  output logic        s_axil_bvalid,
  input  logic        s_axil_bready,
  output logic [1:0]  s_axil_bresp,
  input  logic        s_axil_arvalid,
  output logic        s_axil_arready,
  input  logic [7:0]  s_axil_araddr,
  output logic        s_axil_rvalid,
  input  logic        s_axil_rready,
  output logic [31:0] s_axil_rdata,
  output logic [1:0]  s_axil_rresp,
  // ---------------- SoC faces ----------------
  output logic        soc_rst_n,      // rst_n & CTRL.run
  input  logic        imem_req,
  input  logic [31:0] imem_addr,
  output logic [31:0] imem_rdata,
  input  logic        dmem_req,
  input  logic        dmem_we,
  input  logic [3:0]  dmem_be,
  input  logic [31:0] dmem_addr,
  input  logic [31:0] dmem_wdata,
  output logic [31:0] dmem_rdata,
  output logic        dmem_ack,
  // ---------------- AXI4 master (carveout DRAM) ----------------
  output logic        m_axi_awvalid,
  input  logic        m_axi_awready,
  output logic [31:0] m_axi_awaddr,
  output logic [7:0]  m_axi_awlen,
  output logic [2:0]  m_axi_awsize,
  output logic [1:0]  m_axi_awburst,
  output logic        m_axi_wvalid,
  input  logic        m_axi_wready,
  output logic [31:0] m_axi_wdata,
  output logic [3:0]  m_axi_wstrb,
  output logic        m_axi_wlast,
  input  logic        m_axi_bvalid,
  output logic        m_axi_bready,
  input  logic [1:0]  m_axi_bresp,
  output logic        m_axi_arvalid,
  input  logic        m_axi_arready,
  output logic [31:0] m_axi_araddr,
  output logic [7:0]  m_axi_arlen,
  output logic [2:0]  m_axi_arsize,
  output logic [1:0]  m_axi_arburst,
  input  logic        m_axi_rvalid,
  output logic        m_axi_rready,
  input  logic [31:0] m_axi_rdata,
  input  logic        m_axi_rlast,
  input  logic [1:0]  m_axi_rresp
);
  localparam logic [15:0] MAGIC = 16'h05D0;
  localparam logic [7:0]  VERS  = 8'h01;

  // ---------------- control registers ----------------
  logic        run_q;
  logic [63:0] mcycle_q;
  logic [31:0] mcyc_hi_lat;           // HI latched on LO read
  logic [IAW-1:0] iw_addr, iw_wa;
  assign soc_rst_n = rst_n & run_q;

  // ---------------- imem BRAM ----------------
  logic [31:0] imem [IMEM_KW];
  logic        iw_we;
  logic [31:0] iw_data;
  always_ff @(posedge clk) begin
    imem_rdata <= imem[imem_addr[IAW+1:2]];  // core: rdata next cycle
    if (iw_we) imem[iw_wa] <= iw_data;
  end
  wire _unused_im = &{1'b0, imem_req, imem_addr[31:IAW+2], imem_addr[1:0]};

  // ---------------- AXI-Lite slave ----------------
  logic        aw_pend, w_pend;
  logic [7:0]  aw_addr_q;
  logic [31:0] w_data_q;
  logic [3:0]  w_strb_q;
  assign s_axil_bresp = 2'b00;
  assign s_axil_rresp = 2'b00;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      run_q <= 1'b0; mcycle_q <= '0; mcyc_hi_lat <= '0;
      iw_addr <= '0; iw_wa <= '0; iw_we <= 1'b0; iw_data <= '0;
      aw_pend <= 1'b0; w_pend <= 1'b0;
      aw_addr_q <= '0; w_data_q <= '0; w_strb_q <= '0;
      s_axil_awready <= 1'b0; s_axil_wready <= 1'b0;
      s_axil_bvalid <= 1'b0;
      s_axil_arready <= 1'b0; s_axil_rvalid <= 1'b0;
      s_axil_rdata <= '0;
    end else begin
      iw_we <= 1'b0;
      if (run_q) mcycle_q <= mcycle_q + 64'd1;

      // write channel: accept AW and W independently, commit when both
      s_axil_awready <= s_axil_awvalid && !aw_pend && !s_axil_bvalid
                        && !s_axil_awready;
      if (s_axil_awvalid && s_axil_awready) begin
        aw_addr_q <= s_axil_awaddr; aw_pend <= 1'b1;
      end
      s_axil_wready <= s_axil_wvalid && !w_pend && !s_axil_bvalid
                       && !s_axil_wready;
      if (s_axil_wvalid && s_axil_wready) begin
        w_data_q <= s_axil_wdata; w_strb_q <= s_axil_wstrb;
        w_pend <= 1'b1;
      end
      if (aw_pend && w_pend && !s_axil_bvalid) begin
        if (w_strb_q == 4'hF) begin
          unique case (aw_addr_q[7:2])
            6'h00: run_q <= w_data_q[0];             // CTRL
            6'h04: iw_addr <= w_data_q[IAW-1:0];     // IMEM_ADDR (0x10)
            6'h05: begin                             // IMEM_DATA (0x14)
              iw_we <= 1'b1; iw_data <= w_data_q;
              iw_wa <= iw_addr;                      // write THIS slot
              iw_addr <= iw_addr + IAW'(1);
            end
            default: ;
          endcase
        end
        aw_pend <= 1'b0; w_pend <= 1'b0;
        s_axil_bvalid <= 1'b1;
      end
      if (s_axil_bvalid && s_axil_bready) s_axil_bvalid <= 1'b0;

      // read channel
      s_axil_arready <= s_axil_arvalid && !s_axil_rvalid
                        && !s_axil_arready;
      if (s_axil_arvalid && s_axil_arready) begin
        unique case (s_axil_araddr[7:2])
          6'h00: s_axil_rdata <= {31'd0, run_q};
          6'h01: s_axil_rdata <= {MAGIC, VERS, 8'(IMEM_KW / 8192)};
          6'h02: begin                               // MCYCLE_LO (0x08)
            s_axil_rdata <= mcycle_q[31:0];
            mcyc_hi_lat  <= mcycle_q[63:32];
          end
          6'h03: s_axil_rdata <= mcyc_hi_lat;        // MCYCLE_HI (0x0C)
          6'h04: s_axil_rdata <= 32'(iw_addr);
          default: s_axil_rdata <= 32'hDEAD_BEEF;
        endcase
        s_axil_rvalid <= 1'b1;
      end
      if (s_axil_rvalid && s_axil_rready) s_axil_rvalid <= 1'b0;
    end
  end

  // ---------------- dmem credit face -> AXI4 master ----------------
  typedef struct packed {
    logic        we;
    logic [3:0]  be;
    logic [31:0] addr;
    logic [31:0] wdata;
  } dq_t;
  dq_t         dq [8];
  logic [3:0]  dq_wi, dq_ri;          // wrap-guard index pair
  wire  [3:0]  dq_occ = dq_wi - dq_ri;
  wire         dq_empty = (dq_occ == 4'd0);
  dq_t         cur;

  typedef enum logic [2:0] {D_IDLE, D_AR, D_R, D_AW, D_B} dst_e;
  dst_e dst;
  logic aw_done, w_done;

  assign m_axi_awlen = 8'd0;  assign m_axi_awsize = 3'b010;
  assign m_axi_awburst = 2'b01;
  assign m_axi_arlen = 8'd0;  assign m_axi_arsize = 3'b010;
  assign m_axi_arburst = 2'b01;
  assign m_axi_wlast = 1'b1;
  assign m_axi_bready = 1'b1;
  assign m_axi_rready = 1'b1;
  assign m_axi_awaddr = CARVE_BASE + cur.addr;
  assign m_axi_araddr = CARVE_BASE + cur.addr;
  assign m_axi_wdata  = cur.wdata;
  assign m_axi_wstrb  = cur.be;
  wire _unused_ax = &{1'b0, m_axi_bresp, m_axi_rresp, m_axi_rlast,
                      cur.we, s_axil_araddr[1:0], aw_addr_q[1:0]};

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dq_wi <= '0; dq_ri <= '0; dst <= D_IDLE;
      cur <= '0; aw_done <= 1'b0; w_done <= 1'b0;
      m_axi_awvalid <= 1'b0; m_axi_wvalid <= 1'b0;
      m_axi_arvalid <= 1'b0;
      dmem_ack <= 1'b0; dmem_rdata <= '0;
    end else begin
      dmem_ack <= 1'b0;
      // enqueue: the credit contract bounds occupancy at 8
      if (dmem_req) begin
        dq[dq_wi[2:0]] <= '{we: dmem_we, be: dmem_be,
                            addr: dmem_addr, wdata: dmem_wdata};
        dq_wi <= dq_wi + 4'd1;
      end
      unique case (dst)
        D_IDLE: if (!dq_empty) begin
          cur <= dq[dq_ri[2:0]];
          dq_ri <= dq_ri + 4'd1;
          if (dq[dq_ri[2:0]].we) begin
            m_axi_awvalid <= 1'b1; m_axi_wvalid <= 1'b1;
            aw_done <= 1'b0; w_done <= 1'b0;
            dst <= D_AW;
          end else begin
            m_axi_arvalid <= 1'b1;
            dst <= D_AR;
          end
        end
        D_AR: if (m_axi_arready) begin
          m_axi_arvalid <= 1'b0; dst <= D_R;
        end
        D_R: if (m_axi_rvalid) begin
          dmem_rdata <= m_axi_rdata; dmem_ack <= 1'b1;
          dst <= D_IDLE;
        end
        D_AW: begin
          if (m_axi_awready) begin m_axi_awvalid <= 1'b0; aw_done <= 1'b1; end
          if (m_axi_wready)  begin m_axi_wvalid  <= 1'b0; w_done  <= 1'b1; end
          if ((aw_done || m_axi_awready) && (w_done || m_axi_wready))
            dst <= D_B;
        end
        D_B: if (m_axi_bvalid) begin
          dmem_ack <= 1'b1;
          dst <= D_IDLE;
        end
        default: dst <= D_IDLE;
      endcase
    end
  end

`ifndef SYNTHESIS
  /* verilator lint_off SYNCASYNCNET */
  always_ff @(posedge clk) if (rst_n) begin
    // the credit contract: the face never overruns the 8-deep queue
    assert (dq_occ <= 4'd8) else $fatal(1, "SG0: dmem credit overrun");
  end
  /* verilator lint_on SYNCASYNCNET */
`endif
endmodule
