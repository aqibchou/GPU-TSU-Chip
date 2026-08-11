// SG0 top (docs/FPGA_IMPLEMENTATION.md#sg0-fpga-bridge): the bitstream's RTL payload.
// PLAIN VERILOG on purpose: Vivado's bd module-reference refuses a
// SystemVerilog top file (filemgmt 56-195); children stay SV. —
// the certified simt_soc behind the certified sg0_bridge, nothing
// else. Pure wiring: the bridge owns reset (CTRL.run), imem (BRAM +
// load window), and the dmem credit face -> AXI4 master; the SoC is
// byte-identical to the battery-certified build. Observability
// outputs (cmt/mcmt, op counters) dangle and prune in synthesis —
// on silicon the referees are the DRAM-resident mailboxes, exactly
// as in sim.
module sg0_top #(
  parameter PROFILE    = 0,
  parameter SAMP_NB    = 10,     // D-023 device config
  parameter IMEM_KW    = 32768,
  parameter [31:0] CARVE_BASE = 32'h4000_0000
)(
  input  wire        clk,            // PS pl_clk0 @ 125 MHz
  input  wire        rst_n,          // PS peripheral reset
  // AXI-Lite slave (PS M_AXI_HPM0) — bridge control
  input  wire        s_axil_awvalid,
  output wire        s_axil_awready,
  input  wire [7:0]  s_axil_awaddr,
  input  wire        s_axil_wvalid,
  output wire        s_axil_wready,
  input  wire [31:0] s_axil_wdata,
  input  wire [3:0]  s_axil_wstrb,
  output wire        s_axil_bvalid,
  input  wire        s_axil_bready,
  output wire [1:0]  s_axil_bresp,
  input  wire        s_axil_arvalid,
  output wire        s_axil_arready,
  input  wire [7:0]  s_axil_araddr,
  output wire        s_axil_rvalid,
  input  wire        s_axil_rready,
  output wire [31:0] s_axil_rdata,
  output wire [1:0]  s_axil_rresp,
  // AXI4 master (PS S_AXI_HP0) — carveout DRAM
  output wire        m_axi_awvalid,
  input  wire        m_axi_awready,
  output wire [31:0] m_axi_awaddr,
  output wire [7:0]  m_axi_awlen,
  output wire [2:0]  m_axi_awsize,
  output wire [1:0]  m_axi_awburst,
  output wire        m_axi_wvalid,
  input  wire        m_axi_wready,
  output wire [31:0] m_axi_wdata,
  output wire [3:0]  m_axi_wstrb,
  output wire        m_axi_wlast,
  input  wire        m_axi_bvalid,
  output wire        m_axi_bready,
  input  wire [1:0]  m_axi_bresp,
  output wire        m_axi_arvalid,
  input  wire        m_axi_arready,
  output wire [31:0] m_axi_araddr,
  output wire [7:0]  m_axi_arlen,
  output wire [2:0]  m_axi_arsize,
  output wire [1:0]  m_axi_arburst,
  input  wire        m_axi_rvalid,
  output wire        m_axi_rready,
  input  wire [31:0] m_axi_rdata,
  input  wire        m_axi_rlast,
  input  wire [1:0]  m_axi_rresp
);
  wire        soc_rst_n;
  wire        imem_req;
  wire [31:0] imem_addr, imem_rdata;
  wire        dmem_req, dmem_we, dmem_ack;
  wire [3:0]  dmem_be;
  wire [31:0] dmem_addr, dmem_wdata, dmem_rdata;

  sg0_bridge #(
    .IMEM_KW(IMEM_KW), .CARVE_BASE(CARVE_BASE)
  ) u_bridge (
    .clk(clk), .rst_n(rst_n),
    .s_axil_awvalid(s_axil_awvalid), .s_axil_awready(s_axil_awready),
    .s_axil_awaddr(s_axil_awaddr),
    .s_axil_wvalid(s_axil_wvalid), .s_axil_wready(s_axil_wready),
    .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb),
    .s_axil_bvalid(s_axil_bvalid), .s_axil_bready(s_axil_bready),
    .s_axil_bresp(s_axil_bresp),
    .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready),
    .s_axil_araddr(s_axil_araddr),
    .s_axil_rvalid(s_axil_rvalid), .s_axil_rready(s_axil_rready),
    .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp),
    .soc_rst_n(soc_rst_n),
    .imem_req(imem_req), .imem_addr(imem_addr), .imem_rdata(imem_rdata),
    .dmem_req(dmem_req), .dmem_we(dmem_we), .dmem_be(dmem_be),
    .dmem_addr(dmem_addr), .dmem_wdata(dmem_wdata),
    .dmem_rdata(dmem_rdata), .dmem_ack(dmem_ack),
    .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
    .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
    .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
    .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
    .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb),
    .m_axi_wlast(m_axi_wlast),
    .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
    .m_axi_bresp(m_axi_bresp),
    .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
    .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
    .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
    .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
    .m_axi_rdata(m_axi_rdata), .m_axi_rlast(m_axi_rlast),
    .m_axi_rresp(m_axi_rresp)
  );

  simt_soc #(
    .PROFILE(PROFILE), .SAMP_NB(SAMP_NB)
  ) u_soc (
    .clk(clk), .rst_n(soc_rst_n),
    .imem_req(imem_req), .imem_addr(imem_addr), .imem_rdata(imem_rdata),
    .dmem_req(dmem_req), .dmem_we(dmem_we), .dmem_addr(dmem_addr),
    .dmem_wdata(dmem_wdata), .dmem_be(dmem_be),
    .dmem_rdata(dmem_rdata), .dmem_ack(dmem_ack),
    /* verilator lint_off PINCONNECTEMPTY */
    .cmt_valid(), .cmt_warp(), .cmt_pc(), .cmt_instr(), .cmt_mask(),
    .cmt_rd(), .cmt_wdata(),
    .mcmt_valid(), .mcmt_warp(), .mcmt_pc(), .mcmt_instr(),
    .mcmt_mask(), .mcmt_rd(), .mcmt_wdata(),
    .tensor_ops(), .sampler_ops(), .tensor_busy()
    /* verilator lint_on PINCONNECTEMPTY */
  );
endmodule
