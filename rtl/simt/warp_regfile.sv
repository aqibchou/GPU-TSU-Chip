// Warp-indexed register file (M13): W x 31 x 32b, x0 hardwired zero.
// One read pair (D stage) + one write port (WB stage) — sufficient by
// construction: stages hold distinct warps (INV-1), so no same-cycle
// same-warp read/write hazard exists; a WB write lands strictly before
// that warp's next D read (>= W-3 cycles later). INV-5/6/13 live here
// (docs/HARDWARE_ARCHITECTURE.md#simt-core).
module warp_regfile #(
  parameter int W  = 8,
  localparam int WB = $clog2(W)
) (
  input  logic          clk,
  // D-stage read pair
  input  logic [WB-1:0] r_warp,
  input  logic [4:0]    rs1,
  input  logic [4:0]    rs2,
  output logic [31:0]   rs1_data,
  output logic [31:0]   rs2_data,
  // WB-stage write
  input  logic          w_en,
  input  logic          w_stage_valid,   // WB stage occupancy (INV-6 check)
  input  logic [WB-1:0] w_warp,
  input  logic [4:0]    w_rd,
  input  logic [31:0]   w_data
);
  logic [31:0] rf [W][32];

  assign rs1_data = (rs1 == 5'd0) ? 32'd0 : rf[r_warp][rs1];
  assign rs2_data = (rs2 == 5'd0) ? 32'd0 : rf[r_warp][rs2];

  always_ff @(posedge clk) begin
    if (w_en && w_rd != 5'd0)
      rf[w_warp][w_rd] <= w_data;
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk) begin
    // INV-6 (bubble hygiene): a write requires WB-stage occupancy
    if (w_en) assert (w_stage_valid)
      else $fatal(1, "INV-6: regfile write from a bubble");
    // INV-5 shape: writes to x0 are dropped (checked by read paths too)
  end
`endif
endmodule
