// Lane register file (M14): W warps x L lanes x 31x32b, x0 hardwired.
// Read side: D stage reads rs1/rs2 for ALL lanes of one warp.
// Write port A (WB stage): masked per-lane write, one warp, one rd.
// Write port B (memory unit): single-lane write during a load drain.
// A and B can fire the same cycle but NEVER for the same warp: port B's
// warp is busy (out of the pipe) while port A's warp occupies WB — the
// INV-17 companion assertion enforces it.
module simt_regfile #(
  parameter int W = 8,
  parameter int L = 8,
  localparam int WB_ = $clog2(W)
) (
  input  logic                  clk,
  // D-stage read: one warp, all lanes
  input  logic [WB_-1:0]        r_warp,
  input  logic [4:0]            rs1,
  input  logic [4:0]            rs2,
  output logic [L-1:0][31:0]    rs1_data,
  output logic [L-1:0][31:0]    rs2_data,
  // port A: WB masked write
  input  logic                  a_en,
  input  logic [WB_-1:0]        a_warp,
  input  logic [L-1:0]          a_mask,
  input  logic [4:0]            a_rd,
  input  logic [L-1:0][31:0]    a_data,
  // port B: memory-drain single-lane write
  input  logic                  b_en,
  input  logic [WB_-1:0]        b_warp,
  input  logic [$clog2(L)-1:0]  b_lane,
  input  logic [4:0]            b_rd,
  input  logic [31:0]           b_data
);
  // One bank per (warp, lane): 32x32b, ONE synchronous write port and
  // async reads — the LUTRAM shape (G8σ/D-022: the monolithic
  // two-port rf[W][L][32] synthesized as 63.5k flops + a 74k-LUT read
  // mux cone). Ports A and B are warp-disjoint (INV-17), so per bank
  // the two writes reduce to one mux — cycle-identical to the
  // original: same-edge writes, same-cycle async reads.
  logic [L-1:0][31:0] rs1_bank [W];
  logic [L-1:0][31:0] rs2_bank [W];

  for (genvar w = 0; w < W; w++) begin : g_warp
    for (genvar l = 0; l < L; l++) begin : g_lane
      logic [31:0] mem [32];
      wire a_hit = a_en && (a_warp == WB_'(w)) && a_mask[l]
                   && (a_rd != 5'd0);
      wire b_hit = b_en && (b_warp == WB_'(w))
                   && (b_lane == $clog2(L)'(l)) && (b_rd != 5'd0);
      always_ff @(posedge clk)
        if (a_hit || b_hit)
          mem[b_hit ? b_rd : a_rd] <= b_hit ? b_data : a_data[l];
      assign rs1_bank[w][l] = mem[rs1];
      assign rs2_bank[w][l] = mem[rs2];
    end
  end

  always_comb begin
    for (int l = 0; l < L; l++) begin
      rs1_data[l] = (rs1 == 5'd0) ? 32'd0 : rs1_bank[r_warp][l];
      rs2_data[l] = (rs2 == 5'd0) ? 32'd0 : rs2_bank[r_warp][l];
    end
  end

`ifndef SYNTHESIS
  /* verilator lint_off SYNCASYNCNET */
  always_ff @(posedge clk) begin
    // INV-17 companion: the two write ports are warp-disjoint
    if (a_en && b_en) assert (a_warp != b_warp)
      else $fatal(1, "regfile ports A/B hit the same warp");
  end
  /* verilator lint_on SYNCASYNCNET */
`endif
endmodule
