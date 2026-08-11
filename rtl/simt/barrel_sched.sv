// Barrel scheduler (M13, Epoch D): deterministic round-robin warp rotation
// with the per-warp busy bit as the ONLY interlock — the doctrine's move 1.
// Spec + invariant catalog: docs/HARDWARE_ARCHITECTURE.md#simt-core (INV-1..4, INV-11 live here).
// Rotation NEVER skips: a busy warp's slot carries a bubble, so stage tags
// are a pure function of the cycle phase (INV-2) and same-warp overlap is
// structurally impossible given W > DEPTH (INV-4, asserted anyway).
module barrel_sched #(
  parameter int W     = 8,
  parameter int DEPTH = 5,
  localparam int WB   = $clog2(W)
) (
  input  logic          clk,
  input  logic          rst_n,
  // busy-bit contract: set fires for a warp currently in the pipe that
  // issued a long-latency op; clr fires on that op's unique completion.
  input  logic          set_busy,
  input  logic [WB-1:0] set_warp,
  input  logic          clr_busy,
  input  logic [WB-1:0] clr_warp,
  // D-037: per-warp fetch hold (reconvergence walker in flight). A
  // held warp's slot carries a bubble exactly like a busy one; the
  // rotation and stage-tag invariants are untouched.
  input  logic [W-1:0]  hold,
  // fetch-slot decision for this cycle
  output logic          issue_valid,
  output logic [WB-1:0] issue_warp,
  output logic [W-1:0]  busy,
  // stage occupancy pipe (index 0 = F .. DEPTH-1 = WB)
  output logic [DEPTH-1:0]         stage_valid,
  output logic [DEPTH-1:0][WB-1:0] stage_warp
);
  logic [WB-1:0] phase /*verilator public_flat_rd*/;  // leg-A pilot tap

  assign issue_warp  = phase;
  assign issue_valid = ~busy[phase] & ~hold[phase];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      phase       <= '0;
      busy        <= '0;
      stage_valid <= '0;
      for (int s = 0; s < DEPTH; s++) stage_warp[s] <= '0;
    end else begin
      phase <= (phase == WB'(W - 1)) ? '0 : phase + WB'(1);

      // shift the occupancy pipe; F takes this cycle's decision
      for (int s = DEPTH - 1; s > 0; s--) begin
        stage_valid[s] <= stage_valid[s - 1];
        stage_warp[s]  <= stage_warp[s - 1];
      end
      stage_valid[0] <= issue_valid;
      stage_warp[0]  <= issue_warp;

      if (set_busy) busy[set_warp] <= 1'b1;
      if (clr_busy) busy[clr_warp] <= 1'b0;
    end
  end

`ifndef SYNTHESIS
  // verification-only: sync rst_n use here is intentional
  /* verilator lint_off SYNCASYNCNET */
  // ------ invariant assertions (docs/HARDWARE_ARCHITECTURE.md#simt-core) ------
  always_ff @(posedge clk) if (rst_n) begin
    // INV-2: valid stage tags are the phase-derived rotation slots
    for (int s = 0; s < DEPTH; s++)
      if (stage_valid[s])
        assert (stage_warp[s] ==
                WB'((int'(phase) - 1 - s + 2 * W) % W))
          else $fatal(1, "INV-2 violated at stage %0d", s);
    // INV-1: pairwise-distinct tags among occupied stages
    for (int a = 0; a < DEPTH; a++)
      for (int b2 = a + 1; b2 < DEPTH; b2++)
        if (stage_valid[a] && stage_valid[b2])
          assert (stage_warp[a] != stage_warp[b2])
            else $fatal(1, "INV-1 violated (%0d,%0d)", a, b2);
    // INV-4: <=1 in-flight per warp
    for (int w = 0; w < W; w++) begin
      automatic int c = 0;
      for (int s = 0; s < DEPTH; s++)
        if (stage_valid[s] && stage_warp[s] == WB'(w)) c++;
      assert (c <= 1) else $fatal(1, "INV-4 violated warp %0d", w);
    end
    // INV-3 contract half: no set on an already-busy warp, no clr on an
    // idle warp, no same-cycle set+clr of the same warp
    if (set_busy) assert (!busy[set_warp])
      else $fatal(1, "INV-3: set on busy warp");
    if (clr_busy) assert (busy[clr_warp])
      else $fatal(1, "INV-3: clr on idle warp");
    if (set_busy && clr_busy) assert (set_warp != clr_warp)
      else $fatal(1, "INV-3: set+clr same warp same cycle");
  end
  /* verilator lint_on SYNCASYNCNET */
`endif
endmodule
