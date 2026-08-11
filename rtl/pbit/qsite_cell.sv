// q-site cell (QSITE S2 q4; S4 q8, spec §12): q-candidate MAC +
// Gumbel-max. Bit-true golden: golden/qsite_golden.py (GLUT frozen
// s3.4; score = beta_raw*acc[a] + (GLUT[byte_a] << 5) on the shared
// 1/512 grid; ties break to the smallest candidate). Binary mode
// never uses this cell; pbit_cell is untouched (Q1).
//
// Area posture (qsite_card Amendment 3 fruit, all three applied):
// (1) GLUT reads are SYNCHRONOUS BLOCK-RAM fetches — pass 1 prefetches
//     on the LAST MAC cycle (the farm word is stable through the MAC
//     phase), so q4 pays no extra cycle; pass 2 fetches during the
//     first sample state and scores a cycle later (the fabric adds
//     G_SMP2B for q8 only).
// (2) The argmax tree is SEQUENTIALLY REUSED: one depth-2 four-way
//     tree scores pass 1 (registered as best1) and pass 2, plus one
//     final compare — 4 comparators instead of 7. Left wins ties at
//     every node and pass 1 wins the final tie, so the winner is the
//     smallest-index maximum — bit-identical to a full scan.
// (3) Only 4 score registers live (pass 2 overwrites pass 1; best1
//     carries pass 1 forward).
module qsite_cell #(
  parameter string GLUT_FILE = "glut.mem"
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        arity8,      // 0 = q4
  // MAC phase — dual-slot pairs, mirroring pbit_cell: a contribution
  // adds J to exactly ONE candidate accumulator (delta coupling)
  input  logic        acc_clear,   // acc[a] <= bias lane a (b0 = 0)
  input  logic [69:0] bias,        // {b7, b6..b1} s1.6.3 lanes
  input  logic        acc_en,
  input  logic [9:0]  j_val,       // s1.6.3
  input  logic [2:0]  s_in,        // neighbor VALUE (slot 0)
  input  logic        acc_en2,
  input  logic [9:0]  j_val2,
  input  logic [2:0]  s_in2,
  // sample phase
  input  logic        glut_pre,    // last MAC cycle: prefetch word-1 GLUT
  input  logic        sample_en,   // pass 1 scores (q4: the only pass)
  input  logic        sample_en2,  // q8: best1 latch + word-2 GLUT fetch
  input  logic [7:0]  beta,        // u2.6
  input  logic [31:0] rnd,         // byte a = candidate a (spec §12)
  output logic        s_valid,     // 1-cycle pulse into G_WR
  output logic [2:0]  s_out
);
  (* rom_style = "block" *) logic [7:0] glut [256];
  initial $readmemh(GLUT_FILE, glut);

  // ---------------- MAC: q candidate accumulators ----------------
  logic signed [13:0] acc [8];
  wire  signed [13:0] j_ext  = {{4{j_val[9]}},  j_val};
  wire  signed [13:0] j_ext2 = {{4{j_val2[9]}}, j_val2};

  // ---------------- GLUT: synchronous BRAM fetch ----------------
  // pass 1 sampled at the last-MAC edge (rnd = farm word 1, stable
  // since the previous chunk's steps); pass 2 sampled at the
  // sample_en2 edge (rnd = word 2, visible after the SMP1 step)
  logic [7:0] glut_q [4];
  always_ff @(posedge clk)
    if (glut_pre || sample_en2)
      for (int a = 0; a < 4; a++)
        glut_q[a] <= glut[rnd[8 * a +: 8]];

  // ---------------- scores: one 4-wide pass, time-multiplexed ----
  // beta*acc is s22 on the 1/512 real grid; the GLUT lane (s3.4)
  // lands on the same grid via << 5. Pass 2 muxes acc[4+a] onto the
  // SAME 4 DSP mult-adds.
  logic        st_p2;              // pass-2 scoring cycle (q8)
  logic        st_fin;             // final-compare cycle
  (* use_dsp = "yes" *) logic signed [22:0] score_d [4];
  always_comb
    for (int a = 0; a < 4; a++)
      score_d[a] = 23'($signed(st_p2 ? acc[4 + a] : acc[a])
                       * $signed({1'b0, beta}))
                   + (23'($signed(glut_q[a])) <<< 5);
  logic signed [22:0] score_q [4];

  // ---------------- shared depth-2 argmax tree ----------------
  // (left wins ties; indices within a pass are 0..3)
  logic [1:0]         tree_i;
  logic signed [22:0] tree_s;
  always_comb begin
    automatic logic [1:0] ia, ib;
    automatic logic signed [22:0] sa, sb;
    if (score_q[1] > score_q[0]) begin ia = 2'd1; sa = score_q[1]; end
    else begin ia = 2'd0; sa = score_q[0]; end
    if (score_q[3] > score_q[2]) begin ib = 2'd3; sb = score_q[3]; end
    else begin ib = 2'd2; sb = score_q[2]; end
    if (sb > sa) begin tree_i = ib; tree_s = sb; end
    else begin tree_i = ia; tree_s = sa; end
  end
  logic [1:0]         best1_i;
  logic signed [22:0] best1_s;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int a = 0; a < 8; a++) acc[a] <= 14'sd0;
      for (int a = 0; a < 4; a++) score_q[a] <= 23'sd0;
      st_p2 <= 1'b0; st_fin <= 1'b0;
      best1_i <= 2'd0; best1_s <= 23'sd0;
      s_valid <= 1'b0;
      s_out   <= 3'd0;
    end else begin
      if (acc_clear) begin
        acc[0] <= 14'sd0;
        for (int a = 1; a < 8; a++)
          acc[a] <= {{4{bias[10 * a - 1]}}, bias[10 * (a - 1) +: 10]};
      end else if (acc_en || acc_en2) begin
        if (acc_en)
          acc[s_in] <= acc[s_in] + j_ext
                       + ((acc_en2 && s_in2 == s_in) ? j_ext2 : 14'sd0);
        if (acc_en2 && !(acc_en && s_in2 == s_in))
          acc[s_in2] <= acc[s_in2] + j_ext2;
      end

      // pass-1 scores latch at the sample_en edge (q4 and q8 alike);
      // q8 pass-2 scores latch one cycle after sample_en2 (st_p2)
      if (sample_en || st_p2)
        for (int a = 0; a < 4; a++)
          score_q[a] <= score_d[a];
      // q8: best-of-pass-1 registered while pass-2 GLUT data lands
      if (sample_en2) begin
        best1_i <= tree_i;
        best1_s <= tree_s;
      end
      st_p2  <= sample_en2;
      // q4 finishes one cycle after sample_en; q8 one after pass 2
      st_fin <= arity8 ? st_p2 : sample_en;
      if (st_fin) begin
        if (arity8)
          // strict >: pass 1 (smaller indices) wins ties
          s_out <= (tree_s > best1_s) ? {1'b1, tree_i}
                                      : {1'b0, best1_i};
        else
          s_out <= {1'b0, tree_i};
      end
      s_valid <= st_fin;
    end
  end
endmodule
