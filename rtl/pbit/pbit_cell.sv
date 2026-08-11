// p-bit cell (D13): serial MAC + sigmoid LUT + comparator. Bit-true golden:
// golden/pbit.py (formats frozen there: J/h s1.6.3, acc s1.10.3, beta u2.6,
// sign-magnitude half-away rounding to u4.8, LUT [0,12) step 1/256, u0.16).
// Sample is 2-stage (registered LUT index -> registered decision) so the LUT
// maps to a synchronous BRAM without re-verification later.
module pbit_cell #(
  parameter string LUT_FILE = "sigmoid_lut.mem"
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        bipolar,     // 1: spins are +/-1 (contribution +/-J)
  // MAC phase — dual-slot (2026-07-12): two contributions per cycle.
  // Grouping is bit-exact by the degree contract: the golden asserts
  // the accumulator never leaves s14, so addition order is free.
  input  logic        acc_clear,   // acc <= bias
  input  logic [9:0]  bias,        // s1.6.3
  input  logic        acc_en,      // slot-0 contribution enable
  input  logic [9:0]  j_val,       // s1.6.3
  input  logic        s_in,
  input  logic        acc_en2,     // slot-1 contribution enable
  input  logic [9:0]  j_val2,      // s1.6.3
  input  logic        s_in2,
  // sample phase
  input  logic        sample_en,
  input  logic [7:0]  beta,        // u2.6
  input  logic [31:0] rnd,
  output logic        s_valid,     // 1-cycle pulse, 2 cycles after sample_en
  output logic        s_out,
  output logic [16:0] p_dbg        // p17 behind the decision
);
  localparam int LUT_N = 3072;

  // decision consumes rnd[31:16] only; [15:0] is reserved so a future farm
  // revision can feed two cells from one 32-bit word
  wire _unused_rnd_lo = &{1'b0, rnd[15:0]};

  // the 2-stage sample design (registered index -> registered
  // decision) exists precisely so this maps to synchronous BRAM;
  // as logic-LUTs it costs ~550 LUTs/lane (measured, qsite Q2 run)
  (* rom_style = "block" *) logic [15:0] rom [LUT_N];
  initial $readmemh(LUT_FILE, rom);

  // ---------------- MAC ----------------
  logic signed [13:0] acc;
  wire  signed [13:0] j_ext  = {{4{j_val[9]}},  j_val};
  wire  signed [13:0] j_ext2 = {{4{j_val2[9]}}, j_val2};
  logic signed [13:0] contrib, contrib2;
  always_comb begin
    if (bipolar)      contrib  = s_in  ? j_ext  : -j_ext;
    else              contrib  = s_in  ? j_ext  : 14'sd0;
    if (bipolar)      contrib2 = s_in2 ? j_ext2 : -j_ext2;
    else              contrib2 = s_in2 ? j_ext2 : 14'sd0;
  end
  wire signed [13:0] c0 = acc_en  ? contrib  : 14'sd0;
  wire signed [13:0] c1 = acc_en2 ? contrib2 : 14'sd0;

  // ---------------- sample stage 1: z = round(acc*beta), sign-magnitude ----
  wire signed [22:0] prod = acc * $signed({1'b0, beta});   // s14 x u8 -> s22
  wire               neg  = prod < 0;
  wire        [21:0] mag  = neg ? $unsigned(-prod[21:0]) : $unsigned(prod[21:0]);
  wire        [20:0] magr = 21'((mag + 22'd1) >> 1);       // half-away round
  wire        [11:0] idx  = (magr > 21'(LUT_N - 1)) ? 12'(LUT_N - 1) : magr[11:0];

  logic        st1_v;
  logic        neg_q;
  logic [15:0] rnd16_q;

  // ---------------- sample stage 2: LUT + compare ----------------
  // the ROM read IS the stage boundary: a synchronous, enabled read
  // (pl_q) so the 3072x16 table maps to BRAM instead of ~550 LUTs of
  // fabric per lane (measured on the qsite Q2 run; behavior identical
  // — pl_q equals rom[idx captured at sample_en], as before)
  logic [15:0] pl_q;
  wire [16:0] p17 = neg_q ? (17'd65536 - {1'b0, pl_q}) : {1'b0, pl_q};

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc     <= 14'sd0;
      st1_v   <= 1'b0;
      neg_q   <= 1'b0;
      rnd16_q <= 16'd0;
      s_valid <= 1'b0;
      s_out   <= 1'b0;
      p_dbg   <= 17'd0;
    end else begin
      if (acc_clear)               acc <= {{4{bias[9]}}, bias};
      else if (acc_en || acc_en2)  acc <= acc + c0 + c1;

      st1_v <= sample_en;
      if (sample_en) begin
        neg_q   <= neg;
        rnd16_q <= rnd[31:16];
      end

      s_valid <= st1_v;
      if (st1_v) begin
        s_out <= ({1'b0, rnd16_q} < p17);
        p_dbg <= p17;
      end
    end
  end
  always_ff @(posedge clk)
    if (sample_en) pl_q <= rom[idx];
endmodule
