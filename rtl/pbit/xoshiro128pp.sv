// xoshiro128++ — one stream of the D16 PRNG farm. Golden: golden/xoshiro.py.
// State is loaded word-by-word from outside (jump-ahead seeding is a software
// job: golden computes each stream's initial state; hardware never jumps).
// rnd is the ++ scrambler of the CURRENT state; step advances the engine.
module xoshiro128pp (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        seed_we,
  input  logic [1:0]  seed_sel,     // which of s0..s3 to load
  input  logic [31:0] seed_word,
  input  logic        step,
  output logic [31:0] rnd
);
  logic [31:0] s0, s1, s2, s3;

  wire [31:0] sum  = s0 + s3;
  assign rnd = ({sum[24:0], sum[31:25]}) + s0;   // rotl(s0+s3,7) + s0

  wire [31:0] t     = s1 << 9;
  wire [31:0] s2_a  = s2 ^ s0;
  wire [31:0] s3_a  = s3 ^ s1;
  wire [31:0] s1_n  = s1 ^ s2_a;
  wire [31:0] s0_n  = s0 ^ s3_a;
  wire [31:0] s2_n  = s2_a ^ t;
  wire [31:0] s3_n  = {s3_a[20:0], s3_a[31:21]}; // rotl(s3',11)

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // non-zero reset state; real seeding always overwrites it
      s0 <= 32'hDEADBEEF; s1 <= 32'h12345678;
      s2 <= 32'h9E3779B9; s3 <= 32'hC0FFEE01;
    end else if (seed_we) begin
      unique case (seed_sel)
        2'd0: s0 <= seed_word;
        2'd1: s1 <= seed_word;
        2'd2: s2 <= seed_word;
        default: s3 <= seed_word;
      endcase
    end else if (step) begin
      s0 <= s0_n; s1 <= s1_n; s2 <= s2_n; s3 <= s3_n;
    end
  end
endmodule
