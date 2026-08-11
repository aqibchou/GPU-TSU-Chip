// Monte-Carlo wrapper: one real farm stream feeding one p-bit cell — the
// true hardware randomness path for the M4 Bernoulli-rate gate.
// The PRNG steps exactly when a sample consumes a word (step := sample_en).
module pbit_mc_top #(
  parameter string LUT_FILE = "sigmoid_lut.mem"
) (
  input  logic        clk,
  input  logic        rst_n,
  // PRNG seeding (one stream)
  input  logic        seed_we,
  input  logic [1:0]  seed_sel,
  input  logic [31:0] seed_word,
  // cell controls
  input  logic        bipolar,
  input  logic        acc_clear,
  input  logic [9:0]  bias,
  input  logic        acc_en,
  input  logic [9:0]  j_val,
  input  logic        s_in,
  input  logic        sample_en,
  input  logic [7:0]  beta,
  output logic        s_valid,
  output logic        s_out,
  output logic [16:0] p_dbg
);
  logic [31:0] rnd;

  xoshiro128pp u_prng (
    .clk(clk), .rst_n(rst_n),
    .seed_we(seed_we), .seed_sel(seed_sel), .seed_word(seed_word),
    .step(sample_en),
    .rnd(rnd)
  );

  pbit_cell #(.LUT_FILE(LUT_FILE)) u_cell (
    .clk(clk), .rst_n(rst_n), .bipolar(bipolar),
    .acc_clear(acc_clear), .bias(bias),
    .acc_en(acc_en), .j_val(j_val), .s_in(s_in),
    .acc_en2(1'b0), .j_val2(10'd0), .s_in2(1'b0),   // single-slot legacy
    .sample_en(sample_en), .beta(beta), .rnd(rnd),
    .s_valid(s_valid), .s_out(s_out), .p_dbg(p_dbg)
  );
endmodule
