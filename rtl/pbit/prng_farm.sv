// D16 PRNG farm: NSTREAMS xoshiro128++ engines, one word each per step.
// Seeding: host/golden writes stream-by-stream (states pre-jumped 2^64 apart
// by golden/xoshiro.py). Golden: same file, FarmVec.
module prng_farm #(
  parameter int NSTREAMS = 16
) (
  input  logic                          clk,
  input  logic                          rst_n,
  input  logic                          seed_we,
  input  logic [$clog2(NSTREAMS)-1:0]   seed_stream,
  input  logic [1:0]                    seed_sel,
  input  logic [31:0]                   seed_word,
  input  logic                          step,
  output logic [NSTREAMS-1:0][31:0]     rnd
);
  for (genvar i = 0; i < NSTREAMS; i++) begin : g_stream
    xoshiro128pp u_x (
      .clk      (clk),
      .rst_n    (rst_n),
      .seed_we  (seed_we && (seed_stream == ($clog2(NSTREAMS))'(i))),
      .seed_sel (seed_sel),
      .seed_word(seed_word),
      .step     (step),
      .rnd      (rnd[i])
    );
  end
endmodule
