// Pipelined unsigned multiplier, STAGES-cycle latency. Golden: golden/mul.py.
// p is don't-care while out_valid is low.
module mul_pipe #(
  parameter int W      = 16,
  parameter int STAGES = 3
) (
  input  logic           clk,
  input  logic           rst_n,
  input  logic           in_valid,
  input  logic [W-1:0]   a,
  input  logic [W-1:0]   b,
  output logic           out_valid,
  output logic [2*W-1:0] p
);
  // stage 0 registers operands; stage 1 registers the product;
  // remaining stages are a shift line — total latency = STAGES.
  logic [W-1:0]   a_q, b_q;
  logic [2*W-1:0] p_pipe [STAGES-1];
  logic [STAGES-1:0] v_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_q <= '0;
      b_q <= '0;
      v_q <= '0;
      for (int i = 0; i < STAGES - 1; i++) p_pipe[i] <= '0;
    end else begin
      a_q       <= a;
      b_q       <= b;
      p_pipe[0] <= a_q * b_q;
      for (int i = 1; i < STAGES - 1; i++) p_pipe[i] <= p_pipe[i-1];
      v_q       <= {v_q[STAGES-2:0], in_valid};
    end
  end

  assign out_valid = v_q[STAGES-1];
  assign p         = p_pipe[STAGES-2];
endmodule
