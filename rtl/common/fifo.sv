// Synchronous show-ahead FIFO. Golden: golden/fifo.py.
// Write accepted iff wr_en && !full; read iff rd_en && !empty (flags judged
// pre-edge). rd_data shows the head combinationally; don't-care when empty.
module fifo #(
  parameter int WIDTH = 8,
  parameter int DEPTH = 16  // power of two
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    wr_en,
  input  logic [WIDTH-1:0]        wr_data,
  input  logic                    rd_en,
  output logic [WIDTH-1:0]        rd_data,
  output logic                    full,
  output logic                    empty,
  output logic [$clog2(DEPTH):0]  count
);
  localparam int PW = $clog2(DEPTH);
  localparam int CW = PW + 1;

  logic [WIDTH-1:0] mem [DEPTH];
  logic [PW-1:0]    wr_ptr, rd_ptr;
  logic [CW-1:0]    cnt_q;

  wire do_wr = wr_en && !full;
  wire do_rd = rd_en && !empty;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_ptr <= '0;
      rd_ptr <= '0;
      cnt_q  <= '0;
    end else begin
      if (do_wr) begin
        mem[wr_ptr] <= wr_data;
        wr_ptr      <= wr_ptr + PW'(1);
      end
      if (do_rd) begin
        rd_ptr <= rd_ptr + PW'(1);
      end
      unique case ({do_wr, do_rd})
        2'b10:   cnt_q <= cnt_q + CW'(1);
        2'b01:   cnt_q <= cnt_q - CW'(1);
        default: cnt_q <= cnt_q;
      endcase
    end
  end

  assign rd_data = mem[rd_ptr];
  assign count   = cnt_q;
  assign full    = (cnt_q == CW'(DEPTH));
  assign empty   = (cnt_q == '0);
endmodule
