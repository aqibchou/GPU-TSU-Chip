// 8N1 UART, TX + RX, fixed divisor. Golden: golden/uart.py (encode/decode_strict).
// TX: accept on tx_valid && tx_ready; start bit begins the cycle after accept.
// RX: falling-edge start detect, mid-bit sampling at CLKS_PER_BIT/2, LSB first;
//     a frame with a bad stop bit is dropped silently (no valid pulse).
module uart #(
  parameter int CLKS_PER_BIT = 4  // >= 2
) (
  input  logic       clk,
  input  logic       rst_n,
  // TX
  input  logic       tx_valid,
  input  logic [7:0] tx_data,
  output logic       tx_ready,
  output logic       txd,
  // RX
  input  logic       rxd,
  output logic       rx_valid,   // 1-cycle pulse
  output logic [7:0] rx_data
);
  localparam int CB  = CLKS_PER_BIT;
  localparam int MID = CB / 2;
  localparam int CW  = (CB <= 2) ? 2 : $clog2(CB + 1);

  // ---------------- TX ----------------
  typedef enum logic [1:0] { T_IDLE, T_START, T_DATA, T_STOP } tx_state_e;
  tx_state_e     tx_st;
  logic [7:0]    tx_sh;
  logic [CW-1:0] tx_cnt;
  logic [2:0]    tx_bit;

  assign tx_ready = (tx_st == T_IDLE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tx_st  <= T_IDLE;
      tx_sh  <= '0;
      tx_cnt <= '0;
      tx_bit <= '0;
      txd    <= 1'b1;
    end else begin
      unique case (tx_st)
        T_IDLE: begin
          txd <= 1'b1;
          if (tx_valid) begin
            tx_sh  <= tx_data;
            tx_st  <= T_START;
            tx_cnt <= CW'(1);
            txd    <= 1'b0;               // start bit begins next cycle edge
          end
        end
        T_START: begin
          if (tx_cnt == CW'(CB)) begin
            tx_st  <= T_DATA;
            tx_cnt <= CW'(1);
            tx_bit <= '0;
            txd    <= tx_sh[0];
          end else begin
            tx_cnt <= tx_cnt + CW'(1);
          end
        end
        T_DATA: begin
          if (tx_cnt == CW'(CB)) begin
            tx_cnt <= CW'(1);
            if (tx_bit == 3'd7) begin
              tx_st <= T_STOP;
              txd   <= 1'b1;              // stop bit
            end else begin
              tx_bit <= tx_bit + 3'd1;
              txd    <= tx_sh[3'(tx_bit + 3'd1)];
            end
          end else begin
            tx_cnt <= tx_cnt + CW'(1);
          end
        end
        T_STOP: begin
          if (tx_cnt == CW'(CB)) begin
            tx_st <= T_IDLE;
            txd   <= 1'b1;
          end else begin
            tx_cnt <= tx_cnt + CW'(1);
          end
        end
        default: tx_st <= T_IDLE;
      endcase
    end
  end

  // ---------------- RX ----------------
  typedef enum logic [1:0] { R_IDLE, R_START, R_DATA, R_STOP } rx_state_e;
  rx_state_e     rx_st;
  logic [7:0]    rx_sh;
  logic [CW-1:0] rx_cnt;
  logic [2:0]    rx_bit;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rx_st    <= R_IDLE;
      rx_sh    <= '0;
      rx_cnt   <= '0;
      rx_bit   <= '0;
      rx_valid <= 1'b0;
      rx_data  <= '0;
    end else begin
      rx_valid <= 1'b0;
      unique case (rx_st)
        R_IDLE: begin
          if (!rxd) begin                 // start-bit falling edge seen this cycle
            rx_st  <= R_START;
            rx_cnt <= CW'(1);
          end
        end
        R_START: begin
          if (rx_cnt == CW'(MID)) begin   // mid-start: must still be low
            if (!rxd) begin
              rx_st  <= R_DATA;
              rx_cnt <= CW'(1);
              rx_bit <= '0;
            end else begin
              rx_st <= R_IDLE;            // glitch — abandon
            end
          end else begin
            rx_cnt <= rx_cnt + CW'(1);
          end
        end
        R_DATA: begin
          if (rx_cnt == CW'(CB)) begin    // CB cycles after last sample = mid-bit
            rx_sh  <= {rxd, rx_sh[7:1]};  // LSB first
            rx_cnt <= CW'(1);
            if (rx_bit == 3'd7) rx_st <= R_STOP;
            else                rx_bit <= rx_bit + 3'd1;
          end else begin
            rx_cnt <= rx_cnt + CW'(1);
          end
        end
        R_STOP: begin
          if (rx_cnt == CW'(CB)) begin    // mid-stop: must be high
            if (rxd) begin
              rx_valid <= 1'b1;
              rx_data  <= rx_sh;
            end
            rx_st <= R_IDLE;              // bad stop: drop frame silently
          end else begin
            rx_cnt <= rx_cnt + CW'(1);
          end
        end
        default: rx_st <= R_IDLE;
      endcase
    end
  end
endmodule
