// Data cache (M16): 2-way set-associative, 32 B lines, LRU, write-back +
// write-allocate, blocking, one outstanding request on both faces
// (docs/HARDWARE_ARCHITECTURE.md#memory-hierarchy §1/§3, frozen). Bit-true golden: golden/cache.py —
// the miss flow order (LRU victim -> dirty writeback -> fill -> access ->
// LRU touch) is architectural and must match exactly. Values must be
// invariant to memory-face latency (bench proves at lat 1 and 40).
module dcache #(
  parameter int SIZE_BYTES = 4096,
  localparam int LINE   = 32,
  localparam int WAYS   = 2,
  localparam int SETS   = SIZE_BYTES / (LINE * WAYS),
  localparam int SB     = $clog2(SETS),
  localparam int WORDS  = LINE / 4,          // words per line
  localparam int TB     = 32 - SB - 5        // tag bits (5 = line offset)
) (
  input  logic        clk,
  input  logic        rst_n,
  // core face (mem_spec §1)
  input  logic        c_valid,
  input  logic        c_we,
  input  logic [31:0] c_addr,
  input  logic [31:0] c_wdata,
  input  logic [3:0]  c_be,
  output logic        c_ack,
  output logic [31:0] c_rdata,
  // memory face (same contract, backed by axi_pessimistic in the bench)
  output logic        m_valid,
  output logic        m_we,
  output logic [31:0] m_addr,
  output logic [31:0] m_wdata,
  output logic [3:0]  m_be,
  input  logic        m_ack,
  input  logic [31:0] m_rdata,
  // observability (G-gate fuel)
  output logic [31:0] cnt_hit,
  output logic [31:0] cnt_miss,
  output logic [31:0] cnt_wb
);
  logic [TB-1:0] tag   [SETS][WAYS];
  logic          valid [SETS][WAYS];
  logic          dirty [SETS][WAYS];
  logic          lru   [SETS];               // way to evict next
  logic [31:0]   data  [SETS][WAYS][WORDS];

  // request decode (registered at accept)
  logic        q_we;
  logic [31:0] q_addr, q_wdata;
  logic [3:0]  q_be;
  wire [SB-1:0] q_set = q_addr[5 +: SB];
  wire [TB-1:0] q_tag = q_addr[31 -: TB];
  wire [2:0]    q_word = q_addr[4:2];
  wire _unused_qa = &{1'b0, q_addr[1:0]};

  logic       hit0, hit1, hitv;
  logic       hway;
  always_comb begin
    hit0 = valid[q_set][0] && (tag[q_set][0] == q_tag);
    hit1 = valid[q_set][1] && (tag[q_set][1] == q_tag);
    hitv = hit0 || hit1;
    hway = hit1;
  end

  typedef enum logic [2:0] {IDLE, LOOKUP, WB, FILL, RESP} st_e;
  st_e st;
  logic       vway;                          // victim/serving way on miss
  logic [2:0] beat;
  logic       m_pending;

  assign m_wdata = data[q_set][vway][beat];
  assign m_be    = 4'hF;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= IDLE; c_ack <= 1'b0; c_rdata <= '0;
      q_we <= 1'b0; q_addr <= '0; q_wdata <= '0; q_be <= '0;
      vway <= 1'b0; beat <= '0;
      m_valid <= 1'b0; m_we <= 1'b0; m_addr <= '0; m_pending <= 1'b0;
      cnt_hit <= '0; cnt_miss <= '0; cnt_wb <= '0;
      for (int s = 0; s < SETS; s++) begin
        lru[s] <= 1'b0;
        for (int w = 0; w < WAYS; w++) begin
          valid[s][w] <= 1'b0; dirty[s][w] <= 1'b0; tag[s][w] <= '0;
        end
      end
    end else begin
      c_ack <= 1'b0;

      unique case (st)
        IDLE: if (c_valid) begin
          q_we <= c_we; q_addr <= c_addr; q_wdata <= c_wdata; q_be <= c_be;
          st <= LOOKUP;
        end
        LOOKUP: begin
          if (hitv) begin
            cnt_hit <= cnt_hit + 32'd1;
            if (q_we) begin
              for (int b = 0; b < 4; b++)
                if (q_be[b])
                  data[q_set][hway][q_word][8*b +: 8]
                    <= q_wdata[8*b +: 8];
              dirty[q_set][hway] <= 1'b1;
            end else
              c_rdata <= data[q_set][hway][q_word];
            lru[q_set] <= ~hway;
            c_ack <= 1'b1;
            st <= IDLE;
          end else begin
            cnt_miss <= cnt_miss + 32'd1;
            vway <= lru[q_set];
            beat <= '0;
            if (valid[q_set][lru[q_set]] && dirty[q_set][lru[q_set]]) begin
              cnt_wb <= cnt_wb + 32'd1;
              st <= WB;
            end else
              st <= FILL;
          end
        end
        WB: begin
          if (!m_pending && !m_valid) begin
            m_valid <= 1'b1; m_we <= 1'b1;
            m_addr <= {tag[q_set][vway], q_set, beat, 2'b00};
            m_pending <= 1'b1;
          end
          if (m_pending && m_ack) begin
            m_valid <= 1'b0; m_pending <= 1'b0;
            if (beat == 3'(WORDS - 1)) begin
              beat <= '0;
              st <= FILL;
            end else
              beat <= beat + 3'd1;
          end
        end
        FILL: begin
          if (!m_pending && !m_valid) begin
            m_valid <= 1'b1; m_we <= 1'b0;
            m_addr <= {q_tag, q_set, beat, 2'b00};
            m_pending <= 1'b1;
          end
          if (m_pending && m_ack) begin
            m_valid <= 1'b0; m_pending <= 1'b0;
            data[q_set][vway][beat] <= m_rdata;
            if (beat == 3'(WORDS - 1)) begin
              valid[q_set][vway] <= 1'b1;
              dirty[q_set][vway] <= 1'b0;
              tag[q_set][vway]   <= q_tag;
              st <= RESP;
            end else
              beat <= beat + 3'd1;
          end
        end
        RESP: begin
          // perform the original access on the freshly filled line
          if (q_we) begin
            for (int b = 0; b < 4; b++)
              if (q_be[b])
                data[q_set][vway][q_word][8*b +: 8] <= q_wdata[8*b +: 8];
            dirty[q_set][vway] <= 1'b1;
          end else
            c_rdata <= data[q_set][vway][q_word];
          lru[q_set] <= ~vway;
          c_ack <= 1'b1;
          st <= IDLE;
        end
        default: st <= IDLE;
      endcase
    end
  end

`ifndef SYNTHESIS
  /* verilator lint_off SYNCASYNCNET */
  always_ff @(posedge clk) if (rst_n) begin
    // one-outstanding contract on the memory face
    if (m_valid) assert (st == WB || st == FILL)
      else $fatal(1, "dcache: m_valid outside WB/FILL");
  end
  /* verilator lint_on SYNCASYNCNET */
`endif
endmodule
