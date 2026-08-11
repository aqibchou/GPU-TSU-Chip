// p-dit cell (M10): 27-way categorical sampler, CDF-scan design — settled
// against golden/pdit.py study (worst TVD 1.1e-4; 16 prng bits/sample).
// Bit-true golden: golden/pdit.py z_from_acc() + sample_symbol().
// z pipeline (acc s1.10.3 x beta u2.6, sign-magnitude half-away to units
// 2^-8, |z| <= 3071) is IDENTICAL to pbit_cell stage 1 — frozen contract.
//
// Protocol (symbol-serial):
//   load:   z_load=1 with z_idx, acc, beta  (one symbol per cycle; cell
//           computes and stores signed z_k, tracks running max)
//   clear:  z_clear=1 resets the store + max tracker
//   sample: sample_en=1 with rnd; K_ACTIVE symbols assumed loaded.
//           FSM: WSUM (K cycles, w_k = ROM[min(zmax-z_k, 3071)], S += w_k,
//           w regfile) -> TMUL (T = (u16*S) >> 16) -> SCAN (cum += w_k,
//           emit first k with cum > T). sym_valid pulses with sym.
module pdit_cell #(
  parameter string LUT_FILE = "exp_lut.mem",
  parameter int    K        = 27
) (
  input  logic        clk,
  input  logic        rst_n,
  // z load phase
  input  logic        z_clear,
  input  logic        z_load,
  input  logic [4:0]  z_idx,
  input  logic [13:0] acc,        // s1.10.3
  input  logic [7:0]  beta,       // u2.6
  // sample phase
  input  logic        sample_en,
  input  logic [31:0] rnd,
  output logic        sym_valid,  // 1-cycle pulse
  output logic [4:0]  sym,
  output logic [21:0] s_dbg       // S (sum of weights) behind the decision
);
  localparam int LUT_N = 3072;

  // decision consumes rnd[31:16] only (same reservation as pbit_cell)
  wire _unused_rnd_lo = &{1'b0, rnd[15:0]};

  logic [15:0] rom [LUT_N];
  initial $readmemh(LUT_FILE, rom);

  // ---------------- z store + running max ----------------
  logic signed [12:0] zst [K];       // |z| <= 3071 fits s13
  logic signed [12:0] zmax;
  logic               have_z;

  // z stage: prod/round identical to pbit_cell stage 1
  wire signed [22:0] prod = $signed(acc) * $signed({1'b0, beta});
  wire               neg  = prod < 0;
  wire        [21:0] mag  = neg ? $unsigned(-prod[21:0]) : $unsigned(prod[21:0]);
  wire        [20:0] magr = 21'((mag + 22'd1) >> 1);
  wire        [11:0] msat = (magr > 21'(LUT_N - 1)) ? 12'(LUT_N - 1) : magr[11:0];
  wire signed [12:0] z_in = neg ? -$signed({1'b0, msat}) : $signed({1'b0, msat});

  // ---------------- sample FSM ----------------
  typedef enum logic [1:0] {IDLE, WSUM, TMUL, SCAN} st_e;
  st_e                st;
  logic [4:0]         k;
  logic [15:0]        wreg [K];
  logic [21:0]        ssum;          // <= 27*65535 < 2^21, keep a margin bit
  logic [15:0]        u16;
  logic [21:0]        tthr;
  logic [22:0]        cum;

  // zmax - z_k reaches 6142 (two saturated extremes) — needs 14 signed bits
  wire signed [13:0] zdiff = {zmax[12], zmax} - {zst[k][12], zst[k]};
  wire        [11:0] widx  = (zdiff > 14'sd3071) ? 12'd3071 : zdiff[11:0];
  wire        [15:0] wk    = rom[widx];
  wire [37:0] tprod = u16 * ssum;    // 16 x 22 -> 38 (T = top 22)
  wire _unused_tprod = &{1'b0, tprod[15:0]};

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st        <= IDLE;
      k         <= 5'd0;
      ssum      <= 22'd0;
      u16       <= 16'd0;
      tthr      <= 22'd0;
      cum       <= 23'd0;
      zmax      <= 13'sh1000;        // -4096: below any reachable z
      have_z    <= 1'b0;
      sym_valid <= 1'b0;
      sym       <= 5'd0;
      s_dbg     <= 22'd0;
    end else begin
      sym_valid <= 1'b0;

      if (z_clear) begin
        zmax   <= 13'sh1000;
        have_z <= 1'b0;
      end else if (z_load) begin
        zst[z_idx] <= z_in;
        if (!have_z || z_in > zmax) zmax <= z_in;
        have_z <= 1'b1;
      end

      unique case (st)
        IDLE: if (sample_en) begin
          u16  <= rnd[31:16];
          ssum <= 22'd0;
          k    <= 5'd0;
          st   <= WSUM;
        end
        WSUM: begin
          wreg[k] <= wk;
          ssum    <= ssum + {6'd0, wk};
          if (k == 5'(K - 1)) st <= TMUL;
          else                k  <= k + 5'd1;
        end
        TMUL: begin
          tthr <= tprod[37:16];
          cum  <= 23'd0;
          k    <= 5'd0;
          st   <= SCAN;
        end
        SCAN: begin
          if (cum + {7'd0, wreg[k]} > {1'b0, tthr}) begin
            sym       <= k;
            sym_valid <= 1'b1;
            s_dbg     <= ssum;
            st        <= IDLE;
          end else begin
            cum <= cum + {7'd0, wreg[k]};
            k   <= k + 5'd1;
          end
        end
        default: st <= IDLE;
      endcase
    end
  end
endmodule
