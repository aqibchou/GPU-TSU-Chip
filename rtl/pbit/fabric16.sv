// 4x4 king-graph p-bit fabric with chromatic scheduler (D14, M5).
// Golden: golden/gibbs.py (semantics frozen there and in golden/pbit.py).
// One sweep = colors 0..3; per color: CLR(bias) -> 8 MAC slots -> 2-cycle
// sample -> write-back. Every PRNG stream advances once per color phase;
// site i consumes its word only on phase color(i). Clamped sites never
// update but consume randomness normally.
module fabric16 #(
  parameter string LUT_FILE = "sigmoid_lut.mem"
) (
  input  logic        clk,
  input  logic        rst_n,
  // coupling/bias config: slot 0-7 = J for topology slot k, slot 8 = bias
  input  logic        cfg_we,
  input  logic [3:0]  cfg_site,
  input  logic [3:0]  cfg_slot,
  input  logic [9:0]  cfg_data,
  // run-time controls
  input  logic [15:0] clamp,
  input  logic        s_load,       // IDLE only
  input  logic [15:0] s_init,
  input  logic [7:0]  beta,
  input  logic        bipolar,
  // PRNG farm seeding (stream i = site i)
  input  logic        seed_we,
  input  logic [3:0]  seed_stream,
  input  logic [1:0]  seed_sel,
  input  logic [31:0] seed_word,
  // run control
  input  logic        start,
  input  logic [31:0] sweeps,
  output logic        busy,
  output logic        sweep_pulse,  // 1-cycle pulse at each sweep boundary
  output logic        done,         // 1-cycle pulse when run completes
  output logic [15:0] s_out
);
  // ---------------- topology (elaboration-time constants) ----------------
  function automatic logic [4:0] nbr_f(input int i, input logic [2:0] k);
    int r, c, rr, cc;
    int dr[8] = '{-1, -1, -1, 0, 0, 1, 1, 1};
    int dc[8] = '{-1, 0, 1, -1, 1, -1, 0, 1};
    r = i / 4; c = i % 4;
    rr = r + dr[k]; cc = c + dc[k];
    if (rr < 0 || rr > 3 || cc < 0 || cc > 3) return 5'b0_0000;
    return {1'b1, 4'((rr * 4 + cc))};
  endfunction

  function automatic logic [1:0] color_f(input int i);
    return {1'((i / 4) & 1), 1'(i & 1)};
  endfunction

  // ---------------- storage ----------------
  logic [9:0] jmem [16][9];
  logic [15:0] state;

  always_ff @(posedge clk) begin
    if (cfg_we && cfg_slot <= 4'd8) jmem[cfg_site][cfg_slot] <= cfg_data;
  end

  // ---------------- PRNG farm ----------------
  logic        farm_step;
  logic [15:0][31:0] rnd;
  prng_farm #(.NSTREAMS(16)) u_farm (
    .clk(clk), .rst_n(rst_n),
    .seed_we(seed_we), .seed_stream(seed_stream),
    .seed_sel(seed_sel), .seed_word(seed_word),
    .step(farm_step), .rnd(rnd)
  );

  // ---------------- scheduler FSM ----------------
  typedef enum logic [2:0] { F_IDLE, F_CLR, F_MAC, F_SMP1, F_SMP2, F_WR } fst_e;
  fst_e        fst;
  logic [1:0]  cur_color;
  logic [2:0]  slot;
  logic [31:0] sweeps_q, sweep_cnt;

  wire mac_last = (slot == 3'd7);

  // ---------------- cells ----------------
  logic [15:0] cell_clear, cell_accen, cell_smpen, cell_sval, cell_s;

  for (genvar i = 0; i < 16; i++) begin : g_cell
    localparam logic [1:0] COL = color_f(i);
    // per-slot neighbor constants
    logic       nv;
    logic [3:0] ni;
    always_comb begin
      logic [4:0] t;
      t  = nbr_f(i, slot);
      nv = t[4];
      ni = t[3:0];
    end
    wire active = (cur_color == COL);
    assign cell_clear[i] = (fst == F_CLR)  && active;
    assign cell_accen[i] = (fst == F_MAC)  && active && nv;
    assign cell_smpen[i] = (fst == F_SMP1) && active;

    logic [16:0] p_unused;
    pbit_cell #(.LUT_FILE(LUT_FILE)) u_cell (
      .clk(clk), .rst_n(rst_n), .bipolar(bipolar),
      .acc_clear(cell_clear[i]), .bias(jmem[i][8]),
      .acc_en(cell_accen[i]), .j_val(jmem[i][int'(slot)]), .s_in(state[ni]),
      .acc_en2(1'b0), .j_val2(10'd0), .s_in2(1'b0),   // single-slot legacy
      .sample_en(cell_smpen[i]), .beta(beta), .rnd(rnd[i]),
      .s_valid(cell_sval[i]), .s_out(cell_s[i]), .p_dbg(p_unused)
    );
    wire _unused_p = &{1'b0, p_unused};
  end

  assign farm_step = (fst == F_SMP1);
  assign s_out = state;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fst <= F_IDLE; state <= '0; cur_color <= 2'd0; slot <= 3'd0;
      sweeps_q <= '0; sweep_cnt <= '0;
      busy <= 1'b0; done <= 1'b0; sweep_pulse <= 1'b0;
    end else begin
      done <= 1'b0;
      sweep_pulse <= 1'b0;
      unique case (fst)
        F_IDLE: begin
          if (s_load) state <= s_init;
          if (start) begin
            sweeps_q  <= sweeps;
            sweep_cnt <= '0;
            cur_color <= 2'd0;
            busy      <= 1'b1;
            fst       <= (sweeps == 0) ? F_IDLE : F_CLR;
            if (sweeps == 0) begin busy <= 1'b0; done <= 1'b1; end
          end
        end
        F_CLR:  begin slot <= 3'd0; fst <= F_MAC; end
        F_MAC:  begin
          if (mac_last) fst <= F_SMP1;
          slot <= slot + 3'd1;
        end
        F_SMP1: fst <= F_SMP2;
        F_SMP2: fst <= F_WR;
        F_WR: begin
          for (int i = 0; i < 16; i++) begin
            if (cell_sval[i] && !clamp[i]) state[i] <= cell_s[i];
          end
          if (cur_color == 2'd3) begin
            sweep_pulse <= 1'b1;
            if (sweep_cnt + 1 == sweeps_q) begin
              busy <= 1'b0;
              done <= 1'b1;
              fst  <= F_IDLE;
            end else begin
              sweep_cnt <= sweep_cnt + 1;
              cur_color <= 2'd0;
              fst       <= F_CLR;
            end
          end else begin
            cur_color <= cur_color + 2'd1;
            fst       <= F_CLR;
          end
        end
        default: fst <= F_IDLE;
      endcase
    end
  end
endmodule
