// Scaled sparse-graph p-bit engine (M6): N<=8192 sites, arbitrary topology
// with degree<=8, P time-multiplexed lanes over sync-read URAM/BRAM
// store, chromatic schedule via a precomputed color-major order list,
// multi-entry annealing schedule, flip/update/cycle counters (S3/S4 fuel).
// Golden: golden/gibbs_grid.py BitTrueGridSampler (contract frozen there).
// Clamped sites are excluded from the order list by the loader.
//
// Kria-fit trick #1 (2026-07-11): the flat state register + per-lane
// dynamic bit-indexing synthesized to 84.6k LUTs at NB=12 — 98% of the
// module (ci/logs/g8 autopsy). State now lives in per-lane replicated
// distributed-RAM copies (32 x 1-bit banks, word-writable), fed by ONE
// broadcast write stream. G_WR latches the chunk's <=P writes into a
// write-behind queue that drains 1/cycle under the next chunk's
// fetch/MAC cycles. Correctness by the coloring theorem: within a
// color no site reads a same-color site, so pending writes are never
// read; a color-boundary stall in G_ORD holds a NEW color until the
// queue is dry, and done is only raised from G_DRN (queue empty), so
// idle-time readers always see committed state. Cycle counts shift
// slightly (D-026: cycles are not contract). state persists across
// rst_n (was: reset to zero) — the loader contract writes STATE before
// any run, which every bench and the S7 image format honor.
module fabric_grid #(
  parameter string LUT_FILE = "sigmoid_lut.mem",
  parameter string GLUT_FILE = "glut.mem",
  parameter int NB = 13,                  // site index bits
  parameter int P  = 8                    // update lanes
) (
  input  logic          clk,
  input  logic          rst_n,
  // graph rows: slots 0-7 = {valid, nbr[NB-1:0], J[9:0]} in row_data[23:0];
  // slot 8 = bias (row_data[9:0])
  input  logic          row_we,
  input  logic [NB-1:0] row_addr,
  input  logic [3:0]    row_slot,
  input  logic [29:0]   row_data,   // QSITE S2: bias word carries 3
                                     // 10b lanes; slots use [23:0]
  input  logic          arity_q4,   // 0 = frozen binary path
  input  logic          arity_q8,   // q8 (S4): third plane, 2 draws
  // color-major order list + per-color segment bounds (up to 16 colors)
  input  logic          ord_we,
  input  logic [NB-1:0] ord_addr,
  input  logic [NB-1:0] ord_data,
  input  logic          cb_we,
  input  logic [3:0]    cb_idx,
  input  logic [NB-1:0] cb_start,
  input  logic [NB-1:0] cb_end,
  input  logic [4:0]    n_colors,
  // annealing schedule (up to 32 entries)
  input  logic          sch_we,
  input  logic [4:0]    sch_idx,
  input  logic [7:0]    sch_beta,
  input  logic [23:0]   sch_sweeps,
  input  logic [5:0]    n_sched,
  input  logic          bipolar,
  // state init, 32-bit words
  input  logic          stw_we,
  input  logic [8:0]    stw_addr,  // q4 STATE: 16 sites/word
  //                       (NB < 13 leaves the upper bits unread)
  input  logic [31:0]   stw_data,
  // PRNG farm seeding (P streams, one per lane)
  input  logic          seed_we,
  input  logic [$clog2(P)-1:0] seed_stream,
  input  logic [1:0]    seed_sel,
  input  logic [31:0]   seed_word,
  // idle-time scan/drain ports (async reads, valid while gst is idle;
  // served by lane 0/1 read ports — s_cluster scans and PDRAIN only
  // run while the engine is idle, per the one-command rule)
  input  logic [NB-1:0] sA_addr,
  input  logic [NB-1:0] sB_addr,
  output logic [31:0]   sA_word,
  output logic          sA_bit,
  output logic          sB_bit,
  output logic [2:0]    sA_val,     // q-value taps (p2,p1,p0)
  output logic [2:0]    sB_val,
  output logic [31:0]   sA_word2,   // plane-1 row word (STATE-q drain)
  output logic [31:0]   sA_word3,   // plane-2 row word (q8)
  // run control + observability
  input  logic          start,
  output logic          busy,
  output logic          sweep_pulse,
  output logic          done,
  output logic [47:0]   upd_cnt,
  output logic [47:0]   flip_cnt,
  output logic [63:0]   cycle_cnt,
  // SIM-ONLY mirror of committed state for the legacy direct benches;
  // tied off in synthesis (s_cluster uses the sA/sB ports instead)
  output logic [(1<<NB)-1:0] state_flat,
  output logic [(1<<NB)-1:0] state_flat2,
  output logic [(1<<NB)-1:0] state_flat3
);
  generate if (NB < 13) begin : g_stwslack
    wire _unused_stw = &{1'b0, stw_addr[8:NB-4]};
  end endgenerate
  localparam int NMAX = 1 << NB;

  // ---------------- memories (D-026: sync-read URAM/BRAM class) -------
  // One private copy per lane guarantees simple-dual-port inference
  // (one write stream, one registered read); async multi-port reads
  // synthesized as a 408k-LUT mux forest at NB=10 (G8 evidence).
  logic [NB-1:0] cb_s [16], cb_e [16];
  logic [7:0]  sch_b [32];
  logic [23:0] sch_n [32];

  always_ff @(posedge clk) begin
    if (cb_we) begin cb_s[cb_idx] <= cb_start; cb_e[cb_idx] <= cb_end; end
    if (sch_we) begin sch_b[sch_idx] <= sch_beta; sch_n[sch_idx] <= sch_sweeps; end
  end

  // ---------------- PRNG farm ----------------
  logic farm_step;
  logic [P-1:0][31:0] rnd;
  prng_farm #(.NSTREAMS(P)) u_farm (
    .clk(clk), .rst_n(rst_n),
    .seed_we(seed_we), .seed_stream(seed_stream),
    .seed_sel(seed_sel), .seed_word(seed_word),
    .step(farm_step), .rnd(rnd)
  );

  // ---------------- scheduler ----------------
  // G_ORD fetches the chunk's order entries (sync read); G_BIA fetches
  // bias + slot 0; G_MAC consumes slot s while presenting s+1. Read
  // addresses are chunk-stable, so the read registers self-refresh.
  typedef enum logic [3:0] { G_IDLE, G_ORD, G_BIA, G_CLR, G_MAC,
                             G_SMP1, G_SMP1B, G_SMP2, G_SMP2B, G_WR,
                             G_DRN } gst_e;
  gst_e          gst;
  logic [4:0]    entry;
  logic [23:0]   sweep_in_entry;
  logic [3:0]    cur_color;
  logic [NB-1:0] pos;
  logic [7:0]    beta_q;

  // lane bindings for the current chunk
  logic [P-1:0]        lane_v;
  always_comb
    for (int l = 0; l < P; l++)
      lane_v[l] = ((pos + NB'(l)) < cb_e[cur_color]);

  // ---------------- write-behind state store ----------------
  // queue: the chunk's <=P site writes, latched at G_WR
  logic [NB-1:0] wq_a [P];
  logic [2:0]    wq_d [P];
  logic [P-1:0]  wq_v;
  logic [3:0]    wq_color;
  logic [$clog2(P)-1:0] wq_head;

  // post-reset zeroing sweep: the frozen contract (golden + S7 gate,
  // whose images may omit STATE) starts every life at state==0; the
  // dRAM banks cannot async-reset, so a counter walks every word
  // through the broadcast stream right after rst_n (<= NMAX/32 cycles,
  // over long before any command can reach the loader)
  logic [NB-5:0] init_cnt;
  wire init_run = ~init_cnt[NB-5];

  // one broadcast write stream into every replicated copy:
  // reset sweep, queue drain (single-bit), or IDLE-time word init
  logic          stv_we;
  logic [31:0]   stv_mask;
  logic [NB-6:0] stv_waddr;
  logic [31:0]   stv_wdata;
  logic [31:0]   stv_wdata2;                 // plane 1 (QSITE)
  logic [31:0]   stv_wdata3;                 // plane 2 (q8, S4)
  logic [15:0]   stw_p0, stw_p1;             // q4 STATE-word deinterleave
  logic [7:0]    st8_p0, st8_p1, st8_p2;     // q8: 8 sites x 4b lanes
  always_comb begin
    for (int i = 0; i < 16; i++) begin
      stw_p0[i] = stw_data[2 * i];
      stw_p1[i] = stw_data[2 * i + 1];
    end
    for (int i = 0; i < 8; i++) begin
      st8_p0[i] = stw_data[4 * i];
      st8_p1[i] = stw_data[4 * i + 1];
      st8_p2[i] = stw_data[4 * i + 2];
    end
    wq_head = '0;
    for (int k = P - 1; k >= 0; k--)
      if (wq_v[k]) wq_head = ($clog2(P))'(k);
    stv_we     = 1'b0;
    stv_mask   = '0;
    stv_waddr  = '0;
    stv_wdata  = '0;
    stv_wdata2 = '0;
    stv_wdata3 = '0;
    if (init_run) begin
      stv_we    = 1'b1;
      stv_waddr = init_cnt[NB-6:0];
      stv_mask  = '1;
      stv_wdata = '0;
    end else if (wq_v != '0) begin
      stv_we     = 1'b1;
      stv_waddr  = wq_a[wq_head][NB-1:5];
      stv_mask   = 32'd1 << wq_a[wq_head][4:0];
      stv_wdata  = {32{wq_d[wq_head][0]}};
      stv_wdata2 = {32{wq_d[wq_head][1]}};
      stv_wdata3 = {32{wq_d[wq_head][2]}};
    end else if (stw_we && gst == G_IDLE) begin
      stv_we    = 1'b1;
      if (arity_q8) begin
        // spec §12: a q8 STATE word covers 8 sites (4b lanes, top bit
        // 0); four words fill one 32-site store row
        stv_waddr  = {1'b0, stw_addr[NB-5:2]};
        stv_mask   = 32'($unsigned(32'h000000FF)) << {stw_addr[1:0], 3'b0};
        stv_wdata  = 32'(st8_p0) << {stw_addr[1:0], 3'b0};
        stv_wdata2 = 32'(st8_p1) << {stw_addr[1:0], 3'b0};
        stv_wdata3 = 32'(st8_p2) << {stw_addr[1:0], 3'b0};
      end else if (arity_q4) begin
        // spec §12: a q4 STATE word covers 16 sites (2b lanes); two
        // words fill one 32-site store row
        stv_waddr  = stw_addr[NB-5:1];
        stv_mask   = stw_addr[0] ? 32'hFFFF0000 : 32'h0000FFFF;
        stv_wdata  = stw_addr[0] ? {stw_p0, 16'b0} : {16'b0, stw_p0};
        stv_wdata2 = stw_addr[0] ? {stw_p1, 16'b0} : {16'b0, stw_p1};
      end else begin
        stv_waddr = stw_addr[NB-6:0];
        stv_mask  = '1;
        stv_wdata = stw_data;
      end
    end
  end

  // per-lane replicated sync-read memories + state-store copies.
  // Dual-slot MAC (2026-07-12): slots are consumed in PAIRS —
  // (0,1)(2,3)(4,5)(6,7) — via smem's second URAM port (write-only
  // during config, free during runs). Fetch pipeline: pair p's rows +
  // neighbor bits are STAGED one cycle before consumption, so the
  // async state-store read ends at a register instead of the
  // accumulator (the -0.128 ns path). Chunk: 15 -> 10 cycles.
  logic [1:0] mslot;
  wire [1:0] rd_pair = (gst == G_CLR)                  ? 2'd1
                     : (gst == G_MAC && mslot == 2'd0) ? 2'd2
                     : (gst == G_MAC && mslot == 2'd1) ? 2'd3
                     : 2'd0;
  logic [23:0]   slot_rq  [P];
  logic [23:0]   slot_rq2 [P];
  logic [69:0]   bias_rq [P];
  logic [NB-1:0] lane_site [P];       // = ord_rq, chunk-stable
  logic [P-1:0]  st_bitE, st_bitO;
  logic [2:0]    st_valE [P];
  logic [2:0]    st_valO [P];
  logic [31:0]   st_wordE [P];
  logic [31:0]   st_wordO [P];
  logic [31:0]   st_wordE2 [P];
  logic [31:0]   st_wordO2 [P];
  logic [31:0]   st_wordE3 [P];
  logic [31:0]   st_wordO3 [P];
  logic [9:0]    mac_j0 [P];
  logic [9:0]    mac_j1 [P];
  logic [P-1:0]  mac_v0, mac_v1;
  logic [2:0]    mac_s0 [P];
  logic [2:0]    mac_s1 [P];
  // q8 ROWS bias assembly (spec §12: 3 bias words/site): slots 8,9
  // stash 30b lanes each; slot 10 carries b7 and commits the 70b row
  logic [59:0]   bias_stash;
  always_ff @(posedge clk)
    if (row_we) begin
      if (row_slot == 4'd8) bias_stash[29:0]  <= row_data[29:0];
      if (row_slot == 4'd9) bias_stash[59:30] <= row_data[29:0];
    end
  wire        bm_we = row_we && ((row_slot == 4'd8 && !arity_q8)
                                 || row_slot == 4'd10);
  wire [69:0] bm_wd = arity_q8 ? {row_data[9:0], bias_stash}
                               : {40'b0, row_data[29:0]};
  for (genvar l = 0; l < P; l++) begin : g_mem
    // slot-PAIR packing: one 48b word holds slots (2k, 2k+1), so ONE
    // SDP URAM read port fetches a pair per cycle — no second port
    // (smem already occupies all 64 device URAMs as SDP; TDP/BRAM
    // fallbacks blow the device, DRC-caught twice on 2026-07-12).
    // The ROWS loader streams slots 0..7 in order (image format,
    // gate-verified), so writes pair up through a one-word stash.
    (* ram_style = "ultra" *) logic [47:0] smem [NMAX*4];
    // 70b = the q8 bias payload; URAM is 72b native, so the q4-era
    // 30b row grows free in URAM count (S4 pre-registration)
    (* ram_style = "ultra" *) logic [69:0]   bmem [NMAX];
    (* ram_style = "ultra" *) logic [NB-1:0] omem [NMAX];
    logic [23:0] row_stash;
    logic [47:0] pair_rq;
    always_ff @(posedge clk) begin
      if (row_we && row_slot < 4'd8) begin
        if (!row_slot[0])
          row_stash <= row_data[23:0];          // even slot: stash
        else
          smem[{row_addr, row_slot[2:1]}] <= {row_data[23:0], row_stash};
      end else begin
        pair_rq <= smem[{lane_site[l], rd_pair}];
      end
    end
    always_comb begin
      slot_rq[l]  = pair_rq[23:0];              // even slot
      slot_rq2[l] = pair_rq[47:24];             // odd slot
    end
    always_ff @(posedge clk) begin
      if (bm_we) bmem[row_addr] <= bm_wd;
      if (ord_we) omem[ord_addr] <= ord_data;
      lane_site[l] <= omem[NB'(pos + NB'(l))];
      bias_rq[l]   <= bmem[lane_site[l]];
    end
    // state copy: TWO async 1-bit reads per lane (even/odd slot
    // neighbors during the MAC fetch; port E doubles for the G_WR
    // flip-compare and, on lanes 0/1, the idle-time scan/drain
    // ports), one broadcast write port. Inference replicates banks.
    logic [NB-1:0] rd_idxE, rd_idxO;
    always_comb begin
      rd_idxE = (gst == G_WR) ? lane_site[l] : slot_rq[l][10 +: NB];
      if (gst == G_IDLE) begin
        if (l == 0) rd_idxE = sA_addr;
        if (l == 1) rd_idxE = sB_addr;
      end
      rd_idxO = slot_rq2[l][10 +: NB];
    end
    for (genvar b = 0; b < 32; b++) begin : g_st
      (* ram_style = "distributed" *) logic stb  [NMAX/32];
      (* ram_style = "distributed" *) logic stb2 [NMAX/32];
      (* ram_style = "distributed" *) logic stb3 [NMAX/32];
      initial for (int i = 0; i < NMAX/32; i++) begin
        stb[i] = 1'b0;
        stb2[i] = 1'b0;
        stb3[i] = 1'b0;
      end
      always_ff @(posedge clk)
        if (stv_we && stv_mask[b]) begin
          stb[stv_waddr]  <= stv_wdata[b];
          stb2[stv_waddr] <= stv_wdata2[b];
          stb3[stv_waddr] <= stv_wdata3[b];
        end
      assign st_wordE[l][b]  = stb[rd_idxE[NB-1:5]];
      assign st_wordO[l][b]  = stb[rd_idxO[NB-1:5]];
      assign st_wordE2[l][b] = stb2[rd_idxE[NB-1:5]];
      assign st_wordO2[l][b] = stb2[rd_idxO[NB-1:5]];
      assign st_wordE3[l][b] = stb3[rd_idxE[NB-1:5]];
      assign st_wordO3[l][b] = stb3[rd_idxO[NB-1:5]];
    end
    assign st_bitE[l] = st_wordE[l][rd_idxE[4:0]];
    assign st_bitO[l] = st_wordO[l][rd_idxO[4:0]];
    assign st_valE[l] = {st_wordE3[l][rd_idxE[4:0]],
                         st_wordE2[l][rd_idxE[4:0]], st_bitE[l]};
    assign st_valO[l] = {st_wordO3[l][rd_idxO[4:0]],
                         st_wordO2[l][rd_idxO[4:0]], st_bitO[l]};
    // MAC stage: pair p's (J, valid, neighbor bit) registered one
    // cycle ahead of consumption (values are junk outside the
    // CLR..MAC window and never consumed there)
    always_ff @(posedge clk) begin
      mac_j0[l] <= slot_rq[l][9:0];
      mac_v0[l] <= slot_rq[l][23];
      mac_s0[l] <= st_valE[l];
      mac_j1[l] <= slot_rq2[l][9:0];
      mac_v1[l] <= slot_rq2[l][23];
      mac_s1[l] <= st_valO[l];
    end
  end
  assign sA_word = st_wordE[0];
  assign sA_word2 = st_wordE2[0];
  assign sA_word3 = st_wordE3[0];
  assign sA_bit  = st_bitE[0];
  assign sB_bit  = st_bitE[P > 1 ? 1 : 0];
  assign sA_val  = st_valE[0];
  assign sB_val  = st_valE[P > 1 ? 1 : 0];

  wire chunk_last = ((pos + NB'(P)) >= cb_e[cur_color]);
  wire color_last = ({1'b0, cur_color} + 5'd1 >= n_colors);

  // ---------------- lanes ----------------
  wire q_on = arity_q4 | arity_q8;
  logic [P-1:0] l_clear, l_accen, l_accen2, l_smpen, l_smpen2, l_sval,
                l_s;
  logic [P-1:0] lq_sval;
  logic [2:0]   lq_s [P];
  logic [2:0]   l_val [P];            // committed-value mux (arity)
  for (genvar l = 0; l < P; l++) begin : g_lane
    assign l_clear[l]  = (gst == G_CLR)  && lane_v[l];
    assign l_accen[l]  = (gst == G_MAC)  && lane_v[l] && mac_v0[l];
    assign l_accen2[l] = (gst == G_MAC)  && lane_v[l] && mac_v1[l];
    assign l_smpen[l]  = (gst == G_SMP1) && lane_v[l];
    assign l_smpen2[l] = (gst == G_SMP1B) && lane_v[l];
    // GLUT prefetch on the last MAC cycle (farm word 1 is stable
    // through the whole MAC phase — the BRAM read costs no cycle)
    wire l_gpre = (gst == G_MAC) && (mslot == 2'd3);
    logic [16:0] p_nc;
    pbit_cell #(.LUT_FILE(LUT_FILE)) u_cell (
      .clk(clk), .rst_n(rst_n), .bipolar(bipolar),
      .acc_clear(l_clear[l] && !q_on), .bias(bias_rq[l][9:0]),
      .acc_en(l_accen[l] && !q_on), .j_val(mac_j0[l]),
      .s_in(mac_s0[l][0]),
      .acc_en2(l_accen2[l] && !q_on), .j_val2(mac_j1[l]),
      .s_in2(mac_s1[l][0]),
      .sample_en(l_smpen[l] && !q_on), .beta(beta_q), .rnd(rnd[l]),
      .s_valid(l_sval[l]), .s_out(l_s[l]), .p_dbg(p_nc)
    );
    qsite_cell #(.GLUT_FILE(GLUT_FILE)) u_qcell (
      .clk(clk), .rst_n(rst_n), .arity8(arity_q8),
      .acc_clear(l_clear[l] && q_on), .bias(bias_rq[l]),
      .acc_en(l_accen[l] && q_on), .j_val(mac_j0[l]),
      .s_in(mac_s0[l]),
      .acc_en2(l_accen2[l] && q_on), .j_val2(mac_j1[l]),
      .s_in2(mac_s1[l]),
      .glut_pre(l_gpre),
      .sample_en(l_smpen[l] && q_on), .sample_en2(l_smpen2[l]),
      .beta(beta_q), .rnd(rnd[l]),
      .s_valid(lq_sval[l]), .s_out(lq_s[l])
    );
    assign l_val[l] = q_on ? lq_s[l] : {2'b00, l_s[l]};
    wire _unused_p = &{1'b0, p_nc};
  end

  // D-014-q8 (spec §12): the farm advances TWICE per non-empty chunk
  // at q8 — both words drawn in order before any lane decides
  assign farm_step  = (gst == G_SMP1) || (gst == G_SMP1B);

  // sim-only committed-state mirror for the legacy direct benches;
  // written by the same broadcast stream the copies consume
`ifdef SYNTHESIS
  assign state_flat = '0;
  assign state_flat2 = '0;
  assign state_flat3 = '0;
`else
  logic [NMAX-1:0] state_mirror;
  logic [NMAX-1:0] state_mirror2;
  logic [NMAX-1:0] state_mirror3;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_mirror <= '0;
      state_mirror2 <= '0;
      state_mirror3 <= '0;
    end else if (stv_we)
      for (int b = 0; b < 32; b++)
        if (stv_mask[b]) begin
          state_mirror[32'({stv_waddr, 5'b0}) + b] <= stv_wdata[b];
          state_mirror2[32'({stv_waddr, 5'b0}) + b] <= stv_wdata2[b];
          state_mirror3[32'({stv_waddr, 5'b0}) + b] <= stv_wdata3[b];
        end
  end
  assign state_flat = state_mirror;
  assign state_flat2 = state_mirror2;
  assign state_flat3 = state_mirror3;
  // tripwires: queue dry by the next chunk's G_WR; nothing may write
  // or start while the reset sweep is still walking the store
  /* verilator lint_off SYNCASYNCNET */
  always @(posedge clk) begin
    if (rst_n && gst == G_WR && wq_v != '0)
      $fatal(1, "fabric_grid: write-behind queue not drained at G_WR");
    if (rst_n && init_run && (stw_we || start))
      $fatal(1, "fabric_grid: stw/start during the post-reset zero sweep");
  end
  /* verilator lint_on SYNCASYNCNET */
`endif

  // per-chunk counter increments (combinational)
  logic [3:0] cnt_upd, cnt_flip;
  always_comb begin
    cnt_upd  = '0;
    cnt_flip = '0;
    if (gst == G_WR) begin
      for (int l = 0; l < P; l++) begin
        if ((q_on ? lq_sval[l] : l_sval[l]) && lane_v[l]) begin
          cnt_upd = cnt_upd + 4'd1;
          if (st_valE[l] != l_val[l]) cnt_flip = cnt_flip + 4'd1;
        end
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      gst <= G_IDLE; entry <= '0; sweep_in_entry <= '0;
      cur_color <= '0; pos <= '0; mslot <= '0; beta_q <= '0;
      busy <= 1'b0; done <= 1'b0; sweep_pulse <= 1'b0;
      upd_cnt <= '0; flip_cnt <= '0; cycle_cnt <= '0;
      wq_v <= '0; wq_color <= '0;
      init_cnt <= '0;
    end else begin
      done <= 1'b0;
      sweep_pulse <= 1'b0;
      if (init_run) init_cnt <= init_cnt + 1'b1;
      if (busy) cycle_cnt <= cycle_cnt + 64'd1;

      // queue drain: one broadcast write per cycle (G_WR refills below)
      if (wq_v != '0) wq_v[wq_head] <= 1'b0;

      unique case (gst)
        G_IDLE: if (start) begin
          entry <= '0; sweep_in_entry <= '0; cur_color <= '0;
          beta_q <= sch_b[0];
          pos <= cb_s[0];
          busy <= 1'b1;
          gst  <= G_ORD;
        end
        G_ORD:  begin
          if (wq_v != '0 && cur_color != wq_color) begin
            // color-boundary drain stall: a NEW color may read sites
            // the queue has not committed yet; hold until dry
          end else if (pos >= cb_e[cur_color]) begin
            // EMPTY color segment (e.g. every site of a color clamped):
            // golden consumes no PRNG for it (gibbs_grid sweep skips
            // lo==hi), so skip without stepping the farm. Latent since M6
            // — every prior trajectory-diffed config had non-empty colors;
            // found by the T2 grammar rig (all 12 visibles = one color).
            if (!color_last) begin
              cur_color <= cur_color + 4'd1;
              pos <= cb_s[cur_color + 4'd1];
            end else begin
              sweep_pulse <= 1'b1;
              cur_color <= '0;
              pos <= cb_s[0];
              if (sweep_in_entry + 24'd1 < sch_n[entry]) begin
                sweep_in_entry <= sweep_in_entry + 24'd1;
              end else if ({1'b0, entry} + 6'd1 < n_sched) begin
                entry <= entry + 5'd1;
                sweep_in_entry <= '0;
                beta_q <= sch_b[entry + 5'd1];
              end else begin
                gst <= G_DRN;
              end
            end
          end else gst <= G_BIA;
        end
        G_BIA: gst <= G_CLR;             // bias/slot0 reads in flight
        G_CLR: begin
          mslot <= '0;
          gst <= G_MAC;
        end
        G_MAC:  begin
          if (mslot == 2'd3) gst <= G_SMP1;
          mslot <= mslot + 2'd1;
        end
        G_SMP1: gst <= arity_q8 ? G_SMP1B : G_SMP2;
        G_SMP1B: gst <= G_SMP2;         // q8: best1 + word-2 GLUT fetch
        G_SMP2: gst <= arity_q8 ? G_SMP2B : G_WR;
        G_SMP2B: gst <= G_WR;           // q8: pass-2 argmax + final
        G_WR: begin
          upd_cnt  <= upd_cnt  + 48'(cnt_upd);
          flip_cnt <= flip_cnt + 48'(cnt_flip);
          for (int l = 0; l < P; l++) begin
            wq_a[l] <= lane_site[l];
            wq_d[l] <= l_val[l];
            wq_v[l] <= (q_on ? lq_sval[l] : l_sval[l]) && lane_v[l];
          end
          wq_color <= cur_color;
          if (!chunk_last) begin
            pos <= pos + NB'(P);
            gst <= G_ORD;
          end else if (!color_last) begin
            cur_color <= cur_color + 4'd1;
            pos <= cb_s[cur_color + 4'd1];
            gst <= G_ORD;
          end else begin
            sweep_pulse <= 1'b1;
            cur_color <= '0;
            pos <= cb_s[0];
            if (sweep_in_entry + 24'd1 < sch_n[entry]) begin
              sweep_in_entry <= sweep_in_entry + 24'd1;
              gst <= G_ORD;
            end else if ({1'b0, entry} + 6'd1 < n_sched) begin
              entry <= entry + 5'd1;
              sweep_in_entry <= '0;
              beta_q <= sch_b[entry + 5'd1];
              gst <= G_ORD;
            end else begin
              gst <= G_DRN;
            end
          end
        end
        G_DRN: if (wq_v == '0) begin
          // all writes committed: idle-time readers see final state
          busy <= 1'b0;
          done <= 1'b1;
          gst  <= G_IDLE;
        end
        default: gst <= G_IDLE;
      endcase
    end
  end
endmodule
