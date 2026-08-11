// s_cluster.sv — the sampling ISA engine in the D4 socket (M21, S7σ).
// Implements docs/HARDWARE_ARCHITECTURE.md#sampling-isa exactly; bit-frozen oracle is
// golden/sampling_isa.py. Construction decisions: docs/HARDWARE_ARCHITECTURE.md#sampling-cluster-construction
// (fabric_grid driven ONE SWEEP AT A TIME; shadow config for D30/D33;
// shares the sidecar's arbiter port upstream via an engine mux).
//
// Ops (T_OP): 8 PCONFIG (DMA image -> fabric ports + shadows, D33 work
// scans around ROWS), 9 PSAMPLE (IMM/SCHED sweeps, D19 accumulate at
// sweep ends), 10 PDRAIN (stream §6 layouts to memory; mutates nothing).
//
// ACCWALK (D-034, sampling_isa_spec §13): the D19 accumulate is a
// SNAPSHOT + OVERLAPPED WALK — P_SNAP copies the post-sweep planes
// (idle sA port, one 32-site word/strobe), the next sweep starts
// immediately, and a 2-site/cycle walker (§13 v2) commits the same §4 math
// into the slot-banked m1/m2 while it runs. Drained values are
// bit-identical to the retired serial P_ACC by construction; busy
// holds until the tail walk commits (the PDRAIN visibility contract).
//
// Memory face is the v2 edge/credit contract (mem_spec §1b, D-032c):
// m_req is a one-cycle pulse per beat, completions return strictly in
// order, CRED beats in flight. Both walkers are credit-native
// (D-032c stages C1 + C3): the PDRAIN streamer never waits for a
// completion, and the PCONFIG walker streams each section's beats
// while its consume logic fires on completion pulses (sections
// barrier at data-dependent boundaries by construction — lengths are
// latched at dispatch). busy holds until every beat has returned, so
// !busy means memory-visible.
module s_cluster #(
  parameter int NB = 13,                    // site index bits (n <= 8192)
  parameter int P  = 8,
  parameter int unsigned CRED = 4,          // memory-face credits (1..4)
  parameter string LUT_FILE = "sigmoid_lut.mem",
  parameter string GLUT_FILE = "glut.mem"
) (
  input  logic        clk,
  input  logic        rst_n,
  // command interface (from the simt_core T-CSR block; GO routed here
  // for T_OP 8-10; one outstanding globally per spec I-1/I-2)
  input  logic        go,
  input  logic [2:0]  op,                   // 0=PCONFIG 1=PSAMPLE 2=PDRAIN (pre-decoded T_OP-8)
  input  logic [31:0] t_a,                  // PCONFIG image base
  input  logic [31:0] t_c,                  // PDRAIN dest base
  input  logic [23:0] t_m,                  // PCONFIG words / IMM sweeps
  input  logic [7:0]  t_k,                  // IMM beta
  input  logic [8:0]  t_flags,              // per-op flag map (spec §1)
  output logic        busy,
  // memory face — v2 edge/credit (mem_spec §1b); dedicated arbiter
  // port, in-order completion pulses
  output logic        m_req,
  output logic        m_we,
  output logic [31:0] m_addr,
  output logic [31:0] m_wdata,
  output logic [3:0]  m_be,
  input  logic        m_rsp_valid,
  input  logic [31:0] m_rdata,
  // observability
  output logic [31:0] cnt_sops
);
  localparam int NMAX = 1 << NB;

  // ---------------- T_FLAGS bit map (spec §1) ----------------
  localparam int CF_ROWS = 0, CF_WORK_RESET = 8;
  // section flag bits 1-5 (ORDER/SCHED/SEEDS/STATE/RID) are consumed
  // ordinally by the C_SECT dispatcher walk
  localparam int SF_RECORD = 0, SF_STATS_RESET = 1, SF_IMM = 2;

  // ---------------- header state (persistent) ----------------
  logic [15:0] n_sites;
  logic        bipolar_r, work_track_r;     // header bits 16/17 (rewritten每 PCONFIG)
  logic [7:0]  n_sched_r, n_replicas_r;

  // ---------------- shadow config (D30/D33 need read access) ----------
  // D-028: shadows + accumulators are sync-read memories (URAM/BRAM);
  // the flat-FF forms were the fit killer (m2 alone = 262k FF at
  // NB=10). One write + one sync read port each; every reader below
  // became a 2-phase walker (fetch/commit) or the phased drain
  // pipeline, and SF_STATS_RESET became the P_ZERO walk (D-026:
  // cycles are not contract). sh_rid is word-organized (4 rids/word)
  // because the RID section writes 4 per beat.
  // sh_slot is {PARITY, SLOT}-BANKED (D-034 §13 v2, WALK_W=2): 16
  // banks x NMAX/2 rows, same total bits; the walker reads a full
  // SITE-PAIR row (both sites' 8 slot words) per cycle; every legacy
  // single-word reader routes through the registered bank select —
  // bank = the full address's low 4 bits, values/timing identical.
  // sh_slot memory lives per-bank in g_bank below ({valid,nbr,J} words):
  // the 2-D-array form (one logical array, 16 banks looped in one
  // always_ff) is NOT a RAM template — Vivado dissolved it into 191k
  // FFs + 77k mux LUTs (Synth 8-7186 "incorrect usage"; placement
  // fail, 2026-07-16 diag on the VM). Per-bank generate blocks give
  // each bank exactly one sync read + one decoded write.
  (* ram_style = "block" *) logic [69:0] sh_bias [NMAX]; // 7 lanes (§12;
  // q4 uses lanes 1..3, q8 all 7 — b0 == 0 always)
  (* ram_style = "block" *) logic [31:0] sh_ridw [NMAX/4];
  logic [7:0]  sch_beta_q [32];
  logic [23:0] sch_sweeps_q [32];

  // ---------------- accumulators (D19) ----------------
  logic [31:0] acc_cnt;
  // m1 is {site, lane[2:0]}: binary uses lane 0 (counts state 1);
  // q4 uses lanes 0..2 (states 1..3); q8 lanes 0..6 (states 1..7).
  // Drain words stay bit-identical for binary/q4 images (§12 / Q1) —
  // the lane field widened for S4 but the (site, lane) values are
  // unchanged.
  // m1 and m2 are {PARITY, LANE/SLOT}-BANKED (§13 v2): 16 banks x
  // NMAX/2 each, same bits; the walker commits BOTH pair sites' RMWs
  // per cycle (2 m1 + up to 16 m2) and P_ZERO zeroes all 32 banks in
  // lock-step (65,536 -> 4,096 cycles per RESET at NB=13); drains
  // keep identical wire order via the 4-bit selects.
  // m1/m2 memories live per-bank in g_bank below (same reason as
  // sh_slot: per-bank generate = inferable 1R+1W simple dual port).

  // sync read ports (one per memory; addresses muxed by state below).
  // shs_ra/m2_ra stay FULL {site, slot} addresses for every legacy
  // reader; the banks read row addr[NB+2:3] and the registered
  // selects reconstruct the addressed word — bit- and cycle-identical.
  logic [NB+2:0] shs_ra, m2_ra;
  logic [NB+2:0] m1_ra;
  logic [NB-1:0] shb_ra;
  logic [NB-3:0] shr_ra;
  logic [23:0] shs_rqb [16];
  logic [31:0] m2_rqb [16];
  logic [31:0] m1_rqb [16];
  logic [3:0]  shs_sel_q, m2_sel_q, m1_sel_q;
  wire  [23:0] shs_rq = shs_rqb[shs_sel_q];
  // NB < 13 leaves the slot word's nbr-field slack bits unread
  generate if (NB < 13) begin : g_shslack
    wire _unused_shs = &{1'b0, shs_rq[22:10+NB]};
  end endgenerate
  wire  [31:0] m2_rq  = m2_rqb[m2_sel_q];
  wire  [31:0] m1_rq  = m1_rqb[m1_sel_q];
  logic [31:0] shr_rq;
  logic [69:0] shb_rq;
  always_ff @(posedge clk) begin
    shs_sel_q <= shs_ra[3:0];
    m2_sel_q  <= m2_ra[3:0];
    m1_sel_q  <= m1_ra[3:0];
    shb_rq <= sh_bias[shb_ra];
    shr_rq <= sh_ridw[shr_ra];
  end

  // registered write ports. The memories must live in reset-free
  // blocks — a body under "posedge clk or negedge rst_n" cannot infer
  // block/ultra RAM (Vivado Synth 8-3391; benches never see it). The
  // one-cycle write lag is read-safe by schedule: every consumer of a
  // written address is >= 2 cycles downstream (walker cadences, E-scan
  // entry, command boundaries); same-cycle port collisions land only
  // in read registers that are re-fetched before consumption.
  logic        shs_we, shb_we, shr_we;
  logic [15:0] m1_web, m2_web;         // per-bank strobes (§13 v2)
  logic [NB+2:0] shs_wa;
  logic [NB-2:0] m1_wab [16];
  logic [NB-2:0] m2_wab [16];
  logic [31:0]   m1_wdb [16];
  logic [31:0]   m2_wdb [16];
  logic [NB-1:0] shb_wa;
  logic [NB-3:0] shr_wa;
  logic [23:0] shs_wd;
  logic [31:0] shr_wd;
  logic [69:0] shb_wd;
  logic [59:0] shb_stash;              // q8 ROWS: slots 8,9 lanes
  always_ff @(posedge clk) begin
    if (shb_we) sh_bias[shb_wa] <= shb_wd;
    if (shr_we) sh_ridw[shr_wa] <= shr_wd;
  end

  // Per-bank banked memories (§13 v2). Each generate block owns ONE
  // physical memory per array with exactly one sync read and one
  // decoded write — the inferable simple-dual-port shape. Reads and
  // writes are both nonblocking on the same edge, so a same-cycle
  // read of a written row returns OLD data — identical to the retired
  // two-block form (collisions are schedule-safe per the note above).
  // sh_slot's writer keeps its scalar face: bank decode = wa[3:0].
  for (genvar gb = 0; gb < 16; gb++) begin : g_bank
    logic [23:0] shs_mem [NMAX/2];       // {valid,nbr,J}
    logic [31:0] m1_mem  [NMAX/2];
    logic [31:0] m2_mem  [NMAX/2];
    always_ff @(posedge clk) begin
      shs_rqb[gb] <= shs_mem[shs_ra[NB+2:4]];
      m2_rqb[gb]  <= m2_mem[m2_ra[NB+2:4]];
      m1_rqb[gb]  <= m1_mem[m1_ra[NB+2:4]];
      if (shs_we && (shs_wa[3:0] == 4'(gb)))
        shs_mem[shs_wa[NB+2:4]] <= shs_wd;
      if (m1_web[gb]) m1_mem[m1_wab[gb]] <= m1_wdb[gb];
      if (m2_web[gb]) m2_mem[m2_wab[gb]] <= m2_wdb[gb];
    end
  end

  // ---------------- work registers (D33) ----------------
  logic signed [63:0] work_q [64];
  logic signed [63:0] e2_old [64];          // scratch during ROWS rewrite

  // ---------------- telemetry ----------------
  logic [31:0] cfg_words_q, drain_words_q, sweeps_done_q;

  // ---------------- fabric graft ----------------
  logic        f_row_we, f_ord_we, f_cb_we, f_sch_we, f_stw_we, f_seed_we;
  logic [NB-1:0] f_row_addr, f_ord_addr, f_ord_data, f_cb_start, f_cb_end;
  logic [3:0]  f_row_slot, f_cb_idx;
  logic [29:0] f_row_data;
  logic [23:0] f_sch_sweeps;
  logic [4:0]  f_sch_idx;
  logic [7:0]  f_sch_beta;
  logic [8:0]  f_stw_addr;
  logic [31:0] f_stw_data, f_seed_word;
  logic [4:0]  f_n_colors;
  logic [5:0]  f_n_sched;
  logic [$clog2(P)-1:0] f_seed_stream;
  logic [1:0]  f_seed_sel;
  logic        f_start, f_done, f_sweep_pulse;
  logic        f_busy;
  logic        unused_sinks;
  assign unused_sinks = f_busy & f_sweep_pulse
                        & (|dma_left) & (|q_a) & (|q_op);
  logic [47:0] f_upd_cnt, f_flip_cnt;
  // idle-time fabric state reads (async ports into the write-behind
  // store; the fabric is idle in every state that drives these)
  logic [NB-1:0] fs_a, fs_b;
  logic [31:0]   fs_word;
  logic          fs_abit, fs_bbit;
  logic [2:0]    fs_aval, fs_bval;   // q-value taps (§12)
  logic [31:0]   fs_word2;           // plane-1 row word (STATE-q)
  logic [31:0]   fs_word3;           // plane-2 row word (q8, S4)
  logic          arity_q4;           // header word2[31:24] == 4
  logic          arity_q8;           // header word2[31:24] == 8
  wire           q_on = arity_q4 | arity_q8;
  // m1-region words per site in the MOMENTS drain (q-1; binary 1)
  wire [2:0]     qm1 = arity_q8 ? 3'd7 : arity_q4 ? 3'd3 : 3'd1;

  fabric_grid #(.NB(NB), .P(P), .LUT_FILE(LUT_FILE),
                .GLUT_FILE(GLUT_FILE)) u_fabric (
    .clk(clk), .rst_n(rst_n),
    .row_we(f_row_we), .row_addr(f_row_addr), .row_slot(f_row_slot),
    .row_data(f_row_data), .arity_q4(arity_q4), .arity_q8(arity_q8),
    .sA_val(fs_aval), .sB_val(fs_bval), .sA_word2(fs_word2),
    .sA_word3(fs_word3),
    /* verilator lint_off PINCONNECTEMPTY */
    .state_flat2(), .state_flat3(),
    /* verilator lint_on PINCONNECTEMPTY */
    .ord_we(f_ord_we), .ord_addr(f_ord_addr), .ord_data(f_ord_data),
    .cb_we(f_cb_we), .cb_idx(f_cb_idx), .cb_start(f_cb_start),
    .cb_end(f_cb_end), .n_colors(f_n_colors),
    .sch_we(f_sch_we), .sch_idx(f_sch_idx), .sch_beta(f_sch_beta),
    .sch_sweeps(f_sch_sweeps), .n_sched(f_n_sched),
    .bipolar(bipolar_r),
    .stw_we(f_stw_we), .stw_addr(f_stw_addr), .stw_data(f_stw_data),
    .seed_we(f_seed_we), .seed_stream(f_seed_stream),
    .seed_sel(f_seed_sel), .seed_word(f_seed_word),
    .start(f_start), .busy(f_busy), .sweep_pulse(f_sweep_pulse),
    .done(f_done),
    .sA_addr(fs_a), .sB_addr(fs_b),
    .sA_word(fs_word), .sA_bit(fs_abit), .sB_bit(fs_bbit),
        /* verilator lint_off PINCONNECTEMPTY */
    .upd_cnt(f_upd_cnt), .flip_cnt(f_flip_cnt), .cycle_cnt(),
    .state_flat()
    /* verilator lint_on PINCONNECTEMPTY */
  );

  // ==================== part 2: the FSMs ====================
  // The walkers share the single v2 face: PCONFIG one word at a time
  // through the lv shim; PDRAIN credit-pipelined (D-032c). All
  // indices/counters registered.

  typedef enum logic [4:0] {
    S_IDLE,
    // PCONFIG
    C_HDR0, C_HDR1, C_HDR2, C_SECT,
    C_ROWS, C_ORD_N, C_ORD_CB, C_ORD_IDX, C_SCHED, C_SEEDS, C_STATE,
    C_RID, C_FIN,
    // PSAMPLE
    P_ZERO, P_ENTRY, P_LOADSCH, P_START, P_WAIT, P_ACC, P_NEXT,
    // PDRAIN
    D_RUN,
    // shared E2 scan engine (bracketing ROWS writes); E_COPY is the
    // settle cycle so e2_old captures the LAST site's NBA update too
    E_SCAN, E_COPY,
    // ACCWALK (D-034): the post-sweep snapshot copy. P_ACC above is
    // RETIRED (never entered) but keeps its encoding so the PSTAT tap's
    // public state numbering remains stable.
    P_SNAP
  } st_e;
  st_e st /*verilator public_flat_rd*/;  // PSTAT tap (armed-only host
                                         // histogram; psample probe mirrors
                                         // st_e order)
  st_e e2_ret;                           // e2_ret: state to return to

  // latched command
  logic [2:0]  q_op;
  logic [31:0] q_a, q_c;
  logic [23:0] q_m;
  logic [7:0]  q_k;
  logic [8:0]  q_fl;

  // DMA cursor
  logic [31:0] dma_addr;
  logic [23:0] dma_left;                 // words remaining (PCONFIG guard)

  // PCONFIG walker registers
  logic [5:0]  c_sect;                   // section bit under scan
  logic [15:0] c_site;                   // site cursor
  logic [3:0]  c_slot;                   // slot-within-site cursor
  logic [15:0] c_nord;
  logic [15:0] c_idx;                    // generic index cursor
  logic        rows_flagged, track_pre;  // D33 gating (pre-command)

  // PSAMPLE registers
  logic [5:0]  p_entry;
  logic [23:0] p_left;                   // sweeps left in entry
  logic [7:0]  p_beta;

  // ---------------- ACCWALK (D-034, spec §13) ----------------
  // Snapshot: the post-sweep state planes, copied via the idle sA
  // port during P_SNAP (one 32-site word per strobe), word-granular
  // async-read (distributed) — the walker's ONLY state source during
  // overlap; the fs_* ports belong to the running next sweep and are
  // never touched while the fabric is busy.
  (* ram_style = "distributed" *) logic [31:0] snap1 [NMAX/32];
  (* ram_style = "distributed" *) logic [31:0] snap2 [NMAX/32];
  (* ram_style = "distributed" *) logic [31:0] snap3 [NMAX/32];
  logic          snap_we;
  logic [NB-6:0] snap_wa, snap_w;        // strobe address / copy cursor
  // Walker: 1 site/cycle (WALK_W=1, §13c), 2-stage F->C pipeline.
  // F drives the shadow/counter row reads; C commits one m1 RMW plus
  // 8 parallel m2 bank-RMWs — the retired P_ACC math verbatim, fed
  // from the snapshot instead of the live fabric.
  logic          walk_busy;
  logic [15:0]   w_site;                 // F-stage EVEN site of the pair
  logic          w_fdone;                // F exhausted, pipe draining
  logic          w_cv;                   // C-stage valid
  logic [NB-2:0] w_cs;                   // C-stage PAIR row (site>>1)
  logic          w_cv1;                  // C-stage odd site exists
  logic [2:0]    w_cval0, w_cval1;       // C-stage site values (q)
  logic          w_cbit0, w_cbit1;       // C-stage site bits (binary)
  logic [1:0]    w_tail;
  function automatic logic [2:0] snap_val(input logic [NB-1:0] a);
    snap_val = {snap3[a[NB-1:5]][a[4:0]],
                snap2[a[NB-1:5]][a[4:0]],
                snap1[a[NB-1:5]][a[4:0]]};
  endfunction
  // WALK_W=2 (§13 v2): F reads the site PAIR (w_site, w_site+1)
  wire [2:0] w_sval0_f = snap_val(w_site[NB-1:0]);
  wire [2:0] w_sval1_f = snap_val(w_site[NB-1:0] | NB'(1));
  // ACCWALK snapshot write (its own reset-free block: RAM inference)
  always_ff @(posedge clk) begin
    if (snap_we) begin
      snap1[snap_wa] <= fs_word;
      snap2[snap_wa] <= fs_word2;
      snap3[snap_wa] <= fs_word3;
    end
  end

  // E2 scan engine (per-site row scan; int64 per replica in e2_acc)
  logic [15:0] e_site;
  logic [3:0]  e_slot;
  logic        e_ph;                     // D-028 fetch/commit
  logic signed [31:0] e_pair;            // sum J_raw*s_nbr for site
  logic signed [63:0] e2_acc [64];
  logic        e2_is_old;                // accumulating old or new

  // PDRAIN streamer
  logic [2:0]  d_mode;
  logic [17:0] d_idx;                    // word index within the drain
  logic [17:0] d_total;
  logic [NB-1:0] d_nbr;                  // COV neighbor addr (D-032c:
                                         // survives credit stalls)
  // D-028 drain fetch pipeline: phases before a word can be composed
  // (COV: site trio, then the DEPENDENT m1[neighbor] read; MOMENTS:
  // one fetch; STATE/WORK/TELEMETRY: none)
  logic [1:0]  d_ph;
  logic        d_shv;                    // latched row valid bit (COV)
  logic [31:0] d_m1s, d_m2v;             // latched m1[site], m2[site][k]
  logic [31:0] d_mnb;                    // latched m1[neighbor] (timing
  //                                        stage — see d_need note)
  logic [NB+3:0] z_cnt;                  // P_ZERO walk cursor
  logic [17:0] cov_rem;
  logic [NB-1:0] cov_site;
  logic [4:0]  cov_fld;
  logic [2:0]  cov_k;
  always_comb begin
    cov_rem  = d_idx - 18'd1;
    cov_site = NB'(cov_rem / 18);
    cov_fld  = 5'(cov_rem % 18);
    cov_k    = 3'(cov_fld[4:1] - 4'd1);
  end
  // COV takes THREE fetch phases: trio read, neighbor read, then a
  // register stage for m1[neighbor] (d_mnb) so the emit's 64-bit
  // multiply-subtract launches from flops only — the BRAM+bank-mux →
  // DSP → CARRY8 chain missed 8ns by 0.414 as one cycle (2026-07-16
  // fix re-hearing; D-026: cycles are not contract, values identical).
  wire [1:0] d_need = (d_mode == 3'd2 && d_idx != 0) ? 2'd3 :
                      (d_mode == 3'd1 && d_idx != 0) ? 2'd1 : 2'd0;
  wire [5:0] shr_byte = 6'(shr_rq[8 * e_site[1:0] +: 8]);
  // MOMENTS-q addressing (§12): m1 region is site-major state-minor
  // (constant dividers per arity, muxed — the /18 COV precedent)
  logic [17:0]   q_rem;
  logic [NB-1:0] q_site3;
  logic [2:0]    q_lane3;
  always_comb begin
    q_rem   = d_idx - 18'd1;
    q_site3 = arity_q8 ? NB'(q_rem / 7) : NB'(q_rem / 3);
    q_lane3 = arity_q8 ? 3'(q_rem % 7) : 3'(q_rem % 3);
  end
  // E2-q bias lane for the scanned site's value (b0 == 0; lanes 1..7)
  logic [9:0] shb_lane;
  always_comb begin
    shb_lane = shb_rq[9:0];
    for (int a = 2; a < 8; a++)
      if (fs_aval == 3'(a)) shb_lane = shb_rq[10 * (a - 1) +: 10];
  end

  // fabric state read-address driver: port A = own site (or drain
  // word base), port B = neighbor via the shadow rows. The fabric
  // ports stay async; fs_b now comes from the REGISTERED shadow read
  // (D-028) and is consumed only in commit/late phases.
  always_comb begin
    fs_a = '0; fs_b = '0;
    case (st)
      P_SNAP: fs_a = {snap_wa, 5'b00000};  // ACCWALK copy (fabric idle)
      E_SCAN: begin
        fs_a = e_site[NB-1:0];
        fs_b = shs_rq[10 +: NB];
      end
      D_RUN: fs_a = arity_q8 ? {d_idx[NB-4:2], 5'b0}
                  : arity_q4 ? {d_idx[NB-5:1], 5'b0}
                             : {d_idx[NB-6:0], 5'b0};
      default: ;
    endcase
  end

  // memory read addresses (states are mutually exclusive owners; the
  // ACCWALK walker owns them while busy — it only coexists with the
  // PSAMPLE overlap states, which drive none of these, asserted below)
  always_comb begin
    shs_ra = '0; m1_ra = '0; m2_ra = '0; shb_ra = '0; shr_ra = '0;
    if (walk_busy) begin
      // walker F stage: PAIR rows — bank = addr[3:0] convention means
      // {even_site, 4'b0} row-addresses every bank at row site>>1;
      // the C stage picks lanes/slots from the 16 registered rq's.
      shs_ra = {w_site[NB-1:0], 3'b000};
      m2_ra  = {w_site[NB-1:0], 3'b000};
      m1_ra  = {w_site[NB-1:0], 3'b000};
    end else
    case (st)
      E_SCAN: begin
        shs_ra = {e_site[NB-1:0], e_slot[2:0]};
        shb_ra = e_site[NB-1:0];
        shr_ra = e_site[NB-1:2];
      end
      D_RUN: begin
        if (d_mode == 3'd1) begin        // MOMENTS: one independent read
          m1_ra = q_on ? {q_site3, q_lane3}
                       : {NB'(d_idx - 18'd1), 3'b000};
          m2_ra = (NB+3)'(32'(d_idx) - 32'd1
                          - 32'(qm1) * 32'(n_sites));
        end else if (d_mode == 3'd2) begin
          if (d_ph == 2'd0) begin        // site-addressed trio
            shs_ra = {cov_site, cov_k};
            m1_ra  = {cov_site, 3'b000};
            m2_ra  = {cov_site, cov_k};
          end else if (d_ph == 2'd1) begin // dependent neighbor read
            m1_ra  = {shs_rq[10 +: NB], 3'b000};
          end else begin                 // credit-stall reload (D-032c)
            m1_ra  = {d_nbr, 3'b000};
          end
        end
      end
      default: ;
    endcase
  end

  // drain word count per mode (spec §6)
  function automatic logic [17:0] drain_len(input logic [2:0] m);
    case (m)
      3'd0: drain_len = arity_q8 ? 18'((32'(n_sites) + 7) >> 3)
                      : arity_q4 ? 18'((32'(n_sites) + 15) >> 4)
                                 : 18'((32'(n_sites) + 31) >> 5);
      3'd1: drain_len = 18'(1 + 32'(qm1) * 32'(n_sites)
                            + 32'(n_sites) * 8);
      3'd2: drain_len = 18'(1 + 32'(n_sites) * 18);
      3'd3: drain_len = 18'(1 + 32'(n_replicas_r) * 2);
      default: drain_len = 18'd12;
    endcase
  endfunction


  // §6 readback layouts, one word per call. D-028: operands arrive
  // through the sync read ports — MOMENTS uses the live port data
  // (fetched last cycle); COV uses the ph1-latched site trio plus the
  // live dependent m1[neighbor]. craw's 32x32 products still go via a
  // single shared multiply.
  function automatic logic [31:0] drain_word(input logic [2:0] m,
                                             input logic [17:0] idx);
    logic [31:0] w;
    logic [6:0]  rem;
    logic signed [63:0] v64;
    w = '0;
    unique case (m)
      3'd0: begin                              // STATE
        if (arity_q8) begin                      // 8 sites x 4b (§12)
          for (int i8 = 0; i8 < 8; i8++)
            if (32'({idx, 3'b0}) + i8 < 32'(n_sites)) begin
              w[4 * i8]     = fs_word[8 * int'(idx[1:0]) + i8];
              w[4 * i8 + 1] = fs_word2[8 * int'(idx[1:0]) + i8];
              w[4 * i8 + 2] = fs_word3[8 * int'(idx[1:0]) + i8];
            end
        end else if (arity_q4) begin             // 16 sites x 2b (§12)
          for (int i3 = 0; i3 < 16; i3++)
            if (32'({idx, 4'b0}) + i3 < 32'(n_sites)) begin
              w[2 * i3]     = fs_word[16 * int'(idx[0]) + i3];
              w[2 * i3 + 1] = fs_word2[16 * int'(idx[0]) + i3];
            end
        end else begin
          for (int b3 = 0; b3 < 32; b3++)
            if (32'({idx, 5'b0}) + b3 < 32'(n_sites))
              w[b3] = fs_word[b3];
        end
      end
      3'd1: begin                              // MOMENTS (live port data)
        if (idx == 0) w = acc_cnt;
        else if (32'(idx) <= 32'(qm1) * 32'(n_sites))
          w = m1_rq;
        else w = m2_rq;
      end
      3'd2: begin                              // COV: cnt then 9 i64/site
        if (idx == 0) w = acc_cnt;
        else begin
          if (cov_fld[4:1] == 4'd0)            // diag craw
            v64 = 64'(acc_cnt) * 64'(d_m1s)
                  - 64'(d_m1s) * 64'(d_m1s);
          else begin
            v64 = '0;
            if (d_shv)
              v64 = 64'(acc_cnt) * 64'(d_m2v)
                    - 64'(d_m1s) * 64'(d_mnb); // staged m1[neighbor]
          end
          w = cov_fld[0] ? v64[63:32] : v64[31:0];
        end
      end
      3'd3: begin                              // WORK
        if (idx == 0) w = 32'(n_replicas_r);
        else begin
          rem = 7'(idx - 18'd1);
          v64 = work_q[rem[6:1]];
          w = rem[0] ? v64[63:32] : v64[31:0];
        end
      end
      default: begin                           // TELEMETRY
        unique case (idx[3:0])
          4'd0: w = sweeps_done_q;
          4'd1: w = f_upd_cnt[31:0];
          4'd2: w = {16'd0, f_upd_cnt[47:32]};
          4'd3: w = f_flip_cnt[31:0];
          4'd4: w = {16'd0, f_flip_cnt[47:32]};
          4'd5: w = cfg_words_q;
          4'd6: w = drain_words_q;
          4'd7: w = acc_cnt;
          default: w = '0;                     // busy_cycles golden=0
        endcase
      end
    endcase
    return w;
  endfunction

  assign busy = (st != S_IDLE);

  // ------------- v2 memory-face plumbing (D-032c leg C) -------------
  // One per-face credit pool shared by both issuers, each a
  // registered one-cycle strobe per beat: the PDRAIN streamer
  // (d_req_q) and the PCONFIG issue engine (g_req_q, stage C3 — the
  // lv shim is retired; the walker's consume logic fires on
  // completion pulses while the engine streams the current section's
  // addresses, c_iss of c_tot, credit-gated).
  logic       d_req_q, g_req_q;
  logic [2:0] cred;
  logic [23:0] c_iss, c_tot;          // section issue cursor / total
  assign m_req = d_req_q || g_req_q;
  // credits visible to next cycle's issue decision: the pulse now in
  // flight spends one, a completion now returns one
  wire [2:0]  cred_nxt = cred - 3'(m_req) + 3'(m_rsp_valid);
  wire cfg_stream = (st == C_HDR0) || (st == C_HDR1) || (st == C_HDR2)
                 || (st == C_ROWS) || (st == C_ORD_N) || (st == C_ORD_CB)
                 || (st == C_ORD_IDX) || (st == C_SCHED)
                 || (st == C_SEEDS) || (st == C_STATE) || (st == C_RID);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE;
      m_we <= 1'b0; m_addr <= '0; m_wdata <= '0;
      m_be <= '0;
      d_req_q <= 1'b0; g_req_q <= 1'b0; cred <= 3'(CRED);
      c_iss <= '0; c_tot <= '0;
      f_row_we <= 0; f_ord_we <= 0; f_cb_we <= 0; f_sch_we <= 0;
      f_stw_we <= 0; f_seed_we <= 0; f_start <= 0;
      shs_we <= 0; shb_we <= 0; shr_we <= 0;
      m1_web <= '0; m2_web <= '0;
      snap_we <= 0; snap_wa <= '0; snap_w <= '0;
      walk_busy <= 0; w_site <= '0; w_fdone <= 0; w_cv <= 0;
      w_cv1 <= 0; w_cs <= '0; w_cval0 <= '0; w_cval1 <= '0;
      w_cbit0 <= 0; w_cbit1 <= 0; w_tail <= '0;
      for (int b = 0; b < 16; b++) begin
        m1_wab[b] <= '0; m1_wdb[b] <= '0;
        m2_wab[b] <= '0; m2_wdb[b] <= '0;
      end
      f_n_colors <= '0; f_n_sched <= '0;
      n_sites <= '0; bipolar_r <= 0; work_track_r <= 0;
      arity_q4 <= 0; arity_q8 <= 0;
      n_sched_r <= '0; n_replicas_r <= 8'd1;
      acc_cnt <= '0; cnt_sops <= '0;
      cfg_words_q <= '0; drain_words_q <= '0; sweeps_done_q <= '0;
      for (int r = 0; r < 64; r++) begin
        work_q[r] <= '0; e2_old[r] <= '0;
      end
    end else begin
      // one-shot strobes default low
      f_row_we <= 0; f_ord_we <= 0; f_cb_we <= 0; f_sch_we <= 0;
      f_stw_we <= 0; f_seed_we <= 0; f_start <= 0;
      shs_we <= 0; shb_we <= 0; shr_we <= 0;
      m1_web <= '0; m2_web <= '0;
      snap_we <= 0;
      d_req_q <= 0;
      g_req_q <= 0;

      // v2 face bookkeeping (issuers only decide; the pool lives here)
      cred <= cred_nxt;
      // PCONFIG issue engine (stage C3): stream the current section's
      // beats. Section totals are latched at dispatch (all lengths are
      // known by then — the data-dependent ones arrive in earlier
      // beats), so issue never runs past a data-dependent boundary;
      // the last consume of a section implies zero outstanding.
      if (cfg_stream && c_iss != c_tot && cred_nxt != 3'd0) begin
        g_req_q <= 1'b1; m_we <= 1'b0; m_be <= '1;
        m_addr <= dma_addr;
        dma_addr <= dma_addr + 4;
        c_iss <= c_iss + 24'd1;
      end
`ifndef SYNTHESIS
      assert (cred <= 3'(CRED))
        else $fatal(1, "s_cluster: credit pool overflow");
      assert (!(m_req && cred == 3'd0 && !m_rsp_valid))
        else $fatal(1, "s_cluster: beat issued without a credit");
`endif

      unique case (st)
        S_IDLE: if (go) begin
          q_op <= op; q_a <= t_a; q_c <= t_c; q_m <= t_m; q_k <= t_k;
          q_fl <= t_flags;
          cnt_sops <= cnt_sops + 1;
          unique case (op)
            3'd0: begin                       // PCONFIG
              dma_addr <= t_a; dma_left <= t_m;
              track_pre <= work_track_r;      // D33: PRE-command flag
              rows_flagged <= t_flags[CF_ROWS];
              c_iss <= '0; c_tot <= 24'd3;    // the three header beats
              st <= C_HDR0;
            end
            3'd1: begin                       // PSAMPLE
              p_entry <= '0;
              if (t_flags[SF_STATS_RESET]) begin
                acc_cnt <= '0;                // spec: zeroed NOW — via
                z_cnt <= '0;                  // the P_ZERO walk (D-026:
                st <= P_ZERO;                 // cycles are not contract)
              end else st <= P_ENTRY;
            end
            default: begin                    // PDRAIN
`ifndef SYNTHESIS
              assert (!(q_on && t_flags[2:0] == 3'd2))
                else $fatal(1, "COV-q deferred (spec §12)");
`endif
              d_mode <= t_flags[2:0]; d_idx <= '0; d_ph <= '0;
              d_total <= drain_len(t_flags[2:0]);
              st <= D_RUN;
            end
          endcase
        end

        // ---------------- PCONFIG: header ----------------
        // header beats do NOT count toward cfg_words: spec §6 telemetry
        // word 5 is SECTION words only (golden: len(words) - 3)
        C_HDR0: if (m_rsp_valid) begin         // magic
          dma_left <= dma_left - 1;
          st <= C_HDR1;
        end
        C_HDR1: if (m_rsp_valid) begin         // n | bipolar | work_track
          n_sites <= m_rdata[15:0];
          bipolar_r <= m_rdata[16];
          work_track_r <= m_rdata[17];
          dma_left <= dma_left - 1;
          st <= C_HDR2;
        end
        C_HDR2: if (m_rsp_valid) begin         // colors | sched | replicas
          arity_q4 <= (m_rdata[31:24] == 8'd4);
          arity_q8 <= (m_rdata[31:24] == 8'd8);
`ifndef SYNTHESIS
          assert (m_rdata[31:24] inside {8'd0, 8'd2, 8'd4, 8'd8})
            else $fatal(1, "s_cluster: arity %0d unsupported",
                        m_rdata[31:24]);
`endif
          // the golden's color schedule and loaded annealing schedule
          // PERSIST across images without their sections: header
          // n_colors/n_sched are section lengths, applied only when
          // the section is present (caught by the S8σ gate on a
          // ROWS-only-rewrite-then-sample shape — latent since M21;
          // every prior diffed image carried ORDER+SCHED)
          if (q_fl[1]) f_n_colors <= m_rdata[4:0];
          if (q_fl[2]) n_sched_r  <= m_rdata[15:8];
          n_replicas_r <= (m_rdata[23:16] == 0) ? 8'd1 : m_rdata[23:16];
          // golden resize: entries at/above the OLD count read 0 if the
          // replica count later grows back over them
          for (int r = 0; r < 64; r++)
            if (8'(r) >= n_replicas_r) work_q[r] <= '0;
          dma_left <= dma_left - 1;
          c_sect <= '0; c_site <= '0; c_slot <= '0; c_idx <= '0;
          // D33: bracket scan BEFORE rows apply
          if (rows_flagged && track_pre) begin
            e_site <= '0; e_slot <= '0; e_pair <= '0; e_ph <= '0;
            e2_is_old <= 1;
            for (int r = 0; r < 64; r++) e2_acc[r] <= '0;
            e2_ret <= C_SECT; st <= E_SCAN;
          end else st <= C_SECT;
        end

        // section dispatcher: walk flag bits 0..5 in order
        C_SECT: begin
          if (c_sect > 5) st <= C_FIN;
          else if (q_fl[c_sect[3:0]]) begin
            c_site <= '0; c_slot <= '0; c_idx <= '0;
            c_iss <= '0;                      // issue engine restarts
            unique case (c_sect)
              0: begin
                c_tot <= 24'(32'(n_sites) * (arity_q8 ? 32'd11 : 32'd9));
                st <= C_ROWS;
              end
              1: begin c_tot <= 24'd1;  st <= C_ORD_N; end
              2: begin c_tot <= 24'(n_sched_r); st <= C_SCHED; end
              3: begin c_tot <= 24'd32; st <= C_SEEDS; end
              4: begin
                c_tot <= 24'((32'(n_sites)
                              + (arity_q8 ? 32'd7
                                 : arity_q4 ? 32'd15 : 32'd31))
                             >> (arity_q8 ? 3 : arity_q4 ? 4 : 5));
                st <= C_STATE;
              end
              default: begin
                c_tot <= 24'((32'(n_sites) + 3) >> 2);
                st <= C_RID;
              end
            endcase
          end else c_sect <= c_sect + 1;
        end

        C_ROWS: if (m_rsp_valid) begin
          dma_left <= dma_left - 1;
          cfg_words_q <= cfg_words_q + 1;
          f_row_we <= 1'b1;
          f_row_addr <= c_site[NB-1:0];
          f_row_slot <= c_slot;
          f_row_data <= m_rdata[29:0];
          if (c_slot < 8) begin
            shs_we <= 1'b1; shs_wa <= {c_site[NB-1:0], c_slot[2:0]};
            shs_wd <= m_rdata[23:0];
          end else if (arity_q8) begin
            // q8: slots 8,9 stash 30b lanes; slot 10 commits 70b (§12)
            if (c_slot == 8) shb_stash[29:0]  <= m_rdata[29:0];
            if (c_slot == 9) shb_stash[59:30] <= m_rdata[29:0];
            if (c_slot == 10) begin
              shb_we <= 1'b1; shb_wa <= c_site[NB-1:0];
              shb_wd <= {m_rdata[9:0], shb_stash};
            end
          end else begin
            shb_we <= 1'b1; shb_wa <= c_site[NB-1:0];
            shb_wd <= {40'b0, m_rdata[29:0]};
          end
          if (c_slot == (arity_q8 ? 4'd10 : 4'd8)) begin
            c_slot <= '0;
            if (c_site + 1 == n_sites) begin
              c_sect <= c_sect + 1;
              // spec §5: E2_new right after the ROWS write, BEFORE the
              // STATE/RID sections — both scans see command-entry state+rid
              if (track_pre) begin
                e_site <= '0; e_slot <= '0; e_pair <= '0; e_ph <= '0;
                e2_is_old <= 0;
                for (int r = 0; r < 64; r++) e2_acc[r] <= '0;
                e2_ret <= C_SECT; st <= E_SCAN;
              end else st <= C_SECT;
            end else c_site <= c_site + 1;
          end else c_slot <= c_slot + 1;
        end

        C_ORD_N: if (m_rsp_valid) begin
          c_nord <= m_rdata[15:0];
          dma_left <= dma_left - 1;
          cfg_words_q <= cfg_words_q + 1;
          c_idx <= '0;
          c_iss <= '0; c_tot <= 24'd16;      // CB sub-phase dispatch
          st <= C_ORD_CB;
        end
        C_ORD_CB: if (m_rsp_valid) begin       // 16 colour-bound words
          dma_left <= dma_left - 1;
          cfg_words_q <= cfg_words_q + 1;
          f_cb_we <= 1'b1;
          f_cb_idx <= c_idx[3:0];
          f_cb_start <= m_rdata[NB-1:0];
          f_cb_end <= m_rdata[16 +: NB];
          if (c_idx == 15) begin
            c_idx <= '0;
            if (c_nord == 0) begin
              c_sect <= c_sect + 1; st <= C_SECT;
            end else begin
              c_iss <= '0; c_tot <= 24'(c_nord);   // IDX sub-phase
              st <= C_ORD_IDX;
            end
          end else c_idx <= c_idx + 1;
        end
        C_ORD_IDX: if (m_rsp_valid) begin
          dma_left <= dma_left - 1;
          cfg_words_q <= cfg_words_q + 1;
          f_ord_we <= 1'b1;
          f_ord_addr <= c_idx[NB-1:0];
          f_ord_data <= m_rdata[NB-1:0];
          if (c_idx + 1 == c_nord) begin
            c_sect <= c_sect + 1; st <= C_SECT;
          end else c_idx <= c_idx + 1;
        end

        C_SCHED: if (m_rsp_valid) begin
          dma_left <= dma_left - 1;
          cfg_words_q <= cfg_words_q + 1;
          sch_beta_q[c_idx[4:0]] <= m_rdata[7:0];
          sch_sweeps_q[c_idx[4:0]] <= m_rdata[31:8];
          if (c_idx + 1 == 16'(n_sched_r)) begin
            c_sect <= c_sect + 1; st <= C_SECT;
          end else c_idx <= c_idx + 1;
        end

        C_SEEDS: if (m_rsp_valid) begin        // 32 words, stream-major
          dma_left <= dma_left - 1;
          cfg_words_q <= cfg_words_q + 1;
          f_seed_we <= 1'b1;
          f_seed_stream <= c_idx[4:2];
          f_seed_sel <= c_idx[1:0];
          f_seed_word <= m_rdata;
          if (c_idx == 31) begin
            c_sect <= c_sect + 1; st <= C_SECT;
          end else c_idx <= c_idx + 1;
        end

        C_STATE: if (m_rsp_valid) begin
          dma_left <= dma_left - 1;
          cfg_words_q <= cfg_words_q + 1;
          f_stw_we <= 1'b1;
          f_stw_addr <= c_idx[8:0];
          f_stw_data <= m_rdata;
          if (32'(c_idx) + 1 >= (arity_q8
                                 ? (32'(n_sites) + 7) >> 3
                                 : arity_q4
                                 ? (32'(n_sites) + 15) >> 4
                                 : (32'(n_sites) + 31) >> 5)) begin
            c_sect <= c_sect + 1; st <= C_SECT;
          end else c_idx <= c_idx + 1;
        end

        C_RID: if (m_rsp_valid) begin
          dma_left <= dma_left - 1;
          cfg_words_q <= cfg_words_q + 1;
          // word write (4 rids/beat); bytes past n_sites land in the
          // word but are never read back (the E2 scan stops at n_sites)
          shr_we <= 1'b1; shr_wa <= c_idx[NB-3:0]; shr_wd <= m_rdata;
          if (32'(c_idx) + 1 >= (32'(n_sites) + 3) >> 2) begin
            c_sect <= c_sect + 1; st <= C_SECT;
          end else c_idx <= c_idx + 1;
        end

        C_FIN: begin
          // apply the bracketed D33 update, then WORK_RESET if flagged
          if (rows_flagged && track_pre)
            for (int r = 0; r < 64; r++)
              work_q[r] <= (q_fl[CF_WORK_RESET]) ? 64'sd0
                           : work_q[r] + (e2_acc[r] - e2_old[r]);
          else if (q_fl[CF_WORK_RESET])
            for (int r = 0; r < 64; r++) work_q[r] <= '0;
          st <= S_IDLE;
        end

        // ---------------- PSAMPLE: entry sequencing ----------------
        P_ZERO: begin                         // SF_STATS_RESET zero walk
          // §13 v2 (S2): all 32 counter banks zero in lock-step over
          // the NMAX/2 pair rows — 65,536 -> 4,096 cycles per RESET
          // at NB=13 (D-026: cycles are not contract; values = zeros)
          m1_web <= '1; m2_web <= '1;
          for (int b = 0; b < 16; b++) begin
            m1_wab[b] <= z_cnt[NB-2:0]; m1_wdb[b] <= '0;
            m2_wab[b] <= z_cnt[NB-2:0]; m2_wdb[b] <= '0;
          end
          if (z_cnt == (NB+4)'(NMAX / 2 - 1)) st <= P_ENTRY;
          z_cnt <= z_cnt + 1'b1;
        end
        P_ENTRY: begin
          if (q_fl[SF_IMM]) begin
            p_beta <= q_k; p_left <= q_m;
            if (q_m == 0) begin
              if (!walk_busy) st <= S_IDLE;   // D-034 tail-walk gate
            end else st <= P_LOADSCH;
          end else if (p_entry >= 6'(n_sched_r)) begin
            if (!walk_busy) st <= S_IDLE;     // D-034 tail-walk gate
          end else begin
            p_beta <= sch_beta_q[p_entry[4:0]];
            p_left <= sch_sweeps_q[p_entry[4:0]];
            st <= (sch_sweeps_q[p_entry[4:0]] == 0) ? P_NEXT : P_LOADSCH;
          end
        end
        P_LOADSCH: begin                      // one-sweep fabric entry
          f_sch_we <= 1'b1; f_sch_idx <= '0;
          f_sch_beta <= p_beta; f_sch_sweeps <= 24'd1;
          f_n_sched <= 6'd1;
          st <= P_START;
        end
        P_START: begin
          f_start <= 1'b1;
          st <= P_WAIT;
        end
        P_WAIT: if (f_done) begin
          sweeps_done_q <= sweeps_done_q + 1;
          if (q_fl[SF_RECORD]) begin
            snap_w <= '0;
            acc_cnt <= acc_cnt + 1;             // the old P_ACC edge
            st <= P_SNAP;
          end else st <= P_NEXT;
        end
        P_SNAP: begin                    // ACCWALK (D-034 §13a): copy
          // the post-sweep planes through the idle sA port, one
          // 32-site word per strobe. The PREVIOUS sweep's walk must
          // drain first — its commits land in the same counters the
          // next walk RMWs (and PSTAT charges that stall here).
          if (!walk_busy) begin
            if (32'(snap_w) < ((32'(n_sites) + 31) >> 5)) begin
              snap_we <= 1'b1; snap_wa <= snap_w;
              snap_w <= snap_w + 1'b1;
            end else begin
              // the last strobe's write lands at this edge; the
              // walker reads the arrays from the next cycle on
              walk_busy <= 1'b1;
              w_site <= '0; w_fdone <= 1'b0; w_tail <= '0;
              st <= P_NEXT;
            end
          end
        end
        P_ACC: st <= S_IDLE;   // RETIRED (D-034): unreachable; encoding
                               // kept for PSTAT index stability
        P_NEXT: begin
          if (p_left > 1) begin p_left <= p_left - 1; st <= P_LOADSCH; end
          else if (q_fl[SF_IMM]) begin
            if (!walk_busy) st <= S_IDLE;       // D-034 tail-walk gate
          end else begin p_entry <= p_entry + 1; st <= P_ENTRY; end
        end

        // ---------------- E2 scan (D33; binary mode) ----------------
        // E2[r] = -( sum_i s_i*pair_i + 2*sum_i b_i*s_i ), raw ints
        E_SCAN: begin
          if (!e_ph) e_ph <= 1'b1;            // fetch (D-028)
          else begin
            e_ph <= 1'b0;
            if (e_slot < 8) begin
              if (shs_rq[23] && (q_on ? (fs_bval == fs_aval)
                                      : fs_bbit))
                e_pair <= e_pair + 32'(signed'(shs_rq[9:0]));
              e_slot <= e_slot + 1;
            end else begin
              if (q_on) begin                    // E2-q: ungated (§12)
                e2_acc[shr_byte]
                  <= e2_acc[shr_byte]
                     - 64'(e_pair)
                     - (fs_aval == 3'd0 ? 64'sd0
                        : 64'(signed'({shb_lane, 1'b0})));
              end else if (fs_abit)
                e2_acc[shr_byte]
                  <= e2_acc[shr_byte]
                     - 64'(e_pair)
                     - 64'(signed'({shb_rq[9:0], 1'b0}));
              e_pair <= '0; e_slot <= '0;
              if (e_site + 1 == n_sites) st <= E_COPY;
              else e_site <= e_site + 1;
            end
          end
        end
        E_COPY: begin
          if (e2_is_old)
            for (int r = 0; r < 64; r++) e2_old[r] <= e2_acc[r];
          st <= e2_ret;
        end

        // ------- PDRAIN streamer (D-032c leg C: credit-native) -------
        // One beat per cycle credits permitting, never waiting for a
        // completion. The exit waits for the credit pool to refill
        // with no pulse in flight, so busy deasserts only once every
        // drain word is memory-visible — the T_STATUS poll contract.
        D_RUN: begin
          if (d_idx >= d_total) begin
            if (!m_req && cred == 3'(CRED)) begin
              drain_words_q <= drain_words_q + 32'(d_total);
              st <= S_IDLE;
            end
          end else if (d_ph != d_need) begin  // D-028 fetch phases
            if (d_ph == 2'd1) begin           // COV: latch the site trio
              d_shv <= shs_rq[23]; d_m1s <= m1_rq; d_m2v <= m2_rq;
              d_nbr <= shs_rq[10 +: NB];      // credit-stall-stable nbr
            end
            if (d_ph == 2'd2) d_mnb <= m1_rq; // COV: m1[neighbor] stage
            d_ph <= d_ph + 2'd1;
          end else if (cred_nxt != 3'd0) begin
            d_req_q <= 1'b1; m_we <= 1'b1; m_be <= '1;
            m_addr <= q_c + {12'd0, d_idx, 2'b00};
            m_wdata <= drain_word(d_mode, d_idx);
            d_ph <= '0;
            d_idx <= d_idx + 1;
          end
        end
      endcase

      // ---------------- ACCWALK walker (D-034 §13c) ----------------
      // Runs concurrently with the PSAMPLE overlap states only (the
      // assert below); owns the shadow/counter read rows while busy
      // (comb mux above) and commits here, AFTER the case, so its
      // strobes cannot collide with any state's (P_ZERO/P_ACC-era
      // writers are unreachable while walk_busy). Math = the retired
      // P_ACC verbatim, fed from the snapshot instead of the live
      // fabric; every (site,lane)/(site,slot) is visited once per
      // walk, so the pipelined RMWs are hazard-free.
      if (walk_busy) begin
        if (w_cv) begin : g_wcommit
          // §13 v2: BOTH pair sites commit — banks {parity, lane/slot}
          // are disjoint by construction, one write port each.
          automatic logic [3:0] l0, l1;
          l0 = {1'b0, q_on ? ((w_cval0 == 3'd0) ? 3'd0 : (w_cval0 - 3'd1))
                           : 3'b000};
          l1 = {1'b1, q_on ? ((w_cval1 == 3'd0) ? 3'd0 : (w_cval1 - 3'd1))
                           : 3'b000};
          if (q_on ? (w_cval0 != 3'd0) : 1'b1) begin
            m1_web[l0] <= 1'b1;
            m1_wab[l0] <= w_cs;
            m1_wdb[l0] <= m1_rqb[l0] + (q_on ? 32'd1 : 32'(w_cbit0));
          end
          if (w_cv1 && (q_on ? (w_cval1 != 3'd0) : 1'b1)) begin
            m1_web[l1] <= 1'b1;
            m1_wab[l1] <= w_cs;
            m1_wdb[l1] <= m1_rqb[l1] + (q_on ? 32'd1 : 32'(w_cbit1));
          end
          for (int b = 0; b < 8; b++) begin : g_wslot
            automatic logic [NB-1:0] n0, n1;
            automatic logic [2:0]    v0, v1;
            n0 = shs_rqb[{1'b0, 3'(b)}][10 +: NB];
            v0 = snap_val(n0);
            if (shs_rqb[{1'b0, 3'(b)}][23]
                && (q_on ? (w_cval0 == v0)
                         : (w_cbit0 && v0[0]))) begin
              m2_web[{1'b0, 3'(b)}] <= 1'b1;
              m2_wab[{1'b0, 3'(b)}] <= w_cs;
              m2_wdb[{1'b0, 3'(b)}] <= m2_rqb[{1'b0, 3'(b)}] + 32'd1;
            end
            n1 = shs_rqb[{1'b1, 3'(b)}][10 +: NB];
            v1 = snap_val(n1);
            if (w_cv1 && shs_rqb[{1'b1, 3'(b)}][23]
                && (q_on ? (w_cval1 == v1)
                         : (w_cbit1 && v1[0]))) begin
              m2_web[{1'b1, 3'(b)}] <= 1'b1;
              m2_wab[{1'b1, 3'(b)}] <= w_cs;
              m2_wdb[{1'b1, 3'(b)}] <= m2_rqb[{1'b1, 3'(b)}] + 32'd1;
            end
          end
        end
        // F -> C hand-off (the pair's snapshot values, F-aligned)
        w_cv    <= !w_fdone;
        w_cv1   <= !w_fdone && (w_site + 1 < n_sites);
        w_cs    <= w_site[NB-1:1];
        w_cval0 <= w_sval0_f;
        w_cval1 <= w_sval1_f;
        w_cbit0 <= w_sval0_f[0];
        w_cbit1 <= w_sval1_f[0];
        // F advance by PAIR, then a 2-cycle drain
        if (!w_fdone) begin
          if (w_site + 2 >= n_sites) w_fdone <= 1'b1;
          else w_site <= w_site + 16'd2;
        end else if (!w_cv) begin
          w_tail <= w_tail + 1'b1;
          if (w_tail == 2'd1) begin
            walk_busy <= 1'b0;
            w_tail <= '0;
          end
        end
      end
`ifndef SYNTHESIS
      assert (!(walk_busy && !(st inside {P_SNAP, P_LOADSCH, P_START,
                                          P_WAIT, P_NEXT, P_ENTRY})))
        else $fatal(1, "s_cluster: walker outside PSAMPLE states (st=%0d)",
                    st);
`endif
    end
  end
endmodule
