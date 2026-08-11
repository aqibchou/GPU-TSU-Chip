// Barrel SIMT core v1 (M13): W scalar RV32I+Zicsr harts through one
// 5-stage pipeline (F D EX M WB), deterministic rotation, busy-bit-only
// interlock (unused in v1: both memories are fixed 1-cycle BRAM ports).
// Spec + invariants: docs/HARDWARE_ARCHITECTURE.md#simt-core. Execute semantics are the PROVEN
// scalar core's (rtl/simt/rv32i_core.sv, P1 spike-lockstep green) ported
// stage-wise; golden remains golden/iss.py per warp, mhartid = warp id.
// ALL architectural updates (regfile, CSRs, PC, commit) land in WB (INV-7).
module barrel_core #(
  parameter int W = 8,
  parameter logic [31:0] RESET_PC = 32'h0000_0000,
  localparam int WB_ = $clog2(W)
) (
  input  logic        clk,
  input  logic        rst_n,
  // imem: req this cycle -> rdata next cycle, always ready (spec §6)
  output logic        imem_req,
  output logic [31:0] imem_addr,
  input  logic [31:0] imem_rdata,
  // dmem: same contract, request only from M (INV-12)
  output logic        dmem_req,
  output logic        dmem_we,
  output logic [3:0]  dmem_be,
  output logic [31:0] dmem_addr,
  output logic [31:0] dmem_wdata,
  input  logic [31:0] dmem_rdata,
  // commit bus (INV-9): one record per retirement, per-warp lockstep
  output logic        cmt_valid,
  output logic [WB_-1:0] cmt_warp,
  output logic [31:0] cmt_pc,
  output logic [31:0] cmt_instr,
  output logic [4:0]  cmt_rd,
  output logic [31:0] cmt_wdata,
  output logic        cmt_st_valid,
  output logic [31:0] cmt_st_addr,
  output logic [31:0] cmt_st_data,
  output logic [1:0]  cmt_st_size
);
  localparam logic [31:0] MSTATUS_WMASK = 32'h0000_1888;
  localparam int DEPTH = 5;

  // ---------------- scheduler: the single source of stage tags ----------
  logic          issue_valid;
  logic [WB_-1:0] issue_warp;
  logic [W-1:0]  busy;
  // sched pipe convention: index s = the warp NOW IN STAGE s+1
  // (entry 0 latches at the F-ending edge, i.e. the D resident);
  // so D=sv[0], EX=sv[1], M=sv[2], WB=sv[3]; sv[4] = retired.
  logic [DEPTH-1:0]          sv;
  logic [DEPTH-1:0][WB_-1:0] sw;
  barrel_sched #(.W(W), .DEPTH(DEPTH)) u_sched (
    .clk(clk), .rst_n(rst_n),
    .set_busy(1'b0), .set_warp('0),      // v1: fixed-latency memories
    .clr_busy(1'b0), .clr_warp('0), .hold('0),
    .issue_valid(issue_valid), .issue_warp(issue_warp), .busy(busy),
    .stage_valid(sv), .stage_warp(sw)
  );
  wire _unused_busy = &{1'b0, busy, sv[4], sv[1:0], sw[4], sw[2], sw[0]};

  // ---------------- per-warp architectural state ----------------
  logic [31:0] pc      [W];
  logic [31:0] mstatus [W], mie [W], mtvec [W], mscratch [W];
  logic [31:0] mepc    [W], mcause [W], mtval [W];

  // ---------------- F: fetch request ----------------
  assign imem_req  = issue_valid;
  assign imem_addr = pc[issue_warp];

  // ---------------- D: instruction arrival + regfile read --------------
  // imem_rdata corresponds to the F request of last cycle = stage D now.
  logic [31:0] d_pc;
  wire  [31:0] d_ir = imem_rdata;

  logic [31:0] rs1_data, rs2_data;
  warp_regfile #(.W(W)) u_rf (
    .clk(clk),
    .r_warp(sw[0]), .rs1(d_ir[19:15]), .rs2(d_ir[24:20]),
    .rs1_data(rs1_data), .rs2_data(rs2_data),
    .w_en(wb_rf_we), .w_stage_valid(sv[3]),
    .w_warp(sw[3]), .w_rd(wb_rd), .w_data(wb_rf_wdata)
  );

  // ---------------- EX stage registers ----------------
  logic [31:0] ex_pc, ex_ir, ex_a, ex_b;

  // decode fields (EX)
  wire [6:0] opc = ex_ir[6:0];
  wire [2:0] f3  = ex_ir[14:12];
  wire [4:0] rs1 = ex_ir[19:15];
  wire [6:0] f7  = ex_ir[31:25];
  wire [31:0] a = ex_a, b = ex_b;
  wire [4:0]  shamt = ex_ir[24:20];

  wire [31:0] imm_i = {{20{ex_ir[31]}}, ex_ir[31:20]};
  wire [31:0] imm_s = {{20{ex_ir[31]}}, ex_ir[31:25], ex_ir[11:7]};
  wire [31:0] imm_b = {{19{ex_ir[31]}}, ex_ir[31], ex_ir[7], ex_ir[30:25], ex_ir[11:8], 1'b0};
  wire [31:0] imm_u = {ex_ir[31:12], 12'b0};
  wire [31:0] imm_j = {{11{ex_ir[31]}}, ex_ir[31], ex_ir[19:12], ex_ir[20], ex_ir[30:21], 1'b0};

  wire [11:0] csr_a = ex_ir[31:20];
  wire [WB_-1:0] ex_w = sw[1];

  function automatic logic csr_dummy(input logic [11:0] adr);
    return (adr == 12'h310) || (adr == 12'h344) || (adr == 12'h320)
        || (adr == 12'hB00) || (adr == 12'hB02) || (adr == 12'hB80) || (adr == 12'hB82)
        || (adr >= 12'h3A0 && adr <= 12'h3A3)
        || (adr >= 12'h3B0 && adr <= 12'h3BF);
  endfunction
  function automatic logic csr_ro(input logic [11:0] adr);
    return (adr == 12'hF11) || (adr == 12'hF12) || (adr == 12'hF13)
        || (adr == 12'hF14) || (adr == 12'hF15) || (adr == 12'h301);
  endfunction
  function automatic logic csr_stored(input logic [11:0] adr);
    return (adr == 12'h300) || (adr == 12'h304) || (adr == 12'h305)
        || (adr == 12'h340) || (adr == 12'h341) || (adr == 12'h342) || (adr == 12'h343);
  endfunction

  logic [31:0] csr_rval;
  always_comb begin
    unique case (csr_a)
      12'h300: csr_rval = mstatus[ex_w];
      12'h304: csr_rval = mie[ex_w];
      12'h305: csr_rval = mtvec[ex_w];
      12'h340: csr_rval = mscratch[ex_w];
      12'h341: csr_rval = mepc[ex_w];
      12'h342: csr_rval = mcause[ex_w];
      12'h343: csr_rval = mtval[ex_w];
      12'h301: csr_rval = 32'h4000_0100;
      12'hF14: csr_rval = {{(32-WB_){1'b0}}, ex_w};   // mhartid = warp id
      default: csr_rval = 32'd0;
    endcase
  end

  // ---------------- EX: the proven combinational execute ----------------
  logic        ex_illegal, ex_trap;
  logic [31:0] ex_cause, ex_tval;
  logic        ex_wb_en;
  logic [31:0] ex_wb_val, ex_next_pc;
  logic        ex_is_load, ex_is_store;
  logic [31:0] ex_maddr;
  logic        ex_csr_we;
  logic [31:0] ex_csr_wval;

  wire [31:0] sum_i    = a + imm_i;
  wire [31:0] br_tgt   = ex_pc + imm_b;
  wire [31:0] jal_tgt  = ex_pc + imm_j;
  wire [31:0] jalr_tgt = (a + imm_i) & 32'hFFFF_FFFE;

  logic br_taken;
  always_comb begin
    unique case (f3)
      3'b000: br_taken = (a == b);
      3'b001: br_taken = (a != b);
      3'b100: br_taken = ($signed(a) < $signed(b));
      3'b101: br_taken = ($signed(a) >= $signed(b));
      3'b110: br_taken = (a < b);
      3'b111: br_taken = (a >= b);
      default: br_taken = 1'b0;
    endcase
  end

  wire [31:0] st_mask = (f3[1:0] == 2'd0) ? 32'h0000_00FF
                      : (f3[1:0] == 2'd1) ? 32'h0000_FFFF : 32'hFFFF_FFFF;

  always_comb begin
    ex_illegal  = 1'b0;
    ex_trap     = 1'b0;
    ex_cause    = 32'd0;
    ex_tval     = 32'd0;
    ex_wb_en    = 1'b0;
    ex_wb_val   = 32'd0;
    ex_next_pc  = ex_pc + 32'd4;
    ex_is_load  = 1'b0;
    ex_is_store = 1'b0;
    ex_maddr    = 32'd0;
    ex_csr_we   = 1'b0;
    ex_csr_wval = 32'd0;

    unique case (opc)
      7'b0110111: begin ex_wb_en = 1'b1; ex_wb_val = imm_u; end
      7'b0010111: begin ex_wb_en = 1'b1; ex_wb_val = ex_pc + imm_u; end
      7'b1101111: begin
        if (jal_tgt[1:0] != 2'b00) begin
          ex_trap = 1'b1; ex_cause = 32'd0; ex_tval = jal_tgt;
        end else begin
          ex_wb_en = 1'b1; ex_wb_val = ex_pc + 32'd4; ex_next_pc = jal_tgt;
        end
      end
      7'b1100111: begin
        if (f3 != 3'b000) ex_illegal = 1'b1;
        else if (jalr_tgt[1:0] != 2'b00) begin
          ex_trap = 1'b1; ex_cause = 32'd0; ex_tval = jalr_tgt;
        end else begin
          ex_wb_en = 1'b1; ex_wb_val = ex_pc + 32'd4; ex_next_pc = jalr_tgt;
        end
      end
      7'b1100011: begin
        if (f3 == 3'b010 || f3 == 3'b011) ex_illegal = 1'b1;
        else if (br_taken) begin
          if (br_tgt[1:0] != 2'b00) begin
            ex_trap = 1'b1; ex_cause = 32'd0; ex_tval = br_tgt;
          end else ex_next_pc = br_tgt;
        end
      end
      7'b0000011: begin
        if (f3 == 3'b011 || f3 == 3'b110 || f3 == 3'b111) ex_illegal = 1'b1;
        else begin
          ex_maddr = sum_i;
          if ((f3[1:0] == 2'd1 && sum_i[0]) || (f3[1:0] == 2'd2 && sum_i[1:0] != 2'b00)) begin
            ex_trap = 1'b1; ex_cause = 32'd4; ex_tval = sum_i;
          end else ex_is_load = 1'b1;
        end
      end
      7'b0100011: begin
        if (f3 > 3'b010) ex_illegal = 1'b1;
        else begin
          ex_maddr = a + imm_s;
          if ((f3[1:0] == 2'd1 && ex_maddr[0]) || (f3[1:0] == 2'd2 && ex_maddr[1:0] != 2'b00)) begin
            ex_trap = 1'b1; ex_cause = 32'd6; ex_tval = ex_maddr;
          end else ex_is_store = 1'b1;
        end
      end
      7'b0010011: begin
        ex_wb_en = 1'b1;
        unique case (f3)
          3'b000: ex_wb_val = sum_i;
          3'b010: ex_wb_val = {31'd0, $signed(a) < $signed(imm_i)};
          3'b011: ex_wb_val = {31'd0, a < imm_i};
          3'b100: ex_wb_val = a ^ imm_i;
          3'b110: ex_wb_val = a | imm_i;
          3'b111: ex_wb_val = a & imm_i;
          3'b001: begin
            if (f7 != 7'd0) begin ex_illegal = 1'b1; ex_wb_en = 1'b0; end
            else ex_wb_val = a << shamt;
          end
          3'b101: begin
            if (f7 == 7'd0)       ex_wb_val = a >> shamt;
            else if (f7 == 7'h20) ex_wb_val = $unsigned($signed(a) >>> shamt);
            else begin ex_illegal = 1'b1; ex_wb_en = 1'b0; end
          end
        endcase
      end
      7'b0110011: begin
        ex_wb_en = 1'b1;
        unique case ({f7, f3})
          {7'd0,  3'b000}: ex_wb_val = a + b;
          {7'h20, 3'b000}: ex_wb_val = a - b;
          {7'd0,  3'b001}: ex_wb_val = a << b[4:0];
          {7'd0,  3'b010}: ex_wb_val = {31'd0, $signed(a) < $signed(b)};
          {7'd0,  3'b011}: ex_wb_val = {31'd0, a < b};
          {7'd0,  3'b100}: ex_wb_val = a ^ b;
          {7'd0,  3'b101}: ex_wb_val = a >> b[4:0];
          {7'h20, 3'b101}: ex_wb_val = $unsigned($signed(a) >>> b[4:0]);
          {7'd0,  3'b110}: ex_wb_val = a | b;
          {7'd0,  3'b111}: ex_wb_val = a & b;
          default: begin ex_illegal = 1'b1; ex_wb_en = 1'b0; end
        endcase
      end
      7'b0001111: begin
        if (f3 > 3'b001) ex_illegal = 1'b1;
      end
      7'b1110011: begin
        if (f3 == 3'b000) begin
          unique case (ex_ir)
            32'h0000_0073: begin ex_trap = 1'b1; ex_cause = 32'd11; end
            32'h0010_0073: begin ex_trap = 1'b1; ex_cause = 32'd3; ex_tval = ex_pc; end
            32'h3020_0073: ex_next_pc = mepc[ex_w];
            default: ex_illegal = 1'b1;
          endcase
        end else if (f3 == 3'b100) ex_illegal = 1'b1;
        else begin
          logic [31:0] src;
          logic        wr;
          src = f3[2] ? {27'd0, rs1} : a;
          wr  = (f3[1:0] == 2'b01) || (rs1 != 5'd0);
          if (!(csr_stored(csr_a) || csr_ro(csr_a) || csr_dummy(csr_a)))
            ex_illegal = 1'b1;
          else if (wr && csr_ro(csr_a))
            ex_illegal = 1'b1;
          else begin
            ex_wb_en  = 1'b1;
            ex_wb_val = csr_rval;
            ex_csr_we = wr && csr_stored(csr_a);
            unique case (f3[1:0])
              2'b01:   ex_csr_wval = src;
              2'b10:   ex_csr_wval = csr_rval | src;
              default: ex_csr_wval = csr_rval & ~src;
            endcase
          end
        end
      end
      default: ex_illegal = 1'b1;
    endcase

    if (ex_illegal) begin
      ex_trap  = 1'b1;
      ex_cause = 32'd2;
      ex_tval  = ex_ir;
      ex_wb_en = 1'b0;
      ex_is_load = 1'b0;
      ex_is_store = 1'b0;
      ex_csr_we = 1'b0;
      ex_next_pc = ex_pc + 32'd4;
    end
  end

  logic [31:0] csr_wval_legal;
  always_comb begin
    csr_wval_legal = ex_csr_wval;
    if (csr_a == 12'h300)      csr_wval_legal = ex_csr_wval & MSTATUS_WMASK;
    else if (csr_a == 12'h305) csr_wval_legal = ex_csr_wval & 32'hFFFF_FFFC;
    else if (csr_a == 12'h341) csr_wval_legal = ex_csr_wval & 32'hFFFF_FFFC;
  end

  wire [1:0]  st_off  = ex_maddr[1:0];   // used by st_be shift
  logic [3:0] st_be;
  always_comb begin
    unique case (f3[1:0])
      2'd0:    st_be = 4'b0001 << st_off;
      2'd1:    st_be = 4'b0011 << st_off;
      default: st_be = 4'b1111;
    endcase
  end

  // ---------------- M stage registers (EX results carried) --------------
  logic [31:0] m_pc, m_ir, m_next_pc, m_wb_val, m_maddr, m_stdata;
  logic        m_trap, m_wb_en, m_is_load, m_is_store, m_csr_we, m_is_mret;
  logic [31:0] m_cause, m_tval, m_csr_wval;
  logic [11:0] m_csr_a;
  logic [3:0]  m_st_be;

  // dmem request: registered, fires during M (INV-12)
  assign dmem_req   = sv[2] && (m_is_load || m_is_store);
  assign dmem_we    = m_is_store;
  assign dmem_be    = m_is_store ? m_st_be : 4'd0;
  assign dmem_addr  = {m_maddr[31:2], 2'b00};
  assign dmem_wdata = m_stdata << {m_maddr[1:0], 3'b000};

  // ---------------- WB stage registers ----------------
  logic [31:0] wb_pc, wb_ir, wb_next_pc, wb_wb_val, wb_maddr, wb_stdata;
  logic        wb_trap, wb_en_q, wb_is_load, wb_is_store, wb_csr_we, wb_is_mret;
  logic [31:0] wb_cause, wb_tval, wb_csr_wval;
  logic [11:0] wb_csr_a;

  wire [2:0]  wb_f3 = wb_ir[14:12];
  wire [4:0]  wb_rd_f = wb_ir[11:7];
  wire [31:0] ld_shifted = dmem_rdata >> {wb_maddr[1:0], 3'b000};
  logic [31:0] ld_val;
  always_comb begin
    unique case (wb_f3)
      3'b000:  ld_val = {{24{ld_shifted[7]}},  ld_shifted[7:0]};
      3'b001:  ld_val = {{16{ld_shifted[15]}}, ld_shifted[15:0]};
      3'b100:  ld_val = {24'd0, ld_shifted[7:0]};
      3'b101:  ld_val = {16'd0, ld_shifted[15:0]};
      default: ld_val = ld_shifted;
    endcase
  end

  wire        wb_v      = sv[3];
  wire [WB_-1:0] wb_w   = sw[3];
  wire [31:0] wb_result = wb_is_load ? ld_val : wb_wb_val;
  wire        wb_rf_we  = wb_v && !wb_trap && (wb_en_q || wb_is_load)
                          && (wb_rd_f != 5'd0);
  wire [4:0]  wb_rd     = wb_rd_f;
  wire [31:0] wb_rf_wdata = wb_result;

  // ---------------- pipeline + architectural update ----------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      d_pc <= 32'd0;
      ex_pc <= 32'd0; ex_ir <= 32'd0; ex_a <= 32'd0; ex_b <= 32'd0;
      m_pc <= 32'd0; m_ir <= 32'd0; m_next_pc <= 32'd0; m_wb_val <= 32'd0;
      m_maddr <= 32'd0; m_stdata <= 32'd0; m_trap <= 1'b0; m_wb_en <= 1'b0;
      m_is_load <= 1'b0; m_is_store <= 1'b0; m_csr_we <= 1'b0;
      m_is_mret <= 1'b0; m_cause <= 32'd0; m_tval <= 32'd0;
      m_csr_wval <= 32'd0; m_csr_a <= 12'd0; m_st_be <= 4'd0;
      wb_pc <= 32'd0; wb_ir <= 32'd0; wb_next_pc <= 32'd0; wb_wb_val <= 32'd0;
      wb_maddr <= 32'd0; wb_stdata <= 32'd0; wb_trap <= 1'b0; wb_en_q <= 1'b0;
      wb_is_load <= 1'b0; wb_is_store <= 1'b0; wb_csr_we <= 1'b0;
      wb_is_mret <= 1'b0; wb_cause <= 32'd0; wb_tval <= 32'd0;
      wb_csr_wval <= 32'd0; wb_csr_a <= 12'd0;
      cmt_valid <= 1'b0; cmt_warp <= '0; cmt_pc <= 32'd0; cmt_instr <= 32'd0;
      cmt_rd <= 5'd0; cmt_wdata <= 32'd0; cmt_st_valid <= 1'b0;
      cmt_st_addr <= 32'd0; cmt_st_data <= 32'd0; cmt_st_size <= 2'd0;
      for (int w = 0; w < W; w++) begin
        pc[w] <= RESET_PC;
        mstatus[w] <= 32'd0; mie[w] <= 32'd0; mtvec[w] <= 32'd0;
        mscratch[w] <= 32'd0; mepc[w] <= 32'd0; mcause[w] <= 32'd0;
        mtval[w] <= 32'd0;
      end
    end else begin
      cmt_valid <= 1'b0;
      cmt_st_valid <= 1'b0;

      // F -> D
      d_pc <= pc[issue_warp];
      // D -> EX
      ex_pc <= d_pc; ex_ir <= d_ir; ex_a <= rs1_data; ex_b <= rs2_data;
      // EX -> M
      m_pc <= ex_pc; m_ir <= ex_ir; m_next_pc <= ex_next_pc;
      m_wb_val <= ex_wb_val; m_maddr <= ex_maddr;
      m_stdata <= b & st_mask;
      m_trap <= ex_trap; m_wb_en <= ex_wb_en;
      m_is_load <= ex_is_load; m_is_store <= ex_is_store;
      m_csr_we <= ex_csr_we; m_is_mret <= (ex_ir == 32'h3020_0073) && !ex_trap;
      m_cause <= ex_cause; m_tval <= ex_tval;
      m_csr_wval <= csr_wval_legal; m_csr_a <= csr_a; m_st_be <= st_be;
      // M -> WB
      wb_pc <= m_pc; wb_ir <= m_ir; wb_next_pc <= m_next_pc;
      wb_wb_val <= m_wb_val; wb_maddr <= m_maddr; wb_stdata <= m_stdata;
      wb_trap <= m_trap; wb_en_q <= m_wb_en;
      wb_is_load <= m_is_load; wb_is_store <= m_is_store;
      wb_csr_we <= m_csr_we; wb_is_mret <= m_is_mret;
      wb_cause <= m_cause; wb_tval <= m_tval;
      wb_csr_wval <= m_csr_wval; wb_csr_a <= m_csr_a;

      // WB: the only place architectural state changes (INV-7)
      if (wb_v) begin
        if (wb_trap) begin
          mepc[wb_w]   <= wb_pc;
          mcause[wb_w] <= wb_cause;
          mtval[wb_w]  <= wb_tval;
          mstatus[wb_w] <= ({mstatus[wb_w][31:8], 1'b0, mstatus[wb_w][6:4],
                             1'b0, mstatus[wb_w][2:0]} & ~32'h0000_1880)
                          | {28'd0, mstatus[wb_w][3], 3'd0} << 4
                          | 32'h0000_1800;
          pc[wb_w] <= mtvec[wb_w] & 32'hFFFF_FFFC;
        end else begin
          if (wb_is_mret)
            mstatus[wb_w] <= ({28'd0, mstatus[wb_w][7], 3'd0}
                              | 32'h0000_0080) & MSTATUS_WMASK;
          if (wb_csr_we) begin
            unique case (wb_csr_a)
              12'h300: mstatus[wb_w]  <= wb_csr_wval;
              12'h304: mie[wb_w]      <= wb_csr_wval;
              12'h305: mtvec[wb_w]    <= wb_csr_wval;
              12'h340: mscratch[wb_w] <= wb_csr_wval;
              12'h341: mepc[wb_w]     <= wb_csr_wval;
              12'h342: mcause[wb_w]   <= wb_csr_wval;
              12'h343: mtval[wb_w]    <= wb_csr_wval;
              default: ;
            endcase
          end
          pc[wb_w] <= wb_next_pc;
          cmt_valid <= 1'b1;
          cmt_warp  <= wb_w;
          cmt_pc    <= wb_pc;
          cmt_instr <= wb_ir;
          cmt_rd    <= wb_rf_we ? wb_rd : 5'd0;
          cmt_wdata <= wb_result;
          cmt_st_valid <= wb_is_store;
          cmt_st_addr  <= wb_maddr;
          cmt_st_data  <= wb_stdata;
          cmt_st_size  <= wb_ir[13:12];
        end
      end
    end
  end

`ifndef SYNTHESIS
  /* verilator lint_off SYNCASYNCNET */
  always_ff @(posedge clk) if (rst_n) begin
    // INV-6: bubbles are inert
    if (!sv[2]) assert (!dmem_req)
      else $fatal(1, "INV-6: dmem req from bubble M stage");
    // INV-12: dmem request only from the M-stage warp (by construction —
    // the assert documents it for the formal flow)
  end
  /* verilator lint_on SYNCASYNCNET */
`endif
endmodule
