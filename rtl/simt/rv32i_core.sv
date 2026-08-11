// RV32I + Zicsr + Zifencei, in-order multi-cycle FSM core. Golden: golden/iss.py.
// Deliberately boring (M2): FETCH -> EXEC -> [MEM] at 3-4 cycles/instr; later
// recycled as p-bit fabric control. CSR model and trap behavior mirror
// `spike --priv=m` exactly — see the iss.py docstring for the table.
// Commit interface: registered 1-cycle pulse per retired instruction
// (trapped instructions do not commit, matching spike --log-commits).
module rv32i_core (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] reset_pc,
  // memory: req for 1 cycle, ack+rdata the following cycle (BRAM-style)
  output logic        mem_req,
  output logic        mem_we,
  output logic [3:0]  mem_be,
  output logic [31:0] mem_addr,
  output logic [31:0] mem_wdata,
  input  logic [31:0] mem_rdata,
  input  logic        mem_ack,
  // commit log
  output logic        cmt_valid,
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

  typedef enum logic [2:0] { S_FREQ, S_FWAIT, S_EXEC, S_MWAIT } state_e;
  state_e st;

  logic [31:0] pc, ir;
  logic [31:0] rf [32];

  // CSRs with storage
  logic [31:0] mstatus, mie, mtvec, mscratch, mepc, mcause, mtval;

  // decode fields
  wire [6:0] opc = ir[6:0];
  wire [4:0] rd_f = ir[11:7];
  wire [2:0] f3  = ir[14:12];
  wire [4:0] rs1 = ir[19:15];
  wire [4:0] rs2 = ir[24:20];
  wire [6:0] f7  = ir[31:25];

  wire [31:0] a = (rs1 == 5'd0) ? 32'd0 : rf[rs1];
  wire [31:0] b = (rs2 == 5'd0) ? 32'd0 : rf[rs2];

  wire [31:0] imm_i = {{20{ir[31]}}, ir[31:20]};
  wire [31:0] imm_s = {{20{ir[31]}}, ir[31:25], ir[11:7]};
  wire [31:0] imm_b = {{19{ir[31]}}, ir[31], ir[7], ir[30:25], ir[11:8], 1'b0};
  wire [31:0] imm_u = {ir[31:12], 12'b0};
  wire [31:0] imm_j = {{11{ir[31]}}, ir[31], ir[19:12], ir[20], ir[30:21], 1'b0};

  // ---------------- CSR read/write plumbing ----------------
  wire [11:0] csr_a = ir[31:20];

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
      12'h300: csr_rval = mstatus;
      12'h304: csr_rval = mie;
      12'h305: csr_rval = mtvec;
      12'h340: csr_rval = mscratch;
      12'h341: csr_rval = mepc;
      12'h342: csr_rval = mcause;
      12'h343: csr_rval = mtval;
      12'h301: csr_rval = 32'h4000_0100;   // misa: RV32I
      default: csr_rval = 32'd0;           // RO-zero ids + dummies
    endcase
  end

  // ---------------- execute (combinational) ----------------
  logic        ex_illegal;
  logic        ex_trap;        // any trap (incl. illegal/ecall/ebreak/misalign)
  logic [31:0] ex_cause, ex_tval;
  logic        ex_wb_en;
  logic [31:0] ex_wb_val;
  logic [31:0] ex_next_pc;
  logic        ex_is_load, ex_is_store;
  logic [31:0] ex_maddr;
  logic        ex_csr_we;
  logic [31:0] ex_csr_wval;

  wire [4:0]  shamt   = rs2;
  wire [31:0] sum_i   = a + imm_i;
  wire [31:0] br_tgt  = pc + imm_b;
  wire [31:0] jal_tgt = pc + imm_j;
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

  wire [1:0] ld_size_l2 = f3[1:0];             // 0=B 1=H 2=W (f3 masked)
  wire [31:0] st_mask = (f3[1:0] == 2'd0) ? 32'h0000_00FF
                      : (f3[1:0] == 2'd1) ? 32'h0000_FFFF : 32'hFFFF_FFFF;

  always_comb begin
    ex_illegal  = 1'b0;
    ex_trap     = 1'b0;
    ex_cause    = 32'd0;
    ex_tval     = 32'd0;
    ex_wb_en    = 1'b0;
    ex_wb_val   = 32'd0;
    ex_next_pc  = pc + 32'd4;
    ex_is_load  = 1'b0;
    ex_is_store = 1'b0;
    ex_maddr    = 32'd0;
    ex_csr_we   = 1'b0;
    ex_csr_wval = 32'd0;

    unique case (opc)
      7'b0110111: begin ex_wb_en = 1'b1; ex_wb_val = imm_u; end            // LUI
      7'b0010111: begin ex_wb_en = 1'b1; ex_wb_val = pc + imm_u; end       // AUIPC
      7'b1101111: begin                                                    // JAL
        if (jal_tgt[1:0] != 2'b00) begin
          ex_trap = 1'b1; ex_cause = 32'd0; ex_tval = jal_tgt;
        end else begin
          ex_wb_en = 1'b1; ex_wb_val = pc + 32'd4; ex_next_pc = jal_tgt;
        end
      end
      7'b1100111: begin                                                    // JALR
        if (f3 != 3'b000) ex_illegal = 1'b1;
        else if (jalr_tgt[1:0] != 2'b00) begin
          ex_trap = 1'b1; ex_cause = 32'd0; ex_tval = jalr_tgt;
        end else begin
          ex_wb_en = 1'b1; ex_wb_val = pc + 32'd4; ex_next_pc = jalr_tgt;
        end
      end
      7'b1100011: begin                                                    // branches
        if (f3 == 3'b010 || f3 == 3'b011) ex_illegal = 1'b1;
        else if (br_taken) begin
          if (br_tgt[1:0] != 2'b00) begin
            ex_trap = 1'b1; ex_cause = 32'd0; ex_tval = br_tgt;
          end else ex_next_pc = br_tgt;
        end
      end
      7'b0000011: begin                                                    // loads
        if (f3 == 3'b011 || f3 == 3'b110 || f3 == 3'b111) ex_illegal = 1'b1;
        else begin
          ex_maddr = sum_i;
          if ((f3[1:0] == 2'd1 && sum_i[0]) || (f3[1:0] == 2'd2 && sum_i[1:0] != 2'b00)) begin
            ex_trap = 1'b1; ex_cause = 32'd4; ex_tval = sum_i;
          end else ex_is_load = 1'b1;
        end
      end
      7'b0100011: begin                                                    // stores
        if (f3 > 3'b010) ex_illegal = 1'b1;
        else begin
          ex_maddr = a + imm_s;
          if ((f3[1:0] == 2'd1 && ex_maddr[0]) || (f3[1:0] == 2'd2 && ex_maddr[1:0] != 2'b00)) begin
            ex_trap = 1'b1; ex_cause = 32'd6; ex_tval = ex_maddr;
          end else ex_is_store = 1'b1;
        end
      end
      7'b0010011: begin                                                    // OP-IMM
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
            if (f7 == 7'd0)          ex_wb_val = a >> shamt;
            else if (f7 == 7'h20)    ex_wb_val = $unsigned($signed(a) >>> shamt);
            else begin ex_illegal = 1'b1; ex_wb_en = 1'b0; end
          end
        endcase
      end
      7'b0110011: begin                                                    // OP
        ex_wb_en = 1'b1;
        unique case ({f7, f3})
          {7'd0,   3'b000}: ex_wb_val = a + b;
          {7'h20,  3'b000}: ex_wb_val = a - b;
          {7'd0,   3'b001}: ex_wb_val = a << b[4:0];
          {7'd0,   3'b010}: ex_wb_val = {31'd0, $signed(a) < $signed(b)};
          {7'd0,   3'b011}: ex_wb_val = {31'd0, a < b};
          {7'd0,   3'b100}: ex_wb_val = a ^ b;
          {7'd0,   3'b101}: ex_wb_val = a >> b[4:0];
          {7'h20,  3'b101}: ex_wb_val = $unsigned($signed(a) >>> b[4:0]);
          {7'd0,   3'b110}: ex_wb_val = a | b;
          {7'd0,   3'b111}: ex_wb_val = a & b;
          default: begin ex_illegal = 1'b1; ex_wb_en = 1'b0; end
        endcase
      end
      7'b0001111: begin                                                    // FENCE/FENCE.I
        if (f3 > 3'b001) ex_illegal = 1'b1;
      end
      7'b1110011: begin                                                    // SYSTEM
        if (f3 == 3'b000) begin
          unique case (ir)
            32'h0000_0073: begin ex_trap = 1'b1; ex_cause = 32'd11; end    // ECALL (M)
            32'h0010_0073: begin ex_trap = 1'b1; ex_cause = 32'd3; ex_tval = pc; end // EBREAK
            32'h3020_0073: ex_next_pc = mepc;                              // MRET
            default: ex_illegal = 1'b1;
          endcase
        end else if (f3 == 3'b100) ex_illegal = 1'b1;
        else begin                                                         // Zicsr
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
      ex_tval  = ir;
      ex_wb_en = 1'b0;
      ex_is_load = 1'b0;
      ex_is_store = 1'b0;
      ex_csr_we = 1'b0;
      ex_next_pc = pc + 32'd4;
    end
  end

  // CSR write legalization (mirrors iss.py _csr_write)
  logic [31:0] csr_wval_legal;
  always_comb begin
    csr_wval_legal = ex_csr_wval;
    if (csr_a == 12'h300)      csr_wval_legal = ex_csr_wval & MSTATUS_WMASK;
    else if (csr_a == 12'h305) csr_wval_legal = ex_csr_wval & 32'hFFFF_FFFC;
    else if (csr_a == 12'h341) csr_wval_legal = ex_csr_wval & 32'hFFFF_FFFC;
  end

  // store datapath
  wire [1:0]  st_off  = ex_maddr[1:0];
  wire [31:0] st_data = (b & st_mask) << {st_off, 3'b000};
  logic [3:0] st_be;
  always_comb begin
    unique case (f3[1:0])
      2'd0:    st_be = 4'b0001 << st_off;
      2'd1:    st_be = 4'b0011 << st_off;
      default: st_be = 4'b1111;
    endcase
  end

  // load extraction (in S_MWAIT, from registered address low bits)
  logic [1:0] ld_off_q;
  wire [31:0] ld_shifted = mem_rdata >> {ld_off_q, 3'b000};
  logic [31:0] ld_val;
  always_comb begin
    unique case (f3)
      3'b000:  ld_val = {{24{ld_shifted[7]}},  ld_shifted[7:0]};   // LB
      3'b001:  ld_val = {{16{ld_shifted[15]}}, ld_shifted[15:0]};  // LH
      3'b100:  ld_val = {24'd0, ld_shifted[7:0]};                  // LBU
      3'b101:  ld_val = {16'd0, ld_shifted[15:0]};                 // LHU
      default: ld_val = ld_shifted;                                // LW
    endcase
  end

  // ---------------- FSM + state update ----------------
  logic [31:0] maddr_q, stdata_q;
  logic        isload_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_FREQ;
      pc <= reset_pc;
      ir <= 32'd0;
      for (int i = 0; i < 32; i++) rf[i] <= 32'd0;
      mstatus <= 32'd0; mie <= 32'd0; mtvec <= 32'd0; mscratch <= 32'd0;
      mepc <= 32'd0; mcause <= 32'd0; mtval <= 32'd0;
      mem_req <= 1'b0; mem_we <= 1'b0; mem_be <= 4'd0;
      mem_addr <= 32'd0; mem_wdata <= 32'd0;
      cmt_valid <= 1'b0; cmt_pc <= 32'd0; cmt_instr <= 32'd0;
      cmt_rd <= 5'd0; cmt_wdata <= 32'd0;
      cmt_st_valid <= 1'b0; cmt_st_addr <= 32'd0; cmt_st_data <= 32'd0;
      cmt_st_size <= 2'd0;
      maddr_q <= 32'd0; stdata_q <= 32'd0; isload_q <= 1'b0; ld_off_q <= 2'd0;
    end else begin
      cmt_valid <= 1'b0;
      cmt_st_valid <= 1'b0;
      mem_req <= 1'b0;

      unique case (st)
        S_FREQ: begin
          mem_req  <= 1'b1;
          mem_we   <= 1'b0;
          mem_be   <= 4'd0;
          mem_addr <= pc;
          st       <= S_FWAIT;
        end
        S_FWAIT: if (mem_ack) begin
          ir <= mem_rdata;
          st <= S_EXEC;
        end
        S_EXEC: begin
          if (ex_trap) begin
            mepc    <= pc;
            mcause  <= ex_cause;
            mtval   <= ex_tval;
            mstatus <= ({mstatus[31:8], 1'b0, mstatus[6:4], 1'b0, mstatus[2:0]}
                        & ~32'h0000_1880)
                       | {28'd0, mstatus[3], 3'd0} << 4     // MPIE <- MIE (bit 7)
                       | 32'h0000_1800;                     // MPP <- 11
            pc      <= mtvec & 32'hFFFF_FFFC;
            st      <= S_FREQ;
          end else if (ex_is_load || ex_is_store) begin
            mem_req   <= 1'b1;
            mem_we    <= ex_is_store;
            mem_be    <= ex_is_store ? st_be : 4'd0;
            mem_addr  <= {ex_maddr[31:2], 2'b00};
            mem_wdata <= st_data;
            maddr_q   <= ex_maddr;
            stdata_q  <= b & st_mask;
            isload_q  <= ex_is_load;
            ld_off_q  <= ex_maddr[1:0];
            st        <= S_MWAIT;
          end else begin
            if (ir == 32'h3020_0073) begin                  // MRET mstatus update
              mstatus <= ({28'd0, mstatus[7], 3'd0}         // MIE <- MPIE
                          | 32'h0000_0080) & MSTATUS_WMASK; // MPIE <- 1, MPP <- 0
            end
            if (ex_csr_we) begin
              unique case (csr_a)
                12'h300: mstatus  <= csr_wval_legal;
                12'h304: mie      <= csr_wval_legal;
                12'h305: mtvec    <= csr_wval_legal;
                12'h340: mscratch <= csr_wval_legal;
                12'h341: mepc     <= csr_wval_legal;
                12'h342: mcause   <= csr_wval_legal;
                12'h343: mtval    <= csr_wval_legal;
                default: ;
              endcase
            end
            if (ex_wb_en && rd_f != 5'd0) rf[rd_f] <= ex_wb_val;
            cmt_valid <= 1'b1;
            cmt_pc    <= pc;
            cmt_instr <= ir;
            cmt_rd    <= (ex_wb_en && rd_f != 5'd0) ? rd_f : 5'd0;
            cmt_wdata <= ex_wb_val;
            pc        <= ex_next_pc;
            st        <= S_FREQ;
          end
        end
        S_MWAIT: if (mem_ack) begin
          if (isload_q && rd_f != 5'd0) rf[rd_f] <= ld_val;
          cmt_valid    <= 1'b1;
          cmt_pc       <= pc;
          cmt_instr    <= ir;
          cmt_rd       <= (isload_q && rd_f != 5'd0) ? rd_f : 5'd0;
          cmt_wdata    <= ld_val;
          cmt_st_valid <= !isload_q;
          cmt_st_addr  <= maddr_q;
          cmt_st_data  <= stdata_q;
          cmt_st_size  <= ld_size_l2;
          pc           <= pc + 32'd4;
          st           <= S_FREQ;
        end
        default: st <= S_FREQ;
      endcase
    end
  end
endmodule
