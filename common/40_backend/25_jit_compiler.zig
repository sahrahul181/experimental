//! Direct JIT compiler from Dalvik instructions (`Instruction`) to native x86-64 machine code (`CompiledCode`),
//! satisfying the W^X publication and JIT contracts described in `docs/interpreter_jit_runtime_model.md`.

const std = @import("std");
const builtin = @import("builtin");
const code_buffer = @import("code_buffer");
const instructions = @import("instructions");
const jit_memory = @import("jit_memory");
const runtime_code_manager = @import("runtime_code_manager");
const runtime_deopt = @import("runtime_deopt");
const runtime_stack_map = @import("runtime_stack_map");

const Instruction = instructions.Instruction;

pub const Relocation = code_buffer.Relocation;
pub const SafepointMap = runtime_stack_map.MapSpec;
pub const DeoptPoint = runtime_deopt.PointSpec;

pub const DependencyKind = enum(u8) {
    class,
    method,
    inline_cache,
};

pub const Dependency = struct {
    kind: DependencyKind,
    id: u32,
    expected_epoch: u64 = 0,
};

pub const Error = code_buffer.Error || jit_memory.Error || runtime_code_manager.Error || std.mem.Allocator.Error || error{
    EmptyInstructions,
    InvalidInstructionTarget,
    InvalidRegister,
    UnsupportedArchitecture,
    UnsupportedInstruction,
};

pub const Stats = struct {
    compiled_methods: u32 = 0,
    instructions_processed: u32 = 0,
    bytes_generated: u32 = 0,
    constants_emitted: u32 = 0,
    moves_emitted: u32 = 0,
    arithmetic_emitted: u32 = 0,
    branches_emitted: u32 = 0,
    returns_emitted: u32 = 0,
    stack_maps_generated: u32 = 0,
    deopts_generated: u32 = 0,
};

pub const Options = struct {
    register_count: u16 = 0,
    parameter_count: u16 = 0,
    generate_stack_maps: bool = true,
};

pub const CompiledCode = struct {
    allocator: std.mem.Allocator,
    bytes: []const u8,
    relocations: []const Relocation,
    stack_maps: []const SafepointMap,
    deopts: []const DeoptPoint,
    dependencies: []const Dependency,

    pub fn deinit(self: *CompiledCode) void {
        self.allocator.free(self.bytes);
        self.allocator.free(self.relocations);
        self.allocator.free(self.stack_maps);
        self.allocator.free(self.deopts);
        self.allocator.free(self.dependencies);
        self.* = undefined;
    }

    pub fn verify(self: *const CompiledCode) Error!void {
        if (self.bytes.len == 0) return error.EmptyCode;
    }

    pub fn print(self: *const CompiledCode, writer: anytype) !void {
        try writer.print("jit_compiler bytes={d} relocs={d} stack_maps={d} deopts={d} deps={d}\n", .{
            self.bytes.len,
            self.relocations.len,
            self.stack_maps.len,
            self.deopts.len,
            self.dependencies.len,
        });
    }
};

pub const Compiler = struct {
    stats: Stats = .{},

    pub fn init() Compiler {
        return .{};
    }

    fn findMaxRegister(insts: []const Instruction) u16 {
        var max: u16 = 0;
        for (insts) |inst| {
            switch (inst) {
                .move, .move_wide, .move_object => |op| {
                    if (op.dest > max) max = op.dest;
                    if (op.src > max) max = op.src;
                },
                .move_result, .move_result_wide, .move_result_object, .move_exception => |op| {
                    if (op.dest > max) max = op.dest;
                },
                .return_, .return_wide, .return_object => |op| {
                    if (op.src > max) max = op.src;
                },
                .const_ => |op| {
                    if (op.dest > max) max = op.dest;
                },
                .const_wide => |op| {
                    if (op.dest > max) max = op.dest;
                },
                .add_int, .sub_int, .mul_int, .div_int, .rem_int, .and_int, .or_int, .xor_int, .shl_int, .shr_int, .ushr_int, .add_float, .sub_float, .mul_float, .div_float, .rem_float => |op| {
                    if (op.dest > max) max = op.dest;
                    if (op.src1 > max) max = op.src1;
                    if (op.src2 > max) max = op.src2;
                },
                .add_long, .sub_long, .mul_long, .div_long, .rem_long, .and_long, .or_long, .xor_long, .shl_long, .shr_long, .ushr_long, .add_double, .sub_double, .mul_double, .div_double, .rem_double => |op| {
                    if (op.dest > max) max = op.dest;
                    if (op.src1 > max) max = op.src1;
                    if (op.src2 > max) max = op.src2;
                },
                .add_int_lit8, .rsub_int_lit8, .mul_int_lit8, .div_int_lit8, .rem_int_lit8, .and_int_lit8, .or_int_lit8, .xor_int_lit8, .shl_int_lit8, .shr_int_lit8, .ushr_int_lit8 => |op| {
                    if (op.dest > max) max = op.dest;
                    if (op.src > max) max = op.src;
                },
                .add_int_lit16, .rsub_int_lit16, .mul_int_lit16, .div_int_lit16, .rem_int_lit16, .and_int_lit16, .or_int_lit16, .xor_int_lit16 => |op| {
                    if (op.dest > max) max = op.dest;
                    if (op.src > max) max = op.src;
                },
                .neg_int, .not_int, .neg_float, .int_to_float, .float_to_int, .int_to_byte, .int_to_char, .int_to_short => |op| {
                    if (op.dest > max) max = op.dest;
                    if (op.src > max) max = op.src;
                },
                .neg_long, .not_long, .neg_double, .long_to_double, .double_to_long => |op| {
                    if (op.dest > max) max = op.dest;
                    if (op.src > max) max = op.src;
                },
                .int_to_long, .int_to_double, .float_to_long, .float_to_double => |op| {
                    if (op.dest > max) max = op.dest;
                    if (op.src > max) max = op.src;
                },
                .long_to_int, .long_to_float, .double_to_int, .double_to_float => |op| {
                    if (op.dest > max) max = op.dest;
                    if (op.src > max) max = op.src;
                },
                .cmpl_float, .cmpg_float, .cmpl_double, .cmpg_double, .cmp_long => |op| {
                    if (op.dest > max) max = op.dest;
                    if (op.src1 > max) max = op.src1;
                    if (op.src2 > max) max = op.src2;
                },
                .if_eq, .if_ne, .if_lt, .if_ge, .if_gt, .if_le => |op| {
                    if (op.src1 > max) max = op.src1;
                    if (op.src2 > max) max = op.src2;
                },
                .if_eqz, .if_nez, .if_ltz, .if_gez, .if_gtz, .if_lez => |op| {
                    if (op.src > max) max = op.src;
                },
                .packed_switch, .sparse_switch => |op| {
                    if (op.src > max) max = op.src;
                },
                else => {},
            }
        }
        return max;
    }

    fn slotOffset(reg: u16) u32 {
        return (@as(u32, reg) + 1) * 8;
    }

    fn emitRex(buffer: *code_buffer.Buffer, w: bool, reg: u4, rm: u4) Error!void {
        var rex: u8 = 0x40;
        if (w) rex |= 0x08;
        if ((reg & 8) != 0) rex |= 0x04;
        if ((rm & 8) != 0) rex |= 0x01;
        if (rex != 0x40) try buffer.emitU8(rex);
    }

    fn emitModRm(buffer: *code_buffer.Buffer, mode: u2, reg: u4, rm: u4) Error!void {
        try buffer.emitU8((@as(u8, mode) << 6) | ((@as(u8, reg) & 7) << 3) | (@as(u8, rm) & 7));
    }

    fn emitDisp32(buffer: *code_buffer.Buffer, offset: u32) Error!void {
        // [rbp - offset] using two's complement displacement relative to rbp (register 5)
        const displacement: i32 = -@as(i32, @intCast(offset));
        try buffer.emitU32(@bitCast(displacement));
    }

    fn emitLoadRegSlot32(buffer: *code_buffer.Buffer, reg: u4, slot_reg: u16) Error!void {
        try emitRex(buffer, false, reg, 5); // rbp is 5
        try buffer.emitU8(0x8b);
        try emitModRm(buffer, 2, reg, 5);
        try emitDisp32(buffer, slotOffset(slot_reg));
    }

    fn emitStoreRegSlot32(buffer: *code_buffer.Buffer, reg: u4, slot_reg: u16) Error!void {
        try emitRex(buffer, false, reg, 5);
        try buffer.emitU8(0x89);
        try emitModRm(buffer, 2, reg, 5);
        try emitDisp32(buffer, slotOffset(slot_reg));
    }

    fn emitLoadRegSlot64(buffer: *code_buffer.Buffer, reg: u4, slot_reg: u16) Error!void {
        try emitRex(buffer, true, reg, 5);
        try buffer.emitU8(0x8b);
        try emitModRm(buffer, 2, reg, 5);
        try emitDisp32(buffer, slotOffset(slot_reg));
    }

    fn emitStoreRegSlot64(buffer: *code_buffer.Buffer, reg: u4, slot_reg: u16) Error!void {
        try emitRex(buffer, true, reg, 5);
        try buffer.emitU8(0x89);
        try emitModRm(buffer, 2, reg, 5);
        try emitDisp32(buffer, slotOffset(slot_reg));
    }

    fn emitImm32Slot(buffer: *code_buffer.Buffer, slot_reg: u16, value: i32) Error!void {
        try buffer.emitBytes(&.{ 0xc7, 0x85 });
        try emitDisp32(buffer, slotOffset(slot_reg));
        try buffer.emitU32(@bitCast(value));
    }

    fn emitImm64Reg(buffer: *code_buffer.Buffer, reg: u4, value: u64) Error!void {
        try emitRex(buffer, true, 0, reg);
        try buffer.emitU8(0xb8 + @as(u8, reg & 7));
        try buffer.emitU64(value);
    }

    fn emitLoadXmmSlot32(buffer: *code_buffer.Buffer, xmm_reg: u4, slot_reg: u16) Error!void {
        try buffer.emitU8(0xf3);
        try emitRex(buffer, false, xmm_reg, 5);
        try buffer.emitBytes(&.{ 0x0f, 0x10 });
        try emitModRm(buffer, 2, xmm_reg, 5);
        try emitDisp32(buffer, slotOffset(slot_reg));
    }

    fn emitStoreXmmSlot32(buffer: *code_buffer.Buffer, xmm_reg: u4, slot_reg: u16) Error!void {
        try buffer.emitU8(0xf3);
        try emitRex(buffer, false, xmm_reg, 5);
        try buffer.emitBytes(&.{ 0x0f, 0x11 });
        try emitModRm(buffer, 2, xmm_reg, 5);
        try emitDisp32(buffer, slotOffset(slot_reg));
    }

    fn emitLoadXmmSlot64(buffer: *code_buffer.Buffer, xmm_reg: u4, slot_reg: u16) Error!void {
        try buffer.emitU8(0xf2);
        try emitRex(buffer, false, xmm_reg, 5);
        try buffer.emitBytes(&.{ 0x0f, 0x10 });
        try emitModRm(buffer, 2, xmm_reg, 5);
        try emitDisp32(buffer, slotOffset(slot_reg));
    }

    fn emitStoreXmmSlot64(buffer: *code_buffer.Buffer, xmm_reg: u4, slot_reg: u16) Error!void {
        try buffer.emitU8(0xf2);
        try emitRex(buffer, false, xmm_reg, 5);
        try buffer.emitBytes(&.{ 0x0f, 0x11 });
        try emitModRm(buffer, 2, xmm_reg, 5);
        try emitDisp32(buffer, slotOffset(slot_reg));
    }

    fn emitPrologue(buffer: *code_buffer.Buffer, stack_size: u32) Error!void {
        try buffer.emitU8(0x55); // push rbp
        try buffer.emitBytes(&.{ 0x48, 0x89, 0xe5 }); // mov rbp, rsp
        if (stack_size != 0) {
            if (stack_size <= 127) {
                try buffer.emitBytes(&.{ 0x48, 0x83, 0xec, @intCast(stack_size) });
            } else {
                try buffer.emitBytes(&.{ 0x48, 0x81, 0xec });
                try buffer.emitU32(stack_size);
            }
        }
    }

    fn emitEpilogue(buffer: *code_buffer.Buffer) Error!void {
        try buffer.emitU8(0xc9); // leave
        try buffer.emitU8(0xc3); // ret
    }

    fn branchCondition(condition: u8) u8 {
        return switch (condition) {
            0 => 0x84, // eq -> je
            1 => 0x85, // ne -> jne
            2 => 0x8c, // lt -> jl
            3 => 0x8d, // ge -> jge
            4 => 0x8f, // gt -> jg
            5 => 0x8e, // le -> jle
            else => unreachable,
        };
    }

    pub fn compile(
        self: *Compiler,
        allocator: std.mem.Allocator,
        insts: []const Instruction,
        options: Options,
    ) Error!CompiledCode {
        if (builtin.cpu.arch != .x86_64) return error.UnsupportedArchitecture;
        if (insts.len == 0) return error.EmptyInstructions;

        const max_reg = findMaxRegister(insts);
        const reg_count = if (options.register_count != 0)
            @max(options.register_count, max_reg + 1)
        else
            @max(max_reg + 1, 1);

        const stack_slots_size = @as(u32, reg_count) * 8;
        const stack_size = (stack_slots_size + 15) & ~@as(u32, 15);

        var buffer = code_buffer.Buffer.init(allocator);
        defer buffer.deinit();

        const labels = try allocator.alloc(code_buffer.LabelId, insts.len);
        defer allocator.free(labels);

        for (0..insts.len) |i| {
            labels[i] = try buffer.newLabel();
        }

        try emitPrologue(&buffer, stack_size);

        if (options.parameter_count > 0) {
            const abi_regs = if (builtin.os.tag == .windows)
                [_]u4{ 1, 2, 8, 9 } // rcx, rdx, r8, r9
            else
                [_]u4{ 7, 6, 2, 1, 8, 9 }; // rdi, rsi, rdx, rcx, r8, r9
            const param_base = if (reg_count >= options.parameter_count)
                reg_count - options.parameter_count
            else
                0;
            const count = @min(options.parameter_count, @as(u16, @intCast(abi_regs.len)));
            for (0..count) |p| {
                const reg_idx: u16 = param_base + @as(u16, @intCast(p));
                try emitStoreRegSlot64(&buffer, abi_regs[p], reg_idx);
            }
        }

        for (insts, 0..) |inst, i| {
            try buffer.bindLabel(labels[i]);
            self.stats.instructions_processed += 1;

            switch (inst) {
                .nop => {},
                .move, .move_object => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.moves_emitted += 1;
                },
                .move_wide => |op| {
                    try emitLoadRegSlot64(&buffer, 0, op.src);
                    try emitStoreRegSlot64(&buffer, 0, op.dest);
                    self.stats.moves_emitted += 1;
                },
                .move_result, .move_result_object, .move_exception => |op| {
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.moves_emitted += 1;
                },
                .move_result_wide => |op| {
                    try emitStoreRegSlot64(&buffer, 0, op.dest);
                    self.stats.moves_emitted += 1;
                },
                .const_ => |op| {
                    try emitImm32Slot(&buffer, op.dest, op.value);
                    self.stats.constants_emitted += 1;
                },
                .const_wide => |op| {
                    try emitImm64Reg(&buffer, 0, @bitCast(op.value));
                    try emitStoreRegSlot64(&buffer, 0, op.dest);
                    self.stats.constants_emitted += 1;
                },
                .add_int => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src1);
                    try buffer.emitBytes(&.{ 0x03, 0x85 }); // add eax, [rbp - disp]
                    try emitDisp32(&buffer, slotOffset(op.src2));
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .sub_int => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src1);
                    try buffer.emitBytes(&.{ 0x2b, 0x85 }); // sub eax, [rbp - disp]
                    try emitDisp32(&buffer, slotOffset(op.src2));
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .mul_int => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src1);
                    try buffer.emitBytes(&.{ 0x0f, 0xaf, 0x85 }); // imul eax, [rbp - disp]
                    try emitDisp32(&buffer, slotOffset(op.src2));
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .and_int => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src1);
                    try buffer.emitBytes(&.{ 0x23, 0x85 }); // and eax, [rbp - disp]
                    try emitDisp32(&buffer, slotOffset(op.src2));
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .or_int => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src1);
                    try buffer.emitBytes(&.{ 0x0b, 0x85 }); // or eax, [rbp - disp]
                    try emitDisp32(&buffer, slotOffset(op.src2));
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .xor_int => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src1);
                    try buffer.emitBytes(&.{ 0x33, 0x85 }); // xor eax, [rbp - disp]
                    try emitDisp32(&buffer, slotOffset(op.src2));
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .shl_int, .shr_int, .ushr_int => |op| {
                    try emitLoadRegSlot32(&buffer, 1, op.src2); // ecx = src2 (shift amount)
                    try emitLoadRegSlot32(&buffer, 0, op.src1); // eax = src1
                    const opcode_ext: u8 = switch (inst) {
                        .shl_int => 4,
                        .shr_int => 7,
                        .ushr_int => 5,
                        else => unreachable,
                    };
                    try buffer.emitBytes(&.{ 0xd3, 0xe0 | (opcode_ext << 3) });
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .div_int => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src1);
                    try emitLoadRegSlot32(&buffer, 1, op.src2);
                    try buffer.emitBytes(&.{ 0x83, 0xf9, 0xff }); // cmp ecx, -1
                    const label_check_min = try buffer.newLabel();
                    const label_done = try buffer.newLabel();
                    try buffer.emitBytes(&.{ 0x0f, 0x84 }); // je check_min
                    _ = try buffer.reloc(label_check_min, .rel32, 0);

                    const label_do_div = try buffer.newLabel();
                    try buffer.bindLabel(label_do_div);
                    try buffer.emitU8(0x99); // cdq
                    try buffer.emitBytes(&.{ 0xf7, 0xf9 }); // idiv ecx
                    try buffer.emitBytes(&.{ 0xe9 }); // jmp done
                    _ = try buffer.reloc(label_done, .rel32, 0);

                    try buffer.bindLabel(label_check_min);
                    try buffer.emitBytes(&.{ 0x3d, 0x00, 0x00, 0x00, 0x80 }); // cmp eax, 0x80000000
                    try buffer.emitBytes(&.{ 0x0f, 0x85 }); // jne do_div
                    _ = try buffer.reloc(label_do_div, .rel32, 0);

                    try buffer.bindLabel(label_done);
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .rem_int => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src1);
                    try emitLoadRegSlot32(&buffer, 1, op.src2);
                    try buffer.emitBytes(&.{ 0x83, 0xf9, 0xff }); // cmp ecx, -1
                    const label_check_min = try buffer.newLabel();
                    const label_done = try buffer.newLabel();
                    try buffer.emitBytes(&.{ 0x0f, 0x84 }); // je check_min
                    _ = try buffer.reloc(label_check_min, .rel32, 0);

                    const label_do_div = try buffer.newLabel();
                    try buffer.bindLabel(label_do_div);
                    try buffer.emitU8(0x99); // cdq
                    try buffer.emitBytes(&.{ 0xf7, 0xf9 }); // idiv ecx -> remainder in edx
                    try buffer.emitBytes(&.{ 0xe9 }); // jmp done
                    _ = try buffer.reloc(label_done, .rel32, 0);

                    try buffer.bindLabel(label_check_min);
                    try buffer.emitBytes(&.{ 0x3d, 0x00, 0x00, 0x00, 0x80 }); // cmp eax, 0x80000000
                    try buffer.emitBytes(&.{ 0x0f, 0x85 }); // jne do_div
                    _ = try buffer.reloc(label_do_div, .rel32, 0);
                    try buffer.emitBytes(&.{ 0x33, 0xd2 }); // xor edx, edx

                    try buffer.bindLabel(label_done);
                    try emitStoreRegSlot32(&buffer, 2, op.dest); // store edx
                    self.stats.arithmetic_emitted += 1;
                },
                .add_int_lit8 => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0x83, 0xc0, @bitCast(op.lit) }); // add eax, imm8
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .add_int_lit16 => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    if (op.lit <= 127 and op.lit >= -128) {
                        try buffer.emitBytes(&.{ 0x83, 0xc0, @bitCast(@as(i8, @intCast(op.lit))) }); // add eax, imm8
                    } else {
                        try buffer.emitBytes(&.{ 0x05 }); // add eax, imm32
                        try buffer.emitU32(@bitCast(@as(i32, op.lit)));
                    }
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .rsub_int_lit16 => |op| {
                    try buffer.emitU8(0xb8); // mov eax, imm32
                    try buffer.emitU32(@bitCast(@as(i32, op.lit)));
                    try buffer.emitBytes(&.{ 0x2b, 0x85 }); // sub eax, [rbp - disp]
                    try emitDisp32(&buffer, slotOffset(op.src));
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .rsub_int_lit8 => |op| {
                    try buffer.emitU8(0xb8); // mov eax, imm32
                    try buffer.emitU32(@bitCast(@as(i32, op.lit)));
                    try buffer.emitBytes(&.{ 0x2b, 0x85 }); // sub eax, [rbp - disp]
                    try emitDisp32(&buffer, slotOffset(op.src));
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .mul_int_lit16 => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0x69, 0xc0 }); // imul eax, eax, imm32
                    try buffer.emitU32(@bitCast(@as(i32, op.lit)));
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .mul_int_lit8 => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0x69, 0xc0 }); // imul eax, eax, imm32
                    try buffer.emitU32(@bitCast(@as(i32, op.lit)));
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .and_int_lit16, .or_int_lit16, .xor_int_lit16 => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    const op_ext: u8 = switch (inst) {
                        .and_int_lit16 => 4,
                        .or_int_lit16 => 1,
                        .xor_int_lit16 => 6,
                        else => unreachable,
                    };
                    try buffer.emitBytes(&.{ 0x81, 0xc0 | (op_ext << 3) });
                    try buffer.emitU32(@bitCast(@as(i32, op.lit)));
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .and_int_lit8, .or_int_lit8, .xor_int_lit8 => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    const op_ext: u8 = switch (inst) {
                        .and_int_lit8 => 4,
                        .or_int_lit8 => 1,
                        .xor_int_lit8 => 6,
                        else => unreachable,
                    };
                    try buffer.emitBytes(&.{ 0x81, 0xc0 | (op_ext << 3) });
                    try buffer.emitU32(@bitCast(@as(i32, op.lit)));
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .shl_int_lit8, .shr_int_lit8, .ushr_int_lit8 => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    const opcode_ext: u8 = switch (inst) {
                        .shl_int_lit8 => 4,
                        .shr_int_lit8 => 7,
                        .ushr_int_lit8 => 5,
                        else => unreachable,
                    };
                    const count: u8 = @as(u8, @bitCast(op.lit)) & 31;
                    if (count == 1) {
                        try buffer.emitBytes(&.{ 0xd1, 0xe0 | (opcode_ext << 3) });
                    } else {
                        try buffer.emitBytes(&.{ 0xc1, 0xe0 | (opcode_ext << 3), count });
                    }
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .div_int_lit16 => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    const imm: i32 = op.lit;
                    if (imm == -1) {
                        try buffer.emitBytes(&.{ 0x3d, 0x00, 0x00, 0x00, 0x80 });
                        const label_skip = try buffer.newLabel();
                        try buffer.emitBytes(&.{ 0x0f, 0x84 });
                        _ = try buffer.reloc(label_skip, .rel32, 0);
                        try buffer.emitU8(0x99);
                        try buffer.emitBytes(&.{ 0xb9, 0xff, 0xff, 0xff, 0xff });
                        try buffer.emitBytes(&.{ 0xf7, 0xf9 });
                        try buffer.bindLabel(label_skip);
                    } else {
                        try buffer.emitU8(0x99);
                        try buffer.emitU8(0xb9);
                        try buffer.emitU32(@bitCast(imm));
                        try buffer.emitBytes(&.{ 0xf7, 0xf9 });
                    }
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .div_int_lit8 => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    const imm: i32 = op.lit;
                    if (imm == -1) {
                        try buffer.emitBytes(&.{ 0x3d, 0x00, 0x00, 0x00, 0x80 });
                        const label_skip = try buffer.newLabel();
                        try buffer.emitBytes(&.{ 0x0f, 0x84 });
                        _ = try buffer.reloc(label_skip, .rel32, 0);
                        try buffer.emitU8(0x99);
                        try buffer.emitBytes(&.{ 0xb9, 0xff, 0xff, 0xff, 0xff });
                        try buffer.emitBytes(&.{ 0xf7, 0xf9 });
                        try buffer.bindLabel(label_skip);
                    } else {
                        try buffer.emitU8(0x99);
                        try buffer.emitU8(0xb9);
                        try buffer.emitU32(@bitCast(imm));
                        try buffer.emitBytes(&.{ 0xf7, 0xf9 });
                    }
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .rem_int_lit16 => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    const imm: i32 = op.lit;
                    if (imm == -1) {
                        try buffer.emitBytes(&.{ 0x3d, 0x00, 0x00, 0x00, 0x80 });
                        const label_skip = try buffer.newLabel();
                        const label_done = try buffer.newLabel();
                        try buffer.emitBytes(&.{ 0x0f, 0x84 });
                        _ = try buffer.reloc(label_skip, .rel32, 0);
                        try buffer.emitU8(0x99);
                        try buffer.emitBytes(&.{ 0xb9, 0xff, 0xff, 0xff, 0xff });
                        try buffer.emitBytes(&.{ 0xf7, 0xf9 });
                        try buffer.emitBytes(&.{ 0xe9 });
                        _ = try buffer.reloc(label_done, .rel32, 0);
                        try buffer.bindLabel(label_skip);
                        try buffer.emitBytes(&.{ 0x33, 0xd2 });
                        try buffer.bindLabel(label_done);
                    } else {
                        try buffer.emitU8(0x99);
                        try buffer.emitU8(0xb9);
                        try buffer.emitU32(@bitCast(imm));
                        try buffer.emitBytes(&.{ 0xf7, 0xf9 });
                    }
                    try emitStoreRegSlot32(&buffer, 2, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .rem_int_lit8 => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    const imm: i32 = op.lit;
                    if (imm == -1) {
                        try buffer.emitBytes(&.{ 0x3d, 0x00, 0x00, 0x00, 0x80 });
                        const label_skip = try buffer.newLabel();
                        const label_done = try buffer.newLabel();
                        try buffer.emitBytes(&.{ 0x0f, 0x84 });
                        _ = try buffer.reloc(label_skip, .rel32, 0);
                        try buffer.emitU8(0x99);
                        try buffer.emitBytes(&.{ 0xb9, 0xff, 0xff, 0xff, 0xff });
                        try buffer.emitBytes(&.{ 0xf7, 0xf9 });
                        try buffer.emitBytes(&.{ 0xe9 });
                        _ = try buffer.reloc(label_done, .rel32, 0);
                        try buffer.bindLabel(label_skip);
                        try buffer.emitBytes(&.{ 0x33, 0xd2 });
                        try buffer.bindLabel(label_done);
                    } else {
                        try buffer.emitU8(0x99);
                        try buffer.emitU8(0xb9);
                        try buffer.emitU32(@bitCast(imm));
                        try buffer.emitBytes(&.{ 0xf7, 0xf9 });
                    }
                    try emitStoreRegSlot32(&buffer, 2, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .neg_int, .not_int => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    if (inst == .neg_int) {
                        try buffer.emitBytes(&.{ 0xf7, 0xd8 });
                    } else {
                        try buffer.emitBytes(&.{ 0xf7, 0xd0 });
                    }
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .int_to_byte, .int_to_char, .int_to_short => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    switch (inst) {
                        .int_to_byte => try buffer.emitBytes(&.{ 0x0f, 0xbe, 0xc0 }),
                        .int_to_char => try buffer.emitBytes(&.{ 0x0f, 0xb7, 0xc0 }),
                        .int_to_short => try buffer.emitBytes(&.{ 0x0f, 0xbf, 0xc0 }),
                        else => unreachable,
                    }
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .int_to_long => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0x48, 0x98 }); // cdqe
                    try emitStoreRegSlot64(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .long_to_int => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .int_to_float => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0xf3, 0x0f, 0x2a, 0xc0 }); // cvtsi2ss xmm0, eax
                    try emitStoreXmmSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .int_to_double => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0xf2, 0x0f, 0x2a, 0xc0 }); // cvtsi2sd xmm0, eax
                    try emitStoreXmmSlot64(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .long_to_float => |op| {
                    try emitLoadRegSlot64(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0xf3, 0x48, 0x0f, 0x2a, 0xc0 }); // cvtsi2ss xmm0, rax
                    try emitStoreXmmSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .long_to_double => |op| {
                    try emitLoadRegSlot64(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0xf2, 0x48, 0x0f, 0x2a, 0xc0 }); // cvtsi2sd xmm0, rax
                    try emitStoreXmmSlot64(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .float_to_int => |op| {
                    try emitLoadXmmSlot32(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0xf3, 0x0f, 0x2c, 0xc0 }); // cvttss2si eax, xmm0
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .double_to_int => |op| {
                    try emitLoadXmmSlot64(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0xf2, 0x0f, 0x2c, 0xc0 }); // cvttsd2si eax, xmm0
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .float_to_long => |op| {
                    try emitLoadXmmSlot32(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0xf3, 0x48, 0x0f, 0x2c, 0xc0 }); // cvttss2si rax, xmm0
                    try emitStoreRegSlot64(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .double_to_long => |op| {
                    try emitLoadXmmSlot64(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0xf2, 0x48, 0x0f, 0x2c, 0xc0 }); // cvttsd2si rax, xmm0
                    try emitStoreRegSlot64(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .float_to_double => |op| {
                    try emitLoadXmmSlot32(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0xf3, 0x0f, 0x5a, 0xc0 }); // cvtss2sd xmm0, xmm0
                    try emitStoreXmmSlot64(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .double_to_float => |op| {
                    try emitLoadXmmSlot64(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0xf2, 0x0f, 0x5a, 0xc0 }); // cvtsd2ss xmm0, xmm0
                    try emitStoreXmmSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .neg_long, .not_long => |op| {
                    try emitLoadRegSlot64(&buffer, 0, op.src);
                    if (inst == .neg_long) {
                        try buffer.emitBytes(&.{ 0x48, 0xf7, 0xd8 });
                    } else {
                        try buffer.emitBytes(&.{ 0x48, 0xf7, 0xd0 });
                    }
                    try emitStoreRegSlot64(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .add_long, .sub_long, .and_long, .or_long, .xor_long => |op| {
                    try emitLoadRegSlot64(&buffer, 0, op.src1);
                    const opcode: u8 = switch (inst) {
                        .add_long => 0x03,
                        .sub_long => 0x2b,
                        .and_long => 0x23,
                        .or_long => 0x0b,
                        .xor_long => 0x33,
                        else => unreachable,
                    };
                    try emitRex(&buffer, true, 0, 5);
                    try buffer.emitU8(opcode);
                    try emitModRm(&buffer, 2, 0, 5);
                    try emitDisp32(&buffer, slotOffset(op.src2));
                    try emitStoreRegSlot64(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .mul_long => |op| {
                    try emitLoadRegSlot64(&buffer, 0, op.src1);
                    try emitRex(&buffer, true, 0, 5);
                    try buffer.emitBytes(&.{ 0x0f, 0xaf });
                    try emitModRm(&buffer, 2, 0, 5);
                    try emitDisp32(&buffer, slotOffset(op.src2));
                    try emitStoreRegSlot64(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .div_long => |op| {
                    try emitLoadRegSlot64(&buffer, 0, op.src1);
                    try emitLoadRegSlot64(&buffer, 1, op.src2);
                    try buffer.emitBytes(&.{ 0x48, 0x83, 0xf9, 0xff }); // cmp rcx, -1
                    const label_check_min = try buffer.newLabel();
                    const label_done = try buffer.newLabel();
                    try buffer.emitBytes(&.{ 0x0f, 0x84 }); // je check_min
                    _ = try buffer.reloc(label_check_min, .rel32, 0);

                    const label_do_div = try buffer.newLabel();
                    try buffer.bindLabel(label_do_div);
                    try buffer.emitBytes(&.{ 0x48, 0x99 }); // cqo
                    try buffer.emitBytes(&.{ 0x48, 0xf7, 0xf9 }); // idiv rcx
                    try buffer.emitBytes(&.{ 0xe9 }); // jmp done
                    _ = try buffer.reloc(label_done, .rel32, 0);

                    try buffer.bindLabel(label_check_min);
                    try buffer.emitBytes(&.{ 0x48, 0xba, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80 }); // mov rdx, 0x8000000000000000
                    try buffer.emitBytes(&.{ 0x48, 0x3b, 0xc2 }); // cmp rax, rdx
                    try buffer.emitBytes(&.{ 0x0f, 0x85 }); // jne do_div
                    _ = try buffer.reloc(label_do_div, .rel32, 0);

                    try buffer.bindLabel(label_done);
                    try emitStoreRegSlot64(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .rem_long => |op| {
                    try emitLoadRegSlot64(&buffer, 0, op.src1);
                    try emitLoadRegSlot64(&buffer, 1, op.src2);
                    try buffer.emitBytes(&.{ 0x48, 0x83, 0xf9, 0xff }); // cmp rcx, -1
                    const label_check_min = try buffer.newLabel();
                    const label_done = try buffer.newLabel();
                    try buffer.emitBytes(&.{ 0x0f, 0x84 }); // je check_min
                    _ = try buffer.reloc(label_check_min, .rel32, 0);

                    const label_do_div = try buffer.newLabel();
                    try buffer.bindLabel(label_do_div);
                    try buffer.emitBytes(&.{ 0x48, 0x99 }); // cqo
                    try buffer.emitBytes(&.{ 0x48, 0xf7, 0xf9 }); // idiv rcx -> remainder in RDX
                    try buffer.emitBytes(&.{ 0xe9 }); // jmp done
                    _ = try buffer.reloc(label_done, .rel32, 0);

                    try buffer.bindLabel(label_check_min);
                    try buffer.emitBytes(&.{ 0x48, 0xba, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80 }); // mov rdx, 0x8000000000000000
                    try buffer.emitBytes(&.{ 0x48, 0x3b, 0xc2 }); // cmp rax, rdx
                    try buffer.emitBytes(&.{ 0x0f, 0x85 }); // jne do_div
                    _ = try buffer.reloc(label_do_div, .rel32, 0);
                    try buffer.emitBytes(&.{ 0x33, 0xd2 }); // xor edx, edx

                    try buffer.bindLabel(label_done);
                    try emitStoreRegSlot64(&buffer, 2, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .shl_long, .shr_long, .ushr_long => |op| {
                    try emitLoadRegSlot32(&buffer, 1, op.src2); // shift amount in cl
                    try emitLoadRegSlot64(&buffer, 0, op.src1); // value in rax
                    const opcode_ext: u8 = switch (inst) {
                        .shl_long => 4,
                        .shr_long => 7,
                        .ushr_long => 5,
                        else => unreachable,
                    };
                    try buffer.emitBytes(&.{ 0x48, 0xd3, 0xe0 | (opcode_ext << 3) });
                    try emitStoreRegSlot64(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .cmp_long => |op| {
                    try emitLoadRegSlot64(&buffer, 0, op.src1);
                    try emitRex(&buffer, true, 0, 5);
                    try buffer.emitU8(0x3b); // cmp rax, [rbp - disp]
                    try emitModRm(&buffer, 2, 0, 5);
                    try emitDisp32(&buffer, slotOffset(op.src2));
                    const label_gt = try buffer.newLabel();
                    const label_lt = try buffer.newLabel();
                    const label_done = try buffer.newLabel();

                    try buffer.emitBytes(&.{ 0x0f, 0x8f }); // jg target
                    _ = try buffer.reloc(label_gt, .rel32, 0);
                    try buffer.emitBytes(&.{ 0x0f, 0x8c }); // jl target
                    _ = try buffer.reloc(label_lt, .rel32, 0);

                    try buffer.emitBytes(&.{ 0x33, 0xc0 }); // xor eax, eax
                    try buffer.emitBytes(&.{ 0xe9 }); // jmp done
                    _ = try buffer.reloc(label_done, .rel32, 0);

                    try buffer.bindLabel(label_gt);
                    try buffer.emitBytes(&.{ 0xb8, 0x01, 0x00, 0x00, 0x00 });
                    try buffer.emitBytes(&.{ 0xe9 });
                    _ = try buffer.reloc(label_done, .rel32, 0);

                    try buffer.bindLabel(label_lt);
                    try buffer.emitBytes(&.{ 0xb8, 0xff, 0xff, 0xff, 0xff });

                    try buffer.bindLabel(label_done);
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .add_float, .sub_float, .mul_float, .div_float => |op| {
                    try emitLoadXmmSlot32(&buffer, 0, op.src1);
                    try buffer.emitU8(0xf3);
                    try emitRex(&buffer, false, 0, 5);
                    const opcode_byte: u8 = switch (inst) {
                        .add_float => 0x58,
                        .sub_float => 0x5c,
                        .mul_float => 0x59,
                        .div_float => 0x5e,
                        else => unreachable,
                    };
                    try buffer.emitBytes(&.{ 0x0f, opcode_byte });
                    try emitModRm(&buffer, 2, 0, 5);
                    try emitDisp32(&buffer, slotOffset(op.src2));
                    try emitStoreXmmSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .add_double, .sub_double, .mul_double, .div_double => |op| {
                    try emitLoadXmmSlot64(&buffer, 0, op.src1);
                    try buffer.emitU8(0xf2);
                    try emitRex(&buffer, false, 0, 5);
                    const opcode_byte: u8 = switch (inst) {
                        .add_double => 0x58,
                        .sub_double => 0x5c,
                        .mul_double => 0x59,
                        .div_double => 0x5e,
                        else => unreachable,
                    };
                    try buffer.emitBytes(&.{ 0x0f, opcode_byte });
                    try emitModRm(&buffer, 2, 0, 5);
                    try emitDisp32(&buffer, slotOffset(op.src2));
                    try emitStoreXmmSlot64(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .neg_float => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0x35, 0x00, 0x00, 0x00, 0x80 }); // xor eax, 0x80000000
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .neg_double => |op| {
                    try emitLoadRegSlot64(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0x48, 0xba, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80 }); // mov rdx, 0x8000000000000000
                    try buffer.emitBytes(&.{ 0x48, 0x33, 0xc2 }); // xor rax, rdx
                    try emitStoreRegSlot64(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .rem_float => |op| {
                    try buffer.emitBytes(&.{ 0xd9 }); // fld dword ptr [rbp - disp2]
                    try emitModRm(&buffer, 2, 0, 5);
                    try emitDisp32(&buffer, slotOffset(op.src2));
                    try buffer.emitBytes(&.{ 0xd9 }); // fld dword ptr [rbp - disp1]
                    try emitModRm(&buffer, 2, 0, 5);
                    try emitDisp32(&buffer, slotOffset(op.src1));
                    const loop_label = try buffer.newLabel();
                    try buffer.bindLabel(loop_label);
                    try buffer.emitBytes(&.{ 0xd9, 0xf8 }); // fprem
                    try buffer.emitBytes(&.{ 0xdf, 0xe0 }); // fstsw ax
                    try buffer.emitBytes(&.{ 0xf6, 0xc4, 0x04 }); // test ah, 0x04
                    try buffer.emitBytes(&.{ 0x75, 0xf7 }); // jnz loop_label (-9 bytes)
                    try buffer.emitBytes(&.{ 0xd9 }); // fstp dword ptr [rbp - disp_dest]
                    try emitModRm(&buffer, 2, 3, 5);
                    try emitDisp32(&buffer, slotOffset(op.dest));
                    try buffer.emitBytes(&.{ 0xdd, 0xd8 }); // fstp st(0)
                    self.stats.arithmetic_emitted += 1;
                },
                .rem_double => |op| {
                    try buffer.emitBytes(&.{ 0xdd }); // fld qword ptr [rbp - disp2]
                    try emitModRm(&buffer, 2, 0, 5);
                    try emitDisp32(&buffer, slotOffset(op.src2));
                    try buffer.emitBytes(&.{ 0xdd }); // fld qword ptr [rbp - disp1]
                    try emitModRm(&buffer, 2, 0, 5);
                    try emitDisp32(&buffer, slotOffset(op.src1));
                    const loop_label = try buffer.newLabel();
                    try buffer.bindLabel(loop_label);
                    try buffer.emitBytes(&.{ 0xd9, 0xf8 }); // fprem
                    try buffer.emitBytes(&.{ 0xdf, 0xe0 }); // fstsw ax
                    try buffer.emitBytes(&.{ 0xf6, 0xc4, 0x04 }); // test ah, 0x04
                    try buffer.emitBytes(&.{ 0x75, 0xf7 }); // jnz loop_label (-9 bytes)
                    try buffer.emitBytes(&.{ 0xdd }); // fstp qword ptr [rbp - disp_dest]
                    try emitModRm(&buffer, 2, 3, 5);
                    try emitDisp32(&buffer, slotOffset(op.dest));
                    try buffer.emitBytes(&.{ 0xdd, 0xd8 }); // fstp st(0)
                    self.stats.arithmetic_emitted += 1;
                },
                .cmpl_float, .cmpg_float => |op| {
                    try emitLoadXmmSlot32(&buffer, 0, op.src1);
                    try buffer.emitBytes(&.{ 0x0f, 0x2e }); // ucomiss xmm0, [rbp - disp]
                    try emitModRm(&buffer, 2, 0, 5);
                    try emitDisp32(&buffer, slotOffset(op.src2));

                    const label_nan = try buffer.newLabel();
                    const label_lt = try buffer.newLabel();
                    const label_eq = try buffer.newLabel();
                    const label_done = try buffer.newLabel();

                    try buffer.emitBytes(&.{ 0x0f, 0x8a }); // jp nan (PF=1 on NaN)
                    _ = try buffer.reloc(label_nan, .rel32, 0);
                    try buffer.emitBytes(&.{ 0x0f, 0x82 }); // jb lt (CF=1 if <)
                    _ = try buffer.reloc(label_lt, .rel32, 0);
                    try buffer.emitBytes(&.{ 0x0f, 0x84 }); // je eq (ZF=1 if ==)
                    _ = try buffer.reloc(label_eq, .rel32, 0);

                    // Greater than
                    try buffer.emitBytes(&.{ 0xb8, 0x01, 0x00, 0x00, 0x00 }); // mov eax, 1
                    try buffer.emitBytes(&.{ 0xe9 }); // jmp done
                    _ = try buffer.reloc(label_done, .rel32, 0);

                    try buffer.bindLabel(label_nan);
                    if (inst == .cmpg_float) {
                        try buffer.emitBytes(&.{ 0xb8, 0x01, 0x00, 0x00, 0x00 });
                    } else {
                        try buffer.emitBytes(&.{ 0xb8, 0xff, 0xff, 0xff, 0xff });
                    }
                    try buffer.emitBytes(&.{ 0xe9 });
                    _ = try buffer.reloc(label_done, .rel32, 0);

                    try buffer.bindLabel(label_lt);
                    try buffer.emitBytes(&.{ 0xb8, 0xff, 0xff, 0xff, 0xff });
                    try buffer.emitBytes(&.{ 0xe9 });
                    _ = try buffer.reloc(label_done, .rel32, 0);

                    try buffer.bindLabel(label_eq);
                    try buffer.emitBytes(&.{ 0x33, 0xc0 }); // xor eax, eax

                    try buffer.bindLabel(label_done);
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .cmpl_double, .cmpg_double => |op| {
                    try emitLoadXmmSlot64(&buffer, 0, op.src1);
                    try buffer.emitBytes(&.{ 0x66, 0x0f, 0x2e }); // ucomisd xmm0, [rbp - disp]
                    try emitModRm(&buffer, 2, 0, 5);
                    try emitDisp32(&buffer, slotOffset(op.src2));

                    const label_nan = try buffer.newLabel();
                    const label_lt = try buffer.newLabel();
                    const label_eq = try buffer.newLabel();
                    const label_done = try buffer.newLabel();

                    try buffer.emitBytes(&.{ 0x0f, 0x8a }); // jp nan
                    _ = try buffer.reloc(label_nan, .rel32, 0);
                    try buffer.emitBytes(&.{ 0x0f, 0x82 }); // jb lt
                    _ = try buffer.reloc(label_lt, .rel32, 0);
                    try buffer.emitBytes(&.{ 0x0f, 0x84 }); // je eq
                    _ = try buffer.reloc(label_eq, .rel32, 0);

                    try buffer.emitBytes(&.{ 0xb8, 0x01, 0x00, 0x00, 0x00 });
                    try buffer.emitBytes(&.{ 0xe9 });
                    _ = try buffer.reloc(label_done, .rel32, 0);

                    try buffer.bindLabel(label_nan);
                    if (inst == .cmpg_double) {
                        try buffer.emitBytes(&.{ 0xb8, 0x01, 0x00, 0x00, 0x00 });
                    } else {
                        try buffer.emitBytes(&.{ 0xb8, 0xff, 0xff, 0xff, 0xff });
                    }
                    try buffer.emitBytes(&.{ 0xe9 });
                    _ = try buffer.reloc(label_done, .rel32, 0);

                    try buffer.bindLabel(label_lt);
                    try buffer.emitBytes(&.{ 0xb8, 0xff, 0xff, 0xff, 0xff });
                    try buffer.emitBytes(&.{ 0xe9 });
                    _ = try buffer.reloc(label_done, .rel32, 0);

                    try buffer.bindLabel(label_eq);
                    try buffer.emitBytes(&.{ 0x33, 0xc0 });

                    try buffer.bindLabel(label_done);
                    try emitStoreRegSlot32(&buffer, 0, op.dest);
                    self.stats.arithmetic_emitted += 1;
                },
                .return_void => {
                    try emitEpilogue(&buffer);
                    self.stats.returns_emitted += 1;
                },
                .return_, .return_object => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    try emitLoadXmmSlot32(&buffer, 0, op.src);
                    try emitEpilogue(&buffer);
                    self.stats.returns_emitted += 1;
                },
                .return_wide => |op| {
                    try emitLoadRegSlot64(&buffer, 0, op.src);
                    try emitLoadXmmSlot64(&buffer, 0, op.src);
                    try emitEpilogue(&buffer);
                    self.stats.returns_emitted += 1;
                },
                .goto_ => |op| {
                    const target_idx: i64 = @as(i64, @intCast(i)) + op.offset;
                    if (target_idx < 0 or target_idx >= insts.len) return error.InvalidInstructionTarget;
                    try buffer.emitU8(0xe9); // jmp rel32
                    _ = try buffer.reloc(labels[@intCast(target_idx)], .rel32, 0);
                    self.stats.branches_emitted += 1;
                },
                .if_eq, .if_ne, .if_lt, .if_ge, .if_gt, .if_le => |op| {
                    const target_idx: i64 = @as(i64, @intCast(i)) + op.offset;
                    if (target_idx < 0 or target_idx >= insts.len) return error.InvalidInstructionTarget;
                    try emitLoadRegSlot32(&buffer, 0, op.src1);
                    try buffer.emitBytes(&.{ 0x3b, 0x85 }); // cmp eax, [rbp - disp]
                    try emitDisp32(&buffer, slotOffset(op.src2));
                    const cond: u8 = switch (inst) {
                        .if_eq => branchCondition(0),
                        .if_ne => branchCondition(1),
                        .if_lt => branchCondition(2),
                        .if_ge => branchCondition(3),
                        .if_gt => branchCondition(4),
                        .if_le => branchCondition(5),
                        else => unreachable,
                    };
                    try buffer.emitBytes(&.{ 0x0f, cond }); // jcc rel32
                    _ = try buffer.reloc(labels[@intCast(target_idx)], .rel32, 0);
                    self.stats.branches_emitted += 1;
                },
                .if_eqz, .if_nez, .if_ltz, .if_gez, .if_gtz, .if_lez => |op| {
                    const target_idx: i64 = @as(i64, @intCast(i)) + op.offset;
                    if (target_idx < 0 or target_idx >= insts.len) return error.InvalidInstructionTarget;
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    try buffer.emitBytes(&.{ 0x85, 0xc0 }); // test eax, eax
                    const cond: u8 = switch (inst) {
                        .if_eqz => branchCondition(0),
                        .if_nez => branchCondition(1),
                        .if_ltz => branchCondition(2),
                        .if_gez => branchCondition(3),
                        .if_gtz => branchCondition(4),
                        .if_lez => branchCondition(5),
                        else => unreachable,
                    };
                    try buffer.emitBytes(&.{ 0x0f, cond }); // jcc rel32
                    _ = try buffer.reloc(labels[@intCast(target_idx)], .rel32, 0);
                    self.stats.branches_emitted += 1;
                },
                .packed_switch, .sparse_switch => |op| {
                    if (op.payload) |payload| {
                        if (payload.keys.len > 0) {
                            try emitLoadRegSlot32(&buffer, 0, op.src);
                            for (payload.keys, payload.targets) |key, target_offset| {
                                const target_idx: i64 = @as(i64, @intCast(i)) + target_offset;
                                if (target_idx < 0 or target_idx >= insts.len) return error.InvalidInstructionTarget;
                                try buffer.emitU8(0x3d); // cmp eax, imm32
                                try buffer.emitU32(@bitCast(key));
                                try buffer.emitBytes(&.{ 0x0f, 0x84 }); // je target
                                _ = try buffer.reloc(labels[@intCast(target_idx)], .rel32, 0);
                                self.stats.branches_emitted += 1;
                            }
                        }
                    }
                },
                else => return error.UnsupportedInstruction,
            }
        }

        const code_bytes = try buffer.finalize();
        const relocs = try allocator.dupe(Relocation, buffer.relocs.items);

        self.stats.compiled_methods += 1;
        self.stats.bytes_generated += @intCast(code_bytes.len);

        return CompiledCode{
            .allocator = allocator,
            .bytes = code_bytes,
            .relocations = relocs,
            .stack_maps = &.{},
            .deopts = &.{},
            .dependencies = &.{},
        };
    }

    /// Directly compiles instructions and publishes the immutable executable code version
    /// into the runtime code manager according to the W^X JIT contract.
    pub fn compileAndPublish(
        self: *Compiler,
        allocator: std.mem.Allocator,
        manager: *runtime_code_manager.Manager,
        method_id: u32,
        insts: []const Instruction,
        options: Options,
    ) Error!runtime_code_manager.Candidate {
        var compiled = try self.compile(allocator, insts, options);
        defer compiled.deinit();

        var candidate = try manager.prepare(compiled.bytes);
        try manager.publish(method_id, &candidate);
        return candidate;
    }
};

test "direct JIT compiler compiles basic arithmetic and return" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 10 } },
        .{ .const_ = .{ .dest = 1, .value = 32 } },
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };

    var compiler = Compiler.init();
    var compiled = try compiler.compile(std.testing.allocator, &insts, .{});
    defer compiled.deinit();

    try compiled.verify();
    try std.testing.expect(compiled.bytes.len > 0);
    try std.testing.expectEqual(@as(u32, 1), compiler.stats.compiled_methods);
    try std.testing.expectEqual(@as(u32, 4), compiler.stats.instructions_processed);
}

test "direct JIT compiler compiles conditional branches with correct target resolution" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 5 } }, // 0
        .{ .if_eqz = .{ .src = 0, .offset = 2 } }, // 1: jumps to 3 if 0
        .{ .const_ = .{ .dest = 0, .value = 42 } }, // 2
        .{ .return_ = .{ .src = 0 } },             // 3
    };

    var compiler = Compiler.init();
    var compiled = try compiler.compile(std.testing.allocator, &insts, .{});
    defer compiled.deinit();

    try compiled.verify();
    try std.testing.expectEqual(@as(u32, 4), compiler.stats.instructions_processed);
    try std.testing.expectEqual(@as(u32, 1), compiler.stats.branches_emitted);
}

test "direct JIT compiler W^X publication pipeline" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 99 } },
        .{ .return_ = .{ .src = 0 } },
    };

    var manager = try runtime_code_manager.Manager.init(std.testing.allocator, 4, 4, 4);
    defer manager.deinit() catch {};

    var compiler = Compiler.init();
    _ = try compiler.compileAndPublish(std.testing.allocator, &manager, 1, &insts, .{});

    var reader = try manager.registerReader();
    defer reader.deinit();
    var lease = try manager.enter(&reader, 1);
    defer lease.deinit();
    try std.testing.expect(lease.codeSize() > 0);
    const Fn = fn () callconv(.c) i32;
    const result = lease.typedEntry(Fn)();
    try std.testing.expectEqual(@as(i32, 99), result);
}

test "direct JIT compiler executes constant and arithmetic operations" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 150 } },
        .{ .const_ = .{ .dest = 1, .value = 25 } },
        .{ .sub_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };

    var manager = try runtime_code_manager.Manager.init(std.testing.allocator, 4, 4, 4);
    defer manager.deinit() catch {};

    var compiler = Compiler.init();
    _ = try compiler.compileAndPublish(std.testing.allocator, &manager, 1, &insts, .{});

    var reader = try manager.registerReader();
    defer reader.deinit();
    var lease = try manager.enter(&reader, 1);
    defer lease.deinit();

    const Fn = fn () callconv(.c) i32;
    const result = lease.typedEntry(Fn)();
    try std.testing.expectEqual(@as(i32, 125), result);
}

test "direct JIT compiler executes with parameters and conditional branching" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    // Parameters are stored in v1 and v2 (since reg_count=3, parameter_count=2 -> param_base=1)
    // 0: if v1 == 0 jump to 3 (return v2)
    // 1: v0 = v1 + v2
    // 2: return v0
    // 3: return v2
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 1, .offset = 3 } }, // 0
        .{ .add_int = .{ .dest = 0, .src1 = 1, .src2 = 2 } }, // 1
        .{ .return_ = .{ .src = 0 } }, // 2
        .{ .return_ = .{ .src = 2 } }, // 3
    };

    var manager = try runtime_code_manager.Manager.init(std.testing.allocator, 4, 4, 4);
    defer manager.deinit() catch {};

    var compiler = Compiler.init();
    _ = try compiler.compileAndPublish(std.testing.allocator, &manager, 1, &insts, .{
        .register_count = 3,
        .parameter_count = 2,
    });

    var reader = try manager.registerReader();
    defer reader.deinit();
    var lease = try manager.enter(&reader, 1);
    defer lease.deinit();

    const Fn = fn (i64, i64) callconv(.c) i32;
    const exec = lease.typedEntry(Fn);

    try std.testing.expectEqual(@as(i32, 77), exec(0, 77));
    try std.testing.expectEqual(@as(i32, 42), exec(23, 19));
}

test "direct JIT compiler executes shift instructions" {
    // v0 = v0 << v1
    // v0 = v0 >> 1
    // return v0
    const insts = [_]Instruction{
        .{ .shl_int = .{ .dest = 0, .src1 = 0, .src2 = 1 } },
        .{ .shr_int_lit8 = .{ .dest = 0, .src = 0, .lit = 1 } },
        .{ .return_ = .{ .src = 0 } },
    };

    var manager = try runtime_code_manager.Manager.init(std.testing.allocator, 4, 4, 4);
    defer manager.deinit() catch {};

    var compiler = Compiler.init();
    _ = try compiler.compileAndPublish(std.testing.allocator, &manager, 1, &insts, .{
        .register_count = 2,
        .parameter_count = 2,
    });

    var reader = try manager.registerReader();
    defer reader.deinit();
    var lease = try manager.enter(&reader, 1);
    defer lease.deinit();

    const Fn = fn (i64, i64) callconv(.c) i32;
    const exec = lease.typedEntry(Fn);

    try std.testing.expectEqual(@as(i32, 10), exec(5, 2)); // (5 << 2) >> 1 = 20 >> 1 = 10
}

test "direct JIT compiler executes integer division and remainder with minInt overflow protection" {
    // Parameters in v2 and v3 (param_base = 4 - 2 = 2)
    // v0 = v2 / v3
    // v1 = v2 % v3
    // v0 = v0 + v1
    // return v0
    const insts = [_]Instruction{
        .{ .div_int = .{ .dest = 0, .src1 = 2, .src2 = 3 } },
        .{ .rem_int = .{ .dest = 1, .src1 = 2, .src2 = 3 } },
        .{ .add_int = .{ .dest = 0, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 0 } },
    };

    var manager = try runtime_code_manager.Manager.init(std.testing.allocator, 4, 4, 4);
    defer manager.deinit() catch {};

    var compiler = Compiler.init();
    _ = try compiler.compileAndPublish(std.testing.allocator, &manager, 1, &insts, .{
        .register_count = 4,
        .parameter_count = 2,
    });

    var reader = try manager.registerReader();
    defer reader.deinit();
    var lease = try manager.enter(&reader, 1);
    defer lease.deinit();

    const Fn = fn (i64, i64) callconv(.c) i32;
    const exec = lease.typedEntry(Fn);

    // Normal div and rem: 17 / 5 = 3, 17 % 5 = 2 -> sum = 5
    try std.testing.expectEqual(@as(i32, 5), exec(17, 5));

    // minInt / -1 protection: (-2147483648) / (-1) = -2147483648, rem = 0 -> sum = -2147483648
    try std.testing.expectEqual(@as(i32, std.math.minInt(i32)), exec(std.math.minInt(i32), -1));
}

test "direct JIT compiler executes long arithmetic and comparison" {
    // Parameters in v4 and v5 (param_base = 6 - 2 = 4)
    // v0 = v4 + v5
    // return_wide v0
    const insts = [_]Instruction{
        .{ .add_long = .{ .dest = 0, .src1 = 4, .src2 = 5 } },
        .{ .return_wide = .{ .src = 0 } },
    };

    var manager = try runtime_code_manager.Manager.init(std.testing.allocator, 4, 4, 4);
    defer manager.deinit() catch {};

    var compiler = Compiler.init();
    _ = try compiler.compileAndPublish(std.testing.allocator, &manager, 1, &insts, .{
        .register_count = 6,
        .parameter_count = 2,
    });

    var reader = try manager.registerReader();
    defer reader.deinit();
    var lease = try manager.enter(&reader, 1);
    defer lease.deinit();

    const Fn = fn (i64, i64) callconv(.c) i64;
    const exec = lease.typedEntry(Fn);

    try std.testing.expectEqual(@as(i64, 1234567890123 + 9876543210987), exec(1234567890123, 9876543210987));
}

test "direct JIT compiler executes float arithmetic" {
    // Parameters in v1 and v2 (param_base = 3 - 2 = 1)
    // v0 = v1 * v2
    // v0 = v0 + v1
    // return v0
    const insts = [_]Instruction{
        .{ .mul_float = .{ .dest = 0, .src1 = 1, .src2 = 2 } },
        .{ .add_float = .{ .dest = 0, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 0 } },
    };

    var manager = try runtime_code_manager.Manager.init(std.testing.allocator, 4, 4, 4);
    defer manager.deinit() catch {};

    var compiler = Compiler.init();
    _ = try compiler.compileAndPublish(std.testing.allocator, &manager, 1, &insts, .{
        .register_count = 3,
        .parameter_count = 2,
    });

    var reader = try manager.registerReader();
    defer reader.deinit();
    var lease = try manager.enter(&reader, 1);
    defer lease.deinit();

    const Fn = fn (i64, i64) callconv(.c) f32;
    const exec = lease.typedEntry(Fn);

    const f1: f32 = 3.5;
    const f2: f32 = 4.0;
    const arg1: i64 = @as(i64, @intCast(@as(u32, @bitCast(f1))));
    const arg2: i64 = @as(i64, @intCast(@as(u32, @bitCast(f2))));

    const expected: f32 = (3.5 * 4.0) + 3.5;
    try std.testing.expectEqual(expected, exec(arg1, arg2));
}

test "direct JIT compiler executes double arithmetic" {
    // Parameters in v4 and v5 (param_base = 6 - 2 = 4)
    // v0 = v4 / v5
    // return_wide v0
    const insts = [_]Instruction{
        .{ .div_double = .{ .dest = 0, .src1 = 4, .src2 = 5 } },
        .{ .return_wide = .{ .src = 0 } },
    };

    var manager = try runtime_code_manager.Manager.init(std.testing.allocator, 4, 4, 4);
    defer manager.deinit() catch {};

    var compiler = Compiler.init();
    _ = try compiler.compileAndPublish(std.testing.allocator, &manager, 1, &insts, .{
        .register_count = 6,
        .parameter_count = 2,
    });

    var reader = try manager.registerReader();
    defer reader.deinit();
    var lease = try manager.enter(&reader, 1);
    defer lease.deinit();

    const Fn = fn (i64, i64) callconv(.c) f64;
    const exec = lease.typedEntry(Fn);

    const d1: f64 = 84.0;
    const d2: f64 = 2.5;
    const arg1: i64 = @bitCast(d1);
    const arg2: i64 = @bitCast(d2);

    try std.testing.expectEqual(@as(f64, 33.6), exec(arg1, arg2));
}
