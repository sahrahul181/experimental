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
                    if (op.dest < std.math.maxInt(u16)) {
                        const next: u16 = op.dest + 1;
                        if (next > max) max = next;
                    }
                },
                .add_int, .sub_int, .mul_int, .div_int, .rem_int, .and_int, .or_int, .xor_int, .shl_int, .shr_int, .ushr_int => |op| {
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
                .if_eq, .if_ne, .if_lt, .if_ge, .if_gt, .if_le => |op| {
                    if (op.src1 > max) max = op.src1;
                    if (op.src2 > max) max = op.src2;
                },
                .if_eqz, .if_nez, .if_ltz, .if_gez, .if_gtz, .if_lez => |op| {
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
                .return_void => {
                    try emitEpilogue(&buffer);
                    self.stats.returns_emitted += 1;
                },
                .return_, .return_object => |op| {
                    try emitLoadRegSlot32(&buffer, 0, op.src);
                    try emitEpilogue(&buffer);
                    self.stats.returns_emitted += 1;
                },
                .return_wide => |op| {
                    try emitLoadRegSlot64(&buffer, 0, op.src);
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
