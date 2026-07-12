//! Pure Interpreter: ALU, Conversions, and Control Flow
//! No heap, no object model, no runtime dependencies.
//! Runtime-dependent opcodes (invoke, field, array, object, exception,
//! monitor) all return error.UnimplementedOpcode and are handled by a
//! higher-level VM driver layer.

const std = @import("std");
const instmod = @import("instructions");
pub const Instruction = instmod.Instruction;

pub const ReturnType = enum(u8) {
    void,
    single,
    wide,
    object,
};
pub const ExecutionResult = extern struct {
    kind: ReturnType,
    value32: u32 = 0,
    value64: u64 = 0,
};

pub const RuntimeError = error{
    DivisionByZero,
    UnimplementedOpcode,
    UnexpectedEndOfCode,
    OutOfMemory,
};

pub const ExecutionFrame = struct {
    pc: u32,
    registers: []u32,
    instructions: []const Instruction,
    result_register: [2]u32 = .{ 0, 0 },
    register_is_ref: []bool = &.{},

    // --- Helper Methods for Type Punning ---
    // Dalvik registers are untyped 32-bit slots. These helpers safely cast
    // the binary representations into Zig's strongly typed math primitives.

    pub inline fn getInt(self: *const ExecutionFrame, reg: u16) i32 {
        return @bitCast(self.registers[reg]);
    }
    pub inline fn setInt(self: *ExecutionFrame, reg: u16, val: i32) void {
        self.registers[reg] = @bitCast(val);
    }

    pub inline fn getFloat(self: *const ExecutionFrame, reg: u16) f32 {
        return @bitCast(self.registers[reg]);
    }
    pub inline fn setFloat(self: *ExecutionFrame, reg: u16, val: f32) void {
        self.registers[reg] = @bitCast(val);
    }

    pub inline fn getWide(self: *const ExecutionFrame, reg: u16) u64 {
        const lo: u64 = self.registers[reg];
        const hi: u64 = self.registers[reg + 1];
        return lo | (hi << 32);
    }
    pub inline fn setWide(self: *ExecutionFrame, reg: u16, val: u64) void {
        self.registers[reg] = @truncate(val);
        self.registers[reg + 1] = @truncate(val >> 32);
    }

    pub inline fn getLong(self: *const ExecutionFrame, reg: u16) i64 {
        return @bitCast(self.getWide(reg));
    }
    pub inline fn setLong(self: *ExecutionFrame, reg: u16, val: i64) void {
        self.setWide(reg, @bitCast(val));
    }

    pub inline fn getDouble(self: *const ExecutionFrame, reg: u16) f64 {
        return @bitCast(self.getWide(reg));
    }
    pub inline fn setDouble(self: *ExecutionFrame, reg: u16, val: f64) void {
        self.setWide(reg, @bitCast(val));
    }
};

inline fn getInt(regs: []const u32, reg: u16) i32 {
    return @bitCast(regs[reg]);
}

inline fn setInt(regs: []u32, reg: u16, val: i32) void {
    regs[reg] = @bitCast(val);
}

inline fn getFloat(regs: []const u32, reg: u16) f32 {
    return @bitCast(regs[reg]);
}

inline fn setFloat(regs: []u32, reg: u16, val: f32) void {
    regs[reg] = @bitCast(val);
}

inline fn getWide(regs: []const u32, reg: u16) u64 {
    const lo: u64 = regs[reg];
    const hi: u64 = regs[reg + 1];
    return lo | (hi << 32);
}

inline fn setWide(regs: []u32, reg: u16, val: u64) void {
    regs[reg] = @truncate(val);
    regs[reg + 1] = @truncate(val >> 32);
}

inline fn getLong(regs: []const u32, reg: u16) i64 {
    return @bitCast(getWide(regs, reg));
}

inline fn setLong(regs: []u32, reg: u16, val: i64) void {
    setWide(regs, reg, @bitCast(val));
}

inline fn getDouble(regs: []const u32, reg: u16) f64 {
    return @bitCast(getWide(regs, reg));
}

inline fn setDouble(regs: []u32, reg: u16, val: f64) void {
    setWide(regs, reg, @bitCast(val));
}

inline fn floatToIntSafe(f: f32) i32 {
    if (std.math.isNan(f)) return 0;
    if (f >= @as(f32, @floatFromInt(std.math.maxInt(i32)))) return std.math.maxInt(i32);
    if (f <= @as(f32, @floatFromInt(std.math.minInt(i32)))) return std.math.minInt(i32);
    return @intFromFloat(f);
}

inline fn floatToLongSafe(f: f32) i64 {
    if (std.math.isNan(f)) return 0;
    if (f >= @as(f32, @floatFromInt(std.math.maxInt(i64)))) return std.math.maxInt(i64);
    if (f <= @as(f32, @floatFromInt(std.math.minInt(i64)))) return std.math.minInt(i64);
    return @intFromFloat(f);
}

inline fn doubleToIntSafe(d: f64) i32 {
    if (std.math.isNan(d)) return 0;
    if (d >= @as(f64, @floatFromInt(std.math.maxInt(i32)))) return std.math.maxInt(i32);
    if (d <= @as(f64, @floatFromInt(std.math.minInt(i32)))) return std.math.minInt(i32);
    return @intFromFloat(d);
}

inline fn doubleToLongSafe(d: f64) i64 {
    if (std.math.isNan(d)) return 0;
    if (d >= @as(f64, @floatFromInt(std.math.maxInt(i64)))) return std.math.maxInt(i64);
    if (d <= @as(f64, @floatFromInt(std.math.minInt(i64)))) return std.math.minInt(i64);
    return @intFromFloat(d);
}

inline fn setRef(regs_is_ref: []bool, reg: u16, val: bool) void {
    if (regs_is_ref.len > 0) {
        regs_is_ref[reg] = val;
    }
}

inline fn setRefWide(regs_is_ref: []bool, reg: u16, val: bool) void {
    if (regs_is_ref.len > 0) {
        regs_is_ref[reg] = val;
        regs_is_ref[reg + 1] = val;
    }
}

pub fn execute(frame: *ExecutionFrame) RuntimeError!ExecutionResult {
    // OPTIMIZATION 5: Safety Stripping
    // We explicitly disable runtime bounds checking inside the hot loop.
    // The DEX parser/verifier guarantees that `pc` and register indices are bounds-safe.
    // Removing these safety checks removes hidden conditional branch instructions from
    // the generated machine code, significantly improving pipeline branch prediction.
    @setRuntimeSafety(false);

    var pc = frame.pc;
    errdefer frame.pc = pc;
    const insts = frame.instructions;
    const regs = frame.registers;
    const regs_is_ref = frame.register_is_ref;

    while (pc < insts.len) {
        const inst = insts[pc];
        pc += 1; // Pre-increment PC for branch offsets

        switch (inst) {
            // ==========================================
            // PHASE 1: Base & Constants
            // ==========================================
            .nop => {},
            .const_ => |op| {
                regs[op.dest] = @bitCast(op.value);
                setRef(regs_is_ref, op.dest, false);
            },
            .const_wide => |op| {
                setWide(regs, op.dest, @bitCast(op.value));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .const_string => |op| {
                regs[op.dest] = op.index;
                setRef(regs_is_ref, op.dest, true);
            },
            .const_class => |op| {
                regs[op.dest] = op.type_idx;
                setRef(regs_is_ref, op.dest, true);
            },
            .const_method_handle => |op| {
                regs[op.dest] = op.index;
                setRef(regs_is_ref, op.dest, true);
            },
            .const_method_type => |op| {
                regs[op.dest] = op.index;
                setRef(regs_is_ref, op.dest, true);
            },

            .move => |op| {
                regs[op.dest] = regs[op.src];
                setRef(regs_is_ref, op.dest, if (regs_is_ref.len > 0) regs_is_ref[op.src] else false);
            },
            .move_wide => |op| {
                setWide(regs, op.dest, getWide(regs, op.src));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .move_object => |op| {
                regs[op.dest] = regs[op.src];
                setRef(regs_is_ref, op.dest, if (regs_is_ref.len > 0) regs_is_ref[op.src] else true);
            },

            .move_result => |op| {
                regs[op.dest] = frame.result_register[0];
                setRef(regs_is_ref, op.dest, false);
            },
            .move_result_object => |op| {
                regs[op.dest] = frame.result_register[0];
                setRef(regs_is_ref, op.dest, true);
            },
            .move_result_wide => |op| {
                const ptr: *u64 = @ptrCast(@alignCast(&frame.result_register));
                setWide(regs, op.dest, ptr.*);
                setRefWide(regs_is_ref, op.dest, false);
            },

            // ==========================================
            // PHASE 2: Unary Math & Conversions
            // ==========================================
            .neg_int => |op| {
                setInt(regs, op.dest, -%getInt(regs, op.src));
                setRef(regs_is_ref, op.dest, false);
            },
            .not_int => |op| {
                regs[op.dest] = ~regs[op.src];
                setRef(regs_is_ref, op.dest, false);
            },
            .neg_long => |op| {
                setLong(regs, op.dest, -%getLong(regs, op.src));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .not_long => |op| {
                setWide(regs, op.dest, ~getWide(regs, op.src));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .neg_float => |op| {
                setFloat(regs, op.dest, -getFloat(regs, op.src));
                setRef(regs_is_ref, op.dest, false);
            },
            .neg_double => |op| {
                setDouble(regs, op.dest, -getDouble(regs, op.src));
                setRefWide(regs_is_ref, op.dest, false);
            },

            .int_to_long => |op| {
                setLong(regs, op.dest, getInt(regs, op.src));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .int_to_float => |op| {
                setFloat(regs, op.dest, @floatFromInt(getInt(regs, op.src)));
                setRef(regs_is_ref, op.dest, false);
            },
            .int_to_double => |op| {
                setDouble(regs, op.dest, @floatFromInt(getInt(regs, op.src)));
                setRefWide(regs_is_ref, op.dest, false);
            },

            .long_to_int => |op| {
                setInt(regs, op.dest, @truncate(getLong(regs, op.src)));
                setRef(regs_is_ref, op.dest, false);
            },
            .long_to_float => |op| {
                setFloat(regs, op.dest, @floatFromInt(getLong(regs, op.src)));
                setRef(regs_is_ref, op.dest, false);
            },
            .long_to_double => |op| {
                setDouble(regs, op.dest, @floatFromInt(getLong(regs, op.src)));
                setRefWide(regs_is_ref, op.dest, false);
            },

            .float_to_int => |op| {
                setInt(regs, op.dest, floatToIntSafe(getFloat(regs, op.src)));
                setRef(regs_is_ref, op.dest, false);
            },
            .float_to_long => |op| {
                setLong(regs, op.dest, floatToLongSafe(getFloat(regs, op.src)));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .float_to_double => |op| {
                setDouble(regs, op.dest, @floatCast(getFloat(regs, op.src)));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .double_to_int => |op| {
                setInt(regs, op.dest, doubleToIntSafe(getDouble(regs, op.src)));
                setRef(regs_is_ref, op.dest, false);
            },
            .double_to_long => |op| {
                setLong(regs, op.dest, doubleToLongSafe(getDouble(regs, op.src)));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .double_to_float => |op| {
                setFloat(regs, op.dest, @floatCast(getDouble(regs, op.src)));
                setRef(regs_is_ref, op.dest, false);
            },

            .int_to_byte => |op| {
                setInt(regs, op.dest, @as(i8, @truncate(getInt(regs, op.src))));
                setRef(regs_is_ref, op.dest, false);
            },
            .int_to_char => |op| {
                regs[op.dest] = @as(u16, @truncate(regs[op.src]));
                setRef(regs_is_ref, op.dest, false);
            },
            .int_to_short => |op| {
                setInt(regs, op.dest, @as(i16, @truncate(getInt(regs, op.src))));
                setRef(regs_is_ref, op.dest, false);
            },

            // ==========================================
            // PHASE 2: Binary Math (Integers)
            // Note: Zig's +%, -%, *% perfectly mirror Java's wrap-on-overflow spec.
            // ==========================================
            .add_int => |op| {
                setInt(regs, op.dest, getInt(regs, op.src1) +% getInt(regs, op.src2));
                setRef(regs_is_ref, op.dest, false);
            },
            .sub_int => |op| {
                setInt(regs, op.dest, getInt(regs, op.src1) -% getInt(regs, op.src2));
                setRef(regs_is_ref, op.dest, false);
            },
            .mul_int => |op| {
                setInt(regs, op.dest, getInt(regs, op.src1) *% getInt(regs, op.src2));
                setRef(regs_is_ref, op.dest, false);
            },
            .and_int => |op| {
                regs[op.dest] = regs[op.src1] & regs[op.src2];
                setRef(regs_is_ref, op.dest, false);
            },
            .or_int => |op| {
                regs[op.dest] = regs[op.src1] | regs[op.src2];
                setRef(regs_is_ref, op.dest, false);
            },
            .xor_int => |op| {
                regs[op.dest] = regs[op.src1] ^ regs[op.src2];
                setRef(regs_is_ref, op.dest, false);
            },

            .shl_int => |op| {
                setInt(regs, op.dest, getInt(regs, op.src1) << @as(u5, @truncate(regs[op.src2])));
                setRef(regs_is_ref, op.dest, false);
            },
            .shr_int => |op| {
                setInt(regs, op.dest, getInt(regs, op.src1) >> @as(u5, @truncate(regs[op.src2])));
                setRef(regs_is_ref, op.dest, false);
            },
            .ushr_int => |op| {
                regs[op.dest] = regs[op.src1] >> @as(u5, @truncate(regs[op.src2]));
                setRef(regs_is_ref, op.dest, false);
            },

            .div_int => |op| {
                const v1 = getInt(regs, op.src1);
                const v2 = getInt(regs, op.src2);
                if (v2 == 0) return error.DivisionByZero;
                if (v1 == std.math.minInt(i32) and v2 == -1) {
                    setInt(regs, op.dest, v1); // JVM div spec: no hardware trap
                } else {
                    setInt(regs, op.dest, @divTrunc(v1, v2));
                }
                setRef(regs_is_ref, op.dest, false);
            },
            .rem_int => |op| {
                const v1 = getInt(regs, op.src1);
                const v2 = getInt(regs, op.src2);
                if (v2 == 0) return error.DivisionByZero;
                if (v1 == std.math.minInt(i32) and v2 == -1) {
                    setInt(regs, op.dest, 0); // JVM rem spec
                } else {
                    setInt(regs, op.dest, @rem(v1, v2));
                }
                setRef(regs_is_ref, op.dest, false);
            },

            // -- Long Math --
            .add_long => |op| {
                setLong(regs, op.dest, getLong(regs, op.src1) +% getLong(regs, op.src2));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .sub_long => |op| {
                setLong(regs, op.dest, getLong(regs, op.src1) -% getLong(regs, op.src2));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .mul_long => |op| {
                setLong(regs, op.dest, getLong(regs, op.src1) *% getLong(regs, op.src2));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .div_long => |op| {
                const v1 = getLong(regs, op.src1);
                const v2 = getLong(regs, op.src2);
                if (v2 == 0) return error.DivisionByZero;
                if (v1 == std.math.minInt(i64) and v2 == -1) {
                    setLong(regs, op.dest, v1);
                } else {
                    setLong(regs, op.dest, @divTrunc(v1, v2));
                }
                setRefWide(regs_is_ref, op.dest, false);
            },
            .rem_long => |op| {
                const v1 = getLong(regs, op.src1);
                const v2 = getLong(regs, op.src2);
                if (v2 == 0) return error.DivisionByZero;
                if (v1 == std.math.minInt(i64) and v2 == -1) {
                    setLong(regs, op.dest, 0);
                } else {
                    setLong(regs, op.dest, @rem(v1, v2));
                }
                setRefWide(regs_is_ref, op.dest, false);
            },
            .and_long => |op| {
                setWide(regs, op.dest, getWide(regs, op.src1) & getWide(regs, op.src2));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .or_long => |op| {
                setWide(regs, op.dest, getWide(regs, op.src1) | getWide(regs, op.src2));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .xor_long => |op| {
                setWide(regs, op.dest, getWide(regs, op.src1) ^ getWide(regs, op.src2));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .shl_long => |op| {
                const v1 = getLong(regs, op.src1);
                const v2 = getInt(regs, op.src2);
                const count = @as(u6, @intCast(@as(u32, @bitCast(v2)) & 63));
                setLong(regs, op.dest, v1 << count);
                setRefWide(regs_is_ref, op.dest, false);
            },
            .shr_long => |op| {
                const v1 = getLong(regs, op.src1);
                const v2 = getInt(regs, op.src2);
                const count = @as(u6, @intCast(@as(u32, @bitCast(v2)) & 63));
                setLong(regs, op.dest, v1 >> count);
                setRefWide(regs_is_ref, op.dest, false);
            },
            .ushr_long => |op| {
                const v1 = getWide(regs, op.src1);
                const v2 = getInt(regs, op.src2);
                const count = @as(u6, @intCast(@as(u32, @bitCast(v2)) & 63));
                setWide(regs, op.dest, v1 >> count);
                setRefWide(regs_is_ref, op.dest, false);
            },

            // -- Float & Double Math --
            .add_float => |op| {
                setFloat(regs, op.dest, getFloat(regs, op.src1) + getFloat(regs, op.src2));
                setRef(regs_is_ref, op.dest, false);
            },
            .sub_float => |op| {
                setFloat(regs, op.dest, getFloat(regs, op.src1) - getFloat(regs, op.src2));
                setRef(regs_is_ref, op.dest, false);
            },
            .mul_float => |op| {
                setFloat(regs, op.dest, getFloat(regs, op.src1) * getFloat(regs, op.src2));
                setRef(regs_is_ref, op.dest, false);
            },
            .div_float => |op| {
                setFloat(regs, op.dest, getFloat(regs, op.src1) / getFloat(regs, op.src2));
                setRef(regs_is_ref, op.dest, false);
            },
            .rem_float => |op| {
                setFloat(regs, op.dest, @rem(getFloat(regs, op.src1), getFloat(regs, op.src2)));
                setRef(regs_is_ref, op.dest, false);
            },

            .add_double => |op| {
                setDouble(regs, op.dest, getDouble(regs, op.src1) + getDouble(regs, op.src2));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .sub_double => |op| {
                setDouble(regs, op.dest, getDouble(regs, op.src1) - getDouble(regs, op.src2));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .mul_double => |op| {
                setDouble(regs, op.dest, getDouble(regs, op.src1) * getDouble(regs, op.src2));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .div_double => |op| {
                setDouble(regs, op.dest, getDouble(regs, op.src1) / getDouble(regs, op.src2));
                setRefWide(regs_is_ref, op.dest, false);
            },
            .rem_double => |op| {
                setDouble(regs, op.dest, @rem(getDouble(regs, op.src1), getDouble(regs, op.src2)));
                setRefWide(regs_is_ref, op.dest, false);
            },

            // ==========================================
            // PHASE 2: Binary Math (Lit16 & Lit8)
            // ==========================================
            .add_int_lit16 => |op| {
                setInt(regs, op.dest, getInt(regs, op.src) +% op.lit);
                setRef(regs_is_ref, op.dest, false);
            },
            .rsub_int_lit16 => |op| {
                setInt(regs, op.dest, @as(i32, op.lit) -% getInt(regs, op.src));
                setRef(regs_is_ref, op.dest, false);
            },
            .mul_int_lit16 => |op| {
                setInt(regs, op.dest, getInt(regs, op.src) *% op.lit);
                setRef(regs_is_ref, op.dest, false);
            },
            .and_int_lit16 => |op| {
                regs[op.dest] = regs[op.src] & @as(u32, @bitCast(@as(i32, op.lit)));
                setRef(regs_is_ref, op.dest, false);
            },
            .or_int_lit16 => |op| {
                regs[op.dest] = regs[op.src] | @as(u32, @bitCast(@as(i32, op.lit)));
                setRef(regs_is_ref, op.dest, false);
            },
            .xor_int_lit16 => |op| {
                regs[op.dest] = regs[op.src] ^ @as(u32, @bitCast(@as(i32, op.lit)));
                setRef(regs_is_ref, op.dest, false);
            },
            .div_int_lit16 => |op| {
                if (op.lit == 0) return error.DivisionByZero;
                const v1 = getInt(regs, op.src);
                if (v1 == std.math.minInt(i32) and op.lit == -1) {
                    setInt(regs, op.dest, v1);
                } else {
                    setInt(regs, op.dest, @divTrunc(v1, @as(i32, op.lit)));
                }
                setRef(regs_is_ref, op.dest, false);
            },
            .rem_int_lit16 => |op| {
                if (op.lit == 0) return error.DivisionByZero;
                const v1 = getInt(regs, op.src);
                if (v1 == std.math.minInt(i32) and op.lit == -1) {
                    setInt(regs, op.dest, 0);
                } else {
                    setInt(regs, op.dest, @rem(v1, @as(i32, op.lit)));
                }
                setRef(regs_is_ref, op.dest, false);
            },

            .add_int_lit8 => |op| {
                setInt(regs, op.dest, getInt(regs, op.src) +% op.lit);
                setRef(regs_is_ref, op.dest, false);
            },
            .rsub_int_lit8 => |op| {
                setInt(regs, op.dest, @as(i32, op.lit) -% getInt(regs, op.src));
                setRef(regs_is_ref, op.dest, false);
            },
            .mul_int_lit8 => |op| {
                setInt(regs, op.dest, getInt(regs, op.src) *% op.lit);
                setRef(regs_is_ref, op.dest, false);
            },
            .div_int_lit8 => |op| {
                if (op.lit == 0) return error.DivisionByZero;
                const v1 = getInt(regs, op.src);
                if (v1 == std.math.minInt(i32) and op.lit == -1) {
                    setInt(regs, op.dest, v1);
                } else {
                    setInt(regs, op.dest, @divTrunc(v1, @as(i32, op.lit)));
                }
                setRef(regs_is_ref, op.dest, false);
            },
            .rem_int_lit8 => |op| {
                if (op.lit == 0) return error.DivisionByZero;
                const v1 = getInt(regs, op.src);
                if (v1 == std.math.minInt(i32) and op.lit == -1) {
                    setInt(regs, op.dest, 0);
                } else {
                    setInt(regs, op.dest, @rem(v1, @as(i32, op.lit)));
                }
                setRef(regs_is_ref, op.dest, false);
            },
            .and_int_lit8 => |op| {
                regs[op.dest] = regs[op.src] & @as(u32, @bitCast(@as(i32, op.lit)));
                setRef(regs_is_ref, op.dest, false);
            },
            .or_int_lit8 => |op| {
                regs[op.dest] = regs[op.src] | @as(u32, @bitCast(@as(i32, op.lit)));
                setRef(regs_is_ref, op.dest, false);
            },
            .xor_int_lit8 => |op| {
                regs[op.dest] = regs[op.src] ^ @as(u32, @bitCast(@as(i32, op.lit)));
                setRef(regs_is_ref, op.dest, false);
            },
            .shl_int_lit8 => |op| {
                setInt(regs, op.dest, getInt(regs, op.src) << @as(u5, @truncate(@as(u8, @bitCast(op.lit)))));
                setRef(regs_is_ref, op.dest, false);
            },
            .shr_int_lit8 => |op| {
                setInt(regs, op.dest, getInt(regs, op.src) >> @as(u5, @truncate(@as(u8, @bitCast(op.lit)))));
                setRef(regs_is_ref, op.dest, false);
            },
            .ushr_int_lit8 => |op| {
                regs[op.dest] = regs[op.src] >> @as(u5, @truncate(@as(u8, @bitCast(op.lit))));
                setRef(regs_is_ref, op.dest, false);
            },

            // -- Comparisons --
            .cmpl_float => |op| {
                const v1 = getFloat(regs, op.src1);
                const v2 = getFloat(regs, op.src2);
                if (std.math.isNan(v1) or std.math.isNan(v2)) {
                    setInt(regs, op.dest, -1);
                } else if (v1 > v2) {
                    setInt(regs, op.dest, 1);
                } else if (v1 < v2) {
                    setInt(regs, op.dest, -1);
                } else {
                    setInt(regs, op.dest, 0);
                }
                setRef(regs_is_ref, op.dest, false);
            },
            .cmpg_float => |op| {
                const v1 = getFloat(regs, op.src1);
                const v2 = getFloat(regs, op.src2);
                if (std.math.isNan(v1) or std.math.isNan(v2)) {
                    setInt(regs, op.dest, 1);
                } else if (v1 > v2) {
                    setInt(regs, op.dest, 1);
                } else if (v1 < v2) {
                    setInt(regs, op.dest, -1);
                } else {
                    setInt(regs, op.dest, 0);
                }
                setRef(regs_is_ref, op.dest, false);
            },
            .cmpl_double => |op| {
                const v1 = getDouble(regs, op.src1);
                const v2 = getDouble(regs, op.src2);
                if (std.math.isNan(v1) or std.math.isNan(v2)) {
                    setInt(regs, op.dest, -1);
                } else if (v1 > v2) {
                    setInt(regs, op.dest, 1);
                } else if (v1 < v2) {
                    setInt(regs, op.dest, -1);
                } else {
                    setInt(regs, op.dest, 0);
                }
                setRef(regs_is_ref, op.dest, false);
            },
            .cmpg_double => |op| {
                const v1 = getDouble(regs, op.src1);
                const v2 = getDouble(regs, op.src2);
                if (std.math.isNan(v1) or std.math.isNan(v2)) {
                    setInt(regs, op.dest, 1);
                } else if (v1 > v2) {
                    setInt(regs, op.dest, 1);
                } else if (v1 < v2) {
                    setInt(regs, op.dest, -1);
                } else {
                    setInt(regs, op.dest, 0);
                }
                setRef(regs_is_ref, op.dest, false);
            },
            .cmp_long => |op| {
                const v1 = getLong(regs, op.src1);
                const v2 = getLong(regs, op.src2);
                if (v1 > v2) {
                    setInt(regs, op.dest, 1);
                } else if (v1 < v2) {
                    setInt(regs, op.dest, -1);
                } else {
                    setInt(regs, op.dest, 0);
                }
                setRef(regs_is_ref, op.dest, false);
            },

            // ==========================================
            // PHASE 3: Control Flow (Branching)
            // ==========================================
            .goto_ => |op| {
                // pc has already advanced, subtract 1 to get instruction's original PC for relative jump
                pc = @intCast(@as(i32, @intCast(pc - 1)) + op.offset);
            },

            .if_eq => |op| if (regs[op.src1] == regs[op.src2]) {
                pc = @intCast(@as(i32, @intCast(pc - 1)) + op.offset);
            },
            .if_ne => |op| if (regs[op.src1] != regs[op.src2]) {
                pc = @intCast(@as(i32, @intCast(pc - 1)) + op.offset);
            },
            .if_lt => |op| if (getInt(regs, op.src1) < getInt(regs, op.src2)) {
                pc = @intCast(@as(i32, @intCast(pc - 1)) + op.offset);
            },
            .if_ge => |op| if (getInt(regs, op.src1) >= getInt(regs, op.src2)) {
                pc = @intCast(@as(i32, @intCast(pc - 1)) + op.offset);
            },
            .if_gt => |op| if (getInt(regs, op.src1) > getInt(regs, op.src2)) {
                pc = @intCast(@as(i32, @intCast(pc - 1)) + op.offset);
            },
            .if_le => |op| if (getInt(regs, op.src1) <= getInt(regs, op.src2)) {
                pc = @intCast(@as(i32, @intCast(pc - 1)) + op.offset);
            },

            .if_eqz => |op| if (regs[op.src] == 0) {
                pc = @intCast(@as(i32, @intCast(pc - 1)) + op.offset);
            },
            .if_nez => |op| if (regs[op.src] != 0) {
                pc = @intCast(@as(i32, @intCast(pc - 1)) + op.offset);
            },
            .if_ltz => |op| if (getInt(regs, op.src) < 0) {
                pc = @intCast(@as(i32, @intCast(pc - 1)) + op.offset);
            },
            .if_gez => |op| if (getInt(regs, op.src) >= 0) {
                pc = @intCast(@as(i32, @intCast(pc - 1)) + op.offset);
            },
            .if_gtz => |op| if (getInt(regs, op.src) > 0) {
                pc = @intCast(@as(i32, @intCast(pc - 1)) + op.offset);
            },
            .if_lez => |op| if (getInt(regs, op.src) <= 0) {
                pc = @intCast(@as(i32, @intCast(pc - 1)) + op.offset);
            },

            .packed_switch => |op| {
                const val = getInt(regs, op.src);
                if (op.payload) |payload| {
                    if (payload.keys.len > 0 and val >= payload.keys[0]) {
                        const idx: usize = @intCast(val - payload.keys[0]);
                        if (idx < payload.keys.len and payload.keys[idx] == val) {
                            pc = @intCast(@as(i32, @intCast(pc - 1)) + payload.targets[idx]);
                        }
                    }
                }
            },
            .sparse_switch => |op| {
                const val = getInt(regs, op.src);
                if (op.payload) |payload| {
                    var lo: usize = 0;
                    var hi: usize = payload.keys.len;
                    while (lo < hi) {
                        const mid = lo + (hi - lo) / 2;
                        const key = payload.keys[mid];
                        if (key == val) {
                            pc = @intCast(@as(i32, @intCast(pc - 1)) + payload.targets[mid]);
                            break;
                        }
                        if (key < val) lo = mid + 1 else hi = mid;
                    }
                }
            },

            // ==========================================
            // PHASE 5: Object Instantiation & Typing
            // (Runtime-dependent — implementation lives in the VM layer)
            // ==========================================
            .new_instance => return error.UnimplementedOpcode,
            .instance_of => return error.UnimplementedOpcode,
            .check_cast => return error.UnimplementedOpcode,

            // ==========================================
            // PHASE 6: Field Access
            // (Runtime-dependent — implementation lives in the VM layer)
            // ==========================================
            // -- Static Fields --
            .sget => return error.UnimplementedOpcode,
            .sget_wide => return error.UnimplementedOpcode,
            .sget_object => return error.UnimplementedOpcode,
            .sget_boolean => return error.UnimplementedOpcode,
            .sget_byte => return error.UnimplementedOpcode,
            .sget_char => return error.UnimplementedOpcode,
            .sget_short => return error.UnimplementedOpcode,

            .sput => return error.UnimplementedOpcode,
            .sput_wide => return error.UnimplementedOpcode,
            .sput_object => return error.UnimplementedOpcode,
            .sput_boolean => return error.UnimplementedOpcode,
            .sput_byte => return error.UnimplementedOpcode,
            .sput_char => return error.UnimplementedOpcode,
            .sput_short => return error.UnimplementedOpcode,

            // -- Instance Fields --
            .iget => return error.UnimplementedOpcode,
            .iget_wide => return error.UnimplementedOpcode,
            .iget_object => return error.UnimplementedOpcode,
            .iget_boolean => return error.UnimplementedOpcode,
            .iget_byte => return error.UnimplementedOpcode,
            .iget_char => return error.UnimplementedOpcode,
            .iget_short => return error.UnimplementedOpcode,

            .iput => return error.UnimplementedOpcode,
            .iput_wide => return error.UnimplementedOpcode,
            .iput_object => return error.UnimplementedOpcode,
            .iput_boolean => return error.UnimplementedOpcode,
            .iput_byte => return error.UnimplementedOpcode,
            .iput_char => return error.UnimplementedOpcode,
            .iput_short => return error.UnimplementedOpcode,

            // -- Quickened Fields --
            .iget_quick => return error.UnimplementedOpcode,
            .iget_wide_quick => return error.UnimplementedOpcode,
            .iget_object_quick => return error.UnimplementedOpcode,
            .iput_quick => return error.UnimplementedOpcode,
            .iput_wide_quick => return error.UnimplementedOpcode,
            .iput_object_quick => return error.UnimplementedOpcode,

            // ==========================================
            // PHASE 7: Method Invocation
            // (Runtime-dependent — implementation lives in the VM layer)
            // ==========================================
            .invoke => return error.UnimplementedOpcode,
            .invoke_virtual_quick => return error.UnimplementedOpcode,
            .invoke_super_quick => return error.UnimplementedOpcode,

            // ==========================================
            // PHASE 8: Exceptions & Threading
            // (Runtime-dependent — implementation lives in the VM layer)
            // ==========================================
            .throw_ => return error.UnimplementedOpcode,
            .move_exception => return error.UnimplementedOpcode,
            .monitor_enter => return error.UnimplementedOpcode,
            .monitor_exit => return error.UnimplementedOpcode,
            // ==========================================
            // PHASE 4: Memory & Arrays
            // (Runtime-dependent — implementation lives in the VM layer)
            // ==========================================
            .array_length => return error.UnimplementedOpcode,
            .new_array => return error.UnimplementedOpcode,
            .filled_new_array => return error.UnimplementedOpcode,
            .fill_array_data => return error.UnimplementedOpcode,

            .aget => return error.UnimplementedOpcode,
            .aget_wide => return error.UnimplementedOpcode,
            .aget_object => return error.UnimplementedOpcode,
            .aget_boolean => return error.UnimplementedOpcode,
            .aget_byte => return error.UnimplementedOpcode,
            .aget_char => return error.UnimplementedOpcode,
            .aget_short => return error.UnimplementedOpcode,

            .aput => return error.UnimplementedOpcode,
            .aput_wide => return error.UnimplementedOpcode,
            .aput_object => return error.UnimplementedOpcode,
            .aput_boolean => return error.UnimplementedOpcode,
            .aput_byte => return error.UnimplementedOpcode,
            .aput_char => return error.UnimplementedOpcode,
            .aput_short => return error.UnimplementedOpcode,

            // ==========================================
            // PHASE 1: Returns
            // ==========================================
            .return_void => {
                frame.pc = pc;
                return ExecutionResult{ .kind = .void };
            },
            .return_ => |op| {
                frame.pc = pc;
                return ExecutionResult{ .kind = .single, .value32 = regs[op.src] };
            },
            .return_wide => |op| {
                frame.pc = pc;
                return ExecutionResult{ .kind = .wide, .value64 = getWide(regs, op.src) };
            },
            .return_object => |op| {
                frame.pc = pc;
                return ExecutionResult{ .kind = .object, .value32 = regs[op.src] };
            },
        }
    }

    return error.UnexpectedEndOfCode;
}

fn testExecute(insts: []const Instruction, regs: []u32) RuntimeError!ExecutionResult {
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = regs,
        .instructions = insts,
    };
    return execute(&frame);
}

test "ExecutionFrame register helpers" {
    var regs = [_]u32{0} ** 8;
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = &regs,
        .instructions = &.{},
    };

    frame.setInt(0, -123);
    try std.testing.expectEqual(@as(i32, -123), frame.getInt(0));

    frame.setFloat(1, -2.5);
    try std.testing.expectEqual(@as(f32, -2.5), frame.getFloat(1));

    frame.setWide(2, 0x8877665544332211);
    try std.testing.expectEqual(@as(u64, 0x8877665544332211), frame.getWide(2));

    frame.setLong(4, -0x123456789);
    try std.testing.expectEqual(@as(i64, -0x123456789), frame.getLong(4));

    frame.setDouble(6, 1234.5);
    try std.testing.expectEqual(@as(f64, 1234.5), frame.getDouble(6));
}

test "execute constants moves move-result and returns" {
    var regs = [_]u32{0} ** 16;
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = &regs,
        .instructions = &[_]Instruction{
            .nop,
            .{ .const_ = .{ .dest = 0, .value = -7 } },
            .{ .const_wide = .{ .dest = 2, .value = -0x100000001 } },
            .{ .const_string = .{ .dest = 4, .index = 99 } },
            .{ .const_class = .{ .dest = 5, .type_idx = 7 } },
            .{ .const_method_handle = .{ .dest = 6, .index = 8 } },
            .{ .const_method_type = .{ .dest = 7, .index = 9 } },
            .{ .move = .{ .dest = 1, .src = 0 } },
            .{ .move_wide = .{ .dest = 8, .src = 2 } },
            .{ .move_object = .{ .dest = 10, .src = 4 } },
            .{ .move_result = .{ .dest = 11 } },
            .{ .move_result_object = .{ .dest = 12 } },
            .{ .move_result_wide = .{ .dest = 13 } },
            .{ .return_wide = .{ .src = 13 } },
        },
        .result_register = .{ 0x11223344, 0x55667788 },
    };

    const result = try execute(&frame);
    try std.testing.expectEqual(ReturnType.wide, result.kind);
    try std.testing.expectEqual(@as(u64, 0x5566778811223344), result.value64);
    try std.testing.expectEqual(@as(i32, -7), @as(i32, @bitCast(regs[1])));
    try std.testing.expectEqual(@as(u64, @bitCast(@as(i64, -0x100000001))), frame.getWide(8));
    try std.testing.expectEqual(@as(u32, 99), regs[10]);
    try std.testing.expectEqual(@as(u32, 8), regs[6]);
    try std.testing.expectEqual(@as(u32, 9), regs[7]);
    try std.testing.expectEqual(@as(u32, 0x11223344), regs[11]);
    try std.testing.expectEqual(@as(u32, 0x11223344), regs[12]);

    const void_result = try testExecute(&[_]Instruction{.return_void}, &regs);
    try std.testing.expectEqual(ReturnType.void, void_result.kind);

    regs[0] = 0xdeadbeef;
    const object_result = try testExecute(&[_]Instruction{.{ .return_object = .{ .src = 0 } }}, &regs);
    try std.testing.expectEqual(ReturnType.object, object_result.kind);
    try std.testing.expectEqual(@as(u32, 0xdeadbeef), object_result.value32);
}

test "execute unary math and conversions with edge cases" {
    var regs = [_]u32{0} ** 32;
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = &regs,
        .instructions = &[_]Instruction{
            .{ .const_ = .{ .dest = 0, .value = -5 } },
            .{ .neg_int = .{ .dest = 1, .src = 0 } },
            .{ .not_int = .{ .dest = 2, .src = 0 } },
            .{ .int_to_long = .{ .dest = 4, .src = 0 } },
            .{ .long_to_int = .{ .dest = 6, .src = 4 } },
            .{ .int_to_float = .{ .dest = 7, .src = 0 } },
            .{ .float_to_int = .{ .dest = 8, .src = 7 } },
            .{ .int_to_double = .{ .dest = 10, .src = 0 } },
            .{ .double_to_int = .{ .dest = 12, .src = 10 } },
            .{ .float_to_double = .{ .dest = 14, .src = 7 } },
            .{ .double_to_float = .{ .dest = 16, .src = 14 } },
            .{ .long_to_float = .{ .dest = 18, .src = 4 } },
            .{ .long_to_double = .{ .dest = 20, .src = 4 } },
            .{ .float_to_long = .{ .dest = 22, .src = 7 } },
            .{ .double_to_long = .{ .dest = 24, .src = 10 } },
            .{ .int_to_byte = .{ .dest = 26, .src = 0 } },
            .{ .int_to_char = .{ .dest = 27, .src = 0 } },
            .{ .int_to_short = .{ .dest = 28, .src = 0 } },
            .{ .neg_long = .{ .dest = 4, .src = 4 } },
            .{ .not_long = .{ .dest = 29, .src = 4 } },
            .{ .neg_float = .{ .dest = 7, .src = 7 } },
            .{ .neg_double = .{ .dest = 10, .src = 10 } },
            .return_void,
        },
    };
    _ = try execute(&frame);

    try std.testing.expectEqual(@as(i32, 5), frame.getInt(1));
    try std.testing.expectEqual(~@as(u32, @bitCast(@as(i32, -5))), regs[2]);
    try std.testing.expectEqual(@as(i32, -5), frame.getInt(6));
    try std.testing.expectEqual(@as(i32, -5), frame.getInt(8));
    try std.testing.expectEqual(@as(i32, -5), frame.getInt(12));
    try std.testing.expectEqual(@as(f32, -5), frame.getFloat(16));
    try std.testing.expectEqual(@as(i64, -5), frame.getLong(22));
    try std.testing.expectEqual(@as(i64, -5), frame.getLong(24));
    try std.testing.expectEqual(@as(i32, -5), frame.getInt(26));
    try std.testing.expectEqual(@as(u32, 0xfffb), regs[27]);
    try std.testing.expectEqual(@as(i32, -5), frame.getInt(28));
    try std.testing.expectEqual(~@as(u64, 5), frame.getWide(29));
    try std.testing.expectEqual(@as(f32, 5), frame.getFloat(7));
    try std.testing.expectEqual(@as(f64, 5), frame.getDouble(10));

    frame.setFloat(0, std.math.nan(f32));
    frame.instructions = &[_]Instruction{ .{ .float_to_int = .{ .dest = 1, .src = 0 } }, .{ .float_to_long = .{ .dest = 2, .src = 0 } }, .return_void };
    frame.pc = 0;
    _ = try execute(&frame);
    try std.testing.expectEqual(@as(i32, 0), frame.getInt(1));
    try std.testing.expectEqual(@as(i64, 0), frame.getLong(2));

    frame.setDouble(4, std.math.inf(f64));
    frame.instructions = &[_]Instruction{ .{ .double_to_int = .{ .dest = 6, .src = 4 } }, .{ .double_to_long = .{ .dest = 8, .src = 4 } }, .return_void };
    frame.pc = 0;
    _ = try execute(&frame);
    try std.testing.expectEqual(std.math.maxInt(i32), frame.getInt(6));
    try std.testing.expectEqual(std.math.maxInt(i64), frame.getLong(8));
}

test "execute integer and long binary math edge cases" {
    var regs = [_]u32{0} ** 32;
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = &regs,
        .instructions = &[_]Instruction{
            .{ .const_ = .{ .dest = 0, .value = 7 } },
            .{ .const_ = .{ .dest = 1, .value = 3 } },
            .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
            .{ .sub_int = .{ .dest = 3, .src1 = 0, .src2 = 1 } },
            .{ .mul_int = .{ .dest = 4, .src1 = 0, .src2 = 1 } },
            .{ .div_int = .{ .dest = 5, .src1 = 0, .src2 = 1 } },
            .{ .rem_int = .{ .dest = 6, .src1 = 0, .src2 = 1 } },
            .{ .and_int = .{ .dest = 7, .src1 = 0, .src2 = 1 } },
            .{ .or_int = .{ .dest = 8, .src1 = 0, .src2 = 1 } },
            .{ .xor_int = .{ .dest = 9, .src1 = 0, .src2 = 1 } },
            .{ .shl_int = .{ .dest = 10, .src1 = 1, .src2 = 1 } },
            .{ .shr_int = .{ .dest = 11, .src1 = 0, .src2 = 1 } },
            .{ .ushr_int = .{ .dest = 12, .src1 = 0, .src2 = 1 } },
            .{ .const_wide = .{ .dest = 14, .value = 9 } },
            .{ .const_wide = .{ .dest = 16, .value = 2 } },
            .{ .add_long = .{ .dest = 18, .src1 = 14, .src2 = 16 } },
            .{ .sub_long = .{ .dest = 20, .src1 = 14, .src2 = 16 } },
            .{ .mul_long = .{ .dest = 22, .src1 = 14, .src2 = 16 } },
            .{ .div_long = .{ .dest = 24, .src1 = 14, .src2 = 16 } },
            .{ .rem_long = .{ .dest = 26, .src1 = 14, .src2 = 16 } },
            .{ .and_long = .{ .dest = 14, .src1 = 14, .src2 = 16 } },
            .{ .or_long = .{ .dest = 16, .src1 = 18, .src2 = 20 } },
            .{ .xor_long = .{ .dest = 28, .src1 = 20, .src2 = 22 } },
            .{ .shl_long = .{ .dest = 20, .src1 = 16, .src2 = 1 } },
            .{ .shr_long = .{ .dest = 22, .src1 = 20, .src2 = 1 } },
            .{ .ushr_long = .{ .dest = 24, .src1 = 20, .src2 = 1 } },
            .return_void,
        },
    };
    _ = try execute(&frame);

    try std.testing.expectEqual(@as(i32, 10), frame.getInt(2));
    try std.testing.expectEqual(@as(i32, 4), frame.getInt(3));
    try std.testing.expectEqual(@as(i32, 21), frame.getInt(4));
    try std.testing.expectEqual(@as(i32, 2), frame.getInt(5));
    try std.testing.expectEqual(@as(i32, 1), frame.getInt(6));
    try std.testing.expectEqual(@as(u32, 7 & 3), regs[7]);
    try std.testing.expectEqual(@as(u32, 7 | 3), regs[8]);
    try std.testing.expectEqual(@as(u32, 7 ^ 3), regs[9]);
    try std.testing.expectEqual(@as(i32, 24), frame.getInt(10));
    try std.testing.expectEqual(@as(i64, 11), frame.getLong(18));
    try std.testing.expectEqual(@as(i64, 120), frame.getLong(20));
    try std.testing.expectEqual(@as(i64, 15), frame.getLong(22));
    try std.testing.expectEqual(@as(u64, 15), frame.getWide(24));

    frame.instructions = &[_]Instruction{.{ .div_int = .{ .dest = 0, .src1 = 1, .src2 = 2 } }};
    frame.pc = 0;
    regs[1] = 1;
    regs[2] = 0;
    try std.testing.expectError(error.DivisionByZero, execute(&frame));

    frame.instructions = &[_]Instruction{ .{ .rem_long = .{ .dest = 0, .src1 = 4, .src2 = 6 } }, .return_void };
    frame.pc = 0;
    frame.setLong(4, std.math.minInt(i64));
    frame.setLong(6, -1);
    _ = try execute(&frame);
    try std.testing.expectEqual(@as(i64, 0), frame.getLong(0));
}

test "execute float double comparisons and literal math" {
    var regs = [_]u32{0} ** 32;
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = &regs,
        .instructions = &[_]Instruction{
            .{ .const_ = .{ .dest = 0, .value = 10 } },
            .{ .add_int_lit16 = .{ .dest = 1, .src = 0, .lit = 2 } },
            .{ .rsub_int_lit16 = .{ .dest = 2, .src = 0, .lit = 2 } },
            .{ .mul_int_lit16 = .{ .dest = 3, .src = 0, .lit = 2 } },
            .{ .div_int_lit16 = .{ .dest = 4, .src = 0, .lit = 3 } },
            .{ .rem_int_lit16 = .{ .dest = 5, .src = 0, .lit = 3 } },
            .{ .and_int_lit16 = .{ .dest = 6, .src = 0, .lit = 3 } },
            .{ .or_int_lit16 = .{ .dest = 7, .src = 0, .lit = 3 } },
            .{ .xor_int_lit16 = .{ .dest = 8, .src = 0, .lit = 3 } },
            .{ .add_int_lit8 = .{ .dest = 9, .src = 0, .lit = 1 } },
            .{ .rsub_int_lit8 = .{ .dest = 10, .src = 0, .lit = 1 } },
            .{ .mul_int_lit8 = .{ .dest = 11, .src = 0, .lit = 2 } },
            .{ .div_int_lit8 = .{ .dest = 12, .src = 0, .lit = 4 } },
            .{ .rem_int_lit8 = .{ .dest = 13, .src = 0, .lit = 4 } },
            .{ .and_int_lit8 = .{ .dest = 14, .src = 0, .lit = 6 } },
            .{ .or_int_lit8 = .{ .dest = 15, .src = 0, .lit = 6 } },
            .{ .xor_int_lit8 = .{ .dest = 16, .src = 0, .lit = 6 } },
            .{ .shl_int_lit8 = .{ .dest = 17, .src = 0, .lit = 1 } },
            .{ .shr_int_lit8 = .{ .dest = 18, .src = 0, .lit = 1 } },
            .{ .ushr_int_lit8 = .{ .dest = 19, .src = 0, .lit = 1 } },
            .return_void,
        },
    };
    _ = try execute(&frame);
    try std.testing.expectEqual(@as(i32, 12), frame.getInt(1));
    try std.testing.expectEqual(@as(i32, -8), frame.getInt(2));
    try std.testing.expectEqual(@as(i32, 20), frame.getInt(3));
    try std.testing.expectEqual(@as(i32, 3), frame.getInt(4));
    try std.testing.expectEqual(@as(i32, 1), frame.getInt(5));
    try std.testing.expectEqual(@as(u32, 10 & 3), regs[6]);
    try std.testing.expectEqual(@as(u32, 10 | 3), regs[7]);
    try std.testing.expectEqual(@as(u32, 10 ^ 3), regs[8]);
    try std.testing.expectEqual(@as(i32, 11), frame.getInt(9));
    try std.testing.expectEqual(@as(i32, -9), frame.getInt(10));
    try std.testing.expectEqual(@as(i32, 20), frame.getInt(11));
    try std.testing.expectEqual(@as(i32, 2), frame.getInt(12));
    try std.testing.expectEqual(@as(i32, 2), frame.getInt(13));
    try std.testing.expectEqual(@as(i32, 20), frame.getInt(17));
    try std.testing.expectEqual(@as(i32, 5), frame.getInt(18));
    try std.testing.expectEqual(@as(u32, 5), regs[19]);

    frame.setFloat(0, 1.5);
    frame.setFloat(1, 0.5);
    frame.setDouble(2, 5.0);
    frame.setDouble(4, 2.0);
    frame.setLong(6, 9);
    frame.setLong(8, 10);
    frame.instructions = &[_]Instruction{
        .{ .add_float = .{ .dest = 10, .src1 = 0, .src2 = 1 } },
        .{ .sub_float = .{ .dest = 11, .src1 = 0, .src2 = 1 } },
        .{ .mul_float = .{ .dest = 12, .src1 = 0, .src2 = 1 } },
        .{ .div_float = .{ .dest = 13, .src1 = 0, .src2 = 1 } },
        .{ .rem_float = .{ .dest = 14, .src1 = 0, .src2 = 1 } },
        .{ .add_double = .{ .dest = 16, .src1 = 2, .src2 = 4 } },
        .{ .sub_double = .{ .dest = 18, .src1 = 2, .src2 = 4 } },
        .{ .mul_double = .{ .dest = 20, .src1 = 2, .src2 = 4 } },
        .{ .div_double = .{ .dest = 22, .src1 = 2, .src2 = 4 } },
        .{ .rem_double = .{ .dest = 24, .src1 = 2, .src2 = 4 } },
        .{ .cmpl_float = .{ .dest = 26, .src1 = 0, .src2 = 1 } },
        .{ .cmpg_float = .{ .dest = 27, .src1 = 0, .src2 = 1 } },
        .{ .cmpl_double = .{ .dest = 28, .src1 = 2, .src2 = 4 } },
        .{ .cmpg_double = .{ .dest = 29, .src1 = 2, .src2 = 4 } },
        .{ .cmp_long = .{ .dest = 30, .src1 = 6, .src2 = 8 } },
        .return_void,
    };
    frame.pc = 0;
    _ = try execute(&frame);
    try std.testing.expectEqual(@as(f32, 2.0), frame.getFloat(10));
    try std.testing.expectEqual(@as(f32, 1.0), frame.getFloat(11));
    try std.testing.expectEqual(@as(f32, 0.75), frame.getFloat(12));
    try std.testing.expectEqual(@as(f32, 3.0), frame.getFloat(13));
    try std.testing.expectEqual(@as(f64, 7.0), frame.getDouble(16));
    try std.testing.expectEqual(@as(f64, 3.0), frame.getDouble(18));
    try std.testing.expectEqual(@as(f64, 10.0), frame.getDouble(20));
    try std.testing.expectEqual(@as(f64, 2.5), frame.getDouble(22));
    try std.testing.expectEqual(@as(i32, 1), frame.getInt(26));
    try std.testing.expectEqual(@as(i32, 1), frame.getInt(27));
    try std.testing.expectEqual(@as(i32, 1), frame.getInt(28));
    try std.testing.expectEqual(@as(i32, 1), frame.getInt(29));
    try std.testing.expectEqual(@as(i32, -1), frame.getInt(30));

    frame.setFloat(0, std.math.nan(f32));
    frame.instructions = &[_]Instruction{ .{ .cmpl_float = .{ .dest = 1, .src1 = 0, .src2 = 0 } }, .{ .cmpg_float = .{ .dest = 2, .src1 = 0, .src2 = 0 } }, .return_void };
    frame.pc = 0;
    _ = try execute(&frame);
    try std.testing.expectEqual(@as(i32, -1), frame.getInt(1));
    try std.testing.expectEqual(@as(i32, 1), frame.getInt(2));
}

fn expectBranchResult(inst: Instruction, initial: []const u32, expected: i32) !void {
    var regs = [_]u32{0} ** 4;
    for (initial, 0..) |value, idx| regs[idx] = value;
    const insts = [_]Instruction{
        inst,
        .{ .const_ = .{ .dest = 3, .value = 1 } },
        .{ .return_ = .{ .src = 3 } },
        .{ .const_ = .{ .dest = 3, .value = 2 } },
        .{ .return_ = .{ .src = 3 } },
    };
    const result = try testExecute(&insts, &regs);
    try std.testing.expectEqual(ReturnType.single, result.kind);
    try std.testing.expectEqual(@as(u32, @bitCast(expected)), result.value32);
}

test "execute branches and switches" {
    try expectBranchResult(.{ .goto_ = .{ .offset = 3 } }, &.{}, 2);
    try expectBranchResult(.{ .if_eq = .{ .src1 = 0, .src2 = 1, .offset = 3 } }, &.{ 5, 5 }, 2);
    try expectBranchResult(.{ .if_ne = .{ .src1 = 0, .src2 = 1, .offset = 3 } }, &.{ 5, 6 }, 2);
    try expectBranchResult(.{ .if_lt = .{ .src1 = 0, .src2 = 1, .offset = 3 } }, &.{ @bitCast(@as(i32, -1)), 1 }, 2);
    try expectBranchResult(.{ .if_ge = .{ .src1 = 0, .src2 = 1, .offset = 3 } }, &.{ 1, 1 }, 2);
    try expectBranchResult(.{ .if_gt = .{ .src1 = 0, .src2 = 1, .offset = 3 } }, &.{ 2, 1 }, 2);
    try expectBranchResult(.{ .if_le = .{ .src1 = 0, .src2 = 1, .offset = 3 } }, &.{ 1, 2 }, 2);
    try expectBranchResult(.{ .if_eqz = .{ .src = 0, .offset = 3 } }, &.{0}, 2);
    try expectBranchResult(.{ .if_nez = .{ .src = 0, .offset = 3 } }, &.{1}, 2);
    try expectBranchResult(.{ .if_ltz = .{ .src = 0, .offset = 3 } }, &.{@bitCast(@as(i32, -1))}, 2);
    try expectBranchResult(.{ .if_gez = .{ .src = 0, .offset = 3 } }, &.{0}, 2);
    try expectBranchResult(.{ .if_gtz = .{ .src = 0, .offset = 3 } }, &.{1}, 2);
    try expectBranchResult(.{ .if_lez = .{ .src = 0, .offset = 3 } }, &.{0}, 2);
    try expectBranchResult(.{ .if_eq = .{ .src1 = 0, .src2 = 1, .offset = 3 } }, &.{ 5, 6 }, 1);

    const packed_payload = instmod.SwitchPayload{
        .keys = &[_]i32{ 10, 11, 12 },
        .targets = &[_]i32{ 1, 3, 5 },
    };
    const sparse_payload = instmod.SwitchPayload{
        .keys = &[_]i32{ -10, 0, 99 },
        .targets = &[_]i32{ 1, 3, 5 },
    };
    var regs = [_]u32{0} ** 4;
    regs[0] = 11;
    var result = try testExecute(&[_]Instruction{
        .{ .packed_switch = .{ .src = 0, .payload_offset = 0, .payload = @constCast(&packed_payload) } },
        .{ .const_ = .{ .dest = 1, .value = 10 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 20 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 30 } },
        .{ .return_ = .{ .src = 1 } },
    }, &regs);
    try std.testing.expectEqual(@as(u32, 20), result.value32);

    regs[0] = @bitCast(@as(i32, -10));
    result = try testExecute(&[_]Instruction{
        .{ .sparse_switch = .{ .src = 0, .payload_offset = 0, .payload = @constCast(&sparse_payload) } },
        .{ .const_ = .{ .dest = 1, .value = 10 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 20 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 30 } },
        .{ .return_ = .{ .src = 1 } },
    }, &regs);
    try std.testing.expectEqual(@as(u32, 10), result.value32);

    regs[0] = 77;
    result = try testExecute(&[_]Instruction{
        .{ .sparse_switch = .{ .src = 0, .payload_offset = 0, .payload = @constCast(&sparse_payload) } },
        .{ .const_ = .{ .dest = 1, .value = 10 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 20 } },
        .{ .return_ = .{ .src = 1 } },
    }, &regs);
    try std.testing.expectEqual(@as(u32, 10), result.value32);
}

test "execute arithmetic trap and overflow edge cases" {
    var regs = [_]u32{0} ** 12;
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = &regs,
        .instructions = &.{},
    };

    frame.setInt(0, std.math.minInt(i32));
    frame.setInt(1, -1);
    frame.instructions = &[_]Instruction{
        .{ .div_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .rem_int = .{ .dest = 3, .src1 = 0, .src2 = 1 } },
        .return_void,
    };
    _ = try execute(&frame);
    try std.testing.expectEqual(std.math.minInt(i32), frame.getInt(2));
    try std.testing.expectEqual(@as(i32, 0), frame.getInt(3));

    frame.setLong(4, std.math.minInt(i64));
    frame.setLong(6, -1);
    frame.pc = 0;
    frame.instructions = &[_]Instruction{
        .{ .div_long = .{ .dest = 8, .src1 = 4, .src2 = 6 } },
        .return_void,
    };
    _ = try execute(&frame);
    try std.testing.expectEqual(std.math.minInt(i64), frame.getLong(8));

    frame.pc = 0;
    frame.setLong(6, 0);
    frame.instructions = &[_]Instruction{.{ .div_long = .{ .dest = 8, .src1 = 4, .src2 = 6 } }};
    try std.testing.expectError(error.DivisionByZero, execute(&frame));

    frame.pc = 0;
    frame.setInt(0, 1);
    frame.instructions = &[_]Instruction{.{ .div_int_lit16 = .{ .dest = 1, .src = 0, .lit = 0 } }};
    try std.testing.expectError(error.DivisionByZero, execute(&frame));

    frame.pc = 0;
    frame.instructions = &[_]Instruction{.{ .rem_int_lit8 = .{ .dest = 1, .src = 0, .lit = 0 } }};
    try std.testing.expectError(error.DivisionByZero, execute(&frame));

    frame.pc = 0;
    frame.setInt(0, std.math.minInt(i32));
    frame.instructions = &[_]Instruction{
        .{ .div_int_lit8 = .{ .dest = 1, .src = 0, .lit = -1 } },
        .{ .rem_int_lit16 = .{ .dest = 2, .src = 0, .lit = -1 } },
        .return_void,
    };
    _ = try execute(&frame);
    try std.testing.expectEqual(std.math.minInt(i32), frame.getInt(1));
    try std.testing.expectEqual(@as(i32, 0), frame.getInt(2));

    frame.pc = 0;
    frame.setDouble(0, std.math.nan(f64));
    frame.instructions = &[_]Instruction{
        .{ .cmpl_double = .{ .dest = 2, .src1 = 0, .src2 = 0 } },
        .{ .cmpg_double = .{ .dest = 3, .src1 = 0, .src2 = 0 } },
        .return_void,
    };
    _ = try execute(&frame);
    try std.testing.expectEqual(@as(i32, -1), frame.getInt(2));
    try std.testing.expectEqual(@as(i32, 1), frame.getInt(3));
}

test "execute integer long overflow and shift mask edges" {
    var regs = [_]u32{0} ** 40;
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = &regs,
        .instructions = &[_]Instruction{
            .{ .const_ = .{ .dest = 0, .value = std.math.maxInt(i32) } },
            .{ .const_ = .{ .dest = 1, .value = 1 } },
            .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
            .{ .sub_int = .{ .dest = 3, .src1 = 2, .src2 = 1 } },
            .{ .const_ = .{ .dest = 4, .value = 0x40000000 } },
            .{ .const_ = .{ .dest = 5, .value = 2 } },
            .{ .mul_int = .{ .dest = 6, .src1 = 4, .src2 = 5 } },
            .{ .const_ = .{ .dest = 7, .value = 32 } },
            .{ .const_ = .{ .dest = 8, .value = -1 } },
            .{ .shl_int = .{ .dest = 9, .src1 = 1, .src2 = 7 } },
            .{ .shl_int = .{ .dest = 10, .src1 = 1, .src2 = 8 } },
            .{ .shr_int = .{ .dest = 11, .src1 = 8, .src2 = 7 } },
            .{ .ushr_int = .{ .dest = 12, .src1 = 8, .src2 = 8 } },
            .{ .const_wide = .{ .dest = 14, .value = std.math.maxInt(i64) } },
            .{ .const_wide = .{ .dest = 16, .value = 1 } },
            .{ .add_long = .{ .dest = 18, .src1 = 14, .src2 = 16 } },
            .{ .sub_long = .{ .dest = 20, .src1 = 18, .src2 = 16 } },
            .{ .const_wide = .{ .dest = 22, .value = 0x4000000000000000 } },
            .{ .const_wide = .{ .dest = 24, .value = 2 } },
            .{ .mul_long = .{ .dest = 26, .src1 = 22, .src2 = 24 } },
            .{ .const_ = .{ .dest = 28, .value = 64 } },
            .{ .const_ = .{ .dest = 29, .value = -1 } },
            .{ .shl_long = .{ .dest = 30, .src1 = 16, .src2 = 28 } },
            .{ .shl_long = .{ .dest = 32, .src1 = 16, .src2 = 29 } },
            .{ .shr_long = .{ .dest = 34, .src1 = 18, .src2 = 28 } },
            .{ .ushr_long = .{ .dest = 36, .src1 = 18, .src2 = 29 } },
            .return_void,
        },
    };

    _ = try execute(&frame);
    try std.testing.expectEqual(std.math.minInt(i32), frame.getInt(2));
    try std.testing.expectEqual(std.math.maxInt(i32), frame.getInt(3));
    try std.testing.expectEqual(@as(i32, std.math.minInt(i32)), frame.getInt(6));
    try std.testing.expectEqual(@as(i32, 1), frame.getInt(9));
    try std.testing.expectEqual(@as(i32, std.math.minInt(i32)), frame.getInt(10));
    try std.testing.expectEqual(@as(i32, -1), frame.getInt(11));
    try std.testing.expectEqual(@as(u32, 1), regs[12]);
    try std.testing.expectEqual(std.math.minInt(i64), frame.getLong(18));
    try std.testing.expectEqual(std.math.maxInt(i64), frame.getLong(20));
    try std.testing.expectEqual(std.math.minInt(i64), frame.getLong(26));
    try std.testing.expectEqual(@as(i64, 1), frame.getLong(30));
    try std.testing.expectEqual(std.math.minInt(i64), frame.getLong(32));
    try std.testing.expectEqual(std.math.minInt(i64), frame.getLong(34));
    try std.testing.expectEqual(@as(u64, 1), frame.getWide(36));
}

test "execute literal overflow and sign extension edges" {
    var regs = [_]u32{0} ** 20;
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = &regs,
        .instructions = &[_]Instruction{
            .{ .const_ = .{ .dest = 0, .value = std.math.maxInt(i32) } },
            .{ .add_int_lit16 = .{ .dest = 1, .src = 0, .lit = 1 } },
            .{ .mul_int_lit16 = .{ .dest = 2, .src = 0, .lit = 2 } },
            .{ .rsub_int_lit16 = .{ .dest = 3, .src = 0, .lit = -1 } },
            .{ .add_int_lit8 = .{ .dest = 4, .src = 0, .lit = 1 } },
            .{ .mul_int_lit8 = .{ .dest = 5, .src = 0, .lit = 2 } },
            .{ .rsub_int_lit8 = .{ .dest = 6, .src = 0, .lit = -1 } },
            .{ .const_ = .{ .dest = 7, .value = -1 } },
            .{ .and_int_lit16 = .{ .dest = 8, .src = 7, .lit = -32768 } },
            .{ .or_int_lit16 = .{ .dest = 9, .src = 7, .lit = -32768 } },
            .{ .xor_int_lit16 = .{ .dest = 10, .src = 7, .lit = -32768 } },
            .{ .and_int_lit8 = .{ .dest = 11, .src = 7, .lit = -128 } },
            .{ .or_int_lit8 = .{ .dest = 12, .src = 7, .lit = -128 } },
            .{ .xor_int_lit8 = .{ .dest = 13, .src = 7, .lit = -128 } },
            .{ .const_ = .{ .dest = 14, .value = 1 } },
            .{ .shl_int_lit8 = .{ .dest = 15, .src = 14, .lit = 32 } },
            .{ .shl_int_lit8 = .{ .dest = 16, .src = 14, .lit = -1 } },
            .return_void,
        },
    };

    _ = try execute(&frame);
    try std.testing.expectEqual(std.math.minInt(i32), frame.getInt(1));
    try std.testing.expectEqual(@as(i32, -2), frame.getInt(2));
    try std.testing.expectEqual(std.math.minInt(i32), frame.getInt(3));
    try std.testing.expectEqual(std.math.minInt(i32), frame.getInt(4));
    try std.testing.expectEqual(@as(i32, -2), frame.getInt(5));
    try std.testing.expectEqual(std.math.minInt(i32), frame.getInt(6));
    try std.testing.expectEqual(@as(u32, 0xffff8000), regs[8]);
    try std.testing.expectEqual(@as(u32, 0xffffffff), regs[9]);
    try std.testing.expectEqual(@as(u32, 0x00007fff), regs[10]);
    try std.testing.expectEqual(@as(u32, 0xffffff80), regs[11]);
    try std.testing.expectEqual(@as(u32, 0xffffffff), regs[12]);
    try std.testing.expectEqual(@as(u32, 0x0000007f), regs[13]);
    try std.testing.expectEqual(@as(i32, 1), frame.getInt(15));
    try std.testing.expectEqual(std.math.minInt(i32), frame.getInt(16));
}

test "execute conversion saturation narrowing and signed zero edges" {
    var regs = [_]u32{0} ** 32;
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = &regs,
        .instructions = &.{},
    };

    frame.setFloat(0, std.math.inf(f32));
    frame.setFloat(1, -std.math.inf(f32));
    frame.setDouble(2, std.math.inf(f64));
    frame.setDouble(4, -std.math.inf(f64));
    frame.instructions = &[_]Instruction{
        .{ .float_to_int = .{ .dest = 6, .src = 0 } },
        .{ .float_to_int = .{ .dest = 7, .src = 1 } },
        .{ .float_to_long = .{ .dest = 8, .src = 0 } },
        .{ .float_to_long = .{ .dest = 10, .src = 1 } },
        .{ .double_to_int = .{ .dest = 12, .src = 2 } },
        .{ .double_to_int = .{ .dest = 13, .src = 4 } },
        .{ .double_to_long = .{ .dest = 14, .src = 2 } },
        .{ .double_to_long = .{ .dest = 16, .src = 4 } },
        .return_void,
    };
    _ = try execute(&frame);
    try std.testing.expectEqual(std.math.maxInt(i32), frame.getInt(6));
    try std.testing.expectEqual(std.math.minInt(i32), frame.getInt(7));
    try std.testing.expectEqual(std.math.maxInt(i64), frame.getLong(8));
    try std.testing.expectEqual(std.math.minInt(i64), frame.getLong(10));
    try std.testing.expectEqual(std.math.maxInt(i32), frame.getInt(12));
    try std.testing.expectEqual(std.math.minInt(i32), frame.getInt(13));
    try std.testing.expectEqual(std.math.maxInt(i64), frame.getLong(14));
    try std.testing.expectEqual(std.math.minInt(i64), frame.getLong(16));

    frame.pc = 0;
    frame.setInt(0, 0x80);
    frame.setInt(1, 0xff);
    frame.setInt(2, 0x8000);
    frame.setInt(3, 0xffff);
    frame.instructions = &[_]Instruction{
        .{ .int_to_byte = .{ .dest = 4, .src = 0 } },
        .{ .int_to_byte = .{ .dest = 5, .src = 1 } },
        .{ .int_to_short = .{ .dest = 6, .src = 2 } },
        .{ .int_to_short = .{ .dest = 7, .src = 3 } },
        .{ .int_to_char = .{ .dest = 8, .src = 3 } },
        .return_void,
    };
    _ = try execute(&frame);
    try std.testing.expectEqual(@as(i32, -128), frame.getInt(4));
    try std.testing.expectEqual(@as(i32, -1), frame.getInt(5));
    try std.testing.expectEqual(@as(i32, -32768), frame.getInt(6));
    try std.testing.expectEqual(@as(i32, -1), frame.getInt(7));
    try std.testing.expectEqual(@as(u32, 0xffff), regs[8]);

    frame.pc = 0;
    frame.setFloat(0, -0.0);
    frame.setDouble(2, -0.0);
    frame.instructions = &[_]Instruction{
        .{ .neg_float = .{ .dest = 1, .src = 0 } },
        .{ .neg_double = .{ .dest = 4, .src = 2 } },
        .return_void,
    };
    _ = try execute(&frame);
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 0.0))), regs[1]);
    try std.testing.expectEqual(@as(u64, @bitCast(@as(f64, 0.0))), frame.getWide(4));
}

test "execute floating point infinities and nan propagation edges" {
    var regs = [_]u32{0} ** 32;
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = &regs,
        .instructions = &.{},
    };

    frame.setFloat(0, std.math.inf(f32));
    frame.setFloat(1, -std.math.inf(f32));
    frame.setFloat(2, 0.0);
    frame.setDouble(4, std.math.inf(f64));
    frame.setDouble(6, -std.math.inf(f64));
    frame.setDouble(8, 0.0);
    frame.instructions = &[_]Instruction{
        .{ .add_float = .{ .dest = 10, .src1 = 0, .src2 = 1 } },
        .{ .mul_float = .{ .dest = 11, .src1 = 0, .src2 = 2 } },
        .{ .div_float = .{ .dest = 12, .src1 = 0, .src2 = 1 } },
        .{ .add_double = .{ .dest = 14, .src1 = 4, .src2 = 6 } },
        .{ .mul_double = .{ .dest = 16, .src1 = 4, .src2 = 8 } },
        .{ .div_double = .{ .dest = 18, .src1 = 4, .src2 = 6 } },
        .return_void,
    };
    _ = try execute(&frame);
    try std.testing.expect(std.math.isNan(frame.getFloat(10)));
    try std.testing.expect(std.math.isNan(frame.getFloat(11)));
    try std.testing.expect(std.math.isNan(frame.getFloat(12)));
    try std.testing.expect(std.math.isNan(frame.getDouble(14)));
    try std.testing.expect(std.math.isNan(frame.getDouble(16)));
    try std.testing.expect(std.math.isNan(frame.getDouble(18)));
}

fn expectSwitchResult(inst: Instruction, value: i32, expected: i32) !void {
    var regs = [_]u32{0} ** 4;
    regs[0] = @bitCast(value);
    const result = try testExecute(&[_]Instruction{
        inst,
        .{ .const_ = .{ .dest = 1, .value = 10 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 20 } },
        .{ .return_ = .{ .src = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 30 } },
        .{ .return_ = .{ .src = 1 } },
    }, &regs);
    try std.testing.expectEqual(@as(u32, @bitCast(expected)), result.value32);
}

test "execute switch boundary null and empty payload edges" {
    const packed_payload = instmod.SwitchPayload{
        .keys = &[_]i32{ 10, 11, 12 },
        .targets = &[_]i32{ 1, 3, 5 },
    };
    const sparse_payload = instmod.SwitchPayload{
        .keys = &[_]i32{ -10, 0, 99 },
        .targets = &[_]i32{ 1, 3, 5 },
    };
    const empty_payload = instmod.SwitchPayload{
        .keys = &[_]i32{},
        .targets = &[_]i32{},
    };

    try expectSwitchResult(.{ .packed_switch = .{ .src = 0, .payload_offset = 0, .payload = @constCast(&packed_payload) } }, 10, 10);
    try expectSwitchResult(.{ .packed_switch = .{ .src = 0, .payload_offset = 0, .payload = @constCast(&packed_payload) } }, 12, 30);
    try expectSwitchResult(.{ .packed_switch = .{ .src = 0, .payload_offset = 0, .payload = @constCast(&packed_payload) } }, 9, 10);
    try expectSwitchResult(.{ .packed_switch = .{ .src = 0, .payload_offset = 0, .payload = @constCast(&packed_payload) } }, 13, 10);
    try expectSwitchResult(.{ .packed_switch = .{ .src = 0, .payload_offset = 0, .payload = @constCast(&empty_payload) } }, 10, 10);
    try expectSwitchResult(.{ .packed_switch = .{ .src = 0, .payload_offset = 0, .payload = null } }, 10, 10);

    try expectSwitchResult(.{ .sparse_switch = .{ .src = 0, .payload_offset = 0, .payload = @constCast(&sparse_payload) } }, -10, 10);
    try expectSwitchResult(.{ .sparse_switch = .{ .src = 0, .payload_offset = 0, .payload = @constCast(&sparse_payload) } }, 0, 20);
    try expectSwitchResult(.{ .sparse_switch = .{ .src = 0, .payload_offset = 0, .payload = @constCast(&sparse_payload) } }, 99, 30);
    try expectSwitchResult(.{ .sparse_switch = .{ .src = 0, .payload_offset = 0, .payload = @constCast(&sparse_payload) } }, -11, 10);
    try expectSwitchResult(.{ .sparse_switch = .{ .src = 0, .payload_offset = 0, .payload = @constCast(&sparse_payload) } }, 100, 10);
    try expectSwitchResult(.{ .sparse_switch = .{ .src = 0, .payload_offset = 0, .payload = @constCast(&empty_payload) } }, 0, 10);
    try expectSwitchResult(.{ .sparse_switch = .{ .src = 0, .payload_offset = 0, .payload = null } }, 0, 10);
}

test "execute updates pc on return and error edges" {
    var regs = [_]u32{0} ** 4;
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = &regs,
        .instructions = &[_]Instruction{
            .nop,
            .return_void,
        },
    };
    _ = try execute(&frame);
    try std.testing.expectEqual(@as(u32, 2), frame.pc);

    frame.pc = 0;
    frame.instructions = &[_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 0 } },
        .{ .div_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
    };
    try std.testing.expectError(error.DivisionByZero, execute(&frame));
    try std.testing.expectEqual(@as(u32, 3), frame.pc);

    frame.pc = 0;
    frame.instructions = &[_]Instruction{.nop};
    try std.testing.expectError(error.UnexpectedEndOfCode, execute(&frame));
    try std.testing.expectEqual(@as(u32, 1), frame.pc);
}

test "execute remaining move const return and pc edge cases" {
    var regs = [_]u32{0} ** 12;
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = &regs,
        .instructions = &[_]Instruction{
            .{ .const_ = .{ .dest = 0, .value = std.math.minInt(i32) } },
            .{ .const_ = .{ .dest = 1, .value = std.math.maxInt(i32) } },
            .{ .const_wide = .{ .dest = 2, .value = std.math.minInt(i64) } },
            .{ .const_wide = .{ .dest = 4, .value = std.math.maxInt(i64) } },
            .{ .move_wide = .{ .dest = 5, .src = 4 } },
            .{ .move = .{ .dest = 7, .src = 0 } },
            .{ .move_object = .{ .dest = 8, .src = 1 } },
            .{ .return_ = .{ .src = 7 } },
        },
    };

    var result = try execute(&frame);
    try std.testing.expectEqual(ReturnType.single, result.kind);
    try std.testing.expectEqual(@as(u32, @bitCast(@as(i32, std.math.minInt(i32)))), result.value32);
    try std.testing.expectEqual(@as(u32, @bitCast(@as(i32, std.math.maxInt(i32)))), regs[8]);
    try std.testing.expectEqual(@as(i64, std.math.maxInt(i64)), frame.getLong(5));
    try std.testing.expectEqual(@as(u32, 8), frame.pc);

    frame.pc = 0;
    frame.instructions = &[_]Instruction{
        .{ .const_wide = .{ .dest = 0, .value = 0x1122334455667788 } },
        .{ .return_wide = .{ .src = 0 } },
    };
    result = try execute(&frame);
    try std.testing.expectEqual(ReturnType.wide, result.kind);
    try std.testing.expectEqual(@as(u64, 0x1122334455667788), result.value64);
    try std.testing.expectEqual(@as(u32, 2), frame.pc);

    frame.pc = 0;
    frame.instructions = &[_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 0x12345678 } },
        .{ .return_object = .{ .src = 0 } },
    };
    result = try execute(&frame);
    try std.testing.expectEqual(ReturnType.object, result.kind);
    try std.testing.expectEqual(@as(u32, 0x12345678), result.value32);
    try std.testing.expectEqual(@as(u32, 2), frame.pc);
}

test "execute remaining comparison equal less greater and nan cases" {
    var regs = [_]u32{0} ** 24;
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = &regs,
        .instructions = &.{},
    };

    frame.setFloat(0, 1.0);
    frame.setFloat(1, 2.0);
    frame.setDouble(2, 1.0);
    frame.setDouble(4, 2.0);
    frame.setLong(6, 5);
    frame.setLong(8, 5);
    frame.setLong(10, 6);
    frame.instructions = &[_]Instruction{
        .{ .cmpl_float = .{ .dest = 12, .src1 = 0, .src2 = 1 } },
        .{ .cmpg_float = .{ .dest = 13, .src1 = 0, .src2 = 0 } },
        .{ .cmpl_double = .{ .dest = 14, .src1 = 2, .src2 = 4 } },
        .{ .cmpg_double = .{ .dest = 15, .src1 = 2, .src2 = 2 } },
        .{ .cmp_long = .{ .dest = 16, .src1 = 6, .src2 = 8 } },
        .{ .cmp_long = .{ .dest = 17, .src1 = 10, .src2 = 6 } },
        .{ .cmp_long = .{ .dest = 18, .src1 = 6, .src2 = 10 } },
        .return_void,
    };
    _ = try execute(&frame);
    try std.testing.expectEqual(@as(i32, -1), frame.getInt(12));
    try std.testing.expectEqual(@as(i32, 0), frame.getInt(13));
    try std.testing.expectEqual(@as(i32, -1), frame.getInt(14));
    try std.testing.expectEqual(@as(i32, 0), frame.getInt(15));
    try std.testing.expectEqual(@as(i32, 0), frame.getInt(16));
    try std.testing.expectEqual(@as(i32, 1), frame.getInt(17));
    try std.testing.expectEqual(@as(i32, -1), frame.getInt(18));

    frame.pc = 0;
    frame.setDouble(2, std.math.nan(f64));
    frame.instructions = &[_]Instruction{
        .{ .double_to_int = .{ .dest = 0, .src = 2 } },
        .{ .double_to_long = .{ .dest = 4, .src = 2 } },
        .return_void,
    };
    _ = try execute(&frame);
    try std.testing.expectEqual(@as(i32, 0), frame.getInt(0));
    try std.testing.expectEqual(@as(i64, 0), frame.getLong(4));
}

test "execute remaining division remainder and shift sign edges" {
    var regs = [_]u32{0} ** 20;
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = &regs,
        .instructions = &.{},
    };

    frame.setInt(0, -8);
    frame.setInt(1, 3);
    frame.setLong(2, -9);
    frame.setLong(4, 4);
    frame.instructions = &[_]Instruction{
        .{ .div_int = .{ .dest = 6, .src1 = 0, .src2 = 1 } },
        .{ .rem_int = .{ .dest = 7, .src1 = 0, .src2 = 1 } },
        .{ .div_long = .{ .dest = 8, .src1 = 2, .src2 = 4 } },
        .{ .rem_long = .{ .dest = 10, .src1 = 2, .src2 = 4 } },
        .{ .shr_int = .{ .dest = 12, .src1 = 0, .src2 = 1 } },
        .{ .ushr_int = .{ .dest = 13, .src1 = 0, .src2 = 1 } },
        .{ .shr_long = .{ .dest = 14, .src1 = 2, .src2 = 1 } },
        .{ .ushr_long = .{ .dest = 16, .src1 = 2, .src2 = 1 } },
        .return_void,
    };
    _ = try execute(&frame);
    try std.testing.expectEqual(@as(i32, -2), frame.getInt(6));
    try std.testing.expectEqual(@as(i32, -2), frame.getInt(7));
    try std.testing.expectEqual(@as(i64, -2), frame.getLong(8));
    try std.testing.expectEqual(@as(i64, -1), frame.getLong(10));
    try std.testing.expectEqual(@as(i32, -1), frame.getInt(12));
    try std.testing.expectEqual(@as(u32, 0x1fffffff), regs[13]);
    try std.testing.expectEqual(@as(i64, -2), frame.getLong(14));
    try std.testing.expectEqual(@as(u64, 0x1ffffffffffffffe), frame.getWide(16));

    frame.pc = 0;
    frame.setInt(0, 1);
    frame.instructions = &[_]Instruction{.{ .div_int_lit8 = .{ .dest = 1, .src = 0, .lit = 0 } }};
    try std.testing.expectError(error.DivisionByZero, execute(&frame));

    frame.pc = 0;
    frame.instructions = &[_]Instruction{.{ .rem_int_lit16 = .{ .dest = 1, .src = 0, .lit = 0 } }};
    try std.testing.expectError(error.DivisionByZero, execute(&frame));
}

test "execute remaining floating point finite infinity and zero cases" {
    var regs = [_]u32{0} ** 32;
    var frame = ExecutionFrame{
        .pc = 0,
        .registers = &regs,
        .instructions = &.{},
    };

    frame.setFloat(0, 1.0);
    frame.setFloat(1, 0.0);
    frame.setFloat(2, -0.0);
    frame.setDouble(4, 1.0);
    frame.setDouble(6, 0.0);
    frame.setDouble(8, -0.0);
    frame.instructions = &[_]Instruction{
        .{ .div_float = .{ .dest = 10, .src1 = 0, .src2 = 1 } },
        .{ .div_float = .{ .dest = 11, .src1 = 0, .src2 = 2 } },
        .{ .rem_float = .{ .dest = 12, .src1 = 0, .src2 = 1 } },
        .{ .div_double = .{ .dest = 14, .src1 = 4, .src2 = 6 } },
        .{ .div_double = .{ .dest = 16, .src1 = 4, .src2 = 8 } },
        .{ .rem_double = .{ .dest = 18, .src1 = 4, .src2 = 6 } },
        .return_void,
    };
    _ = try execute(&frame);
    try std.testing.expect(std.math.isPositiveInf(frame.getFloat(10)));
    try std.testing.expect(std.math.isNegativeInf(frame.getFloat(11)));
    try std.testing.expect(std.math.isNan(frame.getFloat(12)));
    try std.testing.expect(std.math.isPositiveInf(frame.getDouble(14)));
    try std.testing.expect(std.math.isNegativeInf(frame.getDouble(16)));
    try std.testing.expect(std.math.isNan(frame.getDouble(18)));

    frame.pc = 0;
    frame.setFloat(0, @as(f32, @floatFromInt(std.math.maxInt(i32))) + 1024.0);
    frame.setFloat(1, @as(f32, @floatFromInt(std.math.minInt(i32))) - 1024.0);
    frame.setDouble(2, @as(f64, @floatFromInt(std.math.maxInt(i32))) + 1024.0);
    frame.setDouble(4, @as(f64, @floatFromInt(std.math.minInt(i32))) - 1024.0);
    frame.instructions = &[_]Instruction{
        .{ .float_to_int = .{ .dest = 6, .src = 0 } },
        .{ .float_to_int = .{ .dest = 7, .src = 1 } },
        .{ .double_to_int = .{ .dest = 8, .src = 2 } },
        .{ .double_to_int = .{ .dest = 9, .src = 4 } },
        .return_void,
    };
    _ = try execute(&frame);
    try std.testing.expectEqual(std.math.maxInt(i32), frame.getInt(6));
    try std.testing.expectEqual(std.math.minInt(i32), frame.getInt(7));
    try std.testing.expectEqual(std.math.maxInt(i32), frame.getInt(8));
    try std.testing.expectEqual(std.math.minInt(i32), frame.getInt(9));
}

test "execute remaining branch backward loop and false condition edges" {
    var regs = [_]u32{0} ** 6;
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 0 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .const_ = .{ .dest = 2, .value = 3 } },
        .{ .add_int = .{ .dest = 0, .src1 = 0, .src2 = 1 } },
        .{ .if_lt = .{ .src1 = 0, .src2 = 2, .offset = -1 } },
        .{ .return_ = .{ .src = 0 } },
    };
    const result = try testExecute(&insts, &regs);
    try std.testing.expectEqual(@as(u32, 3), result.value32);

    try expectBranchResult(.{ .if_ne = .{ .src1 = 0, .src2 = 1, .offset = 3 } }, &.{ 5, 5 }, 1);
    try expectBranchResult(.{ .if_lt = .{ .src1 = 0, .src2 = 1, .offset = 3 } }, &.{ 2, 1 }, 1);
    try expectBranchResult(.{ .if_ge = .{ .src1 = 0, .src2 = 1, .offset = 3 } }, &.{ 0, 1 }, 1);
    try expectBranchResult(.{ .if_gt = .{ .src1 = 0, .src2 = 1, .offset = 3 } }, &.{ 1, 1 }, 1);
    try expectBranchResult(.{ .if_le = .{ .src1 = 0, .src2 = 1, .offset = 3 } }, &.{ 2, 1 }, 1);
    try expectBranchResult(.{ .if_eqz = .{ .src = 0, .offset = 3 } }, &.{1}, 1);
    try expectBranchResult(.{ .if_nez = .{ .src = 0, .offset = 3 } }, &.{0}, 1);
    try expectBranchResult(.{ .if_ltz = .{ .src = 0, .offset = 3 } }, &.{0}, 1);
    try expectBranchResult(.{ .if_gez = .{ .src = 0, .offset = 3 } }, &.{@bitCast(@as(i32, -1))}, 1);
    try expectBranchResult(.{ .if_gtz = .{ .src = 0, .offset = 3 } }, &.{0}, 1);
    try expectBranchResult(.{ .if_lez = .{ .src = 0, .offset = 3 } }, &.{1}, 1);
}

test "execute reports unexpected end and unimplemented opcode" {
    var regs = [_]u32{0} ** 4;
    try std.testing.expectError(error.UnexpectedEndOfCode, testExecute(&[_]Instruction{.nop}, &regs));
    try std.testing.expectError(error.UnimplementedOpcode, testExecute(&[_]Instruction{.{ .new_instance = .{ .dest = 0, .type_idx = 1 } }}, &regs));
}
