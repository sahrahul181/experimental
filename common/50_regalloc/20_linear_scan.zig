//! Linear-scan physical register allocator.
//!
//! This pass uses linear scan over machine live intervals. It prefers
//! caller-saved registers for leaf code so the fast path has no callee-save
//! frame overhead, and it spills only under real register pressure.

const std = @import("std");
const intervals_mod = @import("intervals");
const machine = @import("machine_bridge");
const typedir = @import("typedir");

pub const Error = intervals_mod.Error || error{
    BadAllocation,
};

pub const RegClass = enum(u8) {
    gp,
    xmm,
};

pub const PhysReg = enum(u8) {
    rax,
    rcx,
    rdx,
    rsi,
    rdi,
    r8,
    r9,
    r10,
    r11,
    xmm0,
    xmm1,
    xmm2,
    xmm3,
    xmm4,
    xmm5,
    xmm6,
    xmm7,

    pub fn class(self: PhysReg) RegClass {
        return switch (self) {
            .rax, .rcx, .rdx, .rsi, .rdi, .r8, .r9, .r10, .r11 => .gp,
            .xmm0, .xmm1, .xmm2, .xmm3, .xmm4, .xmm5, .xmm6, .xmm7 => .xmm,
        };
    }
};

pub const Location = union(enum) {
    none,
    phys: PhysReg,
    spill: u32,

    pub fn isAssigned(self: Location) bool {
        return self != .none;
    }
};

pub const Options = struct {
    gp_registers: []const PhysReg = &DEFAULT_GP_REGS,
    xmm_registers: []const PhysReg = &DEFAULT_XMM_REGS,
    /// Values in this set must never share a physical register even when the
    /// coarse linear intervals appear disjoint. Deoptimization/OSR uses this
    /// to model simultaneous state capture at interior entry points.
    distinct_registers: []const machine.RegId = &.{},
    /// Values consumed by instruction encodings with a private fixed-scratch
    /// ABI may not spill. Ordinary live values remain eligible victims.
    must_registers: []const machine.RegId = &.{},
};

pub const Stats = struct {
    intervals: u32 = 0,
    assigned_phys: u32 = 0,
    spills: u32 = 0,
    spill_slots: u32 = 0,
    max_active: u32 = 0,
};

pub const Allocation = struct {
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    intervals: []intervals_mod.Interval,
    locations: []Location,
    stats: Stats,

    pub fn deinit(self: *Allocation) void {
        self.allocator.free(self.locations);
        self.allocator.free(self.intervals);
        self.* = undefined;
    }

    pub fn locationOf(self: *const Allocation, reg: machine.RegId) ?Location {
        if (reg >= self.locations.len) return null;
        return self.locations[reg];
    }

    pub fn verify(self: *const Allocation) Error!void {
        self.source.verify() catch return error.InvalidMachine;
        if (self.locations.len != self.source.reg_types.len) return error.BadAllocation;
        for (self.intervals) |interval| {
            if (interval.reg >= self.locations.len) return error.BadAllocation;
            const location = self.locations[interval.reg];
            if (!location.isAssigned()) return error.BadAllocation;
            switch (location) {
                .phys => |reg| {
                    if (reg.class() != classForType(interval.ty)) return error.BadAllocation;
                },
                .spill => {},
                .none => return error.BadAllocation,
            }
        }
    }

    pub fn print(self: *const Allocation, writer: anytype) !void {
        try writer.print(
            "linear_scan intervals={d} phys={d} spills={d} spill_slots={d} max_active={d}\n",
            .{
                self.stats.intervals,
                self.stats.assigned_phys,
                self.stats.spills,
                self.stats.spill_slots,
                self.stats.max_active,
            },
        );
        for (self.intervals) |interval| {
            try writer.print("  r{d}:{s} [{d},{d}] -> ", .{ interval.reg, @tagName(interval.ty), interval.start, interval.end });
            switch (self.locations[interval.reg]) {
                .phys => |reg| try writer.print("{s}\n", .{@tagName(reg)}),
                .spill => |slot| try writer.print("spill[{d}]\n", .{slot}),
                .none => try writer.print("<none>\n", .{}),
            }
        }
    }
};

const DEFAULT_GP_REGS = [_]PhysReg{ .r10, .r11, .rax, .rcx, .rdx, .r8, .r9 };
const DEFAULT_XMM_REGS = [_]PhysReg{ .xmm0, .xmm1, .xmm2, .xmm3, .xmm4, .xmm5, .xmm6, .xmm7 };

const Active = struct {
    reg: machine.RegId,
    end: intervals_mod.Position,
    phys: PhysReg,
};

fn classForType(ty: typedir.Type) RegClass {
    return switch (ty) {
        .float, .double => .xmm,
        else => .gp,
    };
}

fn lessThanStart(_: void, a: intervals_mod.Interval, b: intervals_mod.Interval) bool {
    if (a.start == b.start) return a.end < b.end;
    return a.start < b.start;
}

fn expireOld(active: *std.ArrayList(Active), current: intervals_mod.Interval) void {
    var i: usize = 0;
    while (i < active.items.len) {
        if (active.items[i].end >= current.start) {
            i += 1;
            continue;
        }
        _ = active.swapRemove(i);
    }
}

fn isActive(active: []const Active, reg: PhysReg) bool {
    for (active) |item| if (item.phys == reg) return true;
    return false;
}

fn firstFree(active: []const Active, regs: []const PhysReg) ?PhysReg {
    for (regs) |reg| {
        if (!isActive(active, reg)) return reg;
    }
    return null;
}

fn containsRegister(registers: []const machine.RegId, reg: machine.RegId) bool {
    for (registers) |candidate| if (candidate == reg) return true;
    return false;
}

fn farthestSpillable(active: []const Active, class: RegClass, must_registers: []const machine.RegId) ?usize {
    var best: ?usize = null;
    for (active, 0..) |item, i| {
        if (item.phys.class() != class or containsRegister(must_registers, item.reg)) continue;
        if (best == null or item.end > active[best.?].end) best = i;
    }
    return best;
}

fn registersForClass(options: Options, class: RegClass) []const PhysReg {
    return switch (class) {
        .gp => options.gp_registers,
        .xmm => options.xmm_registers,
    };
}

fn nextSpillSlot(stats: *Stats) u32 {
    const slot = stats.spill_slots;
    stats.spill_slots += 1;
    stats.spills += 1;
    return slot;
}

pub fn allocate(allocator: std.mem.Allocator, function: *const machine.Function, options: Options) Error!Allocation {
    var live = try intervals_mod.build(allocator, function);
    defer live.deinit();
    try live.verify();

    const intervals = try allocator.dupe(intervals_mod.Interval, live.intervals);
    errdefer allocator.free(intervals);
    for (options.must_registers, 0..) |reg, index| {
        if (reg >= function.reg_types.len) return error.BadAllocation;
        for (options.must_registers[0..index]) |previous| if (previous == reg) return error.BadAllocation;
    }
    if (options.distinct_registers.len != 0) {
        var group_start = intervals_mod.INVALID_POS;
        var group_end: intervals_mod.Position = 0;
        for (options.distinct_registers, 0..) |reg, index| {
            if (reg >= function.reg_types.len) return error.BadAllocation;
            for (options.distinct_registers[0..index]) |previous| if (previous == reg) return error.BadAllocation;
            var found = false;
            for (intervals) |interval| {
                if (interval.reg != reg) continue;
                group_start = @min(group_start, interval.start);
                group_end = @max(group_end, interval.end);
                found = true;
                break;
            }
            if (!found) return error.BadAllocation;
        }
        for (intervals) |*interval| {
            for (options.distinct_registers) |reg| {
                if (interval.reg != reg) continue;
                interval.start = group_start;
                interval.end = group_end;
                break;
            }
        }
    }
    std.mem.sort(intervals_mod.Interval, intervals, {}, lessThanStart);

    const locations = try allocator.alloc(Location, function.reg_types.len);
    errdefer allocator.free(locations);
    @memset(locations, .none);

    var active: std.ArrayList(Active) = .empty;
    defer active.deinit(allocator);

    var stats: Stats = .{ .intervals = @intCast(intervals.len) };
    for (intervals) |interval| {
        expireOld(&active, interval);
        const class = classForType(interval.ty);
        const regs = registersForClass(options, class);
        if (regs.len == 0) {
            if (containsRegister(options.must_registers, interval.reg)) return error.BadAllocation;
            locations[interval.reg] = .{ .spill = nextSpillSlot(&stats) };
            continue;
        }

        if (firstFree(active.items, regs)) |phys| {
            locations[interval.reg] = .{ .phys = phys };
            try active.append(allocator, .{ .reg = interval.reg, .end = interval.end, .phys = phys });
            stats.assigned_phys += 1;
        } else if (farthestSpillable(active.items, class, options.must_registers)) |spill_i| {
            if (containsRegister(options.must_registers, interval.reg) or active.items[spill_i].end > interval.end) {
                const victim = active.items[spill_i];
                locations[victim.reg] = .{ .spill = nextSpillSlot(&stats) };
                locations[interval.reg] = .{ .phys = victim.phys };
                active.items[spill_i] = .{ .reg = interval.reg, .end = interval.end, .phys = victim.phys };
                stats.assigned_phys += 1;
            } else {
                locations[interval.reg] = .{ .spill = nextSpillSlot(&stats) };
            }
        } else {
            if (containsRegister(options.must_registers, interval.reg)) return error.BadAllocation;
            locations[interval.reg] = .{ .spill = nextSpillSlot(&stats) };
        }
        stats.max_active = @max(stats.max_active, @as(u32, @intCast(active.items.len)));
    }

    var out = Allocation{
        .allocator = allocator,
        .source = function,
        .intervals = intervals,
        .locations = locations,
        .stats = stats,
    };
    try out.verify();
    return out;
}

pub fn allocateDefault(allocator: std.mem.Allocator, function: *const machine.Function) Error!Allocation {
    return allocate(allocator, function, .{});
}

test "linear_scan assigns arithmetic function to physical registers" {
    const optimizer = @import("optimizer");
    const Instruction = @import("instructions").Instruction;
    const insts = [_]Instruction{
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    var allocation = try allocateDefault(std.testing.allocator, &optimized.machine);
    defer allocation.deinit();
    try allocation.verify();
    try std.testing.expect(allocation.stats.assigned_phys >= 3);
    try std.testing.expectEqual(@as(u32, 0), allocation.stats.spills);
}

test "linear_scan spills under constrained register pressure" {
    const optimizer = @import("optimizer");
    const Instruction = @import("instructions").Instruction;
    const insts = [_]Instruction{
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .add_int = .{ .dest = 3, .src1 = 2, .src2 = 1 } },
        .{ .add_int = .{ .dest = 4, .src1 = 3, .src2 = 2 } },
        .{ .add_int = .{ .dest = 5, .src1 = 4, .src2 = 3 } },
        .{ .return_ = .{ .src = 5 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    const one_gp = [_]PhysReg{.r10};
    var allocation = try allocate(std.testing.allocator, &optimized.machine, .{ .gp_registers = &one_gp });
    defer allocation.deinit();
    try allocation.verify();
    try std.testing.expect(allocation.stats.spills > 0);
}

test "linear_scan preserves must-register values by spilling eligible victims" {
    const optimizer = @import("optimizer");
    const Instruction = @import("instructions").Instruction;
    const insts = [_]Instruction{
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .add_int = .{ .dest = 3, .src1 = 2, .src2 = 0 } },
        .{ .return_ = .{ .src = 3 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    const one_gp = [_]PhysReg{.r10};
    const required = [_]machine.RegId{0};
    var allocation = try allocate(std.testing.allocator, &optimized.machine, .{
        .gp_registers = &one_gp,
        .must_registers = &required,
    });
    defer allocation.deinit();
    try std.testing.expectEqual(Location{ .phys = .r10 }, allocation.locationOf(0).?);
    try std.testing.expect(allocation.stats.spills > 0);

    const impossible = [_]machine.RegId{ 0, 1 };
    try std.testing.expectError(error.BadAllocation, allocate(std.testing.allocator, &optimized.machine, .{
        .gp_registers = &one_gp,
        .must_registers = &impossible,
    }));
}

test "linear_scan print helper emits stable summary" {
    const optimizer = @import("optimizer");
    const Instruction = @import("instructions").Instruction;
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .return_ = .{ .src = 0 } },
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    var allocation = try allocateDefault(std.testing.allocator, &optimized.machine);
    defer allocation.deinit();

    var storage: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&storage);
    try allocation.print(&stream);
    const output = stream.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "linear_scan intervals=") != null);
}
