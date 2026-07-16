//! Immutable deoptimization metadata and allocation-free frame reconstruction.
//!
//! Metadata is compiler-owned until copied into Table. Runtime reconstruction
//! first stages every source into caller-provided scratch storage and mutates
//! the interpreter frame only after all reads succeed. A failed handoff is
//! therefore transactional and cannot publish a partially rebuilt frame.

const std = @import("std");
const interpreter = @import("interpreter");
const runtime_value = @import("runtime_value");
const runtime_stack_map = @import("runtime_stack_map");

const Handle = runtime_value.Handle;

pub const Reason = enum(u8) {
    invalidation,
    uncommon_trap,
    exception,
};

pub const ValueKind = enum(u8) {
    scalar32,
    scalar64,
    reference,

    pub fn registerWidth(self: ValueKind) u2 {
        return switch (self) {
            .scalar32, .reference => 1,
            .scalar64 => 2,
        };
    }

    fn byteWidth(self: ValueKind) u8 {
        return switch (self) {
            .scalar32 => 4,
            .scalar64, .reference => 8,
        };
    }
};

pub const Source = union(enum) {
    native_register: u8,
    stack_slot: i32,
    constant: u64,
};

pub const ValueSpec = struct {
    vreg: u16,
    kind: ValueKind,
    source: Source,
};

pub const PointSpec = struct {
    id: u32,
    method_id: u32,
    dex_pc: u32,
    values: []const ValueSpec,
};

pub const ValidationOptions = struct {
    register_count: u16,
    native_register_count: u8,
    max_dex_pc: u32,
    stack_alignment: u8 = 4,
};

pub const Record = struct {
    id: u32,
    method_id: u32,
    dex_pc: u32,
    first_value: u32,
    value_count: u16,
    reserved: u16 = 0,
};

pub const ExceptionState = extern struct {
    kind: u32 = 0,
    dex_pc: u32 = 0,
    payload0: u64 = 0,
    payload1: u64 = 0,

    pub fn isPending(self: ExceptionState) bool {
        return self.kind != 0;
    }
};

/// Interpreter-visible frame plus the managed chain state absent from the
/// frontend's compact ExecutionFrame. All slices are preallocated by the VM.
pub const Frame = struct {
    method_id: u32 = 0,
    execution: interpreter.ExecutionFrame,
    previous: ?*Frame = null,
    exception: ExceptionState = .{},
    reason: Reason = .invalidation,
    active: bool = false,
};

pub const Destination = struct {
    frame: *Frame,
    scratch: []u64,
    previous: ?*Frame = null,
};

pub const Capture = struct {
    native_registers: []const u64,
    stack_base: [*]const u8,
    stack_min_offset: i32,
    /// Exclusive upper bound.
    stack_max_offset: i32,
};

pub const ResumeFn = *const fn (*anyopaque, *Frame) usize;

/// Owner-confined request installed immediately before entering a managed
/// trampoline. The pointed-to table, frame buffers, and callback must outlive
/// that call. No field is accessed by another thread.
pub const Request = struct {
    table: *const Table,
    point_id: u32,
    reason: Reason = .invalidation,
    exception: ExceptionState = .{},
    destination: Destination,
    stack_base: [*]const u8,
    stack_min_offset: i32 = 0,
    stack_max_offset: i32 = 0,
    resume_context: *anyopaque,
    resume_fn: ResumeFn,
};

pub const Error = error{
    DuplicatePoint,
    DuplicateRegister,
    EmptyTable,
    IncompleteFrame,
    InvalidConstant,
    InvalidDexPc,
    InvalidDestination,
    InvalidNativeRegister,
    InvalidReference,
    InvalidStackAlignment,
    MissingPoint,
    MissingDeoptMap,
    OutOfBounds,
    TooManyValues,
    UnsortedPoints,
};

pub const Table = struct {
    allocator: std.mem.Allocator,
    records: []Record,
    values: []ValueSpec,
    register_count: u16,
    native_register_count: u8,

    pub fn init(
        allocator: std.mem.Allocator,
        specs: []const PointSpec,
        options: ValidationOptions,
    ) (Error || std.mem.Allocator.Error)!Table {
        if (specs.len == 0 or options.register_count == 0) return error.EmptyTable;
        if (options.stack_alignment == 0 or !std.math.isPowerOfTwo(options.stack_alignment)) {
            return error.InvalidStackAlignment;
        }

        var total_values: usize = 0;
        for (specs, 0..) |spec, point_index| {
            if (spec.dex_pc > options.max_dex_pc) return error.InvalidDexPc;
            if (point_index > 0) {
                const previous = specs[point_index - 1].id;
                if (previous == spec.id) return error.DuplicatePoint;
                if (previous > spec.id) return error.UnsortedPoints;
            }
            if (spec.values.len > std.math.maxInt(u16) or
                spec.values.len > std.math.maxInt(u32) - total_values) return error.TooManyValues;
            total_values += spec.values.len;
            try validatePoint(spec, options);
        }

        const records = try allocator.alloc(Record, specs.len);
        errdefer allocator.free(records);
        const values = try allocator.alloc(ValueSpec, total_values);
        errdefer allocator.free(values);

        var cursor: usize = 0;
        for (specs, 0..) |spec, index| {
            records[index] = .{
                .id = spec.id,
                .method_id = spec.method_id,
                .dex_pc = spec.dex_pc,
                .first_value = @intCast(cursor),
                .value_count = @intCast(spec.values.len),
            };
            @memcpy(values[cursor..][0..spec.values.len], spec.values);
            cursor += spec.values.len;
        }
        return .{
            .allocator = allocator,
            .records = records,
            .values = values,
            .register_count = options.register_count,
            .native_register_count = options.native_register_count,
        };
    }

    pub fn deinit(self: *Table) void {
        self.allocator.free(self.values);
        self.allocator.free(self.records);
        self.* = undefined;
    }

    pub fn find(self: *const Table, id: u32) Error!*const Record {
        var low: usize = 0;
        var high = self.records.len;
        while (low < high) {
            const mid = low + (high - low) / 2;
            const candidate = &self.records[mid];
            if (candidate.id < id) {
                low = mid + 1;
            } else if (candidate.id > id) {
                high = mid;
            } else {
                return candidate;
            }
        }
        return error.MissingPoint;
    }

    pub fn valuesFor(self: *const Table, record: *const Record) []const ValueSpec {
        const first: usize = record.first_value;
        return self.values[first .. first + record.value_count];
    }

    pub fn reconstruct(
        self: *const Table,
        point_id: u32,
        capture: Capture,
        destination: Destination,
        reason: Reason,
        exception: ExceptionState,
    ) Error!*Frame {
        const record = try self.find(point_id);
        const values = self.valuesFor(record);
        const frame = destination.frame;
        if (capture.native_registers.len < self.native_register_count or
            destination.scratch.len < values.len or
            frame.execution.registers.len != self.register_count or
            frame.execution.reference_registers.len != self.register_count or
            frame.execution.register_is_ref.len != self.register_count)
        {
            return error.InvalidDestination;
        }

        // Stage first. No interpreter-visible state changes before every
        // native/stack source has passed bounds and reference validation.
        for (values, 0..) |value, index| {
            const bits = try readSource(value, capture);
            if (value.kind == .reference and !validHandleBits(bits)) return error.InvalidReference;
            destination.scratch[index] = bits;
        }

        @memset(frame.execution.registers, 0);
        @memset(frame.execution.reference_registers, @as(u64, @bitCast(Handle.none)));
        @memset(frame.execution.register_is_ref, false);
        for (values, 0..) |value, index| {
            const bits = destination.scratch[index];
            switch (value.kind) {
                .scalar32 => frame.execution.registers[value.vreg] = @truncate(bits),
                .scalar64 => {
                    frame.execution.registers[value.vreg] = @truncate(bits);
                    frame.execution.registers[value.vreg + 1] = @truncate(bits >> 32);
                },
                .reference => {
                    frame.execution.registers[value.vreg] = @truncate(bits);
                    frame.execution.reference_registers[value.vreg] = bits;
                    frame.execution.register_is_ref[value.vreg] = true;
                },
            }
        }
        frame.execution.pc = record.dex_pc;
        frame.method_id = record.method_id;
        frame.previous = destination.previous;
        frame.exception = exception;
        frame.reason = reason;
        frame.active = true;
        return frame;
    }

    /// Visits only canonical reference slots named by a deoptimization point.
    /// This lets a managed entry publish incoming handles before a handshake
    /// without treating scalar register contents as conservative roots.
    pub fn visitReferenceSlots(
        self: *const Table,
        point_id: u32,
        capture: Capture,
        context: anytype,
        visitor: anytype,
    ) !void {
        const record = try self.find(point_id);
        for (self.valuesFor(record)) |value| {
            if (value.kind != .reference) continue;
            switch (value.source) {
                .native_register => |register| {
                    if (register >= capture.native_registers.len) return error.OutOfBounds;
                    try visitor(context, @as(*const Handle, @ptrCast(&capture.native_registers[register])));
                },
                .stack_slot => |offset| {
                    const address = try stackAddress(capture, offset, @sizeOf(Handle));
                    try visitor(context, @as(*const Handle, @ptrFromInt(address)));
                },
                .constant => |bits| if (bits != @as(u64, @bitCast(Handle.none))) return error.InvalidConstant,
            }
        }
    }

    /// Validates the immutable cross-link used by a compiled safepoint. When
    /// `require_all` is true every stack-map record must reconstruct; entry
    /// maps may pass false and retain GC-only safepoints.
    pub fn validateStackMaps(
        self: *const Table,
        stack_maps: *const runtime_stack_map.Table,
        require_all: bool,
    ) Error!void {
        for (stack_maps.records) |record| {
            if (record.deopt_id == runtime_stack_map.no_deopt) {
                if (require_all) return error.MissingDeoptMap;
                continue;
            }
            _ = self.find(record.deopt_id) catch return error.MissingDeoptMap;
        }
    }
};

fn validatePoint(spec: PointSpec, options: ValidationOptions) Error!void {
    for (spec.values) |value| {
        const width = value.kind.registerWidth();
        if (value.vreg >= options.register_count or width > options.register_count - value.vreg) {
            return error.OutOfBounds;
        }
        switch (value.source) {
            .native_register => |register| if (register >= options.native_register_count) return error.InvalidNativeRegister,
            .stack_slot => |offset| if (@mod(offset, @as(i32, options.stack_alignment)) != 0) return error.InvalidStackAlignment,
            .constant => |bits| if (value.kind == .reference and bits != @as(u64, @bitCast(Handle.none))) return error.InvalidConstant,
        }
    }

    var register: u16 = 0;
    while (register < options.register_count) : (register += 1) {
        var coverage: u8 = 0;
        for (spec.values) |value| {
            const end = value.vreg + value.kind.registerWidth();
            if (register >= value.vreg and register < end) coverage += 1;
        }
        if (coverage == 0) return error.IncompleteFrame;
        if (coverage > 1) return error.DuplicateRegister;
    }
}

fn readSource(value: ValueSpec, capture: Capture) Error!u64 {
    return switch (value.source) {
        .native_register => |register| if (register < capture.native_registers.len)
            capture.native_registers[register]
        else
            error.OutOfBounds,
        .constant => |bits| bits,
        .stack_slot => |offset| readStack(capture, offset, value.kind.byteWidth()),
    };
}

fn readStack(capture: Capture, offset: i32, width: u8) Error!u64 {
    const address = try stackAddress(capture, offset, width);
    const bytes: [*]const u8 = @ptrFromInt(address);
    return switch (width) {
        4 => std.mem.readInt(u32, bytes[0..4], .little),
        8 => std.mem.readInt(u64, bytes[0..8], .little),
        else => unreachable,
    };
}

fn stackAddress(capture: Capture, offset: i32, width: u8) Error!usize {
    const end = @as(i64, offset) + width;
    if (offset < capture.stack_min_offset or end > capture.stack_max_offset) return error.OutOfBounds;
    const base = @intFromPtr(capture.stack_base);
    const address = if (offset >= 0) blk: {
        const magnitude: usize = @intCast(offset);
        if (magnitude > std.math.maxInt(usize) - base) return error.OutOfBounds;
        break :blk base + magnitude;
    } else blk: {
        const magnitude: usize = @intCast(-@as(i64, offset));
        if (magnitude > base) return error.OutOfBounds;
        break :blk base - magnitude;
    };
    if (!std.mem.isAligned(address, @as(usize, @min(width, @as(u8, 8))))) return error.InvalidStackAlignment;
    return address;
}

fn validHandleBits(bits: u64) bool {
    const handle: Handle = @bitCast(bits);
    return bits == @as(u64, @bitCast(Handle.none)) or (!handle.isNull() and handle.generation != 0);
}

const test_options = ValidationOptions{
    .register_count = 4,
    .native_register_count = 16,
    .max_dex_pc = 32,
};

test "deoptimization reconstructs exact values references exception and frame chain" {
    const handle = Handle{ .index = 7, .generation = 3 };
    const values = [_]ValueSpec{
        .{ .vreg = 0, .kind = .scalar32, .source = .{ .native_register = 1 } },
        .{ .vreg = 1, .kind = .scalar64, .source = .{ .stack_slot = 0 } },
        .{ .vreg = 3, .kind = .reference, .source = .{ .native_register = 2 } },
    };
    const points = [_]PointSpec{.{ .id = 9, .method_id = 41, .dex_pc = 2, .values = &values }};
    var table = try Table.init(std.testing.allocator, &points, test_options);
    defer table.deinit();

    var native: [16]u64 = @splat(0);
    native[1] = 0xaabbccdd;
    native[2] = @bitCast(handle);
    var stack = [_]u64{0x8877665544332211};
    var registers: [4]u32 = @splat(0xeeeeeeee);
    var references: [4]u64 = @splat(0xdddddddddddddddd);
    var reference_kinds: [4]bool = @splat(true);
    var previous_registers: [1]u32 = .{1};
    var previous = Frame{ .execution = .{ .pc = 1, .registers = &previous_registers, .instructions = &.{} } };
    var frame = Frame{ .execution = .{
        .pc = 0,
        .registers = &registers,
        .instructions = &.{},
        .register_is_ref = &reference_kinds,
        .reference_registers = &references,
    } };
    var scratch: [3]u64 = undefined;
    const exception = ExceptionState{ .kind = 1, .dex_pc = 2, .payload0 = 17, .payload1 = 5 };
    const rebuilt = try table.reconstruct(9, .{
        .native_registers = &native,
        .stack_base = @ptrCast(&stack[0]),
        .stack_min_offset = 0,
        .stack_max_offset = @sizeOf(@TypeOf(stack)),
    }, .{ .frame = &frame, .scratch = &scratch, .previous = &previous }, .exception, exception);

    try std.testing.expect(rebuilt.active);
    try std.testing.expectEqual(@as(u32, 41), rebuilt.method_id);
    try std.testing.expectEqual(@as(u32, 2), rebuilt.execution.pc);
    try std.testing.expectEqual(@as(u32, 0xaabbccdd), rebuilt.execution.registers[0]);
    try std.testing.expectEqual(@as(u64, 0x8877665544332211), rebuilt.execution.getWide(1));
    try std.testing.expectEqual(@as(u64, @bitCast(handle)), rebuilt.execution.reference_registers[3]);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, &reference_kinds);
    try std.testing.expect(rebuilt.previous == &previous);
    try std.testing.expectEqual(exception, rebuilt.exception);
}

test "deoptimization metadata rejects incomplete overlapping and unsafe frames" {
    const incomplete = [_]ValueSpec{.{ .vreg = 0, .kind = .scalar32, .source = .{ .constant = 0 } }};
    try std.testing.expectError(error.IncompleteFrame, Table.init(
        std.testing.allocator,
        &.{.{ .id = 1, .method_id = 0, .dex_pc = 0, .values = &incomplete }},
        test_options,
    ));

    const overlap = [_]ValueSpec{
        .{ .vreg = 0, .kind = .scalar64, .source = .{ .constant = 0 } },
        .{ .vreg = 1, .kind = .scalar32, .source = .{ .constant = 0 } },
        .{ .vreg = 2, .kind = .scalar64, .source = .{ .constant = 0 } },
    };
    try std.testing.expectError(error.DuplicateRegister, Table.init(
        std.testing.allocator,
        &.{.{ .id = 1, .method_id = 0, .dex_pc = 0, .values = &overlap }},
        test_options,
    ));

    const unsafe_reference = [_]ValueSpec{
        .{ .vreg = 0, .kind = .reference, .source = .{ .constant = 1 } },
        .{ .vreg = 1, .kind = .scalar64, .source = .{ .constant = 0 } },
        .{ .vreg = 3, .kind = .scalar32, .source = .{ .constant = 0 } },
    };
    try std.testing.expectError(error.InvalidConstant, Table.init(
        std.testing.allocator,
        &.{.{ .id = 1, .method_id = 0, .dex_pc = 0, .values = &unsafe_reference }},
        test_options,
    ));
}

test "failed deoptimization capture leaves destination unchanged" {
    const values = [_]ValueSpec{
        .{ .vreg = 0, .kind = .scalar64, .source = .{ .stack_slot = 8 } },
        .{ .vreg = 2, .kind = .scalar64, .source = .{ .constant = 0 } },
    };
    var table = try Table.init(std.testing.allocator, &.{.{
        .id = 1,
        .method_id = 0,
        .dex_pc = 0,
        .values = &values,
    }}, test_options);
    defer table.deinit();
    var native: [16]u64 = @splat(0);
    var stack: [1]u64 = .{1};
    var registers: [4]u32 = @splat(0xfeedbeef);
    var references: [4]u64 = @splat(0xabababababababab);
    var reference_kinds: [4]bool = @splat(true);
    var frame = Frame{ .execution = .{
        .pc = 7,
        .registers = &registers,
        .instructions = &.{},
        .register_is_ref = &reference_kinds,
        .reference_registers = &references,
    } };
    var scratch: [2]u64 = undefined;
    try std.testing.expectError(error.OutOfBounds, table.reconstruct(1, .{
        .native_registers = &native,
        .stack_base = @ptrCast(&stack[0]),
        .stack_min_offset = 0,
        .stack_max_offset = @sizeOf(@TypeOf(stack)),
    }, .{ .frame = &frame, .scratch = &scratch }, .invalidation, .{}));
    try std.testing.expectEqual(@as(u32, 7), frame.execution.pc);
    try std.testing.expectEqualSlices(u32, &.{ 0xfeedbeef, 0xfeedbeef, 0xfeedbeef, 0xfeedbeef }, &registers);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true }, &reference_kinds);
}

test "deoptimization links stack maps and visits only canonical reference slots" {
    const handle = Handle{ .index = 11, .generation = 4 };
    const values = [_]ValueSpec{
        .{ .vreg = 0, .kind = .scalar32, .source = .{ .native_register = 1 } },
        .{ .vreg = 1, .kind = .reference, .source = .{ .native_register = 2 } },
        .{ .vreg = 2, .kind = .scalar64, .source = .{ .constant = 9 } },
    };
    var table = try Table.init(std.testing.allocator, &.{.{
        .id = 7,
        .method_id = 3,
        .dex_pc = 1,
        .values = &values,
    }}, test_options);
    defer table.deinit();

    const root = [_]runtime_stack_map.RootLocation{runtime_stack_map.RootLocation.nativeRegister(2)};
    var linked_maps = try runtime_stack_map.Table.init(std.testing.allocator, &.{.{
        .pc_offset = 5,
        .roots = &root,
        .deopt_id = 7,
    }}, .{
        .native_register_count = 16,
        .interpreter_register_count = 0,
        .max_frame_depth = 0,
        .max_shadow_roots = 0,
    });
    defer linked_maps.deinit();
    try table.validateStackMaps(&linked_maps, true);

    var native: [16]u64 = @splat(0);
    native[1] = 0xdeadbeef;
    native[2] = @bitCast(handle);
    var anchor: u64 = 0;
    var visited = std.ArrayList(u64).empty;
    defer visited.deinit(std.testing.allocator);
    const Visitor = struct {
        fn add(list: *std.ArrayList(u64), slot: *const Handle) !void {
            try list.append(std.testing.allocator, @bitCast(slot.*));
        }
    };
    try table.visitReferenceSlots(7, .{
        .native_registers = &native,
        .stack_base = @ptrCast(&anchor),
        .stack_min_offset = 0,
        .stack_max_offset = @sizeOf(@TypeOf(anchor)),
    }, &visited, Visitor.add);
    try std.testing.expectEqualSlices(u64, &.{@as(u64, @bitCast(handle))}, visited.items);

    linked_maps.records[0].deopt_id = 99;
    try std.testing.expectError(error.MissingDeoptMap, table.validateStackMaps(&linked_maps, true));
}

fn allocationFailureProbe(allocator: std.mem.Allocator) !void {
    const values = [_]ValueSpec{
        .{ .vreg = 0, .kind = .scalar64, .source = .{ .constant = 1 } },
        .{ .vreg = 2, .kind = .scalar64, .source = .{ .constant = 2 } },
    };
    var table = try Table.init(allocator, &.{.{ .id = 1, .method_id = 2, .dex_pc = 3, .values = &values }}, test_options);
    defer table.deinit();
}

test "deoptimization table construction is allocation-failure safe" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{});
}
