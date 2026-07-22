//! Immutable precise root maps shared by the interpreter, JIT, and collector.

const std = @import("std");
const Handle = @import("runtime_value").Handle;

pub const no_deopt: u32 = std.math.maxInt(u32);

pub const LocationKind = enum(u8) {
    native_register,
    stack_slot,
    interpreter_register,
    shadow_slot,
};

/// Compact, architecture-neutral root location. Runtime lookup never allocates.
pub const RootLocation = packed struct(u64) {
    payload: u32,
    frame_depth: u16 = 0,
    kind: LocationKind,
    flags: u8 = 0,

    pub fn nativeRegister(register: u16) RootLocation {
        return .{ .payload = register, .kind = .native_register };
    }

    pub fn stackSlot(byte_offset: i32) RootLocation {
        return .{ .payload = @bitCast(byte_offset), .kind = .stack_slot };
    }

    pub fn interpreterRegister(frame_depth: u16, register: u16) RootLocation {
        return .{
            .payload = register,
            .frame_depth = frame_depth,
            .kind = .interpreter_register,
        };
    }

    pub fn shadowSlot(index: u32) RootLocation {
        return .{ .payload = index, .kind = .shadow_slot };
    }

    pub fn stackOffset(self: RootLocation) i32 {
        std.debug.assert(self.kind == .stack_slot);
        return @bitCast(self.payload);
    }

    pub inline fn bits(self: RootLocation) u64 {
        return @bitCast(self);
    }
};

pub const MapSpec = struct {
    pc_offset: u32,
    roots: []const RootLocation,
    deopt_id: ?u32 = null,
};

pub const ValidationOptions = struct {
    native_register_count: u16,
    interpreter_register_count: u16,
    max_frame_depth: u16,
    max_shadow_roots: u32,
    stack_alignment: u8 = @sizeOf(Handle),
};

pub const Record = struct {
    pc_offset: u32,
    first_root: u32,
    root_count: u16,
    reserved: u16 = 0,
    deopt_id: u32,
};

pub const Error = error{
    DuplicateRequiredSafepoint,
    DuplicateLocation,
    EmptyTable,
    InvalidFlags,
    InvalidLocation,
    InvalidStackAlignment,
    MissingSafepoint,
    OutOfBounds,
    UnexpectedSafepoint,
    TooManyLocations,
    UnsortedSafepoints,
};

pub const Table = struct {
    allocator: std.mem.Allocator,
    records: []Record,
    locations: []RootLocation,

    pub fn init(
        allocator: std.mem.Allocator,
        specs: []const MapSpec,
        options: ValidationOptions,
    ) (Error || std.mem.Allocator.Error)!Table {
        if (specs.len == 0) return error.EmptyTable;
        if (options.stack_alignment == 0 or !std.math.isPowerOfTwo(options.stack_alignment)) {
            return error.InvalidStackAlignment;
        }

        var total_roots: usize = 0;
        for (specs, 0..) |spec, map_index| {
            if (map_index > 0 and specs[map_index - 1].pc_offset >= spec.pc_offset) {
                return error.UnsortedSafepoints;
            }
            if (spec.roots.len > std.math.maxInt(u16)) return error.TooManyLocations;
            if (spec.roots.len > std.math.maxInt(u32) - total_roots) return error.TooManyLocations;
            total_roots += spec.roots.len;

            for (spec.roots, 0..) |location, location_index| {
                try validateLocation(location, options);
                for (spec.roots[0..location_index]) |previous| {
                    if (previous.bits() == location.bits()) return error.DuplicateLocation;
                }
            }
        }

        const records = try allocator.alloc(Record, specs.len);
        errdefer allocator.free(records);
        const locations = try allocator.alloc(RootLocation, total_roots);
        errdefer allocator.free(locations);

        var cursor: usize = 0;
        for (specs, 0..) |spec, index| {
            records[index] = .{
                .pc_offset = spec.pc_offset,
                .first_root = @intCast(cursor),
                .root_count = @intCast(spec.roots.len),
                .deopt_id = spec.deopt_id orelse no_deopt,
            };
            @memcpy(locations[cursor..][0..spec.roots.len], spec.roots);
            cursor += spec.roots.len;
        }

        return .{
            .allocator = allocator,
            .records = records,
            .locations = locations,
        };
    }

    /// Constructs a table and transactionally proves exact coverage of the
    /// caller's sorted reachable-CFG safepoint manifest.
    pub fn initCovered(
        allocator: std.mem.Allocator,
        specs: []const MapSpec,
        required_sites: []const u32,
        options: ValidationOptions,
    ) (Error || std.mem.Allocator.Error)!Table {
        var table = try init(allocator, specs, options);
        errdefer table.deinit();
        try table.verifyCoverage(required_sites);
        return table;
    }

    pub fn deinit(self: *Table) void {
        self.allocator.free(self.locations);
        self.allocator.free(self.records);
        self.* = undefined;
    }

    pub fn find(self: *const Table, pc_offset: u32) Error!*const Record {
        var low: usize = 0;
        var high: usize = self.records.len;
        while (low < high) {
            const mid = low + (high - low) / 2;
            const candidate = &self.records[mid];
            if (candidate.pc_offset < pc_offset) {
                low = mid + 1;
            } else if (candidate.pc_offset > pc_offset) {
                high = mid;
            } else {
                return candidate;
            }
        }
        return error.MissingSafepoint;
    }

    /// Rejects both missing maps and orphan maps. Required ids must be strictly
    /// increasing, matching the dense manifest proven from reachable machine
    /// blocks; comparison is linear and allocation-free.
    pub fn verifyCoverage(self: *const Table, required_sites: []const u32) Error!void {
        for (required_sites, 0..) |site, index| {
            if (index != 0 and required_sites[index - 1] >= site) return error.DuplicateRequiredSafepoint;
        }
        var required_index: usize = 0;
        var record_index: usize = 0;
        while (required_index < required_sites.len and record_index < self.records.len) {
            const required = required_sites[required_index];
            const actual = self.records[record_index].pc_offset;
            if (actual < required) return error.UnexpectedSafepoint;
            if (actual > required) return error.MissingSafepoint;
            required_index += 1;
            record_index += 1;
        }
        if (required_index != required_sites.len) return error.MissingSafepoint;
        if (record_index != self.records.len) return error.UnexpectedSafepoint;
    }

    pub fn rootsFor(self: *const Table, record: *const Record) []const RootLocation {
        const first: usize = record.first_root;
        return self.locations[first .. first + record.root_count];
    }

    pub fn visitHandles(
        self: *const Table,
        record: *const Record,
        frame: FrameView,
        context: anytype,
        visitor: anytype,
    ) !void {
        for (self.rootsFor(record)) |location| {
            try visitor(context, try frame.read(location));
        }
    }
};

pub const FrameView = struct {
    native_registers: []const Handle,
    stack_base: [*]const u8,
    stack_min_offset: i32,
    stack_max_offset: i32,
    interpreter_frames: []const []const Handle,
    shadow_roots: []const Handle,

    pub fn read(self: FrameView, location: RootLocation) Error!Handle {
        switch (location.kind) {
            .native_register => {
                if (location.payload >= self.native_registers.len) return error.OutOfBounds;
                return self.native_registers[location.payload];
            },
            .stack_slot => {
                const offset = location.stackOffset();
                if (offset < self.stack_min_offset or offset > self.stack_max_offset) {
                    return error.OutOfBounds;
                }
                const base = @intFromPtr(self.stack_base);
                const address = if (offset >= 0) blk: {
                    const magnitude: usize = @intCast(offset);
                    if (magnitude > std.math.maxInt(usize) - base) return error.OutOfBounds;
                    break :blk base + magnitude;
                } else blk: {
                    const magnitude: usize = @intCast(-@as(i64, offset));
                    if (magnitude > base) return error.OutOfBounds;
                    break :blk base - magnitude;
                };
                if (!std.mem.isAligned(address, @alignOf(Handle))) return error.InvalidLocation;
                return @as(*const Handle, @ptrFromInt(address)).*;
            },
            .interpreter_register => {
                if (location.frame_depth >= self.interpreter_frames.len) return error.OutOfBounds;
                const registers = self.interpreter_frames[location.frame_depth];
                if (location.payload >= registers.len) return error.OutOfBounds;
                return registers[location.payload];
            },
            .shadow_slot => {
                if (location.payload >= self.shadow_roots.len) return error.OutOfBounds;
                return self.shadow_roots[location.payload];
            },
        }
    }
};

fn validateLocation(location: RootLocation, options: ValidationOptions) Error!void {
    if (location.flags != 0) return error.InvalidFlags;
    switch (location.kind) {
        .native_register => {
            if (location.payload >= options.native_register_count) return error.InvalidLocation;
            if (location.frame_depth != 0) return error.InvalidLocation;
        },
        .stack_slot => {
            if (location.frame_depth != 0) return error.InvalidLocation;
            const offset = location.stackOffset();
            if (@mod(offset, @as(i32, options.stack_alignment)) != 0) {
                return error.InvalidStackAlignment;
            }
        },
        .interpreter_register => {
            if (location.frame_depth > options.max_frame_depth) return error.InvalidLocation;
            if (location.payload >= options.interpreter_register_count) return error.InvalidLocation;
        },
        .shadow_slot => {
            if (location.frame_depth != 0) return error.InvalidLocation;
            if (location.payload >= options.max_shadow_roots) return error.InvalidLocation;
        },
    }
}

const test_options = ValidationOptions{
    .native_register_count = 16,
    .interpreter_register_count = 32,
    .max_frame_depth = 8,
    .max_shadow_roots = 16,
};

test "root location stays one machine word" {
    try std.testing.expectEqual(@as(usize, @sizeOf(u64)), @sizeOf(RootLocation));
}

test "table validates and performs exact binary lookup" {
    const roots_a = [_]RootLocation{
        RootLocation.nativeRegister(2),
        RootLocation.stackSlot(-16),
    };
    const roots_b = [_]RootLocation{
        RootLocation.interpreterRegister(1, 7),
        RootLocation.shadowSlot(3),
    };
    const specs = [_]MapSpec{
        .{ .pc_offset = 12, .roots = &roots_a },
        .{ .pc_offset = 44, .roots = &roots_b, .deopt_id = 9 },
    };

    var table = try Table.init(std.testing.allocator, &specs, test_options);
    defer table.deinit();

    const record = try table.find(44);
    try std.testing.expectEqual(@as(u32, 9), record.deopt_id);
    try std.testing.expectEqualSlices(RootLocation, &roots_b, table.rootsFor(record));
    try std.testing.expectError(error.MissingSafepoint, table.find(43));
}

test "covered table rejects missing orphan and duplicate CFG sites" {
    const empty_roots = [_]RootLocation{};
    const specs = [_]MapSpec{
        .{ .pc_offset = 2, .roots = &empty_roots },
        .{ .pc_offset = 5, .roots = &empty_roots },
        .{ .pc_offset = 9, .roots = &empty_roots },
    };
    var table = try Table.initCovered(std.testing.allocator, &specs, &.{ 2, 5, 9 }, test_options);
    defer table.deinit();
    try table.verifyCoverage(&.{ 2, 5, 9 });
    try std.testing.expectError(error.MissingSafepoint, table.verifyCoverage(&.{ 2, 4, 5, 9 }));
    try std.testing.expectError(error.UnexpectedSafepoint, table.verifyCoverage(&.{ 2, 9 }));
    try std.testing.expectError(error.DuplicateRequiredSafepoint, table.verifyCoverage(&.{ 2, 5, 5, 9 }));

    try std.testing.expectError(
        error.MissingSafepoint,
        Table.initCovered(std.testing.allocator, &specs, &.{ 2, 4, 5, 9 }, test_options),
    );
}

test "covered table supports concurrent allocation-free verification and lookup" {
    const roots = [_]RootLocation{RootLocation.nativeRegister(1)};
    const specs = [_]MapSpec{
        .{ .pc_offset = 1, .roots = &roots },
        .{ .pc_offset = 3, .roots = &roots },
        .{ .pc_offset = 7, .roots = &roots },
    };
    const required = [_]u32{ 1, 3, 7 };
    var table = try Table.initCovered(std.testing.allocator, &specs, &required, test_options);
    defer table.deinit();
    var failed = std.atomic.Value(bool).init(false);

    const Worker = struct {
        table: *const Table,
        required: []const u32,
        failed: *std.atomic.Value(bool),

        fn run(self: *@This()) void {
            for (0..10_000) |_| {
                self.table.verifyCoverage(self.required) catch {
                    self.failed.store(true, .release);
                    return;
                };
                const record = self.table.find(3) catch {
                    self.failed.store(true, .release);
                    return;
                };
                if (self.table.rootsFor(record).len != 1) {
                    self.failed.store(true, .release);
                    return;
                }
            }
        }
    };
    var workers: [4]Worker = undefined;
    var threads: [4]std.Thread = undefined;
    for (&workers, &threads) |*worker, *thread| {
        worker.* = .{ .table = &table, .required = &required, .failed = &failed };
        thread.* = try std.Thread.spawn(.{}, Worker.run, .{worker});
    }
    for (threads) |thread| thread.join();
    try std.testing.expect(!failed.load(.acquire));
}

test "table rejects ambiguity and malformed locations" {
    const duplicate = [_]RootLocation{
        RootLocation.nativeRegister(1),
        RootLocation.nativeRegister(1),
    };
    const duplicate_specs = [_]MapSpec{.{ .pc_offset = 1, .roots = &duplicate }};
    try std.testing.expectError(
        error.DuplicateLocation,
        Table.init(std.testing.allocator, &duplicate_specs, test_options),
    );

    const bad_stack = [_]RootLocation{RootLocation.stackSlot(3)};
    const bad_stack_specs = [_]MapSpec{.{ .pc_offset = 1, .roots = &bad_stack }};
    try std.testing.expectError(
        error.InvalidStackAlignment,
        Table.init(std.testing.allocator, &bad_stack_specs, test_options),
    );

    const no_roots = [_]RootLocation{};
    const unsorted = [_]MapSpec{
        .{ .pc_offset = 9, .roots = &no_roots },
        .{ .pc_offset = 8, .roots = &no_roots },
    };
    try std.testing.expectError(
        error.UnsortedSafepoints,
        Table.init(std.testing.allocator, &unsorted, test_options),
    );
}

test "frame view visits only declared handles" {
    const roots = [_]RootLocation{
        RootLocation.nativeRegister(0),
        RootLocation.stackSlot(0),
        RootLocation.interpreterRegister(0, 1),
        RootLocation.shadowSlot(0),
    };
    const specs = [_]MapSpec{.{ .pc_offset = 5, .roots = &roots }};
    var table = try Table.init(std.testing.allocator, &specs, test_options);
    defer table.deinit();

    const native = [_]Handle{.{ .index = 1, .generation = 1 }};
    var stack = [_]Handle{.{ .index = 2, .generation = 1 }};
    const interpreted = [_]Handle{
        .{ .index = 30, .generation = 1 },
        .{ .index = 3, .generation = 1 },
    };
    const frames = [_][]const Handle{&interpreted};
    const shadow = [_]Handle{.{ .index = 4, .generation = 1 }};
    const view = FrameView{
        .native_registers = &native,
        .stack_base = @ptrCast(&stack[0]),
        .stack_min_offset = 0,
        .stack_max_offset = 0,
        .interpreter_frames = &frames,
        .shadow_roots = &shadow,
    };

    var visited = std.ArrayList(u32).empty;
    defer visited.deinit(std.testing.allocator);
    const Visitor = struct {
        fn visit(list: *std.ArrayList(u32), handle: Handle) !void {
            try list.append(std.testing.allocator, handle.index);
        }
    };
    try table.visitHandles(try table.find(5), view, &visited, Visitor.visit);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, visited.items);
}

fn allocationFailureInit(allocator: std.mem.Allocator) !void {
    const roots = [_]RootLocation{
        RootLocation.nativeRegister(0),
        RootLocation.stackSlot(0),
    };
    const specs = [_]MapSpec{.{ .pc_offset = 1, .roots = &roots }};
    var table = try Table.initCovered(allocator, &specs, &.{1}, test_options);
    defer table.deinit();
}

test "stack map construction is leak-free at every allocation failure" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        allocationFailureInit,
        .{},
    );
}
