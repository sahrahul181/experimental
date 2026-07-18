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
    xmm_register: u8,
    stack_slot: i32,
    constant: u64,
};

pub const ValueSpec = struct {
    vreg: u16,
    kind: ValueKind,
    source: Source,
};

/// One materialized caller in an inlined activation. Specs are ordered from
/// the oldest outer caller to the immediate caller of the leaf frame.
pub const InlineFrameSpec = struct {
    method_id: u32,
    dex_pc: u32,
    register_count: u16,
    values: []const ValueSpec,
};

pub const PointSpec = struct {
    id: u32,
    method_id: u32,
    dex_pc: u32,
    values: []const ValueSpec,
    inline_frames: []const InlineFrameSpec = &.{},
};

pub const ValidationOptions = struct {
    register_count: u16,
    native_register_count: u8,
    xmm_register_count: u8 = 0,
    max_dex_pc: u32,
    stack_alignment: u8 = 4,
};

pub const Record = struct {
    id: u32,
    method_id: u32,
    dex_pc: u32,
    first_value: u32,
    value_count: u16,
    frame_count: u16,
    first_frame: u32,
};

pub const FrameRecord = struct {
    method_id: u32,
    dex_pc: u32,
    first_value: u32,
    value_count: u16,
    register_count: u16,
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
    /// Leaf frame resumed by the interpreter.
    frame: *Frame,
    /// Outer-to-inner caller buffers. Their length must exactly match the
    /// point's immutable inline depth.
    inline_frames: []Frame = &.{},
    /// One word per value across every reconstructed frame.
    scratch: []u64,
    previous: ?*Frame = null,
};

pub const Capture = struct {
    native_registers: []const u64,
    xmm_registers: []const [16]u8 = &.{},
    stack_base: [*]const u8,
    stack_min_offset: i32,
    /// Exclusive upper bound.
    stack_max_offset: i32,
};

pub const OsrImage = struct {
    native_registers: []u64,
    /// Requires `value_count + native_register_count` words.
    scratch: []u64,
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
    xmm_registers: []const [16]u8 = &.{},
    stack_min_offset: i32 = 0,
    stack_max_offset: i32 = 0,
    resume_context: *anyopaque,
    resume_fn: ResumeFn,
};

pub const Error = error{
    DuplicatePoint,
    DuplicateRegister,
    AliasedOsrRegister,
    EmptyTable,
    IncompleteFrame,
    InvalidConstant,
    InvalidDexPc,
    InvalidDestination,
    InvalidFrameChain,
    InvalidNativeRegister,
    InvalidReference,
    InvalidReferenceLocation,
    InvalidStackAlignment,
    InvalidXmmRegister,
    MissingPoint,
    MissingDeoptMap,
    MissingStackMap,
    OutOfBounds,
    OsrStateMismatch,
    TooManyValues,
    UnsortedPoints,
    UnsupportedOsrSource,
    UnsupportedOsrFrameChain,
};

pub const Table = struct {
    allocator: std.mem.Allocator,
    records: []Record,
    frames: []FrameRecord,
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
        var total_frames: usize = 0;
        for (specs, 0..) |spec, point_index| {
            if (spec.dex_pc > options.max_dex_pc) return error.InvalidDexPc;
            if (point_index > 0) {
                const previous = specs[point_index - 1].id;
                if (previous == spec.id) return error.DuplicatePoint;
                if (previous > spec.id) return error.UnsortedPoints;
            }
            if (spec.inline_frames.len >= std.math.maxInt(u16) or
                spec.inline_frames.len + 1 > std.math.maxInt(u32) - total_frames) return error.TooManyValues;
            total_frames += spec.inline_frames.len + 1;
            for (spec.inline_frames) |frame| {
                if (frame.dex_pc > options.max_dex_pc) return error.InvalidDexPc;
                try addValueCount(&total_values, frame.values.len);
                try validateFrame(frame.register_count, frame.values, options);
            }
            try addValueCount(&total_values, spec.values.len);
            try validateFrame(options.register_count, spec.values, options);
        }

        const records = try allocator.alloc(Record, specs.len);
        errdefer allocator.free(records);
        const frames = try allocator.alloc(FrameRecord, total_frames);
        errdefer allocator.free(frames);
        const values = try allocator.alloc(ValueSpec, total_values);
        errdefer allocator.free(values);

        var value_cursor: usize = 0;
        var frame_cursor: usize = 0;
        for (specs, 0..) |spec, index| {
            const first_frame = frame_cursor;
            for (spec.inline_frames) |frame| {
                frames[frame_cursor] = .{
                    .method_id = frame.method_id,
                    .dex_pc = frame.dex_pc,
                    .first_value = @intCast(value_cursor),
                    .value_count = @intCast(frame.values.len),
                    .register_count = frame.register_count,
                };
                @memcpy(values[value_cursor..][0..frame.values.len], frame.values);
                value_cursor += frame.values.len;
                frame_cursor += 1;
            }
            const leaf_first_value = value_cursor;
            frames[frame_cursor] = .{
                .method_id = spec.method_id,
                .dex_pc = spec.dex_pc,
                .first_value = @intCast(value_cursor),
                .value_count = @intCast(spec.values.len),
                .register_count = options.register_count,
            };
            @memcpy(values[value_cursor..][0..spec.values.len], spec.values);
            value_cursor += spec.values.len;
            frame_cursor += 1;
            records[index] = .{
                .id = spec.id,
                .method_id = spec.method_id,
                .dex_pc = spec.dex_pc,
                .first_value = @intCast(leaf_first_value),
                .value_count = @intCast(spec.values.len),
                .frame_count = @intCast(spec.inline_frames.len + 1),
                .first_frame = @intCast(first_frame),
            };
        }
        return .{
            .allocator = allocator,
            .records = records,
            .frames = frames,
            .values = values,
            .register_count = options.register_count,
            .native_register_count = options.native_register_count,
        };
    }

    pub fn deinit(self: *Table) void {
        self.allocator.free(self.values);
        self.allocator.free(self.frames);
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

    pub fn framesFor(self: *const Table, record: *const Record) []const FrameRecord {
        const first: usize = record.first_frame;
        return self.frames[first .. first + record.frame_count];
    }

    pub fn valuesForFrame(self: *const Table, frame: FrameRecord) []const ValueSpec {
        const first: usize = frame.first_value;
        return self.values[first .. first + frame.value_count];
    }

    pub fn requiredScratchWords(self: *const Table, point_id: u32) Error!usize {
        const record = try self.find(point_id);
        var count: usize = 0;
        for (self.framesFor(record)) |frame| count += frame.value_count;
        return count;
    }

    pub const StackBounds = struct {
        min_offset: i32,
        max_offset: i32,
    };

    pub fn stackBounds(self: *const Table, point_id: u32) Error!StackBounds {
        const record = try self.find(point_id);
        var minimum: i32 = 0;
        var maximum: i32 = 0;
        for (self.framesFor(record)) |frame| {
            for (self.valuesForFrame(frame)) |value| switch (value.source) {
                .stack_slot => |offset| {
                    minimum = @min(minimum, offset);
                    maximum = @max(maximum, offset + @as(i32, value.kind.byteWidth()));
                },
                else => {},
            };
        }
        return .{ .min_offset = minimum, .max_offset = maximum };
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
        const frames = self.framesFor(record);
        if (capture.native_registers.len < self.native_register_count or
            destination.inline_frames.len + 1 != frames.len or
            destination.scratch.len < try self.requiredScratchWords(point_id)) return error.InvalidDestination;
        if (slicesOverlap(destination.scratch, capture.native_registers) or
            slicesOverlap(destination.scratch, capture.xmm_registers)) return error.InvalidDestination;

        for (frames, 0..) |frame_record, frame_index| {
            const frame = destinationFrame(destination, frame_index, frames.len);
            if (frame.execution.registers.len != frame_record.register_count or
                frame.execution.reference_registers.len != frame_record.register_count or
                frame.execution.register_is_ref.len != frame_record.register_count) return error.InvalidDestination;
            if (!validFrameStorage(frame, destination.scratch)) return error.InvalidDestination;
            for (0..frame_index) |prior_index| {
                const prior = destinationFrame(destination, prior_index, frames.len);
                if (prior == frame) return error.InvalidFrameChain;
                if (frameStorageOverlaps(frame, prior)) return error.InvalidDestination;
            }
        }
        try validatePreviousChain(destination, frames.len);

        for (frames) |frame_record| {
            for (self.valuesForFrame(frame_record)) |value| switch (value.source) {
                .stack_slot => |offset| {
                    const address = try stackAddress(capture, offset, value.kind.byteWidth());
                    if (byteRangesOverlap(
                        address,
                        value.kind.byteWidth(),
                        @intFromPtr(destination.scratch.ptr),
                        destination.scratch.len * @sizeOf(u64),
                    )) return error.InvalidDestination;
                },
                else => {},
            };
        }

        // Stage first. No interpreter-visible state changes before every
        // native/stack source has passed bounds and reference validation.
        var scratch_cursor: usize = 0;
        for (frames) |frame_record| {
            for (self.valuesForFrame(frame_record)) |value| {
                const bits = try readSource(value, capture);
                if (value.kind == .reference and !validHandleBits(bits)) return error.InvalidReference;
                destination.scratch[scratch_cursor] = bits;
                scratch_cursor += 1;
            }
        }

        // Commit cannot fail. Link from the existing caller through each
        // materialized inline activation and return the innermost leaf.
        scratch_cursor = 0;
        var previous = destination.previous;
        for (frames, 0..) |frame_record, frame_index| {
            const frame = destinationFrame(destination, frame_index, frames.len);
            const values = self.valuesForFrame(frame_record);
            commitFrame(frame, frame_record, values, destination.scratch[scratch_cursor..][0..values.len]);
            scratch_cursor += values.len;
            frame.previous = previous;
            frame.exception = if (frame_index + 1 == frames.len) exception else .{};
            frame.reason = reason;
            frame.active = true;
            previous = frame;
        }
        return destination.frame;
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
        const frames = self.framesFor(record);
        for (frames, 0..) |frame_record, frame_index| {
            for (self.valuesForFrame(frame_record), 0..) |value, value_index| {
                if (value.kind != .reference) continue;
                if (self.captureLocationSeen(frames, frame_index, value_index, value.source)) continue;
                switch (value.source) {
                    .native_register => |register| {
                        if (register >= capture.native_registers.len) return error.OutOfBounds;
                        try visitor(context, @as(*const Handle, @ptrCast(&capture.native_registers[register])));
                    },
                    .stack_slot => |offset| {
                        const address = try stackAddress(capture, offset, @sizeOf(Handle));
                        try visitor(context, @as(*const Handle, @ptrFromInt(address)));
                    },
                    .xmm_register => return error.InvalidReferenceLocation,
                    .constant => |bits| if (bits != @as(u64, @bitCast(Handle.none))) return error.InvalidConstant,
                }
            }
        }
    }

    /// Publishes references from exactly the newly reconstructed activation,
    /// stopping before the pre-existing caller chain.
    pub fn visitFrameReferenceSlots(
        self: *const Table,
        point_id: u32,
        leaf: *const Frame,
        context: anytype,
        visitor: anytype,
    ) !void {
        const record = try self.find(point_id);
        const frames = self.framesFor(record);
        var current: ?*const Frame = leaf;
        var reverse_index = frames.len;
        while (reverse_index > 0) {
            reverse_index -= 1;
            const frame = current orelse return error.InvalidFrameChain;
            const expected = frames[reverse_index];
            if (!frame.active or frame.method_id != expected.method_id or frame.execution.pc != expected.dex_pc or
                frame.execution.registers.len != expected.register_count or
                frame.execution.reference_registers.len != expected.register_count or
                frame.execution.register_is_ref.len != expected.register_count) return error.InvalidFrameChain;
            for (frame.execution.register_is_ref, 0..) |is_reference, register| {
                if (!is_reference) continue;
                const bits = frame.execution.reference_registers[register];
                if (!validHandleBits(bits)) return error.InvalidReference;
                try visitor(context, @as(*const Handle, @ptrCast(&frame.execution.reference_registers[register])));
            }
            current = frame.previous;
        }
    }

    fn captureLocationSeen(
        self: *const Table,
        frames: []const FrameRecord,
        frame_index: usize,
        value_index: usize,
        source: Source,
    ) bool {
        for (frames[0..frame_index]) |prior_frame| {
            for (self.valuesForFrame(prior_frame)) |prior| {
                if (prior.kind == .reference and sameCaptureLocation(prior.source, source)) return true;
            }
        }
        const current_values = self.valuesForFrame(frames[frame_index]);
        for (current_values[0..value_index]) |prior| {
            if (prior.kind == .reference and sameCaptureLocation(prior.source, source)) return true;
        }
        return false;
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

    pub fn validateAllLinked(
        self: *const Table,
        stack_maps: *const runtime_stack_map.Table,
    ) Error!void {
        for (self.records) |deopt_record| {
            var found = false;
            for (stack_maps.records) |stack_record| {
                if (stack_record.deopt_id == deopt_record.id) {
                    found = true;
                    break;
                }
            }
            if (!found) return error.MissingStackMap;
        }
    }

    /// Transactionally exports an interpreter frame into the physical image
    /// expected at an OSR entry. The same table drives deoptimization in the
    /// reverse direction, preventing format drift between the two paths.
    pub fn exportOsr(
        self: *const Table,
        point_id: u32,
        frame: *const Frame,
        image: OsrImage,
    ) Error!void {
        const record = try self.find(point_id);
        if (record.frame_count != 1) return error.UnsupportedOsrFrameChain;
        const values = self.valuesFor(record);
        const native_count: usize = self.native_register_count;
        if (!frame.active or frame.method_id != record.method_id or frame.execution.pc != record.dex_pc or
            frame.execution.registers.len != self.register_count or
            frame.execution.reference_registers.len != self.register_count or
            frame.execution.register_is_ref.len != self.register_count or
            image.native_registers.len < native_count or
            image.scratch.len < values.len + native_count)
        {
            return error.OsrStateMismatch;
        }

        const staged_values = image.scratch[0..values.len];
        const staged_registers = image.scratch[values.len..][0..native_count];
        @memset(staged_registers, 0);
        var assigned: [std.math.maxInt(u8) + 1]bool = @splat(false);
        for (values, 0..) |value, index| {
            const bits = try readFrameValue(frame, value);
            staged_values[index] = bits;
            switch (value.source) {
                .constant => |expected| if (bits != expected) return error.OsrStateMismatch,
                .stack_slot, .xmm_register => return error.UnsupportedOsrSource,
                .native_register => |register| {
                    if (register >= native_count) return error.OutOfBounds;
                    if (assigned[register] and staged_registers[register] != bits) return error.AliasedOsrRegister;
                    assigned[register] = true;
                    staged_registers[register] = bits;
                },
            }
        }
        @memcpy(image.native_registers[0..native_count], staged_registers);
    }

    pub fn importOsr(
        self: *const Table,
        point_id: u32,
        image: []const u64,
        destination: Destination,
    ) Error!*Frame {
        const record = try self.find(point_id);
        if (record.frame_count != 1) return error.UnsupportedOsrFrameChain;
        var stack_anchor: u64 = 0;
        return self.reconstruct(point_id, .{
            .native_registers = image,
            .stack_base = @ptrCast(&stack_anchor),
            .stack_min_offset = 0,
            .stack_max_offset = @sizeOf(@TypeOf(stack_anchor)),
        }, destination, .uncommon_trap, .{});
    }
};

fn addValueCount(total: *usize, count: usize) Error!void {
    if (count > std.math.maxInt(u16) or count > std.math.maxInt(u32) - total.*) return error.TooManyValues;
    total.* += count;
}

fn validateFrame(register_count: u16, values: []const ValueSpec, options: ValidationOptions) Error!void {
    if (register_count == 0) return error.IncompleteFrame;
    for (values) |value| {
        const width = value.kind.registerWidth();
        if (value.vreg >= register_count or width > register_count - value.vreg) {
            return error.OutOfBounds;
        }
        switch (value.source) {
            .native_register => |register| if (register >= options.native_register_count) return error.InvalidNativeRegister,
            .xmm_register => |register| {
                if (register >= options.xmm_register_count) return error.InvalidXmmRegister;
                if (value.kind == .reference) return error.InvalidReferenceLocation;
            },
            .stack_slot => |offset| {
                if (@mod(offset, @as(i32, options.stack_alignment)) != 0) return error.InvalidStackAlignment;
                if (offset > std.math.maxInt(i32) - @as(i32, value.kind.byteWidth())) return error.OutOfBounds;
            },
            .constant => |bits| if (value.kind == .reference and bits != @as(u64, @bitCast(Handle.none))) return error.InvalidConstant,
        }
    }

    var register: u16 = 0;
    while (register < register_count) : (register += 1) {
        var coverage: u8 = 0;
        for (values) |value| {
            const end = value.vreg + value.kind.registerWidth();
            if (register >= value.vreg and register < end) coverage += 1;
        }
        if (coverage == 0) return error.IncompleteFrame;
        if (coverage > 1) return error.DuplicateRegister;
    }
}

fn destinationFrame(destination: Destination, index: usize, frame_count: usize) *Frame {
    std.debug.assert(index < frame_count);
    if (index + 1 == frame_count) return destination.frame;
    return &destination.inline_frames[index];
}

fn validatePreviousChain(destination: Destination, frame_count: usize) Error!void {
    var cursor = destination.previous;
    var tortoise = destination.previous;
    var hare = destination.previous;
    while (cursor) |frame| {
        for (0..frame_count) |index| {
            if (frame == destinationFrame(destination, index, frame_count)) return error.InvalidFrameChain;
        }
        cursor = frame.previous;
        tortoise = if (tortoise) |value| value.previous else null;
        hare = if (hare) |value| if (value.previous) |next| next.previous else null else null;
        if (tortoise != null and hare != null and tortoise == hare) return error.InvalidFrameChain;
    }
}

fn validFrameStorage(frame: *const Frame, scratch: []const u64) bool {
    if (slicesOverlap(frame.execution.registers, frame.execution.reference_registers) or
        slicesOverlap(frame.execution.registers, frame.execution.register_is_ref) or
        slicesOverlap(frame.execution.reference_registers, frame.execution.register_is_ref) or
        slicesOverlap(frame.execution.registers, scratch) or
        slicesOverlap(frame.execution.reference_registers, scratch) or
        slicesOverlap(frame.execution.register_is_ref, scratch)) return false;
    return !byteRangesOverlap(
        @intFromPtr(frame),
        @sizeOf(Frame),
        @intFromPtr(scratch.ptr),
        scratch.len * @sizeOf(u64),
    );
}

fn frameStorageOverlaps(a: *const Frame, b: *const Frame) bool {
    return slicesOverlap(a.execution.registers, b.execution.registers) or
        slicesOverlap(a.execution.registers, b.execution.reference_registers) or
        slicesOverlap(a.execution.registers, b.execution.register_is_ref) or
        slicesOverlap(a.execution.reference_registers, b.execution.registers) or
        slicesOverlap(a.execution.reference_registers, b.execution.reference_registers) or
        slicesOverlap(a.execution.reference_registers, b.execution.register_is_ref) or
        slicesOverlap(a.execution.register_is_ref, b.execution.registers) or
        slicesOverlap(a.execution.register_is_ref, b.execution.reference_registers) or
        slicesOverlap(a.execution.register_is_ref, b.execution.register_is_ref);
}

fn slicesOverlap(a: anytype, b: anytype) bool {
    return byteRangesOverlap(
        @intFromPtr(a.ptr),
        a.len * @sizeOf(@TypeOf(a[0])),
        @intFromPtr(b.ptr),
        b.len * @sizeOf(@TypeOf(b[0])),
    );
}

fn byteRangesOverlap(a_start: usize, a_len: usize, b_start: usize, b_len: usize) bool {
    if (a_len == 0 or b_len == 0) return false;
    const a_end = std.math.add(usize, a_start, a_len) catch return true;
    const b_end = std.math.add(usize, b_start, b_len) catch return true;
    return a_start < b_end and b_start < a_end;
}

fn commitFrame(frame: *Frame, record: FrameRecord, values: []const ValueSpec, staged: []const u64) void {
    @memset(frame.execution.registers, 0);
    @memset(frame.execution.reference_registers, @as(u64, @bitCast(Handle.none)));
    @memset(frame.execution.register_is_ref, false);
    for (values, staged) |value, bits| {
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
}

fn sameCaptureLocation(a: Source, b: Source) bool {
    return switch (a) {
        .native_register => |left| switch (b) {
            .native_register => |right| left == right,
            else => false,
        },
        .xmm_register => |left| switch (b) {
            .xmm_register => |right| left == right,
            else => false,
        },
        .stack_slot => |left| switch (b) {
            .stack_slot => |right| left == right,
            else => false,
        },
        .constant => false,
    };
}

fn readSource(value: ValueSpec, capture: Capture) Error!u64 {
    return switch (value.source) {
        .native_register => |register| if (register < capture.native_registers.len)
            capture.native_registers[register]
        else
            error.OutOfBounds,
        .xmm_register => |register| if (register < capture.xmm_registers.len)
            readXmm(capture.xmm_registers[register], value.kind.byteWidth())
        else
            error.OutOfBounds,
        .constant => |bits| bits,
        .stack_slot => |offset| readStack(capture, offset, value.kind.byteWidth()),
    };
}

fn readXmm(bytes: [16]u8, width: u8) u64 {
    return switch (width) {
        4 => std.mem.readInt(u32, bytes[0..4], .little),
        8 => std.mem.readInt(u64, bytes[0..8], .little),
        else => unreachable,
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

fn readFrameValue(frame: *const Frame, value: ValueSpec) Error!u64 {
    return switch (value.kind) {
        .scalar32 => blk: {
            if (frame.execution.register_is_ref[value.vreg]) return error.OsrStateMismatch;
            break :blk frame.execution.registers[value.vreg];
        },
        .scalar64 => blk: {
            if (frame.execution.register_is_ref[value.vreg] or frame.execution.register_is_ref[value.vreg + 1]) {
                return error.OsrStateMismatch;
            }
            break :blk frame.execution.getWide(value.vreg);
        },
        .reference => blk: {
            if (!frame.execution.register_is_ref[value.vreg]) return error.OsrStateMismatch;
            const bits = frame.execution.reference_registers[value.vreg];
            if (!validHandleBits(bits)) return error.InvalidReference;
            break :blk bits;
        },
    };
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

test "deoptimization reconstructs exact low XMM lanes and managed spill slots" {
    const values = [_]ValueSpec{
        .{ .vreg = 0, .kind = .scalar32, .source = .{ .xmm_register = 1 } },
        .{ .vreg = 1, .kind = .scalar64, .source = .{ .xmm_register = 2 } },
        .{ .vreg = 3, .kind = .scalar64, .source = .{ .stack_slot = 8 } },
    };
    var table = try Table.init(std.testing.allocator, &.{.{
        .id = 15,
        .method_id = 8,
        .dex_pc = 4,
        .values = &values,
    }}, .{
        .register_count = 5,
        .native_register_count = 16,
        .xmm_register_count = 8,
        .max_dex_pc = 4,
    });
    defer table.deinit();
    try std.testing.expectEqual(Table.StackBounds{ .min_offset = 0, .max_offset = 16 }, try table.stackBounds(15));

    var xmm: [8][16]u8 = @splat(@splat(0));
    std.mem.writeInt(u32, xmm[1][0..4], 0xa1b2c3d4, .little);
    std.mem.writeInt(u64, xmm[2][0..8], 0x1122334455667788, .little);
    var stack: [2]u64 = .{ 0, 0x8877665544332211 };
    var native: [16]u64 = @splat(0);
    var registers: [5]u32 = @splat(0);
    var references: [5]u64 = @splat(@as(u64, @bitCast(Handle.none)));
    var kinds: [5]bool = @splat(false);
    var frame = Frame{ .execution = .{
        .pc = 0,
        .registers = &registers,
        .instructions = &.{},
        .register_is_ref = &kinds,
        .reference_registers = &references,
    } };
    var scratch: [3]u64 = undefined;
    _ = try table.reconstruct(15, .{
        .native_registers = &native,
        .xmm_registers = &xmm,
        .stack_base = @ptrCast(&stack[0]),
        .stack_min_offset = 0,
        .stack_max_offset = @sizeOf(@TypeOf(stack)),
    }, .{ .frame = &frame, .scratch = &scratch }, .invalidation, .{});
    try std.testing.expectEqual(@as(u32, 0xa1b2c3d4), frame.execution.registers[0]);
    try std.testing.expectEqual(@as(u64, 0x1122334455667788), frame.execution.getWide(1));
    try std.testing.expectEqual(@as(u64, 0x8877665544332211), frame.execution.getWide(3));

    var osr_scratch: [19]u64 = undefined;
    try std.testing.expectError(error.UnsupportedOsrSource, table.exportOsr(15, &frame, .{
        .native_registers = &native,
        .scratch = &osr_scratch,
    }));
    const bad_reference = [_]ValueSpec{
        .{ .vreg = 0, .kind = .reference, .source = .{ .xmm_register = 0 } },
        .{ .vreg = 1, .kind = .scalar64, .source = .{ .constant = 0 } },
        .{ .vreg = 3, .kind = .scalar64, .source = .{ .constant = 0 } },
    };
    try std.testing.expectError(error.InvalidReferenceLocation, Table.init(std.testing.allocator, &.{.{
        .id = 16,
        .method_id = 8,
        .dex_pc = 4,
        .values = &bad_reference,
    }}, .{
        .register_count = 5,
        .native_register_count = 16,
        .xmm_register_count = 8,
        .max_dex_pc = 4,
    }));
}

test "deoptimization transactionally reconstructs an exact inlined activation chain" {
    const shared = Handle{ .index = 17, .generation = 5 };
    const leaf_only = Handle{ .index = 19, .generation = 6 };
    const outer_values = [_]ValueSpec{
        .{ .vreg = 0, .kind = .reference, .source = .{ .native_register = 2 } },
        .{ .vreg = 1, .kind = .scalar64, .source = .{ .stack_slot = 0 } },
    };
    const inner_values = [_]ValueSpec{
        .{ .vreg = 0, .kind = .scalar32, .source = .{ .native_register = 3 } },
        .{ .vreg = 1, .kind = .scalar32, .source = .{ .constant = 44 } },
    };
    const leaf_values = [_]ValueSpec{
        .{ .vreg = 0, .kind = .reference, .source = .{ .native_register = 2 } },
        .{ .vreg = 1, .kind = .reference, .source = .{ .native_register = 4 } },
        .{ .vreg = 2, .kind = .scalar64, .source = .{ .constant = 0x8877665544332211 } },
    };
    const callers = [_]InlineFrameSpec{
        .{ .method_id = 10, .dex_pc = 1, .register_count = 3, .values = &outer_values },
        .{ .method_id = 20, .dex_pc = 2, .register_count = 2, .values = &inner_values },
    };
    var table = try Table.init(std.testing.allocator, &.{.{
        .id = 31,
        .method_id = 30,
        .dex_pc = 3,
        .values = &leaf_values,
        .inline_frames = &callers,
    }}, .{ .register_count = 4, .native_register_count = 8, .max_dex_pc = 4 });
    defer table.deinit();
    try std.testing.expectEqual(@as(usize, 3), table.frames.len);
    try std.testing.expectEqual(@as(usize, 7), try table.requiredScratchWords(31));

    var native: [8]u64 = @splat(0);
    native[2] = @bitCast(shared);
    native[3] = 33;
    native[4] = @bitCast(leaf_only);
    var stack = [_]u64{0x1122334455667788};

    var outer_registers: [3]u32 = @splat(0);
    var outer_references: [3]u64 = @splat(@as(u64, @bitCast(Handle.none)));
    var outer_kinds: [3]bool = @splat(false);
    var inner_registers: [2]u32 = @splat(0);
    var inner_references: [2]u64 = @splat(@as(u64, @bitCast(Handle.none)));
    var inner_kinds: [2]bool = @splat(false);
    var leaf_registers: [4]u32 = @splat(0);
    var leaf_references: [4]u64 = @splat(@as(u64, @bitCast(Handle.none)));
    var leaf_kinds: [4]bool = @splat(false);
    var callers_out = [_]Frame{
        .{ .execution = .{ .pc = 90, .registers = &outer_registers, .instructions = &.{}, .register_is_ref = &outer_kinds, .reference_registers = &outer_references } },
        .{ .execution = .{ .pc = 91, .registers = &inner_registers, .instructions = &.{}, .register_is_ref = &inner_kinds, .reference_registers = &inner_references } },
    };
    var leaf = Frame{ .execution = .{
        .pc = 92,
        .registers = &leaf_registers,
        .instructions = &.{},
        .register_is_ref = &leaf_kinds,
        .reference_registers = &leaf_references,
    } };
    var preexisting = Frame{ .execution = .{
        .pc = 99,
        .registers = &.{},
        .instructions = &.{},
        .register_is_ref = &.{},
        .reference_registers = &.{},
    }, .active = true };
    var scratch: [7]u64 = undefined;
    const exception = ExceptionState{ .kind = 2, .dex_pc = 3, .payload0 = 7 };
    const rebuilt = try table.reconstruct(31, .{
        .native_registers = &native,
        .stack_base = @ptrCast(&stack[0]),
        .stack_min_offset = 0,
        .stack_max_offset = @sizeOf(@TypeOf(stack)),
    }, .{
        .frame = &leaf,
        .inline_frames = &callers_out,
        .scratch = &scratch,
        .previous = &preexisting,
    }, .exception, exception);

    try std.testing.expect(rebuilt == &leaf);
    try std.testing.expect(callers_out[0].previous == &preexisting);
    try std.testing.expect(callers_out[1].previous == &callers_out[0]);
    try std.testing.expect(leaf.previous == &callers_out[1]);
    try std.testing.expectEqual(@as(u32, 10), callers_out[0].method_id);
    try std.testing.expectEqual(@as(u64, 0x1122334455667788), callers_out[0].execution.getWide(1));
    try std.testing.expectEqual(@as(u32, 33), callers_out[1].execution.registers[0]);
    try std.testing.expectEqual(@as(u32, 44), callers_out[1].execution.registers[1]);
    try std.testing.expectEqual(@as(u64, @bitCast(shared)), leaf.execution.reference_registers[0]);
    try std.testing.expectEqual(@as(u64, @bitCast(leaf_only)), leaf.execution.reference_registers[1]);
    try std.testing.expectEqual(exception, leaf.exception);
    try std.testing.expect(!callers_out[0].exception.isPending());

    var frame_roots: std.ArrayList(u64) = .empty;
    defer frame_roots.deinit(std.testing.allocator);
    const FrameVisitor = struct {
        fn add(list: *std.ArrayList(u64), slot: *const Handle) !void {
            try list.append(std.testing.allocator, @bitCast(slot.*));
        }
    };
    try table.visitFrameReferenceSlots(31, &leaf, &frame_roots, FrameVisitor.add);
    try std.testing.expectEqualSlices(u64, &.{
        @as(u64, @bitCast(shared)),
        @as(u64, @bitCast(leaf_only)),
        @as(u64, @bitCast(shared)),
    }, frame_roots.items);

    var capture_roots: std.ArrayList(u64) = .empty;
    defer capture_roots.deinit(std.testing.allocator);
    const Visitor = struct {
        fn add(list: *std.ArrayList(u64), slot: *const Handle) !void {
            try list.append(std.testing.allocator, @bitCast(slot.*));
        }
    };
    try table.visitReferenceSlots(31, .{
        .native_registers = &native,
        .stack_base = @ptrCast(&stack[0]),
        .stack_min_offset = 0,
        .stack_max_offset = @sizeOf(@TypeOf(stack)),
    }, &capture_roots, Visitor.add);
    try std.testing.expectEqualSlices(u64, &.{ @as(u64, @bitCast(shared)), @as(u64, @bitCast(leaf_only)) }, capture_roots.items);

    const before_outer_pc = callers_out[0].execution.pc;
    const before_leaf_pc = leaf.execution.pc;
    try std.testing.expectError(error.InvalidDestination, table.reconstruct(31, .{
        .native_registers = &native,
        .stack_base = @ptrCast(&stack[0]),
        .stack_min_offset = 0,
        .stack_max_offset = @sizeOf(@TypeOf(stack)),
    }, .{ .frame = &leaf, .inline_frames = callers_out[0..1], .scratch = &scratch }, .invalidation, .{}));
    try std.testing.expectEqual(before_outer_pc, callers_out[0].execution.pc);
    try std.testing.expectEqual(before_leaf_pc, leaf.execution.pc);

    callers_out[0].execution.pc = 71;
    callers_out[1].execution.pc = 72;
    leaf.execution.pc = 73;
    native[4] = 1;
    try std.testing.expectError(error.InvalidReference, table.reconstruct(31, .{
        .native_registers = &native,
        .stack_base = @ptrCast(&stack[0]),
        .stack_min_offset = 0,
        .stack_max_offset = @sizeOf(@TypeOf(stack)),
    }, .{ .frame = &leaf, .inline_frames = &callers_out, .scratch = &scratch }, .invalidation, .{}));
    try std.testing.expectEqual(@as(u32, 71), callers_out[0].execution.pc);
    try std.testing.expectEqual(@as(u32, 72), callers_out[1].execution.pc);
    try std.testing.expectEqual(@as(u32, 73), leaf.execution.pc);

    native[4] = @bitCast(leaf_only);
    const saved_leaf_references = leaf.execution.reference_registers;
    leaf.execution.reference_registers = scratch[0..4];
    defer leaf.execution.reference_registers = saved_leaf_references;
    try std.testing.expectError(error.InvalidDestination, table.reconstruct(31, .{
        .native_registers = &native,
        .stack_base = @ptrCast(&stack[0]),
        .stack_min_offset = 0,
        .stack_max_offset = @sizeOf(@TypeOf(stack)),
    }, .{ .frame = &leaf, .inline_frames = &callers_out, .scratch = &scratch }, .invalidation, .{}));
    leaf.execution.reference_registers = saved_leaf_references;

    preexisting.previous = &leaf;
    defer preexisting.previous = null;
    try std.testing.expectError(error.InvalidFrameChain, table.reconstruct(31, .{
        .native_registers = &native,
        .stack_base = @ptrCast(&stack[0]),
        .stack_min_offset = 0,
        .stack_max_offset = @sizeOf(@TypeOf(stack)),
    }, .{
        .frame = &leaf,
        .inline_frames = &callers_out,
        .scratch = &scratch,
        .previous = &preexisting,
    }, .invalidation, .{}));
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

test "OSR export and import round trip through the deoptimization format" {
    const handle = Handle{ .index = 21, .generation = 7 };
    const values = [_]ValueSpec{
        .{ .vreg = 0, .kind = .scalar32, .source = .{ .native_register = 1 } },
        .{ .vreg = 1, .kind = .scalar64, .source = .{ .native_register = 2 } },
        .{ .vreg = 3, .kind = .reference, .source = .{ .native_register = 4 } },
    };
    var table = try Table.init(std.testing.allocator, &.{.{
        .id = 12,
        .method_id = 5,
        .dex_pc = 9,
        .values = &values,
    }}, .{ .register_count = 4, .native_register_count = 8, .max_dex_pc = 10 });
    defer table.deinit();

    var source_registers = [_]u32{ 0x12345678, 0x44332211, 0x88776655, @truncate(@as(u64, @bitCast(handle))) };
    var source_references: [4]u64 = @splat(@as(u64, @bitCast(Handle.none)));
    source_references[3] = @bitCast(handle);
    var source_kinds = [_]bool{ false, false, false, true };
    var source_frame = Frame{
        .method_id = 5,
        .execution = .{
            .pc = 9,
            .registers = &source_registers,
            .instructions = &.{},
            .register_is_ref = &source_kinds,
            .reference_registers = &source_references,
        },
        .active = true,
    };
    var native_image: [8]u64 = @splat(0xeeeeeeeeeeeeeeee);
    var export_scratch: [11]u64 = undefined;
    try table.exportOsr(12, &source_frame, .{ .native_registers = &native_image, .scratch = &export_scratch });
    try std.testing.expectEqual(@as(u64, 0x12345678), native_image[1]);
    try std.testing.expectEqual(@as(u64, 0x8877665544332211), native_image[2]);
    try std.testing.expectEqual(@as(u64, @bitCast(handle)), native_image[4]);

    var target_registers: [4]u32 = @splat(0);
    var target_references: [4]u64 = @splat(@as(u64, @bitCast(Handle.none)));
    var target_kinds: [4]bool = @splat(false);
    var target_frame = Frame{ .execution = .{
        .pc = 0,
        .registers = &target_registers,
        .instructions = &.{},
        .register_is_ref = &target_kinds,
        .reference_registers = &target_references,
    } };
    var import_scratch: [3]u64 = undefined;
    _ = try table.importOsr(12, &native_image, .{ .frame = &target_frame, .scratch = &import_scratch });
    try std.testing.expectEqualSlices(u32, &source_registers, &target_registers);
    try std.testing.expectEqual(@as(u64, @bitCast(handle)), target_references[3]);
    try std.testing.expectEqual(Reason.uncommon_trap, target_frame.reason);

    const before = native_image;
    source_frame.execution.register_is_ref[3] = false;
    try std.testing.expectError(error.OsrStateMismatch, table.exportOsr(12, &source_frame, .{
        .native_registers = &native_image,
        .scratch = &export_scratch,
    }));
    try std.testing.expectEqualSlices(u64, &before, &native_image);
}

fn allocationFailureProbe(allocator: std.mem.Allocator) !void {
    const values = [_]ValueSpec{
        .{ .vreg = 0, .kind = .scalar64, .source = .{ .constant = 1 } },
        .{ .vreg = 2, .kind = .scalar64, .source = .{ .constant = 2 } },
    };
    const caller_values = [_]ValueSpec{
        .{ .vreg = 0, .kind = .scalar64, .source = .{ .constant = 3 } },
        .{ .vreg = 2, .kind = .scalar64, .source = .{ .constant = 4 } },
    };
    const callers = [_]InlineFrameSpec{.{
        .method_id = 1,
        .dex_pc = 2,
        .register_count = 4,
        .values = &caller_values,
    }};
    var table = try Table.init(allocator, &.{.{
        .id = 1,
        .method_id = 2,
        .dex_pc = 3,
        .values = &values,
        .inline_frames = &callers,
    }}, test_options);
    defer table.deinit();
}

test "deoptimization table construction is allocation-failure safe" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{});
}
