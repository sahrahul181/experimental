//! Managed-memory attachment for the runtime-independent interpreter.
//!
//! Each callback is owner-confined, allocation-free, and lock-free. Reference
//! stores use the same collector entry points as generated code: exact-slot
//! SATB before mutation, then either an object card or static-root publication.

const std = @import("std");
const interpreter = @import("interpreter");
const runtime_gc = @import("runtime_gc");
const runtime_heap = @import("runtime_heap");
const runtime_value = @import("runtime_value");

const Handle = runtime_value.Handle;

pub const Error = runtime_gc.Error || runtime_value.Error || error{
    InvalidLayout,
    MissingFieldLayout,
    MissingStaticFieldLayout,
    ArrayIndexOutOfBounds,
};

pub const ReferenceArrayLayout = struct {
    length_offset: u32,
    data_offset: u32,
    element_stride: u8 = @sizeOf(Handle),
};

pub const StaticFieldLayout = struct {
    address: usize,
};

pub const Options = struct {
    /// Reference-field payload offsets indexed by resolved field id.
    reference_field_offsets: []const u32 = &.{},
    reference_array_layout: ?ReferenceArrayLayout = null,
    static_field_layouts: []const StaticFieldLayout = &.{},
};

pub const Stats = struct {
    instance_reference_stores: u64,
    array_reference_stores: u64,
    static_reference_stores: u64,
    failures: u64,
};

pub const Context = struct {
    collector: *runtime_gc.ConcurrentCollector,
    satb: *runtime_gc.SatbBuffer,
    reference_field_offsets: []const u32,
    reference_array_layout: ?ReferenceArrayLayout,
    static_field_layouts: []const StaticFieldLayout,
    last_card_destination: u64 = 0,
    instance_reference_stores: u64 = 0,
    array_reference_stores: u64 = 0,
    static_reference_stores: u64 = 0,
    failures: u64 = 0,

    pub fn init(
        collector: *runtime_gc.ConcurrentCollector,
        satb: *runtime_gc.SatbBuffer,
        options: Options,
    ) Error!Context {
        if (!collector.ownsBuffer(satb)) return error.MissingSatbBuffer;
        for (options.reference_field_offsets) |offset| {
            if (!std.mem.isAligned(offset, @alignOf(std.atomic.Value(u64)))) return error.InvalidLayout;
        }
        if (options.reference_array_layout) |layout| try validateArrayLayout(layout);
        for (options.static_field_layouts) |layout| {
            if (layout.address == 0 or
                !std.mem.isAligned(layout.address, @alignOf(std.atomic.Value(u64))) or
                !collector.isStaticRootSlot(layout.address)) return error.InvalidLayout;
        }
        return .{
            .collector = collector,
            .satb = satb,
            .reference_field_offsets = options.reference_field_offsets,
            .reference_array_layout = options.reference_array_layout,
            .static_field_layouts = options.static_field_layouts,
        };
    }

    pub fn attachment(self: *Context) interpreter.ManagedMemory {
        return .{ .context = self, .vtable = &vtable };
    }

    pub fn stats(self: *const Context) Stats {
        return .{
            .instance_reference_stores = self.instance_reference_stores,
            .array_reference_stores = self.array_reference_stores,
            .static_reference_stores = self.static_reference_stores,
            .failures = self.failures,
        };
    }

    fn preWrite(self: *Context, slot_address: usize) Error!void {
        for (0..1024) |_| {
            self.collector.referenceStorePreWrite(self.satb, slot_address, false) catch |err| switch (err) {
                error.SatbQueueFull => {
                    _ = self.collector.drainSatb(1) catch |drain_err| switch (drain_err) {
                        error.RetryBarrier => {
                            std.atomic.spinLoopHint();
                            continue;
                        },
                        else => return drain_err,
                    };
                    continue;
                },
                error.RetryBarrier => {
                    std.atomic.spinLoopHint();
                    continue;
                },
                else => return err,
            };
            return;
        }
        return error.RetryBarrier;
    }

    fn staticPostWrite(self: *Context, slot_address: usize, stored: Handle) Error!void {
        for (0..1024) |_| {
            self.collector.referenceStaticStorePostWrite(slot_address, stored) catch |err| switch (err) {
                error.RetryBarrier => {
                    std.atomic.spinLoopHint();
                    continue;
                },
                else => return err,
            };
            return;
        }
        return error.RetryBarrier;
    }

    fn referenceFieldSlot(self: *Context, object: Handle, field_idx: u32) Error!usize {
        if (object.isNull()) return error.InvalidHandle;
        if (field_idx >= self.reference_field_offsets.len) return error.MissingFieldLayout;
        const base = @intFromPtr(try self.collector.handleTable().resolve(object));
        const offset: usize = self.reference_field_offsets[field_idx];
        if (offset > std.math.maxInt(usize) - base) return error.InvalidLayout;
        const address = base + offset;
        if (!std.mem.isAligned(address, @alignOf(std.atomic.Value(u64)))) return error.InvalidLayout;
        return address;
    }

    fn referenceArraySlot(self: *Context, array: Handle, index: i32) Error!usize {
        if (array.isNull()) return error.InvalidHandle;
        const layout = self.reference_array_layout orelse return error.InvalidLayout;
        const base = @intFromPtr(try self.collector.handleTable().resolve(array));
        const length_address = std.math.add(usize, base, layout.length_offset) catch return error.InvalidLayout;
        const length_slot: *const std.atomic.Value(u32) = @ptrFromInt(length_address);
        const length = length_slot.load(.acquire);
        if (index < 0 or @as(u32, @intCast(index)) >= length) return error.ArrayIndexOutOfBounds;
        const scaled = std.math.mul(usize, @intCast(index), layout.element_stride) catch return error.InvalidLayout;
        const data = std.math.add(usize, base, layout.data_offset) catch return error.InvalidLayout;
        const address = std.math.add(usize, data, scaled) catch return error.InvalidLayout;
        if (!std.mem.isAligned(address, @alignOf(std.atomic.Value(u64)))) return error.InvalidLayout;
        return address;
    }
};

fn validateArrayLayout(layout: ReferenceArrayLayout) Error!void {
    if (!std.mem.isAligned(layout.length_offset, @alignOf(std.atomic.Value(u32))) or
        !std.mem.isAligned(layout.data_offset, @alignOf(std.atomic.Value(u64))) or
        layout.element_stride != @sizeOf(Handle) or
        layout.data_offset < layout.length_offset + @sizeOf(u32)) return error.InvalidLayout;
}

fn failed(context: *Context) interpreter.ManagedMemoryStatus {
    context.failures += 1;
    return .failure;
}

fn storeInstanceReference(raw: *anyopaque, object_bits: u64, field_idx: u32, stored_bits: u64) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    const object: Handle = @bitCast(object_bits);
    if (object.isNull()) return .null_reference;
    const slot_address = context.referenceFieldSlot(object, field_idx) catch return failed(context);
    context.preWrite(slot_address) catch return failed(context);
    const slot: *std.atomic.Value(u64) = @ptrFromInt(slot_address);
    slot.store(stored_bits, .release);
    context.collector.referenceStorePostWrite(object, @bitCast(stored_bits), false, &context.last_card_destination) catch return failed(context);
    context.instance_reference_stores += 1;
    return .ok;
}

fn storeArrayReference(raw: *anyopaque, array_bits: u64, index: i32, stored_bits: u64) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    const array: Handle = @bitCast(array_bits);
    if (array.isNull()) return .null_reference;
    const slot_address = context.referenceArraySlot(array, index) catch |err| switch (err) {
        error.ArrayIndexOutOfBounds => return .array_index_out_of_bounds,
        else => return failed(context),
    };
    context.preWrite(slot_address) catch return failed(context);
    const slot: *std.atomic.Value(u64) = @ptrFromInt(slot_address);
    slot.store(stored_bits, .release);
    context.collector.referenceStorePostWrite(array, @bitCast(stored_bits), false, &context.last_card_destination) catch return failed(context);
    context.array_reference_stores += 1;
    return .ok;
}

fn storeStaticReference(raw: *anyopaque, field_idx: u32, stored_bits: u64) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    if (field_idx >= context.static_field_layouts.len) return failed(context);
    const slot_address = context.static_field_layouts[field_idx].address;
    context.preWrite(slot_address) catch return failed(context);
    const slot: *std.atomic.Value(u64) = @ptrFromInt(slot_address);
    slot.store(stored_bits, .release);
    context.staticPostWrite(slot_address, @bitCast(stored_bits)) catch return failed(context);
    context.static_reference_stores += 1;
    return .ok;
}

const vtable = interpreter.ManagedMemoryVTable{
    .store_instance_reference = storeInstanceReference,
    .store_array_reference = storeArrayReference,
    .store_static_reference = storeStaticReference,
};

fn publishTestObject(
    heap: *runtime_heap.ManagedHeap,
    handles: *runtime_value.HandleTable,
    allocator: *runtime_heap.ThreadAllocator,
    size: usize,
) !Handle {
    const reservation = try allocator.allocate(size, runtime_value.object_alignment);
    const handle = try handles.reserve(0, 0);
    errdefer handles.cancelReservation(handle) catch {};
    try heap.publishObject(reservation, handle);
    return handle;
}

test "interpreter field array and static stores share exact concurrent barriers" {
    var old_region: [256]u8 align(runtime_value.object_alignment) = @splat(0);
    var young_region: [256]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&old_region),
        try runtime_value.Region.fromSlice(&young_region),
    };
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 16, &regions);
    defer handles.deinit();
    var heap = try runtime_heap.ManagedHeap.init(std.testing.allocator, &handles, 128);
    defer heap.deinit();

    var old_allocator = heap.threadAllocator();
    const destination = try publishTestObject(&heap, &handles, &old_allocator, 16);
    const array = try publishTestObject(&heap, &handles, &old_allocator, 32);
    var young_allocator = heap.threadAllocator();
    const old_field = try publishTestObject(&heap, &handles, &young_allocator, 8);
    const old_array = try publishTestObject(&heap, &handles, &young_allocator, 8);
    const old_static = try publishTestObject(&heap, &handles, &young_allocator, 8);
    const new_field = try publishTestObject(&heap, &handles, &young_allocator, 8);
    const new_array = try publishTestObject(&heap, &handles, &young_allocator, 8);
    const new_static = try publishTestObject(&heap, &handles, &young_allocator, 8);
    try std.testing.expectEqual(@as(u8, 0), (try handles.inspect(destination)).region_id);
    try std.testing.expectEqual(@as(u8, 0), (try handles.inspect(array)).region_id);
    try std.testing.expectEqual(@as(u8, 1), (try handles.inspect(old_field)).region_id);

    const destination_address = @intFromPtr(try handles.resolve(destination));
    const field_slot: *std.atomic.Value(u64) = @ptrFromInt(destination_address);
    field_slot.store(@bitCast(old_field), .release);
    const array_address = @intFromPtr(try handles.resolve(array));
    const length_slot: *std.atomic.Value(u32) = @ptrFromInt(array_address);
    length_slot.store(2, .release);
    const array_slot: *std.atomic.Value(u64) = @ptrFromInt(array_address + 16);
    array_slot.store(@bitCast(old_array), .release);
    var static_slot = std.atomic.Value(u64).init(@bitCast(old_static));
    const static_address = @intFromPtr(&static_slot);

    var collector = try runtime_gc.ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 8,
        .max_satb_buffers = 1,
        .card_bytes = 64,
        .static_root_slots = &.{static_address},
    });
    defer collector.deinit() catch unreachable;
    try collector.setRegionKind(0, .old);
    try collector.setRegionKind(1, .young);
    var satb = try runtime_gc.SatbBuffer.init(std.testing.allocator, 4);
    defer satb.deinit() catch unreachable;
    try collector.registerSatbBuffer(&satb);
    defer collector.unregisterSatbBuffer(&satb) catch unreachable;

    const epoch = try collector.beginMark();
    _ = try collector.markHandle(destination);
    _ = try collector.markHandle(array);
    _ = try collector.markHandle(new_field);
    _ = try collector.markHandle(new_array);

    var memory = try Context.init(&collector, &satb, .{
        .reference_field_offsets = &.{0},
        .reference_array_layout = .{ .length_offset = 0, .data_offset = 8 },
        .static_field_layouts = &.{.{ .address = static_address }},
    });
    var registers = [_]u32{ 0, 0, 0, 1, 0, 0 };
    var references = [_]u64{
        @bitCast(destination), @bitCast(new_field), @bitCast(array), 0,
        @bitCast(new_array), @bitCast(new_static),
    };
    var reference_kinds = [_]bool{ true, true, true, false, true, true };
    const insts = [_]interpreter.Instruction{
        .{ .iput_object = .{ .field_idx = 0, .dest_or_src = 1, .obj = 0 } },
        .{ .aput_object = .{ .dest_or_src = 4, .array = 2, .index = 3 } },
        .{ .sput_object = .{ .field_idx = 0, .dest_or_src = 5 } },
        .return_void,
    };
    var frame = interpreter.ExecutionFrame{
        .pc = 0,
        .registers = &registers,
        .instructions = &insts,
        .register_is_ref = &reference_kinds,
        .reference_registers = &references,
        .managed_memory = memory.attachment(),
    };
    _ = try interpreter.execute(&frame);

    try std.testing.expectEqual(@as(u64, @bitCast(new_field)), field_slot.load(.acquire));
    try std.testing.expectEqual(@as(u64, @bitCast(new_array)), array_slot.load(.acquire));
    try std.testing.expectEqual(@as(u64, @bitCast(new_static)), static_slot.load(.acquire));
    try std.testing.expectEqual(@as(usize, 3), satb.pendingCount());
    try std.testing.expect(try collector.isCardDirty(destination));
    try std.testing.expect(try collector.isCardDirty(array));
    try std.testing.expectEqual(@as(u64, 1), collector.stats().static_roots_scanned);
    try std.testing.expectEqual(@as(u64, 1), collector.stats().static_root_writes);
    try std.testing.expectEqual(@as(u64, 1), memory.stats().instance_reference_stores);
    try std.testing.expectEqual(@as(u64, 1), memory.stats().array_reference_stores);
    try std.testing.expectEqual(@as(u64, 1), memory.stats().static_reference_stores);

    try satb.flushForEpoch(epoch);
    try std.testing.expectEqual(@as(usize, 3), try collector.drainSatb(8));
    while (try collector.traceWork(16) != 0) {}
    try collector.tryFinishMark();
    try std.testing.expect(try collector.isMarked(old_field));
    try std.testing.expect(try collector.isMarked(old_array));
    try std.testing.expect(try collector.isMarked(old_static));
    try std.testing.expect(try collector.isMarked(new_static));
}
