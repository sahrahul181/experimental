//! Managed-memory attachment for the runtime-independent interpreter.
//!
//! Access callbacks are owner-confined, allocation-free, and lock-free.
//! Reference stores use the same collector entry points as generated code:
//! exact-slot SATB before mutation, then either an object card or static-root
//! publication. Allocation callbacks use only the owner TLAB slow path.

const std = @import("std");
const interpreter = @import("interpreter");
const runtime_gc = @import("runtime_gc");
const runtime_heap = @import("runtime_heap");
const runtime_monitor = @import("runtime_monitor");
const runtime_method_sync = @import("runtime_method_sync");
const thread_registry = @import("runtime_thread_registry");
const runtime_value = @import("runtime_value");

const Handle = runtime_value.Handle;

pub const Error = runtime_gc.Error || runtime_monitor.Error || runtime_value.Error || error{
    InvalidLayout,
    MissingFieldLayout,
    MissingStaticFieldLayout,
    MissingInstanceLayout,
    MissingThreadBinding,
    ArrayIndexOutOfBounds,
    NegativeArraySize,
    InvalidMethod,
    MissingMethod,
    StackOverflow,
};

pub const ReferenceArrayLayout = struct {
    length_offset: u32,
    data_offset: u32,
    element_stride: u8 = @sizeOf(Handle),
};

pub const StaticFieldLayout = struct {
    address: usize,
};

pub const InstanceLayout = struct {
    payload_size: u32,
    alignment: u16 = runtime_value.object_alignment,
    gc_layout_id: u32 = 0,
    kind: u8 = 0,
    flags: u8 = 0,
};

pub const ReferenceArrayTypeLayout = struct {
    gc_layout_id: u32,
    alignment: u16 = runtime_value.object_alignment,
    kind: u8 = 0,
    flags: u8 = 0,
};

pub const ParameterKind = enum(u8) { scalar, reference };

/// Immutable exception class metadata. Handle.kind identifies the concrete
/// runtime class; `super_type_idx` forms a verified acyclic hierarchy.
pub const ExceptionType = struct {
    type_idx: u32,
    kind: u8,
    super_type_idx: ?u32 = null,
};

/// Immutable, verifier-produced interpreter target. Parameter destinations
/// are explicit Dalvik word registers; wide parameters occupy two scalar
/// entries and reference parameters occupy one full-width Handle entry.
pub const InterpretedMethod = struct {
    id: u32,
    instructions: []const interpreter.Instruction,
    register_count: u16,
    parameter_registers: []const u16 = &.{},
    parameter_kinds: []const ParameterKind = &.{},
    /// Decoded instruction-index ranges and targets, resolved from raw DEX
    /// code-unit PCs before this descriptor is published.
    try_blocks: []const interpreter.TryBlock = &.{},
    return_type: interpreter.ReturnType,
};

/// Fixed-capacity, owner-thread call arena. Nested calls slice this storage in
/// strict LIFO order; no runtime allocator or synchronization is involved.
pub const InterpreterCallStack = struct {
    owner: std.Thread.Id,
    registers: []u32,
    references: []u64,
    reference_kinds: []bool,
    cursor: usize = 0,
    depth: usize = 0,

    pub fn init(registers: []u32, references: []u64, reference_kinds: []bool) Error!InterpreterCallStack {
        if (registers.len != references.len or registers.len != reference_kinds.len) {
            return error.InvalidLayout;
        }
        return .{
            .owner = std.Thread.getCurrentId(),
            .registers = registers,
            .references = references,
            .reference_kinds = reference_kinds,
        };
    }

    const Storage = struct {
        stack: *InterpreterCallStack,
        mark: usize,
        registers: []u32,
        references: []u64,
        reference_kinds: []bool,

        fn release(self: *Storage) void {
            std.debug.assert(std.Thread.getCurrentId() == self.stack.owner);
            std.debug.assert(self.stack.cursor == self.mark + self.registers.len);
            @memset(self.registers, 0);
            @memset(self.references, interpreter.null_reference_bits);
            @memset(self.reference_kinds, false);
            self.stack.cursor = self.mark;
            self.stack.depth -= 1;
        }
    };

    fn acquire(self: *InterpreterCallStack, count: usize) Error!Storage {
        if (std.Thread.getCurrentId() != self.owner) return error.WrongThread;
        if (count > self.registers.len - self.cursor) return error.StackOverflow;
        const mark = self.cursor;
        self.cursor += count;
        self.depth += 1;
        const registers = self.registers[mark..self.cursor];
        const references = self.references[mark..self.cursor];
        const reference_kinds = self.reference_kinds[mark..self.cursor];
        @memset(registers, 0);
        @memset(references, interpreter.null_reference_bits);
        @memset(reference_kinds, false);
        return .{
            .stack = self,
            .mark = mark,
            .registers = registers,
            .references = references,
            .reference_kinds = reference_kinds,
        };
    }
};

pub const ThreadBinding = struct {
    registry: *thread_registry.Registry,
    context: *thread_registry.ThreadContext,
    allocator: *runtime_heap.ThreadAllocator,
};

pub const Options = struct {
    /// Reference-field payload offsets indexed by resolved field id.
    reference_field_offsets: []const u32 = &.{},
    reference_array_layout: ?ReferenceArrayLayout = null,
    static_field_layouts: []const StaticFieldLayout = &.{},
    instance_layouts: []const InstanceLayout = &.{},
    reference_array_types: []const ReferenceArrayTypeLayout = &.{},
    interpreted_methods: []const InterpretedMethod = &.{},
    exception_types: []const ExceptionType = &.{},
    monitor_table: ?*runtime_monitor.MonitorTable = null,
    method_synchronization: ?*const runtime_method_sync.Table = null,
    interpreter_stack: ?*InterpreterCallStack = null,
    thread_binding: ?ThreadBinding = null,
};

pub const Stats = struct {
    instance_reference_stores: u64,
    array_reference_stores: u64,
    static_reference_stores: u64,
    instance_reference_loads: u64,
    array_reference_loads: u64,
    static_reference_loads: u64,
    array_length_loads: u64,
    frames_entered: u64,
    frames_exited: u64,
    polls: u64,
    slow_polls: u64,
    instances_allocated: u64,
    arrays_allocated: u64,
    invocations: u64,
    invocation_returns: u64,
    exceptions_propagated: u64,
    stack_overflows: u64,
    max_call_depth: usize,
    max_frame_roots: usize,
    exception_type_checks: u64,
    exception_type_matches: u64,
    monitor_enters: u64,
    monitor_exits: u64,
    monitor_unwind_exits: u64,
    synchronized_invocations: u64,
    synchronized_returns: u64,
    synchronized_exceptions: u64,
    failures: u64,
};

pub const Context = struct {
    owner: std.Thread.Id,
    collector: *runtime_gc.ConcurrentCollector,
    satb: *runtime_gc.SatbBuffer,
    reference_field_offsets: []const u32,
    reference_array_layout: ?ReferenceArrayLayout,
    static_field_layouts: []const StaticFieldLayout,
    instance_layouts: []const InstanceLayout,
    reference_array_types: []const ReferenceArrayTypeLayout,
    interpreted_methods: []const InterpretedMethod,
    exception_types: []const ExceptionType,
    monitor_table: ?*runtime_monitor.MonitorTable,
    method_synchronization: ?*const runtime_method_sync.Table,
    interpreter_stack: ?*InterpreterCallStack,
    thread_binding: ?ThreadBinding,
    last_card_destination: u64 = 0,
    instance_reference_stores: u64 = 0,
    array_reference_stores: u64 = 0,
    static_reference_stores: u64 = 0,
    instance_reference_loads: u64 = 0,
    array_reference_loads: u64 = 0,
    static_reference_loads: u64 = 0,
    array_length_loads: u64 = 0,
    frames_entered: u64 = 0,
    frames_exited: u64 = 0,
    polls: u64 = 0,
    slow_polls: u64 = 0,
    instances_allocated: u64 = 0,
    arrays_allocated: u64 = 0,
    invocations: u64 = 0,
    invocation_returns: u64 = 0,
    exceptions_propagated: u64 = 0,
    stack_overflows: u64 = 0,
    max_call_depth: usize = 0,
    max_frame_roots: usize = 0,
    exception_type_checks: u64 = 0,
    exception_type_matches: u64 = 0,
    monitor_enters: u64 = 0,
    monitor_exits: u64 = 0,
    monitor_unwind_exits: u64 = 0,
    synchronized_invocations: u64 = 0,
    synchronized_returns: u64 = 0,
    synchronized_exceptions: u64 = 0,
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
        for (options.instance_layouts) |layout| {
            if (layout.payload_size == 0 or
                layout.alignment < runtime_value.object_alignment or
                !std.math.isPowerOfTwo(layout.alignment)) return error.InvalidLayout;
        }
        for (options.reference_array_types) |layout| {
            const array_layout = options.reference_array_layout orelse return error.InvalidLayout;
            if (layout.gc_layout_id == 0 or
                layout.alignment < runtime_value.object_alignment or
                !std.math.isPowerOfTwo(layout.alignment) or
                !collector.supportsTrailingReferenceLayout(
                    layout.gc_layout_id,
                    array_layout.data_offset,
                    array_layout.element_stride,
                )) return error.InvalidLayout;
        }
        try validateExceptionTypes(options.exception_types);
        try validateInterpretedMethods(options.interpreted_methods, options.exception_types);
        if (options.monitor_table) |monitors| {
            if (monitors.handleTable() != collector.handleTable()) return error.InvalidLayout;
            for (monitors.rootSlotAddresses()) |address| {
                if (!collector.isStaticRootSlot(address)) return error.InvalidLayout;
            }
        }
        if (options.method_synchronization) |synchronization| {
            if (synchronization.collectorDomain() != collector or options.monitor_table == null) return error.InvalidMethod;
            for (options.interpreted_methods) |method| {
                const entry = synchronization.find(method.id) orelse continue;
                switch (entry.target) {
                    .instance_parameter => |parameter| {
                        if (parameter >= method.parameter_kinds.len or method.parameter_kinds[parameter] != .reference) {
                            return error.InvalidMethod;
                        }
                    },
                    .static_root_slot => {},
                }
            }
        }
        if (options.interpreted_methods.len != 0) {
            const stack = options.interpreter_stack orelse return error.InvalidMethod;
            if (stack.owner != std.Thread.getCurrentId()) return error.WrongThread;
        } else if (options.interpreter_stack != null) {
            return error.InvalidMethod;
        }
        if (options.thread_binding) |binding| {
            if (!binding.context.isRunning() or
                binding.allocator.heap != collector.heap or
                !collector.ownsThreadBuffer(satb, binding.context)) return error.MissingThreadBinding;
        } else if (options.instance_layouts.len != 0 or
            options.reference_array_types.len != 0 or
            options.interpreted_methods.len != 0 or
            options.exception_types.len != 0 or
            options.monitor_table != null or
            options.method_synchronization != null)
        {
            return error.MissingThreadBinding;
        }
        return .{
            .owner = std.Thread.getCurrentId(),
            .collector = collector,
            .satb = satb,
            .reference_field_offsets = options.reference_field_offsets,
            .reference_array_layout = options.reference_array_layout,
            .static_field_layouts = options.static_field_layouts,
            .instance_layouts = options.instance_layouts,
            .reference_array_types = options.reference_array_types,
            .interpreted_methods = options.interpreted_methods,
            .exception_types = options.exception_types,
            .monitor_table = options.monitor_table,
            .method_synchronization = options.method_synchronization,
            .interpreter_stack = options.interpreter_stack,
            .thread_binding = options.thread_binding,
        };
    }

    pub fn attachment(self: *Context) interpreter.ManagedMemory {
        return .{ .context = self, .vtable = if (self.thread_binding == null) &barrier_vtable else &managed_vtable };
    }

    pub fn stats(self: *const Context) Stats {
        return .{
            .instance_reference_stores = self.instance_reference_stores,
            .array_reference_stores = self.array_reference_stores,
            .static_reference_stores = self.static_reference_stores,
            .instance_reference_loads = self.instance_reference_loads,
            .array_reference_loads = self.array_reference_loads,
            .static_reference_loads = self.static_reference_loads,
            .array_length_loads = self.array_length_loads,
            .frames_entered = self.frames_entered,
            .frames_exited = self.frames_exited,
            .polls = self.polls,
            .slow_polls = self.slow_polls,
            .instances_allocated = self.instances_allocated,
            .arrays_allocated = self.arrays_allocated,
            .invocations = self.invocations,
            .invocation_returns = self.invocation_returns,
            .exceptions_propagated = self.exceptions_propagated,
            .stack_overflows = self.stack_overflows,
            .max_call_depth = self.max_call_depth,
            .max_frame_roots = self.max_frame_roots,
            .exception_type_checks = self.exception_type_checks,
            .exception_type_matches = self.exception_type_matches,
            .monitor_enters = self.monitor_enters,
            .monitor_exits = self.monitor_exits,
            .monitor_unwind_exits = self.monitor_unwind_exits,
            .synchronized_invocations = self.synchronized_invocations,
            .synchronized_returns = self.synchronized_returns,
            .synchronized_exceptions = self.synchronized_exceptions,
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

    const ObjectView = struct { base: usize, size: usize };

    fn objectView(self: *Context, object: Handle) Error!ObjectView {
        if (object.isNull()) return error.InvalidHandle;
        const location = try self.collector.handleTable().inspect(object);
        switch (location.state) {
            .live, .evacuating => {},
            else => return error.InvalidHandle,
        }
        const region = try self.collector.handleTable().regionAt(location.region_id);
        const offset = std.math.mul(
            usize,
            @as(usize, location.offset_units),
            runtime_value.object_alignment,
        ) catch return error.InvalidLayout;
        if (offset >= region.len) return error.InvalidLayout;
        return .{
            .base = std.math.add(usize, region.base, offset) catch return error.InvalidLayout,
            .size = try self.collector.heap.objectSize(location.region_id, offset),
        };
    }

    fn referenceFieldSlot(self: *Context, object: Handle, field_idx: u32) Error!usize {
        if (field_idx >= self.reference_field_offsets.len) return error.MissingFieldLayout;
        const view = try self.objectView(object);
        const offset: usize = self.reference_field_offsets[field_idx];
        if (offset > view.size or @sizeOf(Handle) > view.size - offset) return error.InvalidLayout;
        const address = std.math.add(usize, view.base, offset) catch return error.InvalidLayout;
        if (!std.mem.isAligned(address, @alignOf(std.atomic.Value(u64)))) return error.InvalidLayout;
        return address;
    }

    fn referenceArraySlot(self: *Context, array: Handle, index: i32) Error!usize {
        const state = try self.referenceArrayState(array);
        if (index < 0 or @as(u32, @intCast(index)) >= state.length) return error.ArrayIndexOutOfBounds;
        const scaled = std.math.mul(usize, @intCast(index), state.layout.element_stride) catch return error.InvalidLayout;
        const data = std.math.add(usize, state.view.base, state.layout.data_offset) catch return error.InvalidLayout;
        const address = std.math.add(usize, data, scaled) catch return error.InvalidLayout;
        if (!std.mem.isAligned(address, @alignOf(std.atomic.Value(u64)))) return error.InvalidLayout;
        return address;
    }

    const ReferenceArrayState = struct {
        view: ObjectView,
        layout: ReferenceArrayLayout,
        length: u32,
    };

    fn referenceArrayState(self: *Context, array: Handle) Error!ReferenceArrayState {
        const layout = self.reference_array_layout orelse return error.InvalidLayout;
        const view = try self.objectView(array);
        if (layout.length_offset > view.size or @sizeOf(u32) > view.size - layout.length_offset) {
            return error.InvalidLayout;
        }
        const length_address = std.math.add(usize, view.base, layout.length_offset) catch return error.InvalidLayout;
        const length_slot: *const std.atomic.Value(u32) = @ptrFromInt(length_address);
        const length = length_slot.load(.acquire);
        const payload_bytes = std.math.mul(usize, length, layout.element_stride) catch return error.InvalidLayout;
        const required_size = std.math.add(usize, layout.data_offset, payload_bytes) catch return error.InvalidLayout;
        // The heap may round or reserve trailing zeroed slots. The logical
        // extent must fit; collector scanning safely treats extra slots as
        // null references because every publication zero-initializes them.
        if (required_size > view.size) return error.InvalidLayout;
        return .{ .view = view, .layout = layout, .length = length };
    }

    fn interpretedMethod(self: *const Context, invocation: *const interpreter.Invoke) Error!*const InterpretedMethod {
        if (invocation.call_target) |index| {
            if (index >= self.interpreted_methods.len) return error.MissingMethod;
            return &self.interpreted_methods[index];
        }
        const method_id = switch (invocation.target) {
            .method_idx => |id| id,
            .call_site_idx => return error.MissingMethod,
        };
        var low: usize = 0;
        var high = self.interpreted_methods.len;
        while (low < high) {
            const middle = low + (high - low) / 2;
            const candidate = &self.interpreted_methods[middle];
            if (candidate.id < method_id) {
                low = middle + 1;
            } else if (candidate.id > method_id) {
                high = middle;
            } else return candidate;
        }
        return error.MissingMethod;
    }
};

fn findExceptionType(types: []const ExceptionType, type_idx: u32) ?*const ExceptionType {
    var low: usize = 0;
    var high = types.len;
    while (low < high) {
        const middle = low + (high - low) / 2;
        const candidate = &types[middle];
        if (candidate.type_idx < type_idx) {
            low = middle + 1;
        } else if (candidate.type_idx > type_idx) {
            high = middle;
        } else return candidate;
    }
    return null;
}

fn validateExceptionTypes(types: []const ExceptionType) Error!void {
    var previous_type: u32 = 0;
    for (types, 0..) |entry, index| {
        if (index != 0 and entry.type_idx <= previous_type) return error.InvalidMethod;
        previous_type = entry.type_idx;
        for (types[0..index]) |previous| {
            if (previous.kind == entry.kind) return error.InvalidMethod;
        }
        if (entry.super_type_idx) |parent| {
            if (parent == entry.type_idx or findExceptionType(types, parent) == null) return error.InvalidMethod;
        }
    }
    for (types) |entry| {
        var current = entry.super_type_idx;
        var hops: usize = 0;
        while (current) |type_idx| {
            if (hops >= types.len) return error.InvalidMethod;
            const parent = findExceptionType(types, type_idx) orelse return error.InvalidMethod;
            current = parent.super_type_idx;
            hops += 1;
        }
    }
}

fn validateTryBlocks(method: InterpretedMethod, exception_types: []const ExceptionType) Error!void {
    for (method.try_blocks, 0..) |try_block, block_index| {
        if (try_block.start_pc >= try_block.end_pc or
            try_block.end_pc > method.instructions.len or
            try_block.handlers.len == 0)
        {
            return error.InvalidMethod;
        }
        if (block_index != 0) {
            const previous = method.try_blocks[block_index - 1];
            if (try_block.start_pc < previous.start_pc or
                (try_block.start_pc == previous.start_pc and try_block.end_pc <= previous.end_pc))
            {
                return error.InvalidMethod;
            }
        }
        for (method.try_blocks[0..block_index]) |previous| {
            const overlaps = try_block.start_pc < previous.end_pc and previous.start_pc < try_block.end_pc;
            if (!overlaps) continue;
            const previous_contains = previous.start_pc <= try_block.start_pc and previous.end_pc >= try_block.end_pc;
            const current_contains = try_block.start_pc <= previous.start_pc and try_block.end_pc >= previous.end_pc;
            if (!previous_contains and !current_contains) return error.InvalidMethod;
        }
        for (try_block.handlers, 0..) |handler, handler_index| {
            if (handler.target_pc >= method.instructions.len or
                std.meta.activeTag(method.instructions[handler.target_pc]) != .move_exception)
            {
                return error.InvalidMethod;
            }
            if (handler.type_idx) |type_idx| {
                if (findExceptionType(exception_types, type_idx) == null) return error.InvalidMethod;
                for (try_block.handlers[0..handler_index]) |previous| {
                    if (previous.type_idx != null and previous.type_idx.? == type_idx) return error.InvalidMethod;
                }
            } else if (handler_index + 1 != try_block.handlers.len) {
                return error.InvalidMethod;
            }
        }
    }
}

fn validateInterpretedMethods(methods: []const InterpretedMethod, exception_types: []const ExceptionType) Error!void {
    var previous_id: u32 = 0;
    for (methods, 0..) |method, index| {
        if ((index != 0 and method.id <= previous_id) or
            method.instructions.len == 0 or
            method.parameter_registers.len != method.parameter_kinds.len)
        {
            return error.InvalidMethod;
        }
        previous_id = method.id;
        for (method.parameter_registers, 0..) |destination, parameter_index| {
            if (destination >= method.register_count) return error.InvalidMethod;
            for (method.parameter_registers[0..parameter_index]) |previous| {
                if (previous == destination) return error.InvalidMethod;
            }
        }
        try validateTryBlocks(method, exception_types);
    }
}

test "interpreted method metadata rejects ambiguous frame layouts" {
    const code = [_]interpreter.Instruction{.return_void};
    const destinations = [_]u16{0};
    const duplicate_destinations = [_]u16{ 0, 0 };
    const one_kind = [_]ParameterKind{.scalar};
    const two_kinds = [_]ParameterKind{ .scalar, .reference };
    try validateInterpretedMethods(&.{.{
        .id = 1,
        .instructions = &code,
        .register_count = 1,
        .parameter_registers = &destinations,
        .parameter_kinds = &one_kind,
        .return_type = .void,
    }}, &.{});
    try std.testing.expectError(error.InvalidMethod, validateInterpretedMethods(&.{.{
        .id = 1,
        .instructions = &.{},
        .register_count = 0,
        .return_type = .void,
    }}, &.{}));
    try std.testing.expectError(error.InvalidMethod, validateInterpretedMethods(&.{.{
        .id = 1,
        .instructions = &code,
        .register_count = 1,
        .parameter_registers = &duplicate_destinations,
        .parameter_kinds = &two_kinds,
        .return_type = .void,
    }}, &.{}));
    try std.testing.expectError(error.InvalidMethod, validateInterpretedMethods(&.{
        .{ .id = 2, .instructions = &code, .register_count = 0, .return_type = .void },
        .{ .id = 2, .instructions = &code, .register_count = 0, .return_type = .void },
    }, &.{}));

    const exception_types = [_]ExceptionType{
        .{ .type_idx = 10, .kind = 1 },
        .{ .type_idx = 20, .kind = 2, .super_type_idx = 10 },
    };
    try validateExceptionTypes(&exception_types);
    try std.testing.expectError(error.InvalidMethod, validateExceptionTypes(&.{
        .{ .type_idx = 10, .kind = 1, .super_type_idx = 20 },
        .{ .type_idx = 20, .kind = 2, .super_type_idx = 10 },
    }));
    try std.testing.expectError(error.InvalidMethod, validateExceptionTypes(&.{
        .{ .type_idx = 10, .kind = 1 },
        .{ .type_idx = 20, .kind = 1 },
    }));

    const handler_code = [_]interpreter.Instruction{
        .return_void,
        .return_void,
        .return_void,
        .return_void,
        .return_void,
        .{ .move_exception = .{ .dest = 0 } },
    };
    const valid_handlers = [_]interpreter.CatchHandler{.{ .type_idx = 10, .target_pc = 5 }};
    const valid_tries = [_]interpreter.TryBlock{.{ .start_pc = 0, .end_pc = 1, .handlers = &valid_handlers }};
    try validateInterpretedMethods(&.{.{
        .id = 1,
        .instructions = &handler_code,
        .register_count = 1,
        .try_blocks = &valid_tries,
        .return_type = .void,
    }}, &exception_types);

    const bad_target_handlers = [_]interpreter.CatchHandler{.{ .type_idx = 10, .target_pc = 0 }};
    const bad_target_tries = [_]interpreter.TryBlock{.{ .start_pc = 0, .end_pc = 1, .handlers = &bad_target_handlers }};
    try std.testing.expectError(error.InvalidMethod, validateInterpretedMethods(&.{.{
        .id = 1,
        .instructions = &handler_code,
        .register_count = 1,
        .try_blocks = &bad_target_tries,
        .return_type = .void,
    }}, &exception_types));

    const ordered_handlers = [_]interpreter.CatchHandler{
        .{ .type_idx = null, .target_pc = 5 },
        .{ .type_idx = 10, .target_pc = 5 },
    };
    const badly_ordered_tries = [_]interpreter.TryBlock{.{ .start_pc = 0, .end_pc = 1, .handlers = &ordered_handlers }};
    try std.testing.expectError(error.InvalidMethod, validateInterpretedMethods(&.{.{
        .id = 1,
        .instructions = &handler_code,
        .register_count = 1,
        .try_blocks = &badly_ordered_tries,
        .return_type = .void,
    }}, &exception_types));

    const crossing_tries = [_]interpreter.TryBlock{
        .{ .start_pc = 0, .end_pc = 3, .handlers = &valid_handlers },
        .{ .start_pc = 2, .end_pc = 4, .handlers = &valid_handlers },
    };
    try std.testing.expectError(error.InvalidMethod, validateInterpretedMethods(&.{.{
        .id = 1,
        .instructions = &handler_code,
        .register_count = 1,
        .try_blocks = &crossing_tries,
        .return_type = .void,
    }}, &exception_types));
}

fn validateArrayLayout(layout: ReferenceArrayLayout) Error!void {
    if (layout.length_offset > std.math.maxInt(u32) - @sizeOf(u32)) return error.InvalidLayout;
    if (!std.mem.isAligned(layout.length_offset, @alignOf(std.atomic.Value(u32))) or
        !std.mem.isAligned(layout.data_offset, @alignOf(std.atomic.Value(u64))) or
        layout.element_stride != @sizeOf(Handle) or
        layout.data_offset < layout.length_offset + @sizeOf(u32)) return error.InvalidLayout;
}

fn refreshFrameRoots(
    context: *Context,
    state: *interpreter.ManagedFrameState,
    references: []u64,
    reference_kinds: []const bool,
) Error!void {
    if (!state.active or references.len != reference_kinds.len) return error.InvalidState;
    const binding = context.thread_binding orelse return error.MissingThreadBinding;
    try binding.context.restoreRootMark(state.root_mark);
    errdefer binding.context.restoreRootMark(state.root_mark) catch unreachable;
    for (reference_kinds, 0..) |is_reference, index| {
        if (!is_reference) continue;
        const slot: *const Handle = @ptrCast(&references[index]);
        try binding.context.addRoot(slot);
    }
    context.max_frame_roots = @max(context.max_frame_roots, binding.context.rootCount());
}

fn enterFrame(
    raw: *anyopaque,
    state: *interpreter.ManagedFrameState,
    references: []u64,
    reference_kinds: []const bool,
) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    if (std.Thread.getCurrentId() != context.owner) return .failure;
    if (state.active) return failed(context);
    const binding = context.thread_binding orelse return failed(context);
    state.root_mark = binding.context.rootMark() catch return failed(context);
    state.active = true;
    refreshFrameRoots(context, state, references, reference_kinds) catch {
        binding.context.restoreRootMark(state.root_mark) catch {};
        state.active = false;
        return failed(context);
    };
    context.frames_entered += 1;
    return .ok;
}

fn pollFrame(
    raw: *anyopaque,
    state: *interpreter.ManagedFrameState,
    references: []u64,
    reference_kinds: []const bool,
) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    if (std.Thread.getCurrentId() != context.owner) return .failure;
    const binding = context.thread_binding orelse return failed(context);
    refreshFrameRoots(context, state, references, reference_kinds) catch return failed(context);
    context.polls += 1;
    if (binding.context.observedEpoch() == binding.registry.requestEpoch()) return .ok;
    if (context.collector.phase() == .marking) {
        context.satb.flushForEpoch(context.collector.epoch()) catch return failed(context);
    }
    const waited = binding.registry.poll(binding.context) catch return failed(context);
    context.slow_polls += @intFromBool(waited);
    return .ok;
}

fn leaveFrame(raw: *anyopaque, state: *interpreter.ManagedFrameState) void {
    const context: *Context = @ptrCast(@alignCast(raw));
    if (std.Thread.getCurrentId() != context.owner) return;
    const binding = context.thread_binding orelse {
        context.failures += 1;
        return;
    };
    if (!state.active) {
        context.failures += 1;
        return;
    }
    if (state.held_monitor_count != 0) {
        const monitors = context.monitor_table orelse {
            context.failures += 1;
            return;
        };
        while (state.held_monitor_count != 0) {
            const index = state.held_monitor_count - 1;
            const handle: Handle = @bitCast(state.held_monitors[index]);
            monitors.exit(handle, context.collector, context.satb) catch {
                context.failures += 1;
                return;
            };
            state.held_monitors[index] = interpreter.null_reference_bits;
            state.held_monitor_count -= 1;
            context.monitor_unwind_exits += 1;
        }
    }
    state.synchronized_monitor_owned = false;
    binding.context.restoreRootMark(state.root_mark) catch {
        context.failures += 1;
        return;
    };
    state.active = false;
    context.frames_exited += 1;
}

fn allocateInstanceInner(context: *Context, type_idx: u32) Error!Handle {
    const binding = context.thread_binding orelse return error.MissingThreadBinding;
    if (type_idx >= context.instance_layouts.len) return error.MissingInstanceLayout;
    const layout = context.instance_layouts[type_idx];
    const handle = try context.collector.handleTable().reserve(layout.kind, layout.flags);
    errdefer context.collector.handleTable().cancelReservation(handle) catch {};
    const reservation = try binding.allocator.allocate(layout.payload_size, layout.alignment);
    const bytes: [*]u8 = @ptrCast(reservation.address());
    @memset(bytes[0..reservation.allocated_size], 0);
    for (0..1024) |_| {
        context.collector.publishAllocatedObject(reservation, handle, layout.gc_layout_id) catch |err| switch (err) {
            error.RetryBarrier => {
                std.atomic.spinLoopHint();
                continue;
            },
            else => return err,
        };
        return handle;
    }
    return error.RetryBarrier;
}

fn allocateInstance(raw: *anyopaque, type_idx: u32, output: *u64) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    if (std.Thread.getCurrentId() != context.owner) return .failure;
    const handle = allocateInstanceInner(context, type_idx) catch return failed(context);
    output.* = @bitCast(handle);
    context.instances_allocated += 1;
    return .ok;
}

fn allocateArrayInner(context: *Context, type_idx: u32, length: i32) Error!Handle {
    if (length < 0) return error.NegativeArraySize;
    if (type_idx >= context.reference_array_types.len) return error.InvalidLayout;
    const physical = context.reference_array_layout orelse return error.InvalidLayout;
    const logical = context.reference_array_types[type_idx];
    const payload_bytes = std.math.mul(
        usize,
        @as(usize, @intCast(length)),
        physical.element_stride,
    ) catch return error.InvalidLayout;
    const payload_size = std.math.add(usize, physical.data_offset, payload_bytes) catch return error.InvalidLayout;

    const handle = try context.collector.handleTable().reserve(logical.kind, logical.flags);
    errdefer context.collector.handleTable().cancelReservation(handle) catch {};
    const binding = context.thread_binding orelse return error.MissingThreadBinding;
    const reservation = try binding.allocator.allocate(payload_size, logical.alignment);
    const bytes: [*]u8 = @ptrCast(reservation.address());
    @memset(bytes[0..reservation.allocated_size], 0);
    const length_address = std.math.add(
        usize,
        @intFromPtr(reservation.address()),
        physical.length_offset,
    ) catch return error.InvalidLayout;
    const length_slot: *std.atomic.Value(u32) = @ptrFromInt(length_address);
    length_slot.store(@intCast(length), .monotonic);

    for (0..1024) |_| {
        context.collector.publishAllocatedObject(reservation, handle, logical.gc_layout_id) catch |err| switch (err) {
            error.RetryBarrier => {
                std.atomic.spinLoopHint();
                continue;
            },
            else => return err,
        };
        return handle;
    }
    return error.RetryBarrier;
}

fn allocateArray(raw: *anyopaque, type_idx: u32, length: i32, output: *u64) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    if (std.Thread.getCurrentId() != context.owner) return .failure;
    const handle = allocateArrayInner(context, type_idx, length) catch |err| switch (err) {
        error.NegativeArraySize => return .negative_array_size,
        else => return failed(context),
    };
    output.* = @bitCast(handle);
    context.arrays_allocated += 1;
    return .ok;
}

fn failed(context: *Context) interpreter.ManagedMemoryStatus {
    context.failures += 1;
    return .failure;
}

fn loadInstanceReference(raw: *anyopaque, object_bits: u64, field_idx: u32, output: *u64) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    if (std.Thread.getCurrentId() != context.owner) return .failure;
    const object: Handle = @bitCast(object_bits);
    if (object.isNull()) return .null_reference;
    const slot_address = context.referenceFieldSlot(object, field_idx) catch return failed(context);
    const slot: *const std.atomic.Value(u64) = @ptrFromInt(slot_address);
    output.* = slot.load(.acquire);
    context.instance_reference_loads += 1;
    return .ok;
}

fn loadArrayReference(raw: *anyopaque, array_bits: u64, index: i32, output: *u64) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    if (std.Thread.getCurrentId() != context.owner) return .failure;
    const array: Handle = @bitCast(array_bits);
    if (array.isNull()) return .null_reference;
    const slot_address = context.referenceArraySlot(array, index) catch |err| switch (err) {
        error.ArrayIndexOutOfBounds => return .array_index_out_of_bounds,
        else => return failed(context),
    };
    const slot: *const std.atomic.Value(u64) = @ptrFromInt(slot_address);
    output.* = slot.load(.acquire);
    context.array_reference_loads += 1;
    return .ok;
}

fn loadStaticReference(raw: *anyopaque, field_idx: u32, output: *u64) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    if (std.Thread.getCurrentId() != context.owner) return .failure;
    if (field_idx >= context.static_field_layouts.len) return failed(context);
    const slot: *const std.atomic.Value(u64) = @ptrFromInt(context.static_field_layouts[field_idx].address);
    output.* = slot.load(.acquire);
    context.static_reference_loads += 1;
    return .ok;
}

fn loadArrayLength(raw: *anyopaque, array_bits: u64, output: *u32) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    if (std.Thread.getCurrentId() != context.owner) return .failure;
    const array: Handle = @bitCast(array_bits);
    if (array.isNull()) return .null_reference;
    const state = context.referenceArrayState(array) catch return failed(context);
    output.* = state.length;
    context.array_length_loads += 1;
    return .ok;
}

fn exceptionMatches(raw: *anyopaque, exception_bits: u64, catch_type_idx: u32, output: *bool) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    if (std.Thread.getCurrentId() != context.owner) return .failure;
    const exception: Handle = @bitCast(exception_bits);
    if (exception.isNull()) return .null_reference;
    const location = context.collector.handleTable().inspect(exception) catch return failed(context);
    switch (location.state) {
        .live, .evacuating => {},
        else => return failed(context),
    }
    const catch_type = findExceptionType(context.exception_types, catch_type_idx) orelse return failed(context);
    _ = catch_type;
    var concrete: ?*const ExceptionType = null;
    for (context.exception_types) |*candidate| {
        if (candidate.kind == exception.kind) {
            concrete = candidate;
            break;
        }
    }
    var current = concrete orelse return failed(context);
    output.* = false;
    for (0..context.exception_types.len) |_| {
        context.exception_type_checks += 1;
        if (current.type_idx == catch_type_idx) {
            output.* = true;
            context.exception_type_matches += 1;
            return .ok;
        }
        const parent = current.super_type_idx orelse return .ok;
        current = findExceptionType(context.exception_types, parent) orelse return failed(context);
    }
    return failed(context);
}

fn monitorEnter(raw: *anyopaque, object_bits: u64) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    if (std.Thread.getCurrentId() != context.owner) return .failure;
    const object: Handle = @bitCast(object_bits);
    if (object.isNull()) return .null_reference;
    const monitors = context.monitor_table orelse return failed(context);
    const binding = context.thread_binding orelse return failed(context);
    monitors.enter(object, context.collector, binding.registry, binding.context, context.satb) catch return failed(context);
    context.monitor_enters += 1;
    return .ok;
}

fn monitorExit(raw: *anyopaque, object_bits: u64) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    if (std.Thread.getCurrentId() != context.owner) return .failure;
    const object: Handle = @bitCast(object_bits);
    if (object.isNull()) return .null_reference;
    const monitors = context.monitor_table orelse return failed(context);
    monitors.exit(object, context.collector, context.satb) catch |err| switch (err) {
        error.IllegalMonitorState => return .illegal_monitor_state,
        else => return failed(context),
    };
    context.monitor_exits += 1;
    return .ok;
}

fn invokeInterpretedMethod(
    raw: *anyopaque,
    invocation: *const interpreter.Invoke,
    caller_registers: []const u32,
    caller_references: []const u64,
    caller_reference_kinds: []const bool,
    output: *interpreter.ExecutionResult,
    exception_output: *u64,
) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    if (std.Thread.getCurrentId() != context.owner) return .failure;
    if (caller_registers.len != caller_references.len or
        caller_registers.len != caller_reference_kinds.len)
    {
        return failed(context);
    }
    const method = context.interpretedMethod(invocation) catch return failed(context);
    if (invocation.args.len != method.parameter_registers.len) return failed(context);
    var synchronized_monitor: ?u64 = null;
    if (context.method_synchronization) |synchronization| if (synchronization.find(method.id)) |entry| {
        synchronized_monitor = switch (entry.target) {
            .instance_parameter => |parameter| blk: {
                if (invocation.kind == .static or parameter >= invocation.args.len) return failed(context);
                const source: usize = invocation.args[parameter];
                if (source >= caller_references.len or !caller_reference_kinds[source]) return failed(context);
                const handle: Handle = @bitCast(caller_references[source]);
                if (handle.isNull()) return .null_reference;
                break :blk @bitCast(handle);
            },
            .static_root_slot => blk: {
                if (invocation.kind != .static) return failed(context);
                const handle = synchronization.loadStatic(entry) catch return failed(context);
                break :blk @bitCast(handle);
            },
        };
    };
    if (invocation.kind != .static) {
        if (invocation.args.len == 0) return failed(context);
        const receiver: usize = invocation.args[0];
        if (receiver >= caller_registers.len or !caller_reference_kinds[receiver]) return failed(context);
        const receiver_handle: Handle = @bitCast(caller_references[receiver]);
        if (receiver_handle.isNull()) return .null_reference;
    }

    const stack = context.interpreter_stack orelse return failed(context);
    var storage = stack.acquire(method.register_count) catch |err| switch (err) {
        error.StackOverflow => {
            context.stack_overflows += 1;
            return .stack_overflow;
        },
        else => return failed(context),
    };
    defer storage.release();

    for (
        invocation.args,
        method.parameter_registers,
        method.parameter_kinds,
    ) |source_register, destination_register, parameter_kind| {
        if (source_register >= caller_registers.len) return failed(context);
        switch (parameter_kind) {
            .scalar => {
                if (caller_reference_kinds[source_register]) return failed(context);
                storage.registers[destination_register] = caller_registers[source_register];
            },
            .reference => {
                if (!caller_reference_kinds[source_register]) return failed(context);
                const bits = caller_references[source_register];
                storage.registers[destination_register] = @truncate(bits);
                storage.references[destination_register] = bits;
                storage.reference_kinds[destination_register] = true;
            },
        }
    }

    context.invocations += 1;
    context.max_call_depth = @max(context.max_call_depth, stack.depth);
    var frame = interpreter.ExecutionFrame{
        .pc = 0,
        .registers = storage.registers,
        .instructions = method.instructions,
        .register_is_ref = storage.reference_kinds,
        .reference_registers = storage.references,
        .try_blocks = method.try_blocks,
        .managed_memory = context.attachment(),
        .synchronized_monitor = synchronized_monitor,
    };
    if (synchronized_monitor != null) context.synchronized_invocations += 1;
    const result = interpreter.execute(&frame) catch |err| switch (err) {
        error.ManagedException => {
            if (@as(u32, @truncate(frame.pending_exception)) == std.math.maxInt(u32)) {
                return failed(context);
            }
            exception_output.* = frame.pending_exception;
            context.exceptions_propagated += 1;
            if (synchronized_monitor != null) context.synchronized_exceptions += 1;
            return .managed_exception;
        },
        error.StackOverflow => return .stack_overflow,
        error.NullReference => return .null_reference,
        error.ArrayIndexOutOfBounds => return .array_index_out_of_bounds,
        error.NegativeArraySize => return .negative_array_size,
        else => return failed(context),
    };
    if (result.kind != method.return_type) return failed(context);
    output.* = result;
    context.invocation_returns += 1;
    if (synchronized_monitor != null) context.synchronized_returns += 1;
    return .ok;
}

fn storeInstanceReference(raw: *anyopaque, object_bits: u64, field_idx: u32, stored_bits: u64) interpreter.ManagedMemoryStatus {
    const context: *Context = @ptrCast(@alignCast(raw));
    if (std.Thread.getCurrentId() != context.owner) return .failure;
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
    if (std.Thread.getCurrentId() != context.owner) return .failure;
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
    if (std.Thread.getCurrentId() != context.owner) return .failure;
    if (field_idx >= context.static_field_layouts.len) return failed(context);
    const slot_address = context.static_field_layouts[field_idx].address;
    context.preWrite(slot_address) catch return failed(context);
    const slot: *std.atomic.Value(u64) = @ptrFromInt(slot_address);
    slot.store(stored_bits, .release);
    context.staticPostWrite(slot_address, @bitCast(stored_bits)) catch return failed(context);
    context.static_reference_stores += 1;
    return .ok;
}

const barrier_vtable = interpreter.ManagedMemoryVTable{
    .load_instance_reference = loadInstanceReference,
    .load_array_reference = loadArrayReference,
    .load_static_reference = loadStaticReference,
    .array_length = loadArrayLength,
    .store_instance_reference = storeInstanceReference,
    .store_array_reference = storeArrayReference,
    .store_static_reference = storeStaticReference,
};

const managed_vtable = interpreter.ManagedMemoryVTable{
    .enter_frame = enterFrame,
    .poll_frame = pollFrame,
    .leave_frame = leaveFrame,
    .allocate_instance = allocateInstance,
    .allocate_array = allocateArray,
    .load_instance_reference = loadInstanceReference,
    .load_array_reference = loadArrayReference,
    .load_static_reference = loadStaticReference,
    .array_length = loadArrayLength,
    .invoke_method = invokeInterpretedMethod,
    .exception_matches = exceptionMatches,
    .monitor_enter = monitorEnter,
    .monitor_exit = monitorExit,
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

fn publishTypedTestObject(
    heap: *runtime_heap.ManagedHeap,
    handles: *runtime_value.HandleTable,
    allocator: *runtime_heap.ThreadAllocator,
    size: usize,
    kind: u8,
) !Handle {
    const reservation = try allocator.allocate(size, runtime_value.object_alignment);
    const handle = try handles.reserve(kind, 0);
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
    var registers = [_]u32{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
    var references = [_]u64{
        @bitCast(destination), @bitCast(new_field),  @bitCast(array), 0,
        @bitCast(new_array),   @bitCast(new_static), 0,               0,
        0,                     0,
    };
    var reference_kinds = [_]bool{ true, true, true, false, true, true, false, false, false, false };
    const insts = [_]interpreter.Instruction{
        .{ .iput_object = .{ .field_idx = 0, .dest_or_src = 1, .obj = 0 } },
        .{ .aput_object = .{ .dest_or_src = 4, .array = 2, .index = 3 } },
        .{ .sput_object = .{ .field_idx = 0, .dest_or_src = 5 } },
        .{ .iget_object = .{ .field_idx = 0, .dest_or_src = 6, .obj = 0 } },
        .{ .aget_object = .{ .dest_or_src = 7, .array = 2, .index = 3 } },
        .{ .sget_object = .{ .field_idx = 0, .dest_or_src = 8 } },
        .{ .array_length = .{ .dest = 9, .array = 2 } },
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
    try std.testing.expectEqual(@as(u64, @bitCast(new_field)), references[6]);
    try std.testing.expectEqual(@as(u64, @bitCast(new_array)), references[7]);
    try std.testing.expectEqual(@as(u64, @bitCast(new_static)), references[8]);
    try std.testing.expectEqual(@as(u32, 2), registers[9]);
    try std.testing.expectEqual(@as(usize, 3), satb.pendingCount());
    try std.testing.expect(try collector.isCardDirty(destination));
    try std.testing.expect(try collector.isCardDirty(array));
    try std.testing.expectEqual(@as(u64, 1), collector.stats().static_roots_scanned);
    try std.testing.expectEqual(@as(u64, 1), collector.stats().static_root_writes);
    try std.testing.expectEqual(@as(u64, 1), memory.stats().instance_reference_stores);
    try std.testing.expectEqual(@as(u64, 1), memory.stats().array_reference_stores);
    try std.testing.expectEqual(@as(u64, 1), memory.stats().static_reference_stores);
    try std.testing.expectEqual(@as(u64, 1), memory.stats().instance_reference_loads);
    try std.testing.expectEqual(@as(u64, 1), memory.stats().array_reference_loads);
    try std.testing.expectEqual(@as(u64, 1), memory.stats().static_reference_loads);
    try std.testing.expectEqual(@as(u64, 1), memory.stats().array_length_loads);

    try satb.flushForEpoch(epoch);
    try std.testing.expectEqual(@as(usize, 3), try collector.drainSatb(8));
    while (try collector.traceWork(16) != 0) {}
    try collector.tryFinishMark();
    try std.testing.expect(try collector.isMarked(old_field));
    try std.testing.expect(try collector.isMarked(old_array));
    try std.testing.expect(try collector.isMarked(old_static));
    try std.testing.expect(try collector.isMarked(new_static));
}

fn waitForStage(stage: *const std.atomic.Value(u8), expected: u8) !void {
    for (0..1_000_000) |_| {
        if (stage.load(.acquire) >= expected) return;
        std.atomic.spinLoopHint();
        std.Thread.yield() catch {};
    }
    return error.Timeout;
}

fn finishHandshake(handshake: *runtime_gc.MarkHandshake) !void {
    for (0..1_000_000) |_| {
        if (try handshake.advance()) return;
        std.atomic.spinLoopHint();
        std.Thread.yield() catch {};
    }
    return error.Timeout;
}

test "interpreter allocation and exact frame roots survive a concurrent mark handshake" {
    var storage: [2048]u8 align(runtime_value.object_alignment) = @splat(0xaa);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 8, &regions);
    defer handles.deinit();
    var heap = try runtime_heap.ManagedHeap.init(std.testing.allocator, &handles, 128);
    defer heap.deinit();
    var collector = try runtime_gc.ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 4,
        .max_satb_buffers = 1,
        .card_bytes = 64,
    });
    defer collector.deinit() catch unreachable;
    var registry = try thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;

    var stage = std.atomic.Value(u8).init(0);
    var command = std.atomic.Value(u8).init(0);
    var allocated_bits = std.atomic.Value(u64).init(@bitCast(Handle.none));
    var failure_code = std.atomic.Value(u8).init(0);

    const Worker = struct {
        heap: *runtime_heap.ManagedHeap,
        collector: *runtime_gc.ConcurrentCollector,
        registry: *thread_registry.Registry,
        stage: *std.atomic.Value(u8),
        command: *std.atomic.Value(u8),
        allocated_bits: *std.atomic.Value(u64),
        failure_code: *std.atomic.Value(u8),

        fn fail(self: *@This(), code: u8) void {
            self.failure_code.store(code, .release);
            self.stage.store(255, .release);
        }

        fn run(self: *@This()) void {
            var context = thread_registry.ThreadContext.init(std.testing.allocator, 4) catch return self.fail(1);
            defer context.deinit();
            var satb = runtime_gc.SatbBuffer.init(std.testing.allocator, 4) catch return self.fail(2);
            defer satb.deinit() catch self.fail(9);
            self.collector.registerThreadSatbBuffer(&satb, &context) catch return self.fail(3);
            defer self.collector.unregisterSatbBuffer(&satb) catch self.fail(8);
            self.registry.register(&context) catch return self.fail(4);
            defer self.registry.unregister(&context) catch self.fail(7);
            var worker_allocator = self.heap.threadAllocator();
            const layouts = [_]InstanceLayout{.{ .payload_size = 24 }};
            var runtime = Context.init(self.collector, &satb, .{
                .instance_layouts = &layouts,
                .thread_binding = .{
                    .registry = self.registry,
                    .context = &context,
                    .allocator = &worker_allocator,
                },
            }) catch return self.fail(5);

            var allocation_registers = [_]u32{0};
            var allocation_references = [_]u64{0};
            var allocation_kinds = [_]bool{false};
            const allocation_code = [_]interpreter.Instruction{
                .{ .new_instance = .{ .dest = 0, .type_idx = 0 } },
                .{ .return_object = .{ .src = 0 } },
            };
            var allocation_frame = interpreter.ExecutionFrame{
                .pc = 0,
                .registers = &allocation_registers,
                .instructions = &allocation_code,
                .register_is_ref = &allocation_kinds,
                .reference_registers = &allocation_references,
                .managed_memory = runtime.attachment(),
            };
            const result = interpreter.execute(&allocation_frame) catch return self.fail(6);
            if (result.kind != .object or result.value64 == @as(u64, @bitCast(Handle.none))) return self.fail(10);
            self.allocated_bits.store(result.value64, .release);

            // Slot zero deliberately retains stale handle bits but is scalar.
            // Only slot one may appear in the precise handshake snapshot.
            var registers = [_]u32{ 0, @truncate(result.value64) };
            var references = [_]u64{ result.value64, result.value64 };
            var kinds = [_]bool{ false, true };
            const code = [_]interpreter.Instruction{.{ .return_object = .{ .src = 1 } }};
            var frame = interpreter.ExecutionFrame{
                .pc = 0,
                .registers = &registers,
                .instructions = &code,
                .register_is_ref = &kinds,
                .reference_registers = &references,
                .managed_memory = runtime.attachment(),
            };
            self.stage.store(1, .release);
            while (self.command.load(.acquire) < 1) std.atomic.spinLoopHint();
            _ = interpreter.execute(&frame) catch return self.fail(11);
            if (context.rootCount() != 0 or runtime.stats().frames_entered != runtime.stats().frames_exited) return self.fail(12);
            self.stage.store(2, .release);
            while (self.command.load(.acquire) < 2) std.atomic.spinLoopHint();
        }
    };

    var worker = Worker{
        .heap = &heap,
        .collector = &collector,
        .registry = &registry,
        .stage = &stage,
        .command = &command,
        .allocated_bits = &allocated_bits,
        .failure_code = &failure_code,
    };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    var joined = false;
    defer if (!joined) {
        command.store(2, .release);
        thread.join();
    };
    try waitForStage(&stage, 1);
    try std.testing.expectEqual(@as(u8, 0), failure_code.load(.acquire));

    _ = try collector.beginMark();
    var handshake = try collector.beginThreadHandshake(&registry);
    command.store(1, .release);
    try finishHandshake(&handshake);
    try waitForStage(&stage, 2);

    const allocated: Handle = @bitCast(allocated_bits.load(.acquire));
    try std.testing.expect(try collector.isMarked(allocated));
    try std.testing.expectEqual(@as(u64, 1), collector.stats().roots_discovered);
    try std.testing.expectEqual(@as(usize, 1), try collector.traceWork(4));
    try collector.tryFinishMark();
    command.store(2, .release);
    thread.join();
    joined = true;
    try std.testing.expectEqual(@as(u8, 0), failure_code.load(.acquire));
    const object: [*]const u8 = @ptrCast(try handles.resolve(allocated));
    const zeroes: [24]u8 = @splat(0);
    try std.testing.expectEqualSlices(u8, &zeroes, object[0..24]);
}

test "reference arrays allocate trace relocate and serve concurrent readers" {
    var primary: [4096]u8 align(runtime_value.object_alignment) = @splat(0xaa);
    var relocation: [256]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&primary),
        try runtime_value.Region.fromSlice(&relocation),
    };
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 16, &regions);
    defer handles.deinit();
    var heap = try runtime_heap.ManagedHeap.init(std.testing.allocator, &handles, 256);
    defer heap.deinit();
    var allocator = heap.threadAllocator();
    const first = try publishTestObject(&heap, &handles, &allocator, 8);
    const second = try publishTestObject(&heap, &handles, &allocator, 8);

    const layouts = [_]runtime_gc.LayoutSpec{.{
        .id = 1,
        .minimum_size = 8,
        .trailing_references = .{ .offset = 8 },
    }};
    var collector = try runtime_gc.ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 16,
        .max_satb_buffers = 5,
        .card_bytes = 64,
        .layouts = &layouts,
    });
    defer collector.deinit() catch unreachable;
    try std.testing.expect(collector.supportsTrailingReferenceLayout(1, 8, @sizeOf(Handle)));
    try std.testing.expect(!collector.supportsTrailingReferenceLayout(1, 16, @sizeOf(Handle)));

    var registry = try thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var thread_context = try thread_registry.ThreadContext.init(std.testing.allocator, 8);
    defer thread_context.deinit();
    var satb = try runtime_gc.SatbBuffer.init(std.testing.allocator, 8);
    defer satb.deinit() catch unreachable;
    try collector.registerThreadSatbBuffer(&satb, &thread_context);
    var satb_registered = true;
    defer if (satb_registered) collector.unregisterSatbBuffer(&satb) catch unreachable;
    try registry.register(&thread_context);
    var thread_registered = true;
    defer if (thread_registered) registry.unregister(&thread_context) catch unreachable;

    const array_types = [_]ReferenceArrayTypeLayout{.{ .gc_layout_id = 1, .kind = 3, .flags = 5 }};
    const mismatched_array_types = [_]ReferenceArrayTypeLayout{.{ .gc_layout_id = 2 }};
    try std.testing.expectError(error.InvalidLayout, Context.init(&collector, &satb, .{
        .reference_array_layout = .{ .length_offset = 0, .data_offset = 8 },
        .reference_array_types = &mismatched_array_types,
        .thread_binding = .{
            .registry = &registry,
            .context = &thread_context,
            .allocator = &allocator,
        },
    }));
    var runtime = try Context.init(&collector, &satb, .{
        .reference_array_layout = .{ .length_offset = 0, .data_offset = 8 },
        .reference_array_types = &array_types,
        .thread_binding = .{
            .registry = &registry,
            .context = &thread_context,
            .allocator = &allocator,
        },
    });

    var registers = [_]u32{ 2, 0, 0, 0, 0, 1, 0, 0 };
    var references = [_]u64{ 0, 0, @bitCast(first), 0, @bitCast(second), 0, 0, 0 };
    var kinds = [_]bool{ false, false, true, false, true, false, false, false };
    const code = [_]interpreter.Instruction{
        .{ .new_array = .{ .type_idx = 0, .dest = 1, .size = 0 } },
        .{ .aput_object = .{ .dest_or_src = 2, .array = 1, .index = 3 } },
        .{ .aput_object = .{ .dest_or_src = 4, .array = 1, .index = 5 } },
        .{ .aget_object = .{ .dest_or_src = 6, .array = 1, .index = 5 } },
        .{ .array_length = .{ .dest = 7, .array = 1 } },
        .{ .return_object = .{ .src = 6 } },
    };
    var frame = interpreter.ExecutionFrame{
        .pc = 0,
        .registers = &registers,
        .instructions = &code,
        .register_is_ref = &kinds,
        .reference_registers = &references,
        .managed_memory = runtime.attachment(),
    };
    const result = try interpreter.execute(&frame);
    const array: Handle = @bitCast(references[1]);
    try std.testing.expectEqual(interpreter.ReturnType.object, result.kind);
    try std.testing.expectEqual(@as(u64, @bitCast(second)), result.value64);
    try std.testing.expectEqual(@as(u32, 2), registers[7]);
    try std.testing.expectEqual(@as(u8, 3), array.kind);
    try std.testing.expectEqual(@as(u8, 5), array.flags);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().arrays_allocated);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().array_reference_loads);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().array_length_loads);

    var output: u64 = 0;
    try std.testing.expectEqual(
        interpreter.ManagedMemoryStatus.null_reference,
        loadArrayReference(&runtime, @bitCast(Handle.none), 0, &output),
    );
    try std.testing.expectEqual(
        interpreter.ManagedMemoryStatus.array_index_out_of_bounds,
        loadArrayReference(&runtime, @bitCast(array), -1, &output),
    );
    try std.testing.expectEqual(
        interpreter.ManagedMemoryStatus.array_index_out_of_bounds,
        loadArrayReference(&runtime, @bitCast(array), 2, &output),
    );
    try std.testing.expectEqual(
        interpreter.ManagedMemoryStatus.negative_array_size,
        allocateArray(&runtime, 0, -1, &output),
    );

    const array_address = @intFromPtr(try handles.resolve(array));
    const length_slot: *std.atomic.Value(u32) = @ptrFromInt(array_address);
    length_slot.store(1024, .release);
    try std.testing.expectEqual(
        interpreter.ManagedMemoryStatus.failure,
        loadArrayReference(&runtime, @bitCast(array), 0, &output),
    );
    length_slot.store(2, .release);

    try std.testing.expectError(error.InvalidObject, allocateArrayInner(&runtime, 0, std.math.maxInt(i32)));
    const reusable = try handles.reserve(0, 0);
    try handles.cancelReservation(reusable);

    try registry.unregister(&thread_context);
    thread_registered = false;
    try collector.unregisterSatbBuffer(&satb);
    satb_registered = false;
    _ = try collector.beginMark();
    _ = try collector.markHandle(array);
    while (try collector.traceWork(16) != 0) {}
    try collector.tryFinishMark();
    try std.testing.expect(try collector.isMarked(array));
    try std.testing.expect(try collector.isMarked(first));
    try std.testing.expect(try collector.isMarked(second));

    const ticket = try handles.beginRelocation(second);
    try std.testing.expect(try handles.commitRelocation(ticket, 1, @ptrCast(&relocation[0])));
    try std.testing.expectEqual(@intFromPtr(&relocation[0]), @intFromPtr(try handles.resolve(second)));
    try std.testing.expectEqual(
        interpreter.ManagedMemoryStatus.ok,
        loadArrayReference(&runtime, @bitCast(array), 1, &output),
    );
    try std.testing.expectEqual(@as(u64, @bitCast(second)), output);

    var reader_failed = std.atomic.Value(bool).init(false);
    const Reader = struct {
        collector: *runtime_gc.ConcurrentCollector,
        array: Handle,
        expected: Handle,
        failed: *std.atomic.Value(bool),

        fn registerBuffer(self: *@This(), buffer: *runtime_gc.SatbBuffer) bool {
            for (0..100_000) |_| {
                self.collector.registerSatbBuffer(buffer) catch |err| switch (err) {
                    error.WrongPhase => {
                        std.atomic.spinLoopHint();
                        std.Thread.yield() catch {};
                        continue;
                    },
                    else => return false,
                };
                return true;
            }
            return false;
        }

        fn unregisterBuffer(self: *@This(), buffer: *runtime_gc.SatbBuffer) bool {
            for (0..100_000) |_| {
                self.collector.unregisterSatbBuffer(buffer) catch |err| switch (err) {
                    error.WrongPhase => {
                        std.atomic.spinLoopHint();
                        std.Thread.yield() catch {};
                        continue;
                    },
                    else => return false,
                };
                return true;
            }
            return false;
        }

        fn run(self: *@This()) void {
            var buffer = runtime_gc.SatbBuffer.init(std.testing.allocator, 2) catch {
                self.failed.store(true, .release);
                return;
            };
            defer buffer.deinit() catch self.failed.store(true, .release);
            if (!self.registerBuffer(&buffer)) {
                self.failed.store(true, .release);
                return;
            }
            defer if (!self.unregisterBuffer(&buffer)) self.failed.store(true, .release);
            var local = Context.init(self.collector, &buffer, .{
                .reference_array_layout = .{ .length_offset = 0, .data_offset = 8 },
            }) catch {
                self.failed.store(true, .release);
                return;
            };
            for (0..10_000) |_| {
                var bits: u64 = 0;
                if (loadArrayReference(&local, @bitCast(self.array), 1, &bits) != .ok or
                    bits != @as(u64, @bitCast(self.expected)))
                {
                    self.failed.store(true, .release);
                    return;
                }
            }
        }
    };
    var readers: [4]Reader = undefined;
    var threads: [4]std.Thread = undefined;
    for (&readers, &threads) |*reader, *thread| {
        reader.* = .{ .collector = &collector, .array = array, .expected = second, .failed = &reader_failed };
        thread.* = try std.Thread.spawn(.{}, Reader.run, .{reader});
    }
    for (&threads) |*thread| thread.join();
    try std.testing.expect(!reader_failed.load(.acquire));
}

test "managed invocation nests roots propagates exceptions and recovers from stack overflow" {
    var storage: [2048]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 8, &regions);
    defer handles.deinit();
    var heap = try runtime_heap.ManagedHeap.init(std.testing.allocator, &handles, 128);
    defer heap.deinit();
    var allocator = heap.threadAllocator();
    const exception = try publishTypedTestObject(&heap, &handles, &allocator, 8, 2);
    var collector = try runtime_gc.ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 8,
        .max_satb_buffers = 5,
        .card_bytes = 64,
    });
    defer collector.deinit() catch unreachable;
    var registry = try thread_registry.Registry.init(std.testing.allocator, std.testing.io, 5);
    defer registry.deinit() catch unreachable;

    const scalar_arg = [_]u16{0};
    const reference_arg = [_]u16{0};
    var recursive_invoke = interpreter.Invoke{
        .class_name = "Runtime",
        .method_name = "recursive",
        .signature = "(I)I",
        .args = &scalar_arg,
        .call_target = 0,
        .dest = null,
        .kind = .static,
    };
    var throw_invoke = interpreter.Invoke{
        .class_name = "Runtime",
        .method_name = "thrower",
        .signature = "(Ljava/lang/Object;)V",
        .args = &reference_arg,
        .call_target = 2,
        .dest = null,
        .kind = .static,
    };
    const recursive_code = [_]interpreter.Instruction{
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .if_lez = .{ .src = 0, .offset = 5 } },
        .{ .sub_int = .{ .dest = 0, .src1 = 0, .src2 = 1 } },
        .{ .invoke = &recursive_invoke },
        .{ .move_result = .{ .dest = 3 } },
        .{ .return_ = .{ .src = 3 } },
        .{ .const_ = .{ .dest = 3, .value = 0 } },
        .{ .return_ = .{ .src = 3 } },
    };
    const echo_code = [_]interpreter.Instruction{.{ .return_object = .{ .src = 0 } }};
    const throw_code = [_]interpreter.Instruction{.{ .throw_ = .{ .src = 0 } }};
    const bridge_code = [_]interpreter.Instruction{
        .{ .invoke = &throw_invoke },
        .return_void,
    };
    const typed_handlers = [_]interpreter.CatchHandler{.{ .type_idx = 100, .target_pc = 2 }};
    const caught_tries = [_]interpreter.TryBlock{.{ .start_pc = 0, .end_pc = 1, .handlers = &typed_handlers }};
    const caught_code = [_]interpreter.Instruction{
        .{ .invoke = &throw_invoke },
        .return_void,
        .{ .move_exception = .{ .dest = 0 } },
        .{ .return_object = .{ .src = 0 } },
    };
    const rethrow_code = [_]interpreter.Instruction{
        .{ .invoke = &throw_invoke },
        .return_void,
        .{ .move_exception = .{ .dest = 0 } },
        .{ .throw_ = .{ .src = 0 } },
    };
    const scalar_parameters = [_]u16{0};
    const scalar_kinds = [_]ParameterKind{.scalar};
    const reference_parameters = [_]u16{0};
    const reference_kinds = [_]ParameterKind{.reference};
    const methods = [_]InterpretedMethod{
        .{
            .id = 10,
            .instructions = &recursive_code,
            .register_count = 4,
            .parameter_registers = &scalar_parameters,
            .parameter_kinds = &scalar_kinds,
            .return_type = .single,
        },
        .{
            .id = 20,
            .instructions = &echo_code,
            .register_count = 1,
            .parameter_registers = &reference_parameters,
            .parameter_kinds = &reference_kinds,
            .return_type = .object,
        },
        .{
            .id = 30,
            .instructions = &throw_code,
            .register_count = 1,
            .parameter_registers = &reference_parameters,
            .parameter_kinds = &reference_kinds,
            .return_type = .void,
        },
        .{
            .id = 40,
            .instructions = &bridge_code,
            .register_count = 1,
            .parameter_registers = &reference_parameters,
            .parameter_kinds = &reference_kinds,
            .return_type = .void,
        },
        .{
            .id = 50,
            .instructions = &caught_code,
            .register_count = 1,
            .parameter_registers = &reference_parameters,
            .parameter_kinds = &reference_kinds,
            .try_blocks = &caught_tries,
            .return_type = .object,
        },
        .{
            .id = 60,
            .instructions = &rethrow_code,
            .register_count = 1,
            .parameter_registers = &reference_parameters,
            .parameter_kinds = &reference_kinds,
            .try_blocks = &caught_tries,
            .return_type = .void,
        },
    };
    const exception_types = [_]ExceptionType{
        .{ .type_idx = 100, .kind = 1 },
        .{ .type_idx = 200, .kind = 2, .super_type_idx = 100 },
    };

    var context = try thread_registry.ThreadContext.init(std.testing.allocator, 8);
    defer context.deinit();
    var satb = try runtime_gc.SatbBuffer.init(std.testing.allocator, 4);
    defer satb.deinit() catch unreachable;
    try collector.registerThreadSatbBuffer(&satb, &context);
    var satb_registered = true;
    defer if (satb_registered) collector.unregisterSatbBuffer(&satb) catch unreachable;
    try registry.register(&context);
    var context_registered = true;
    defer if (context_registered) registry.unregister(&context) catch unreachable;
    var call_registers: [16]u32 = undefined;
    var call_references: [16]u64 = undefined;
    var call_kinds: [16]bool = undefined;
    var call_stack = try InterpreterCallStack.init(&call_registers, &call_references, &call_kinds);
    var runtime = try Context.init(&collector, &satb, .{
        .interpreted_methods = &methods,
        .exception_types = &exception_types,
        .interpreter_stack = &call_stack,
        .thread_binding = .{ .registry = &registry, .context = &context, .allocator = &allocator },
    });

    const Driver = struct {
        fn scalar(runtime_context: *Context, invocation: *interpreter.Invoke, value: i32) !i32 {
            var registers = [_]u32{@bitCast(value)};
            var references = [_]u64{interpreter.null_reference_bits};
            var kinds = [_]bool{false};
            const code = [_]interpreter.Instruction{
                .{ .invoke = invocation },
                .{ .move_result = .{ .dest = 0 } },
                .{ .return_ = .{ .src = 0 } },
            };
            var frame = interpreter.ExecutionFrame{
                .pc = 0,
                .registers = &registers,
                .instructions = &code,
                .register_is_ref = &kinds,
                .reference_registers = &references,
                .managed_memory = runtime_context.attachment(),
            };
            return @bitCast((try interpreter.execute(&frame)).value32);
        }

        fn object(runtime_context: *Context, invocation: *interpreter.Invoke, value: Handle) !u64 {
            var registers = [_]u32{@truncate(@as(u64, @bitCast(value)))};
            var references = [_]u64{@bitCast(value)};
            var kinds = [_]bool{true};
            const code = [_]interpreter.Instruction{
                .{ .invoke = invocation },
                .{ .move_result_object = .{ .dest = 0 } },
                .{ .return_object = .{ .src = 0 } },
            };
            var frame = interpreter.ExecutionFrame{
                .pc = 0,
                .registers = &registers,
                .instructions = &code,
                .register_is_ref = &kinds,
                .reference_registers = &references,
                .managed_memory = runtime_context.attachment(),
            };
            return (try interpreter.execute(&frame)).value64;
        }
    };

    var top_recursive = recursive_invoke;
    try std.testing.expectEqual(@as(i32, 0), try Driver.scalar(&runtime, &top_recursive, 2));
    try std.testing.expectEqual(@as(usize, 3), runtime.stats().max_call_depth);
    try std.testing.expectEqual(@as(usize, 0), call_stack.cursor);
    try std.testing.expectEqual(@as(usize, 0), call_stack.depth);
    try std.testing.expectEqual(@as(usize, 0), context.rootCount());

    var echo_invoke = interpreter.Invoke{
        .class_name = "Runtime",
        .method_name = "echo",
        .signature = "(Ljava/lang/Object;)Ljava/lang/Object;",
        .args = &reference_arg,
        .call_target = 1,
        .dest = null,
        .kind = .static,
    };
    try std.testing.expectEqual(@as(u64, @bitCast(exception)), try Driver.object(&runtime, &echo_invoke, exception));
    var symbolic_echo = echo_invoke;
    symbolic_echo.call_target = null;
    symbolic_echo.target = .{ .method_idx = 20 };
    try std.testing.expectEqual(@as(u64, @bitCast(exception)), try Driver.object(&runtime, &symbolic_echo, exception));

    var bridge_invoke = interpreter.Invoke{
        .class_name = "Runtime",
        .method_name = "bridge",
        .signature = "(Ljava/lang/Object;)V",
        .args = &reference_arg,
        .call_target = 3,
        .dest = null,
        .kind = .static,
    };
    var exception_registers = [_]u32{@truncate(@as(u64, @bitCast(exception)))};
    var exception_references = [_]u64{@bitCast(exception)};
    var exception_kinds = [_]bool{true};
    const exception_driver = [_]interpreter.Instruction{.{ .invoke = &bridge_invoke }};
    var exception_frame = interpreter.ExecutionFrame{
        .pc = 0,
        .registers = &exception_registers,
        .instructions = &exception_driver,
        .register_is_ref = &exception_kinds,
        .reference_registers = &exception_references,
        .managed_memory = runtime.attachment(),
    };
    try std.testing.expectError(error.ManagedException, interpreter.execute(&exception_frame));
    try std.testing.expectEqual(@as(u64, @bitCast(exception)), exception_frame.pending_exception);
    try std.testing.expectEqual(@as(usize, 0), call_stack.cursor);
    try std.testing.expectEqual(@as(usize, 0), call_stack.depth);
    try std.testing.expectEqual(@as(usize, 0), context.rootCount());
    try std.testing.expectEqual(@as(u64, 2), runtime.stats().exceptions_propagated);
    try std.testing.expectEqual(@as(usize, 3), runtime.stats().max_frame_roots);

    var caught_invoke = interpreter.Invoke{
        .class_name = "Runtime",
        .method_name = "caught",
        .signature = "(Ljava/lang/Object;)Ljava/lang/Object;",
        .args = &reference_arg,
        .call_target = 4,
        .dest = null,
        .kind = .static,
    };
    try std.testing.expectEqual(@as(u64, @bitCast(exception)), try Driver.object(&runtime, &caught_invoke, exception));
    try std.testing.expectEqual(@as(u64, 2), runtime.stats().exception_type_checks);
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().exception_type_matches);

    var rethrow_invoke = caught_invoke;
    rethrow_invoke.method_name = "rethrow";
    rethrow_invoke.signature = "(Ljava/lang/Object;)V";
    rethrow_invoke.call_target = 5;
    exception_frame.pc = 0;
    exception_frame.pending_exception = interpreter.null_reference_bits;
    exception_frame.instructions = &[_]interpreter.Instruction{.{ .invoke = &rethrow_invoke }};
    try std.testing.expectError(error.ManagedException, interpreter.execute(&exception_frame));
    try std.testing.expectEqual(@as(u64, @bitCast(exception)), exception_frame.pending_exception);
    try std.testing.expectEqual(@as(usize, 0), call_stack.cursor);
    try std.testing.expectEqual(@as(usize, 0), context.rootCount());

    try std.testing.expectError(error.StackOverflow, Driver.scalar(&runtime, &top_recursive, 4));
    try std.testing.expectEqual(@as(u64, 1), runtime.stats().stack_overflows);
    try std.testing.expectEqual(@as(usize, 0), call_stack.cursor);
    try std.testing.expectEqual(@as(usize, 0), call_stack.depth);
    try std.testing.expectEqual(@as(u64, @bitCast(exception)), try Driver.object(&runtime, &echo_invoke, exception));

    var virtual_invoke = echo_invoke;
    virtual_invoke.kind = .virtual;
    try std.testing.expectError(error.NullReference, Driver.object(&runtime, &virtual_invoke, Handle.none));

    try registry.unregister(&context);
    context_registered = false;
    try collector.unregisterSatbBuffer(&satb);
    satb_registered = false;

    var concurrency_failed = std.atomic.Value(bool).init(false);
    const Worker = struct {
        collector: *runtime_gc.ConcurrentCollector,
        registry: *thread_registry.Registry,
        heap: *runtime_heap.ManagedHeap,
        methods: []const InterpretedMethod,
        exception_types: []const ExceptionType,
        invocation: *interpreter.Invoke,
        caught_invocation: *interpreter.Invoke,
        exception: Handle,
        failed: *std.atomic.Value(bool),

        fn run(self: *@This()) void {
            var local_context = thread_registry.ThreadContext.init(std.testing.allocator, 8) catch {
                self.failed.store(true, .release);
                return;
            };
            defer local_context.deinit();
            var local_satb = runtime_gc.SatbBuffer.init(std.testing.allocator, 4) catch {
                self.failed.store(true, .release);
                return;
            };
            defer local_satb.deinit() catch self.failed.store(true, .release);
            self.collector.registerThreadSatbBuffer(&local_satb, &local_context) catch {
                self.failed.store(true, .release);
                return;
            };
            defer self.collector.unregisterSatbBuffer(&local_satb) catch self.failed.store(true, .release);
            self.registry.register(&local_context) catch {
                self.failed.store(true, .release);
                return;
            };
            defer self.registry.unregister(&local_context) catch self.failed.store(true, .release);
            var local_allocator = self.heap.threadAllocator();
            var registers: [16]u32 = undefined;
            var references: [16]u64 = undefined;
            var kinds: [16]bool = undefined;
            var stack = InterpreterCallStack.init(&registers, &references, &kinds) catch {
                self.failed.store(true, .release);
                return;
            };
            var local_runtime = Context.init(self.collector, &local_satb, .{
                .interpreted_methods = self.methods,
                .exception_types = self.exception_types,
                .interpreter_stack = &stack,
                .thread_binding = .{
                    .registry = self.registry,
                    .context = &local_context,
                    .allocator = &local_allocator,
                },
            }) catch {
                self.failed.store(true, .release);
                return;
            };
            for (0..100) |_| {
                const result = Driver.scalar(&local_runtime, self.invocation, 2) catch {
                    self.failed.store(true, .release);
                    return;
                };
                if (result != 0 or stack.cursor != 0 or local_context.rootCount() != 0) {
                    self.failed.store(true, .release);
                    return;
                }
                const caught = Driver.object(&local_runtime, self.caught_invocation, self.exception) catch {
                    self.failed.store(true, .release);
                    return;
                };
                if (caught != @as(u64, @bitCast(self.exception)) or
                    stack.cursor != 0 or local_context.rootCount() != 0)
                {
                    self.failed.store(true, .release);
                    return;
                }
            }
        }
    };
    var workers: [4]Worker = undefined;
    var threads: [4]std.Thread = undefined;
    for (&workers, &threads) |*worker, *thread| {
        worker.* = .{
            .collector = &collector,
            .registry = &registry,
            .heap = &heap,
            .methods = &methods,
            .exception_types = &exception_types,
            .invocation = &top_recursive,
            .caught_invocation = &caught_invoke,
            .exception = exception,
            .failed = &concurrency_failed,
        };
        thread.* = try std.Thread.spawn(.{}, Worker.run, .{worker});
    }
    for (&threads) |*thread| thread.join();
    try std.testing.expect(!concurrency_failed.load(.acquire));
}

test "managed monitors are reentrant unwind safely and join passive GC handshakes" {
    var storage: [4096]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 8, &regions);
    defer handles.deinit();
    var heap = try runtime_heap.ManagedHeap.init(std.testing.allocator, &handles, 128);
    defer heap.deinit();
    var allocator = heap.threadAllocator();
    const object = try publishTestObject(&heap, &handles, &allocator, 8);
    var monitors = try runtime_monitor.MonitorTable.init(std.testing.allocator, std.testing.io, &handles);
    defer monitors.deinit() catch unreachable;
    var collector = try runtime_gc.ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 16,
        .max_satb_buffers = 5,
        .card_bytes = 64,
        .static_root_slots = monitors.rootSlotAddresses(),
    });
    defer collector.deinit() catch unreachable;
    var registry = try thread_registry.Registry.init(std.testing.allocator, std.testing.io, 5);
    defer registry.deinit() catch unreachable;
    var owner_context = try thread_registry.ThreadContext.init(std.testing.allocator, 8);
    defer owner_context.deinit();
    var owner_satb = try runtime_gc.SatbBuffer.init(std.testing.allocator, 4);
    defer owner_satb.deinit() catch unreachable;
    try collector.registerThreadSatbBuffer(&owner_satb, &owner_context);
    defer collector.unregisterSatbBuffer(&owner_satb) catch unreachable;
    try registry.register(&owner_context);
    defer registry.unregister(&owner_context) catch unreachable;
    var owner_allocator = heap.threadAllocator();
    var runtime = try Context.init(&collector, &owner_satb, .{
        .monitor_table = &monitors,
        .thread_binding = .{
            .registry = &registry,
            .context = &owner_context,
            .allocator = &owner_allocator,
        },
    });

    var registers = [_]u32{@truncate(@as(u64, @bitCast(object)))};
    var references = [_]u64{@bitCast(object)};
    var kinds = [_]bool{true};
    const balanced_code = [_]interpreter.Instruction{
        .{ .monitor_enter = .{ .src = 0 } },
        .{ .monitor_enter = .{ .src = 0 } },
        .{ .monitor_exit = .{ .src = 0 } },
        .{ .monitor_exit = .{ .src = 0 } },
        .return_void,
    };
    var frame = interpreter.ExecutionFrame{
        .pc = 0,
        .registers = &registers,
        .instructions = &balanced_code,
        .register_is_ref = &kinds,
        .reference_registers = &references,
        .managed_memory = runtime.attachment(),
    };
    _ = try interpreter.execute(&frame);
    try std.testing.expectEqual(@as(u64, 2), runtime.stats().monitor_enters);
    try std.testing.expectEqual(@as(u64, 2), runtime.stats().monitor_exits);
    try std.testing.expectEqual(@as(u64, 1), monitors.stats().reentrant_enters);

    var capacity_code: [17]interpreter.Instruction = undefined;
    for (&capacity_code) |*instruction| instruction.* = .{ .monitor_enter = .{ .src = 0 } };
    const unwind_before_capacity = runtime.stats().monitor_unwind_exits;
    frame.pc = 0;
    frame.instructions = &capacity_code;
    try std.testing.expectError(error.MonitorCapacityExceeded, interpreter.execute(&frame));
    try std.testing.expectEqual(
        unwind_before_capacity + interpreter.ManagedFrameState.max_held_monitors,
        runtime.stats().monitor_unwind_exits,
    );
    try std.testing.expectEqual(@as(u8, 0), frame.managed_frame_state.held_monitor_count);

    const unwind_before_throw = runtime.stats().monitor_unwind_exits;
    frame.pc = 0;
    frame.instructions = &[_]interpreter.Instruction{
        .{ .monitor_enter = .{ .src = 0 } },
        .{ .throw_ = .{ .src = 0 } },
    };
    try std.testing.expectError(error.ManagedException, interpreter.execute(&frame));
    try std.testing.expectEqual(unwind_before_throw + 1, runtime.stats().monitor_unwind_exits);
    try std.testing.expectEqual(@as(u8, 0), frame.managed_frame_state.held_monitor_count);
    try std.testing.expectEqual(@as(usize, 0), owner_context.rootCount());

    const catch_handlers = [_]interpreter.CatchHandler{.{ .type_idx = null, .target_pc = 3 }};
    const catch_tries = [_]interpreter.TryBlock{.{ .start_pc = 1, .end_pc = 2, .handlers = &catch_handlers }};
    const caught_code = [_]interpreter.Instruction{
        .{ .monitor_enter = .{ .src = 0 } },
        .{ .throw_ = .{ .src = 0 } },
        .return_void,
        .{ .move_exception = .{ .dest = 0 } },
        .{ .monitor_exit = .{ .src = 0 } },
        .return_void,
    };
    frame.pc = 0;
    frame.pending_exception = interpreter.null_reference_bits;
    frame.instructions = &caught_code;
    frame.try_blocks = &catch_tries;
    _ = try interpreter.execute(&frame);
    try std.testing.expectEqual(@as(u8, 0), frame.managed_frame_state.held_monitor_count);

    frame.pc = 0;
    frame.try_blocks = &.{};
    frame.instructions = &[_]interpreter.Instruction{.{ .monitor_exit = .{ .src = 0 } }};
    try std.testing.expectError(error.IllegalMonitorState, interpreter.execute(&frame));

    // Hold the object monitor on the test thread, then remove this thread from
    // handshake membership. Four FIFO contenders can now block passively.
    try std.testing.expectEqual(
        interpreter.ManagedMemoryStatus.ok,
        monitorEnter(&runtime, @bitCast(object)),
    );
    var finish_workers = std.atomic.Value(bool).init(false);
    var worker_failed = std.atomic.Value(bool).init(false);
    var order_cursor = std.atomic.Value(usize).init(0);
    var order: [4]u8 = @splat(255);
    var done: [4]std.atomic.Value(bool) = undefined;
    for (&done) |*entry| entry.* = std.atomic.Value(bool).init(false);
    const Worker = struct {
        id: u8,
        collector: *runtime_gc.ConcurrentCollector,
        registry: *thread_registry.Registry,
        heap: *runtime_heap.ManagedHeap,
        monitors: *runtime_monitor.MonitorTable,
        object: Handle,
        finish: *std.atomic.Value(bool),
        failed: *std.atomic.Value(bool),
        order_cursor: *std.atomic.Value(usize),
        order: *[4]u8,
        done: *std.atomic.Value(bool),

        fn fail(self: *@This()) void {
            self.failed.store(true, .release);
            self.done.store(true, .release);
        }

        fn registerBuffer(self: *@This(), satb: *runtime_gc.SatbBuffer, context: *thread_registry.ThreadContext) bool {
            for (0..100_000) |_| {
                self.collector.registerThreadSatbBuffer(satb, context) catch |err| switch (err) {
                    error.WrongPhase => {
                        std.atomic.spinLoopHint();
                        std.Thread.yield() catch {};
                        continue;
                    },
                    else => return false,
                };
                return true;
            }
            return false;
        }

        fn unregisterBuffer(self: *@This(), satb: *runtime_gc.SatbBuffer) bool {
            for (0..100_000) |_| {
                self.collector.unregisterSatbBuffer(satb) catch |err| switch (err) {
                    error.WrongPhase => {
                        std.atomic.spinLoopHint();
                        std.Thread.yield() catch {};
                        continue;
                    },
                    else => return false,
                };
                return true;
            }
            return false;
        }

        fn run(self: *@This()) void {
            var context = thread_registry.ThreadContext.init(std.testing.allocator, 2) catch return self.fail();
            defer context.deinit();
            var satb = runtime_gc.SatbBuffer.init(std.testing.allocator, 2) catch return self.fail();
            defer satb.deinit() catch self.fail();
            if (!self.registerBuffer(&satb, &context)) return self.fail();
            defer if (!self.unregisterBuffer(&satb)) self.fail();
            self.registry.register(&context) catch return self.fail();
            defer self.registry.unregister(&context) catch self.fail();
            var monitor_allocator = self.heap.threadAllocator();
            var local = Context.init(self.collector, &satb, .{
                .monitor_table = self.monitors,
                .thread_binding = .{
                    .registry = self.registry,
                    .context = &context,
                    .allocator = &monitor_allocator,
                },
            }) catch return self.fail();
            var root = self.object;
            context.addRoot(&root) catch return self.fail();
            if (monitorExit(&local, @bitCast(self.object)) != .illegal_monitor_state) return self.fail();
            if (monitorEnter(&local, @bitCast(self.object)) != .ok) return self.fail();
            const position = self.order_cursor.fetchAdd(1, .acq_rel);
            if (position >= self.order.len) return self.fail();
            self.order[position] = self.id;
            if (monitorExit(&local, @bitCast(self.object)) != .ok) return self.fail();
            if (self.collector.phase() == .marking) {
                satb.flushForEpoch(self.collector.epoch()) catch return self.fail();
            }
            self.done.store(true, .release);
            while (!self.finish.load(.acquire)) std.atomic.spinLoopHint();
        }
    };
    var workers: [4]Worker = undefined;
    var threads: [4]std.Thread = undefined;
    for (&workers, &threads, 0..) |*worker, *thread, index| {
        worker.* = .{
            .id = @intCast(index),
            .collector = &collector,
            .registry = &registry,
            .heap = &heap,
            .monitors = &monitors,
            .object = object,
            .finish = &finish_workers,
            .failed = &worker_failed,
            .order_cursor = &order_cursor,
            .order = &order,
            .done = &done[index],
        };
        thread.* = try std.Thread.spawn(.{}, Worker.run, .{worker});
        const expected_contentions: u64 = @intCast(index + 1);
        for (0..1_000_000) |_| {
            if (monitors.stats().contended_enters >= expected_contentions) break;
            if (worker_failed.load(.acquire)) return error.TestUnexpectedResult;
            std.atomic.spinLoopHint();
            std.Thread.yield() catch {};
        } else return error.Timeout;
    }

    const mark_epoch = try collector.beginMark();
    try collector.enterBlockedForMark(&registry, &owner_context, &owner_satb);
    var handshake = try collector.beginThreadHandshake(&registry);
    try finishHandshake(&handshake);
    try registry.leaveBlocked(&owner_context);
    try std.testing.expectEqual(
        interpreter.ManagedMemoryStatus.ok,
        monitorExit(&runtime, @bitCast(object)),
    );
    for (&done) |*entry| {
        for (0..1_000_000) |_| {
            if (entry.load(.acquire)) break;
            if (worker_failed.load(.acquire)) return error.TestUnexpectedResult;
            std.atomic.spinLoopHint();
            std.Thread.yield() catch {};
        } else return error.Timeout;
    }
    try owner_satb.flushForEpoch(mark_epoch);
    while (try collector.drainSatb(16) != 0) {}
    while (try collector.traceWork(16) != 0) {}
    try collector.tryFinishMark();
    try std.testing.expect(try collector.isMarked(object));
    finish_workers.store(true, .release);
    for (&threads) |*thread| thread.join();
    try std.testing.expect(!worker_failed.load(.acquire));
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 2, 3 }, &order);
    try std.testing.expectEqual(@as(u64, 4), monitors.stats().contended_enters);
    try std.testing.expectEqual(monitors.stats().associations, monitors.stats().disassociations);
}

test "interpreter allocation failure rolls back the constructing handle" {
    var storage: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 2, &regions);
    defer handles.deinit();
    var heap = try runtime_heap.ManagedHeap.init(std.testing.allocator, &handles, 64);
    defer heap.deinit();
    var collector = try runtime_gc.ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 2,
        .max_satb_buffers = 1,
        .card_bytes = 64,
    });
    defer collector.deinit() catch unreachable;
    var registry = try thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try thread_registry.ThreadContext.init(std.testing.allocator, 1);
    defer context.deinit();
    var satb = try runtime_gc.SatbBuffer.init(std.testing.allocator, 2);
    defer satb.deinit() catch unreachable;
    try collector.registerThreadSatbBuffer(&satb, &context);
    defer collector.unregisterSatbBuffer(&satb) catch unreachable;
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    var allocator = heap.threadAllocator();
    const layouts = [_]InstanceLayout{.{ .payload_size = 1024 }};
    var runtime = try Context.init(&collector, &satb, .{
        .instance_layouts = &layouts,
        .thread_binding = .{ .registry = &registry, .context = &context, .allocator = &allocator },
    });

    try std.testing.expectError(error.OutOfRegionMemory, allocateInstanceInner(&runtime, 0));
    const reusable = try handles.reserve(0, 0);
    try handles.cancelReservation(reusable);

    var registers = [_]u32{ 0, 0 };
    var references = [_]u64{ @bitCast(Handle.none), @bitCast(Handle.none) };
    var kinds = [_]bool{ true, true };
    const code = [_]interpreter.Instruction{.return_void};
    var frame = interpreter.ExecutionFrame{
        .pc = 0,
        .registers = &registers,
        .instructions = &code,
        .register_is_ref = &kinds,
        .reference_registers = &references,
        .managed_memory = runtime.attachment(),
    };
    try std.testing.expectError(error.ManagedMemoryFailure, interpreter.execute(&frame));
    try std.testing.expect(!frame.managed_frame_state.active);
    try std.testing.expectEqual(@as(usize, 0), context.rootCount());
}

test "synchronized interpreted methods release on returns exceptions and reentrant cleanup" {
    var storage: [2048]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try runtime_value.HandleTable.init(std.testing.allocator, 4, &regions);
    defer handles.deinit();
    var heap = try runtime_heap.ManagedHeap.init(std.testing.allocator, &handles, 128);
    defer heap.deinit();
    var heap_allocator = heap.threadAllocator();
    const receiver = try publishTestObject(&heap, &handles, &heap_allocator, 8);
    const class_object = try publishTestObject(&heap, &handles, &heap_allocator, 8);
    var monitors = try runtime_monitor.MonitorTable.init(std.testing.allocator, std.testing.io, &handles);
    defer monitors.deinit() catch unreachable;
    var class_root = std.atomic.Value(u64).init(@bitCast(class_object));
    var static_roots: [5]usize = undefined;
    @memcpy(static_roots[0..4], monitors.rootSlotAddresses());
    static_roots[4] = @intFromPtr(&class_root);
    std.mem.sort(usize, &static_roots, {}, struct {
        fn lessThan(_: void, lhs: usize, rhs: usize) bool {
            return lhs < rhs;
        }
    }.lessThan);
    var collector = try runtime_gc.ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 8,
        .max_satb_buffers = 1,
        .card_bytes = 64,
        .static_root_slots = &static_roots,
    });
    defer collector.deinit() catch unreachable;
    var registry = try thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var thread_context = try thread_registry.ThreadContext.init(std.testing.allocator, 16);
    defer thread_context.deinit();
    try registry.register(&thread_context);
    defer registry.unregister(&thread_context) catch unreachable;
    var satb = try runtime_gc.SatbBuffer.init(std.testing.allocator, 8);
    defer satb.deinit() catch unreachable;
    try collector.registerThreadSatbBuffer(&satb, &thread_context);
    defer collector.unregisterSatbBuffer(&satb) catch unreachable;

    const parameter_registers = [_]u16{0};
    const parameter_kinds = [_]ParameterKind{.reference};
    const normal_code = [_]interpreter.Instruction{.{ .return_object = .{ .src = 0 } }};
    const static_code = [_]interpreter.Instruction{.return_void};
    const throw_code = [_]interpreter.Instruction{.{ .throw_ = .{ .src = 0 } }};
    const reentrant_code = [_]interpreter.Instruction{ .{ .monitor_enter = .{ .src = 0 } }, .return_void };
    const methods = [_]InterpretedMethod{
        .{ .id = 10, .instructions = &normal_code, .register_count = 1, .parameter_registers = &parameter_registers, .parameter_kinds = &parameter_kinds, .return_type = .object },
        .{ .id = 20, .instructions = &static_code, .register_count = 1, .return_type = .void },
        .{ .id = 30, .instructions = &throw_code, .register_count = 1, .parameter_registers = &parameter_registers, .parameter_kinds = &parameter_kinds, .return_type = .void },
        .{ .id = 40, .instructions = &reentrant_code, .register_count = 1, .parameter_registers = &parameter_registers, .parameter_kinds = &parameter_kinds, .return_type = .void },
    };
    const synchronization_entries = [_]runtime_method_sync.Entry{
        .{ .method_id = 10, .target = .{ .instance_parameter = 0 } },
        .{ .method_id = 20, .target = .{ .static_root_slot = @intFromPtr(&class_root) } },
        .{ .method_id = 30, .target = .{ .instance_parameter = 0 } },
        .{ .method_id = 40, .target = .{ .instance_parameter = 0 } },
    };
    const synchronization = try runtime_method_sync.Table.init(&synchronization_entries, &collector);
    var stack_registers: [16]u32 = @splat(0);
    var stack_references: [16]u64 = @splat(@as(u64, @bitCast(Handle.none)));
    var stack_kinds: [16]bool = @splat(false);
    var call_stack = try InterpreterCallStack.init(&stack_registers, &stack_references, &stack_kinds);
    var owner_allocator = heap.threadAllocator();
    var runtime = try Context.init(&collector, &satb, .{
        .interpreted_methods = &methods,
        .monitor_table = &monitors,
        .method_synchronization = &synchronization,
        .interpreter_stack = &call_stack,
        .thread_binding = .{ .registry = &registry, .context = &thread_context, .allocator = &owner_allocator },
    });

    var caller_registers = [_]u32{@truncate(@as(u64, @bitCast(receiver)))};
    var caller_references = [_]u64{@bitCast(receiver)};
    var caller_kinds = [_]bool{true};
    const instance_args = [_]u16{0};
    var invocation = interpreter.Invoke{
        .class_name = "Synchronized",
        .method_name = "normal",
        .signature = "()Ljava/lang/Object;",
        .args = &instance_args,
        .call_target = 0,
        .dest = null,
        .kind = .direct,
    };
    var output: interpreter.ExecutionResult = .{ .kind = .void };
    var exception_bits: u64 = @bitCast(Handle.none);
    try std.testing.expectEqual(interpreter.ManagedMemoryStatus.ok, invokeInterpretedMethod(
        &runtime,
        &invocation,
        &caller_registers,
        &caller_references,
        &caller_kinds,
        &output,
        &exception_bits,
    ));
    try std.testing.expectEqual(@as(u64, @bitCast(receiver)), output.value64);

    invocation.call_target = 1;
    invocation.args = &.{};
    invocation.kind = .static;
    try std.testing.expectEqual(interpreter.ManagedMemoryStatus.ok, invokeInterpretedMethod(
        &runtime,
        &invocation,
        &caller_registers,
        &caller_references,
        &caller_kinds,
        &output,
        &exception_bits,
    ));

    invocation.call_target = 2;
    invocation.args = &instance_args;
    invocation.kind = .direct;
    try std.testing.expectEqual(interpreter.ManagedMemoryStatus.managed_exception, invokeInterpretedMethod(
        &runtime,
        &invocation,
        &caller_registers,
        &caller_references,
        &caller_kinds,
        &output,
        &exception_bits,
    ));
    try std.testing.expectEqual(@as(u64, @bitCast(receiver)), exception_bits);

    invocation.call_target = 3;
    try std.testing.expectEqual(interpreter.ManagedMemoryStatus.ok, invokeInterpretedMethod(
        &runtime,
        &invocation,
        &caller_registers,
        &caller_references,
        &caller_kinds,
        &output,
        &exception_bits,
    ));
    const stats = runtime.stats();
    try std.testing.expectEqual(@as(u64, 4), stats.synchronized_invocations);
    try std.testing.expectEqual(@as(u64, 3), stats.synchronized_returns);
    try std.testing.expectEqual(@as(u64, 1), stats.synchronized_exceptions);
    try std.testing.expectEqual(@as(u64, 1), monitors.stats().reentrant_enters);
    try std.testing.expectEqual(monitors.stats().associations, monitors.stats().disassociations);

    // A deoptimized synchronized activation already owns its implicit monitor.
    // Interpreter adoption must validate that ownership without entering it a
    // second time, then release it exactly once on return.
    try monitors.enter(receiver, &collector, &registry, &thread_context, &satb);
    const enters_before_adoption = monitors.stats().enters;
    const exits_before_adoption = monitors.stats().exits;
    var resumed_registers = [_]u32{@truncate(@as(u64, @bitCast(receiver)))};
    var resumed_references = [_]u64{@bitCast(receiver)};
    var resumed_kinds = [_]bool{true};
    var resumed_frame = interpreter.ExecutionFrame{
        .pc = 0,
        .registers = &resumed_registers,
        .instructions = &normal_code,
        .register_is_ref = &resumed_kinds,
        .reference_registers = &resumed_references,
        .managed_memory = runtime.attachment(),
        .managed_frame_state = .{
            .held_monitors = blk: {
                var held: [interpreter.ManagedFrameState.max_held_monitors]u64 =
                    @splat(@as(u64, @bitCast(Handle.none)));
                held[0] = @bitCast(receiver);
                break :blk held;
            },
            .held_monitor_count = 1,
            .synchronized_monitor_owned = true,
        },
        .synchronized_monitor = @bitCast(receiver),
    };
    _ = try interpreter.execute(&resumed_frame);
    try std.testing.expectEqual(enters_before_adoption, monitors.stats().enters);
    try std.testing.expectEqual(exits_before_adoption + 1, monitors.stats().exits);
    try std.testing.expectEqual(@as(u8, 0), resumed_frame.managed_frame_state.held_monitor_count);
    try std.testing.expect(!resumed_frame.managed_frame_state.synchronized_monitor_owned);
    try std.testing.expectEqual(monitors.stats().associations, monitors.stats().disassociations);
    try std.testing.expectEqual(@as(usize, 0), call_stack.cursor);
    try std.testing.expectEqual(@as(usize, 0), thread_context.rootCount());
}
