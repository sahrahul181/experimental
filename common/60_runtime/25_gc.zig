//! Concurrent marking metadata, SATB publication, and remembered-set cards.
//!
//! All storage is fixed at initialization. Mutator barriers never allocate,
//! lock, or wait: bounded saturation and the short termination election are
//! reported to the caller so a safepoint/slow path can retry without losing
//! the pre-write value.

const std = @import("std");
const runtime_value = @import("runtime_value");
const runtime_heap = @import("runtime_heap");
const thread_registry = @import("runtime_thread_registry");

const Handle = runtime_value.Handle;
const HandleTable = runtime_value.HandleTable;
const ManagedHeap = runtime_heap.ManagedHeap;

pub const Error = runtime_heap.Error || error{
    AlreadyRegistered,
    BuffersBusy,
    BuffersNotFlushed,
    CardSequenceExhausted,
    EpochExhausted,
    InvalidCapacity,
    InvalidCardSize,
    InvalidClaim,
    InvalidLayout,
    MarkHandshakeIncomplete,
    MissingSatbBuffer,
    RegistryFull,
    RetryBarrier,
    SatbNotDrained,
    SatbQueueFull,
    StillRegistered,
    UnknownLayout,
    Unregistered,
    WorkNotDrained,
};

pub const Phase = enum(u8) {
    idle,
    preparing,
    marking,
    terminating,
};

pub const RegionKind = enum(u8) {
    young,
    old,
    pinned,
    large,
};

pub const TrailingReferences = struct {
    offset: u32,
    stride: u32 = @sizeOf(Handle),
};

/// Immutable precise scanning metadata. Layout zero is reserved for opaque
/// leaf objects and therefore cannot appear in this table.
pub const LayoutSpec = struct {
    id: u32,
    minimum_size: u32,
    reference_offsets: []const u32 = &.{},
    trailing_references: ?TrailingReferences = null,
};

pub const Options = struct {
    satb_queue_capacity: usize = 1024,
    max_satb_buffers: usize = 64,
    card_bytes: usize = 512,
    initial_region_kind: RegionKind = .young,
    layouts: []const LayoutSpec = &.{},
    /// Strictly address-sorted, permanently pinned atomic Handle slots.
    /// Addresses are copied during initialization; the slots themselves must
    /// outlive the collector.
    static_root_slots: []const usize = &.{},
};

pub const Stats = struct {
    cycles_started: u64,
    first_marks: u64,
    duplicate_marks: u64,
    satb_published: u64,
    satb_drained: u64,
    cards_dirtied: u64,
    cards_claimed: u64,
    termination_retries: u64,
    work_enqueued: u64,
    objects_scanned: u64,
    references_scanned: u64,
    thread_handshakes: u64,
    root_snapshots: u64,
    roots_discovered: u64,
    allocations_blackened: u64,
    satb_repeat_elisions: u64,
    card_repeat_elisions: u64,
    static_roots_scanned: u64,
    static_root_writes: u64,
};

const AtomicStats = struct {
    cycles_started: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    first_marks: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    duplicate_marks: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    satb_published: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    satb_drained: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    cards_dirtied: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    cards_claimed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    termination_retries: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    work_enqueued: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    objects_scanned: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    references_scanned: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    thread_handshakes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    root_snapshots: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    roots_discovered: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    allocations_blackened: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    satb_repeat_elisions: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    card_repeat_elisions: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    static_roots_scanned: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    static_root_writes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
};

const QueueState = enum(u8) {
    empty,
    writing,
    full,
    reading,
};

const QueueSlot = struct {
    state: std.atomic.Value(QueueState),
    value: u64,
};

const SatbQueue = struct {
    slots: []QueueSlot,
    write_cursor: std.atomic.Value(usize),
    read_cursor: std.atomic.Value(usize),

    fn enqueue(self: *SatbQueue, handle: Handle) bool {
        const start = self.write_cursor.fetchAdd(1, .monotonic);
        for (0..self.slots.len) |attempt| {
            const index = (start +% attempt) % self.slots.len;
            const slot = &self.slots[index];
            if (slot.state.cmpxchgStrong(.empty, .writing, .acquire, .monotonic) != null) continue;
            slot.value = @bitCast(handle);
            slot.state.store(.full, .release);
            return true;
        }
        return false;
    }

    /// Mark work has one slot per handle-table entry and each handle is
    /// published only by its first marker, so capacity exhaustion is
    /// impossible. A failed scan means another operation owns a transient
    /// slot; retrying here is lock-free and never runs on a mutator barrier.
    fn enqueueGuaranteed(self: *SatbQueue, handle: Handle) void {
        while (!self.enqueue(handle)) std.atomic.spinLoopHint();
    }

    fn dequeue(self: *SatbQueue) ?Handle {
        const start = self.read_cursor.fetchAdd(1, .monotonic);
        for (0..self.slots.len) |attempt| {
            const index = (start +% attempt) % self.slots.len;
            const slot = &self.slots[index];
            if (slot.state.cmpxchgStrong(.full, .reading, .acquire, .monotonic) != null) continue;
            const handle: Handle = @bitCast(slot.value);
            slot.state.store(.empty, .release);
            return handle;
        }
        return null;
    }

    fn isEmpty(self: *const SatbQueue) bool {
        for (self.slots) |*slot| {
            if (slot.state.load(.acquire) != .empty) return false;
        }
        return true;
    }
};

const CardState = enum(u8) {
    clean,
    dirty,
    claimed,
};

const CardEntry = struct {
    state: std.atomic.Value(CardState),
    sequence: std.atomic.Value(u64),
};

const RegionGc = struct {
    kind: std.atomic.Value(RegionKind),
    mark_words: []std.atomic.Value(u64),
    cards: []CardEntry,
    card_base: usize,
};

const LayoutEntry = struct {
    id: u32,
    minimum_size: u32,
    reference_offsets: []u32,
    trailing_references: ?TrailingReferences,
};

const HandshakeMemberState = enum(u8) {
    pending,
    feeding,
    released,
};

pub const CardClaim = struct {
    region_id: u8,
    card_index: usize,
    sequence: u64,
};

pub const ConcurrentCollector = struct {
    allocator: std.mem.Allocator,
    heap: *ManagedHeap,
    handles: *HandleTable,
    regions: []RegionGc,
    mark_storage: []std.atomic.Value(u64),
    discovered_storage: []std.atomic.Value(u64),
    card_storage: []CardEntry,
    queue_storage: []QueueSlot,
    work_queue_storage: []QueueSlot,
    buffer_slots: []std.atomic.Value(usize),
    queue: SatbQueue,
    work_queue: SatbQueue,
    layouts: []LayoutEntry,
    layout_offset_storage: []u32,
    static_root_slots: []usize,
    handshake_members: []*thread_registry.ThreadContext,
    handshake_states: []std.atomic.Value(HandshakeMemberState),
    phase_value: std.atomic.Value(Phase),
    epoch_value: std.atomic.Value(u64),
    active_operations: std.atomic.Value(usize),
    card_cursor: std.atomic.Value(usize),
    handshake_active: std.atomic.Value(bool),
    completed_handshake_epoch: std.atomic.Value(u64),
    card_bytes: usize,
    counters: AtomicStats = .{},

    pub fn init(
        allocator: std.mem.Allocator,
        heap: *ManagedHeap,
        handles: *HandleTable,
        options: Options,
    ) Error!ConcurrentCollector {
        if (heap.handles != handles) return error.InvalidState;
        if (options.satb_queue_capacity == 0 or options.max_satb_buffers == 0) {
            return error.InvalidCapacity;
        }
        if (options.card_bytes < runtime_value.object_alignment or
            !std.math.isPowerOfTwo(options.card_bytes)) return error.InvalidCardSize;
        const layout_offset_count = try validateLayoutSpecs(options.layouts);
        try validateStaticRootSlots(options.static_root_slots);

        const region_count: usize = handles.regionCount();
        const regions = try allocator.alloc(RegionGc, region_count);
        errdefer allocator.free(regions);
        const layouts = try allocator.alloc(LayoutEntry, options.layouts.len);
        errdefer allocator.free(layouts);
        const layout_offset_storage = try allocator.alloc(u32, layout_offset_count);
        errdefer allocator.free(layout_offset_storage);
        const static_root_slots = try allocator.dupe(usize, options.static_root_slots);
        errdefer allocator.free(static_root_slots);

        var mark_word_count: usize = 0;
        var card_count: usize = 0;
        for (0..region_count) |index| {
            const region = try handles.regionAt(@intCast(index));
            const slots = region.len / runtime_value.object_alignment +
                @intFromBool(region.len % runtime_value.object_alignment != 0);
            const words = slots / 64 + @intFromBool(slots % 64 != 0);
            const cards = region.len / options.card_bytes +
                @intFromBool(region.len % options.card_bytes != 0);
            mark_word_count = try addCapacity(mark_word_count, words);
            card_count = try addCapacity(card_count, cards);
        }

        const mark_storage = try allocator.alloc(std.atomic.Value(u64), mark_word_count);
        errdefer allocator.free(mark_storage);
        const discovered_word_count: usize = handles.entryCapacity() / 64 +
            @intFromBool(handles.entryCapacity() % 64 != 0);
        const discovered_storage = try allocator.alloc(std.atomic.Value(u64), discovered_word_count);
        errdefer allocator.free(discovered_storage);
        const card_storage = try allocator.alloc(CardEntry, card_count);
        errdefer allocator.free(card_storage);
        const queue_storage = try allocator.alloc(QueueSlot, options.satb_queue_capacity);
        errdefer allocator.free(queue_storage);
        const work_queue_storage = try allocator.alloc(QueueSlot, handles.entryCapacity());
        errdefer allocator.free(work_queue_storage);
        const buffer_slots = try allocator.alloc(std.atomic.Value(usize), options.max_satb_buffers);
        errdefer allocator.free(buffer_slots);
        const handshake_members = try allocator.alloc(*thread_registry.ThreadContext, options.max_satb_buffers);
        errdefer allocator.free(handshake_members);
        const handshake_states = try allocator.alloc(
            std.atomic.Value(HandshakeMemberState),
            options.max_satb_buffers,
        );
        errdefer allocator.free(handshake_states);

        for (mark_storage) |*word| word.* = std.atomic.Value(u64).init(0);
        for (discovered_storage) |*word| word.* = std.atomic.Value(u64).init(0);
        for (card_storage) |*card| card.* = .{
            .state = std.atomic.Value(CardState).init(.clean),
            .sequence = std.atomic.Value(u64).init(0),
        };
        for (queue_storage) |*slot| slot.* = .{
            .state = std.atomic.Value(QueueState).init(.empty),
            .value = 0,
        };
        for (work_queue_storage) |*slot| slot.* = .{
            .state = std.atomic.Value(QueueState).init(.empty),
            .value = 0,
        };
        for (buffer_slots) |*slot| slot.* = std.atomic.Value(usize).init(0);
        for (handshake_states) |*state| state.* = std.atomic.Value(HandshakeMemberState).init(.released);

        var layout_cursor: usize = 0;
        for (layouts, options.layouts) |*entry, spec| {
            const offsets = layout_offset_storage[layout_cursor .. layout_cursor + spec.reference_offsets.len];
            @memcpy(offsets, spec.reference_offsets);
            entry.* = .{
                .id = spec.id,
                .minimum_size = spec.minimum_size,
                .reference_offsets = offsets,
                .trailing_references = spec.trailing_references,
            };
            layout_cursor += offsets.len;
        }

        var mark_cursor: usize = 0;
        var card_cursor: usize = 0;
        for (regions, 0..) |*metadata, index| {
            const region = try handles.regionAt(@intCast(index));
            const slots = region.len / runtime_value.object_alignment +
                @intFromBool(region.len % runtime_value.object_alignment != 0);
            const words = slots / 64 + @intFromBool(slots % 64 != 0);
            const cards = region.len / options.card_bytes +
                @intFromBool(region.len % options.card_bytes != 0);
            metadata.* = .{
                .kind = std.atomic.Value(RegionKind).init(options.initial_region_kind),
                .mark_words = mark_storage[mark_cursor .. mark_cursor + words],
                .cards = card_storage[card_cursor .. card_cursor + cards],
                .card_base = card_cursor,
            };
            mark_cursor += words;
            card_cursor += cards;
        }

        return .{
            .allocator = allocator,
            .heap = heap,
            .handles = handles,
            .regions = regions,
            .mark_storage = mark_storage,
            .discovered_storage = discovered_storage,
            .card_storage = card_storage,
            .queue_storage = queue_storage,
            .work_queue_storage = work_queue_storage,
            .buffer_slots = buffer_slots,
            .queue = .{
                .slots = queue_storage,
                .write_cursor = std.atomic.Value(usize).init(0),
                .read_cursor = std.atomic.Value(usize).init(0),
            },
            .work_queue = .{
                .slots = work_queue_storage,
                .write_cursor = std.atomic.Value(usize).init(0),
                .read_cursor = std.atomic.Value(usize).init(0),
            },
            .layouts = layouts,
            .layout_offset_storage = layout_offset_storage,
            .static_root_slots = static_root_slots,
            .handshake_members = handshake_members,
            .handshake_states = handshake_states,
            .phase_value = std.atomic.Value(Phase).init(.idle),
            .epoch_value = std.atomic.Value(u64).init(0),
            .active_operations = std.atomic.Value(usize).init(0),
            .card_cursor = std.atomic.Value(usize).init(0),
            .handshake_active = std.atomic.Value(bool).init(false),
            .completed_handshake_epoch = std.atomic.Value(u64).init(0),
            .card_bytes = options.card_bytes,
        };
    }

    pub fn deinit(self: *ConcurrentCollector) Error!void {
        if (self.phase_value.cmpxchgStrong(.idle, .preparing, .acq_rel, .acquire) != null) {
            return error.WrongPhase;
        }
        for (self.buffer_slots) |*slot| {
            if (slot.load(.acquire) != 0) {
                self.phase_value.store(.idle, .release);
                return error.StillRegistered;
            }
        }
        self.allocator.free(self.buffer_slots);
        self.allocator.free(self.handshake_states);
        self.allocator.free(self.handshake_members);
        self.allocator.free(self.work_queue_storage);
        self.allocator.free(self.queue_storage);
        self.allocator.free(self.card_storage);
        self.allocator.free(self.discovered_storage);
        self.allocator.free(self.mark_storage);
        self.allocator.free(self.layout_offset_storage);
        self.allocator.free(self.static_root_slots);
        self.allocator.free(self.layouts);
        self.allocator.free(self.regions);
        self.* = undefined;
    }

    pub fn phase(self: *const ConcurrentCollector) Phase {
        return self.phase_value.load(.acquire);
    }

    pub fn epoch(self: *const ConcurrentCollector) u64 {
        return self.epoch_value.load(.acquire);
    }

    pub fn beginMark(self: *ConcurrentCollector) Error!u64 {
        if (self.handshake_active.load(.acquire)) return error.HandshakeInProgress;
        if (self.phase_value.cmpxchgStrong(.idle, .preparing, .acq_rel, .acquire) != null) {
            return error.WrongPhase;
        }
        errdefer self.phase_value.store(.idle, .release);
        if (self.active_operations.load(.seq_cst) != 0) return error.BuffersBusy;
        if (!self.queue.isEmpty()) return error.SatbNotDrained;
        if (!self.work_queue.isEmpty()) return error.WorkNotDrained;
        const old_epoch = self.epoch_value.load(.acquire);
        if (old_epoch == std.math.maxInt(u64)) return error.EpochExhausted;
        for (self.mark_storage) |*word| word.store(0, .monotonic);
        for (self.discovered_storage) |*word| word.store(0, .monotonic);
        const next_epoch = old_epoch + 1;
        self.completed_handshake_epoch.store(0, .monotonic);
        self.epoch_value.store(next_epoch, .release);
        // Acquire one credit before exposing `.marking`. A concurrent
        // terminator therefore cannot close the epoch between publication and
        // the first static-root load.
        _ = self.active_operations.fetchAdd(1, .seq_cst);
        self.phase_value.store(.marking, .release);
        _ = self.counters.cycles_started.fetchAdd(1, .monotonic);
        const static_scan = self.scanStaticRootsActive();
        self.leaveConcurrentOperation();
        static_scan catch |err| {
            self.abortMark() catch {};
            return err;
        };
        return next_epoch;
    }

    /// Publishes an allocation under the phase operation credit. Objects
    /// racing mark startup finish before the epoch opens; objects published in
    /// an active epoch are discovered before the credit is released.
    pub fn publishAllocatedObject(
        self: *ConcurrentCollector,
        reservation: runtime_heap.Reservation,
        handle: Handle,
        layout_id: u32,
    ) Error!void {
        _ = self.active_operations.fetchAdd(1, .seq_cst);
        defer self.leaveConcurrentOperation();
        const current_phase = self.phase_value.load(.seq_cst);
        if (current_phase == .preparing or current_phase == .terminating) return error.RetryBarrier;
        try self.heap.publishObjectWithLayout(reservation, handle, layout_id);
        if (current_phase == .marking) {
            _ = try self.markHandleActive(handle);
            _ = self.counters.allocations_blackened.fetchAdd(1, .monotonic);
        }
    }

    pub fn markHandle(self: *ConcurrentCollector, handle: Handle) Error!bool {
        try self.enterMarkOperation();
        defer self.leaveConcurrentOperation();
        return self.markHandleActive(handle);
    }

    fn markHandleActive(self: *ConcurrentCollector, handle: Handle) Error!bool {
        if (handle.isNull()) return false;
        const location = try self.liveLocation(handle);
        const offset: usize = @as(usize, location.offset_units) * runtime_value.object_alignment;
        if (!try self.heap.isObjectStart(location.region_id, offset)) return error.InvalidObject;
        const region = &self.regions[location.region_id];
        const slot = offset / runtime_value.object_alignment;
        const bit: u6 = @intCast(slot % 64);
        _ = region.mark_words[slot / 64].bitSet(bit, .acq_rel);
        const handle_word = &self.discovered_storage[handle.index / 64];
        const handle_bit: u6 = @intCast(handle.index % 64);
        const old = handle_word.bitSet(handle_bit, .acq_rel);
        if (old == 0) {
            self.work_queue.enqueueGuaranteed(handle);
            _ = self.counters.first_marks.fetchAdd(1, .monotonic);
            _ = self.counters.work_enqueued.fetchAdd(1, .monotonic);
            return true;
        }
        _ = self.counters.duplicate_marks.fetchAdd(1, .monotonic);
        return false;
    }

    pub fn isMarked(self: *const ConcurrentCollector, handle: Handle) Error!bool {
        if (handle.isNull()) return false;
        _ = try self.liveLocation(handle);
        const word = self.discovered_storage[handle.index / 64].load(.acquire);
        const bit: u6 = @intCast(handle.index % 64);
        return (word & (@as(u64, 1) << bit)) != 0;
    }

    pub fn drainSatb(self: *ConcurrentCollector, limit: usize) Error!usize {
        try self.enterMarkOperation();
        defer self.leaveConcurrentOperation();
        var drained: usize = 0;
        while (drained < limit) : (drained += 1) {
            const handle = self.queue.dequeue() orelse break;
            _ = try self.markHandleActive(handle);
        }
        _ = self.counters.satb_drained.fetchAdd(drained, .monotonic);
        return drained;
    }

    /// Claims and precisely scans up to `limit` discovered objects. Multiple
    /// collector workers may call this concurrently; queue slots provide
    /// single ownership and the operation credit closes dequeue/termination.
    pub fn traceWork(self: *ConcurrentCollector, limit: usize) Error!usize {
        try self.enterMarkOperation();
        defer self.leaveConcurrentOperation();
        var scanned: usize = 0;
        while (scanned < limit) {
            const handle = self.work_queue.dequeue() orelse break;
            self.scanObjectActive(handle) catch |err| {
                self.work_queue.enqueueGuaranteed(handle);
                return err;
            };
            scanned += 1;
        }
        _ = self.counters.objects_scanned.fetchAdd(scanned, .monotonic);
        return scanned;
    }

    /// Completes only after a stable, barrier-closed second readiness check.
    /// `RetryBarrier` prevents a mutator from committing a store while the
    /// phase is `.terminating`; failed elections reopen `.marking`.
    pub fn tryFinishMark(self: *ConcurrentCollector) Error!void {
        try self.verifyTerminationReady();
        if (self.phase_value.cmpxchgStrong(.marking, .terminating, .seq_cst, .seq_cst) != null) {
            return error.WrongPhase;
        }
        if (self.active_operations.load(.seq_cst) != 0) {
            self.phase_value.store(.marking, .seq_cst);
            _ = self.counters.termination_retries.fetchAdd(1, .monotonic);
            return error.BuffersBusy;
        }
        self.verifyTerminationReady() catch |err| {
            self.phase_value.store(.marking, .seq_cst);
            _ = self.counters.termination_retries.fetchAdd(1, .monotonic);
            return err;
        };
        self.phase_value.store(.idle, .release);
    }

    /// Abandons a failed mark cycle without waiting. The preparing gate stops
    /// new barrier work; if an operation already owns a credit, the caller
    /// retries later. Once closed, no thread can touch owner-local cursors, so
    /// queued and buffered snapshot state can be discarded before returning
    /// to idle.
    pub fn abortMark(self: *ConcurrentCollector) Error!void {
        if (self.handshake_active.load(.acquire)) return error.HandshakeInProgress;
        if (self.phase_value.cmpxchgStrong(.marking, .preparing, .seq_cst, .seq_cst) != null) {
            return error.WrongPhase;
        }
        if (self.active_operations.load(.seq_cst) != 0) {
            self.phase_value.store(.marking, .seq_cst);
            return error.BuffersBusy;
        }
        while (self.queue.dequeue() != null) {}
        while (self.work_queue.dequeue() != null) {}
        const current_epoch = self.epoch_value.load(.acquire);
        for (self.buffer_slots) |*slot| {
            const address = slot.load(.acquire);
            if (address == 0) continue;
            const buffer: *SatbBuffer = @ptrFromInt(address);
            buffer.cursor = 0;
            buffer.pending.store(0, .release);
            buffer.acknowledged_epoch.store(current_epoch, .release);
        }
        self.phase_value.store(.idle, .release);
    }

    /// Starts an allocation-free precise-root/SATB handshake. Membership and
    /// SATB binding are validated atomically before the registry publishes its
    /// request epoch, so setup failure cannot strand a mutator.
    pub fn beginThreadHandshake(
        self: *ConcurrentCollector,
        registry: *thread_registry.Registry,
    ) Error!MarkHandshake {
        if (self.phase_value.load(.acquire) != .marking) return error.WrongPhase;
        if (registry.memberCapacity() > self.handshake_members.len) return error.MemberBufferTooSmall;
        if (self.handshake_active.cmpxchgStrong(false, true, .acq_rel, .acquire) != null) {
            return error.HandshakeInProgress;
        }
        errdefer self.handshake_active.store(false, .release);
        for (self.handshake_states) |*state| state.store(.released, .monotonic);
        const handshake = try registry.beginValidatedHandshake(
            self.handshake_members,
            .{
                .context = @ptrCast(self),
                .validate = validateHandshakeMember,
            },
        );
        for (self.handshake_states[0..handshake.members.len]) |*state| state.store(.pending, .monotonic);
        return .{
            .collector = self,
            .registry_handshake = handshake,
            .mark_epoch = self.epoch_value.load(.acquire),
            .owner = std.Thread.getCurrentId(),
        };
    }

    /// Mutator-side poll for a collector-owned registry handshake. SATB is
    /// flushed before root publication; queue pressure returns to the caller
    /// without publishing an acknowledgement or committing further stores.
    pub fn pollThreadHandshake(
        self: *ConcurrentCollector,
        registry: *thread_registry.Registry,
        context: *thread_registry.ThreadContext,
        buffer: *SatbBuffer,
    ) Error!bool {
        if (!self.handshake_active.load(.acquire)) return error.InvalidState;
        if (buffer.collector != self or buffer.thread_context != context) return error.MissingSatbBuffer;
        if (context.observedEpoch() == registry.requestEpoch()) return false;
        try buffer.flushForEpoch(self.epoch_value.load(.acquire));
        return registry.poll(context);
    }

    /// Flushes owner-local SATB state before the registry makes roots passive.
    pub fn enterBlockedForMark(
        self: *ConcurrentCollector,
        registry: *thread_registry.Registry,
        context: *thread_registry.ThreadContext,
        buffer: *SatbBuffer,
    ) Error!void {
        if (buffer.collector != self or buffer.thread_context != context) return error.MissingSatbBuffer;
        if (self.phase_value.load(.acquire) == .marking) {
            try buffer.flushForEpoch(self.epoch_value.load(.acquire));
        }
        try registry.enterBlocked(context);
    }

    pub fn registerSatbBuffer(self: *ConcurrentCollector, buffer: *SatbBuffer) Error!void {
        return self.registerSatbBufferForThread(buffer, null);
    }

    pub fn registerThreadSatbBuffer(
        self: *ConcurrentCollector,
        buffer: *SatbBuffer,
        context: *thread_registry.ThreadContext,
    ) Error!void {
        return self.registerSatbBufferForThread(buffer, context);
    }

    fn registerSatbBufferForThread(
        self: *ConcurrentCollector,
        buffer: *SatbBuffer,
        context: ?*thread_registry.ThreadContext,
    ) Error!void {
        if (self.phase_value.cmpxchgStrong(.idle, .preparing, .acq_rel, .acquire) != null) {
            return error.WrongPhase;
        }
        defer self.phase_value.store(.idle, .release);
        if (buffer.collector != null) return error.AlreadyRegistered;
        if (buffer.active.load(.acquire) or buffer.pending.load(.acquire) != 0) return error.BuffersBusy;
        if (context) |candidate| {
            if (self.findBufferForContext(candidate) != null) return error.AlreadyRegistered;
        }
        for (self.buffer_slots, 0..) |*slot, index| {
            if (slot.load(.monotonic) != 0) continue;
            buffer.collector = self;
            buffer.thread_context = context;
            buffer.slot_index = index;
            buffer.acknowledged_epoch.store(self.epoch_value.load(.acquire), .release);
            slot.store(@intFromPtr(buffer), .release);
            return;
        }
        return error.RegistryFull;
    }

    pub fn unregisterSatbBuffer(self: *ConcurrentCollector, buffer: *SatbBuffer) Error!void {
        if (self.phase_value.cmpxchgStrong(.idle, .preparing, .acq_rel, .acquire) != null) {
            return error.WrongPhase;
        }
        defer self.phase_value.store(.idle, .release);
        if (buffer.collector != self) return error.Unregistered;
        if (buffer.active.load(.acquire) or buffer.pending.load(.acquire) != 0) return error.BuffersBusy;
        const slot = &self.buffer_slots[buffer.slot_index];
        if (slot.load(.acquire) != @intFromPtr(buffer)) return error.Unregistered;
        slot.store(0, .release);
        buffer.collector = null;
        buffer.thread_context = null;
        buffer.slot_index = 0;
    }

    pub fn setRegionKind(self: *ConcurrentCollector, region_id: u8, kind: RegionKind) Error!void {
        if (self.phase_value.cmpxchgStrong(.idle, .preparing, .acq_rel, .acquire) != null) {
            return error.WrongPhase;
        }
        defer self.phase_value.store(.idle, .release);
        const index: usize = region_id;
        if (index >= self.regions.len) return error.InvalidRegion;
        self.regions[index].kind.store(kind, .release);
    }

    /// Post-write old-to-young barrier. The sequence is advanced before the
    /// dirty publication so a refiner can never erase a racing mutator write.
    pub fn dirtyCardForStore(
        self: *ConcurrentCollector,
        destination: Handle,
        stored: Handle,
    ) Error!bool {
        if (destination.isNull() or stored.isNull()) return false;
        const destination_location = try self.liveLocation(destination);
        const stored_location = try self.liveLocation(stored);
        if (self.regions[destination_location.region_id].kind.load(.acquire) != .old or
            self.regions[stored_location.region_id].kind.load(.acquire) != .young) return false;
        const offset: usize = @as(usize, destination_location.offset_units) * runtime_value.object_alignment;
        if (!try self.heap.isObjectStart(destination_location.region_id, offset)) return error.InvalidObject;
        const card = &self.regions[destination_location.region_id].cards[offset / self.card_bytes];
        try advanceCardSequence(card);
        card.state.store(.dirty, .release);
        _ = self.counters.cards_dirtied.fetchAdd(1, .monotonic);
        return true;
    }

    pub fn claimDirtyCard(self: *ConcurrentCollector) ?CardClaim {
        if (self.card_storage.len == 0) return null;
        const start = self.card_cursor.fetchAdd(1, .monotonic);
        for (0..self.card_storage.len) |attempt| {
            const global = (start +% attempt) % self.card_storage.len;
            const card = &self.card_storage[global];
            if (card.state.cmpxchgStrong(.dirty, .claimed, .acq_rel, .acquire) != null) continue;
            const location = self.cardLocation(global) orelse unreachable;
            _ = self.counters.cards_claimed.fetchAdd(1, .monotonic);
            return .{
                .region_id = location.region_id,
                .card_index = location.card_index,
                .sequence = card.sequence.load(.acquire),
            };
        }
        return null;
    }

    pub fn finishCardClaim(self: *ConcurrentCollector, claim: CardClaim) Error!void {
        const index: usize = claim.region_id;
        if (index >= self.regions.len or claim.card_index >= self.regions[index].cards.len) {
            return error.InvalidClaim;
        }
        const card = &self.regions[index].cards[claim.card_index];
        if (card.sequence.load(.acquire) != claim.sequence) return;
        _ = card.state.cmpxchgStrong(.claimed, .clean, .release, .monotonic);
    }

    pub fn isCardDirty(self: *ConcurrentCollector, object: Handle) Error!bool {
        const location = try self.liveLocation(object);
        const offset: usize = @as(usize, location.offset_units) * runtime_value.object_alignment;
        if (!try self.heap.isObjectStart(location.region_id, offset)) return error.InvalidObject;
        return self.regions[location.region_id].cards[offset / self.card_bytes].state.load(.acquire) != .clean;
    }

    pub fn stats(self: *const ConcurrentCollector) Stats {
        return .{
            .cycles_started = self.counters.cycles_started.load(.acquire),
            .first_marks = self.counters.first_marks.load(.acquire),
            .duplicate_marks = self.counters.duplicate_marks.load(.acquire),
            .satb_published = self.counters.satb_published.load(.acquire),
            .satb_drained = self.counters.satb_drained.load(.acquire),
            .cards_dirtied = self.counters.cards_dirtied.load(.acquire),
            .cards_claimed = self.counters.cards_claimed.load(.acquire),
            .termination_retries = self.counters.termination_retries.load(.acquire),
            .work_enqueued = self.counters.work_enqueued.load(.acquire),
            .objects_scanned = self.counters.objects_scanned.load(.acquire),
            .references_scanned = self.counters.references_scanned.load(.acquire),
            .thread_handshakes = self.counters.thread_handshakes.load(.acquire),
            .root_snapshots = self.counters.root_snapshots.load(.acquire),
            .roots_discovered = self.counters.roots_discovered.load(.acquire),
            .allocations_blackened = self.counters.allocations_blackened.load(.acquire),
            .satb_repeat_elisions = self.counters.satb_repeat_elisions.load(.acquire),
            .card_repeat_elisions = self.counters.card_repeat_elisions.load(.acquire),
            .static_roots_scanned = self.counters.static_roots_scanned.load(.acquire),
            .static_root_writes = self.counters.static_root_writes.load(.acquire),
        };
    }

    fn verifyTerminationReady(self: *ConcurrentCollector) Error!void {
        const current_phase = self.phase_value.load(.acquire);
        if (current_phase != .marking and current_phase != .terminating) return error.WrongPhase;
        const current_epoch = self.epoch_value.load(.acquire);
        if (self.handshake_active.load(.acquire)) return error.HandshakeInProgress;
        if (self.hasBoundBuffers() and self.completed_handshake_epoch.load(.acquire) != current_epoch) {
            return error.MarkHandshakeIncomplete;
        }
        for (self.buffer_slots) |*slot| {
            const address = slot.load(.acquire);
            if (address == 0) continue;
            const buffer: *SatbBuffer = @ptrFromInt(address);
            if (buffer.active.load(.acquire)) return error.BuffersBusy;
            if (buffer.pending.load(.acquire) != 0 or
                buffer.acknowledged_epoch.load(.acquire) != current_epoch) return error.BuffersNotFlushed;
        }
        if (!self.queue.isEmpty()) return error.SatbNotDrained;
        if (!self.work_queue.isEmpty()) return error.WorkNotDrained;
    }

    fn scanObjectActive(self: *ConcurrentCollector, handle: Handle) Error!void {
        const location = try self.liveLocation(handle);
        const offset: usize = @as(usize, location.offset_units) * runtime_value.object_alignment;
        const object_size: usize = try self.heap.objectSize(location.region_id, offset);
        const layout_id = try self.heap.objectLayoutId(location.region_id, offset);
        if (layout_id == 0) return;
        const layout = self.findLayout(layout_id) orelse return error.UnknownLayout;
        if (object_size < layout.minimum_size) return error.InvalidLayout;
        const region = try self.handles.regionAt(location.region_id);
        const payload = region.base + offset;

        for (layout.reference_offsets) |reference_offset| {
            try self.scanReferenceAt(payload, object_size, reference_offset);
        }
        if (layout.trailing_references) |trailing| {
            const start: usize = trailing.offset;
            const stride: usize = trailing.stride;
            if (start > object_size or (object_size - start) % stride != 0) return error.InvalidLayout;
            var reference_offset = start;
            while (reference_offset < object_size) : (reference_offset += stride) {
                try self.scanReferenceAt(payload, object_size, reference_offset);
            }
        }
    }

    fn scanReferenceAt(
        self: *ConcurrentCollector,
        payload: usize,
        object_size: usize,
        reference_offset: usize,
    ) Error!void {
        if (reference_offset > object_size or @sizeOf(Handle) > object_size - reference_offset) {
            return error.InvalidLayout;
        }
        const slot: *const std.atomic.Value(u64) = @ptrFromInt(payload + reference_offset);
        const child: Handle = @bitCast(slot.load(.acquire));
        _ = self.counters.references_scanned.fetchAdd(1, .monotonic);
        if (!child.isNull()) _ = try self.markHandleActive(child);
    }

    fn findLayout(self: *const ConcurrentCollector, id: u32) ?*const LayoutEntry {
        var low: usize = 0;
        var high = self.layouts.len;
        while (low < high) {
            const middle = low + (high - low) / 2;
            const candidate = &self.layouts[middle];
            if (candidate.id < id) {
                low = middle + 1;
            } else if (candidate.id > id) {
                high = middle;
            } else {
                return candidate;
            }
        }
        return null;
    }

    fn findBufferForContext(self: *const ConcurrentCollector, context: *thread_registry.ThreadContext) ?*SatbBuffer {
        for (self.buffer_slots) |*slot| {
            const address = slot.load(.acquire);
            if (address == 0) continue;
            const buffer: *SatbBuffer = @ptrFromInt(address);
            if (buffer.thread_context == context) return buffer;
        }
        return null;
    }

    fn hasBoundBuffers(self: *const ConcurrentCollector) bool {
        for (self.buffer_slots) |*slot| {
            const address = slot.load(.acquire);
            if (address == 0) continue;
            const buffer: *SatbBuffer = @ptrFromInt(address);
            if (buffer.thread_context != null) return true;
        }
        return false;
    }

    pub fn bufferForThread(
        self: *ConcurrentCollector,
        context: *thread_registry.ThreadContext,
    ) ?*SatbBuffer {
        return self.findBufferForContext(context);
    }

    pub fn ownsBuffer(self: *const ConcurrentCollector, buffer: *const SatbBuffer) bool {
        return buffer.collector == self;
    }

    pub fn handleTable(self: *const ConcurrentCollector) *HandleTable {
        return self.handles;
    }

    /// Confirms that a runtime allocator's variable reference payload agrees
    /// with the collector's immutable scanner metadata.
    pub fn supportsTrailingReferenceLayout(
        self: *const ConcurrentCollector,
        layout_id: u32,
        data_offset: u32,
        element_stride: u32,
    ) bool {
        const layout = self.findLayout(layout_id) orelse return false;
        const trailing = layout.trailing_references orelse return false;
        return layout.minimum_size == data_offset and
            trailing.offset == data_offset and
            trailing.stride == element_stride;
    }

    /// Validates the immutable mutator binding consumed by interpreter/JIT
    /// poll adapters. The owner-local buffer cursor remains private.
    pub fn ownsThreadBuffer(
        self: *const ConcurrentCollector,
        buffer: *const SatbBuffer,
        context: *const thread_registry.ThreadContext,
    ) bool {
        return buffer.collector == self and buffer.thread_context == context;
    }

    pub fn isStaticRootSlot(self: *const ConcurrentCollector, slot_address: usize) bool {
        var low: usize = 0;
        var high = self.static_root_slots.len;
        while (low < high) {
            const middle = low + (high - low) / 2;
            const candidate = self.static_root_slots[middle];
            if (candidate < slot_address) {
                low = middle + 1;
            } else if (candidate > slot_address) {
                high = middle;
            } else return true;
        }
        return false;
    }

    pub fn referenceStorePreWrite(
        self: *ConcurrentCollector,
        buffer: *SatbBuffer,
        slot_address: usize,
        repeat_proven: bool,
    ) Error!void {
        if (!self.ownsBuffer(buffer)) return error.MissingSatbBuffer;
        if (!std.mem.isAligned(slot_address, runtime_value.object_alignment)) return error.UnalignedAddress;
        try buffer.recordReferenceSlot(slot_address, repeat_proven);
    }

    pub fn referenceStorePostWrite(
        self: *ConcurrentCollector,
        destination: Handle,
        stored: Handle,
        repeat_proven: bool,
        cached_destination_bits: *u64,
    ) Error!void {
        const destination_bits: u64 = @bitCast(destination);
        if (repeat_proven and cached_destination_bits.* == destination_bits and
            try self.republishDirtyCard(destination))
        {
            _ = self.counters.card_repeat_elisions.fetchAdd(1, .monotonic);
            return;
        }
        _ = try self.dirtyCardForStore(destination, stored);
        cached_destination_bits.* = destination_bits;
    }

    /// Static slots are roots rather than object cards. The initial scan
    /// covers unchanged slots; this insertion barrier covers replacements
    /// racing an active mark epoch without allocating or taking a lock.
    pub fn referenceStaticStorePostWrite(
        self: *ConcurrentCollector,
        slot_address: usize,
        stored: Handle,
    ) Error!void {
        if (!self.isStaticRootSlot(slot_address)) return error.InvalidLayout;
        _ = self.active_operations.fetchAdd(1, .seq_cst);
        defer self.leaveConcurrentOperation();
        switch (self.phase_value.load(.seq_cst)) {
            .idle => return,
            .preparing, .terminating => return error.RetryBarrier,
            .marking => {},
        }
        if (!stored.isNull()) _ = try self.markHandleActive(stored);
        _ = self.counters.static_root_writes.fetchAdd(1, .monotonic);
    }

    /// Caller owns an active mark-operation credit spanning phase publication.
    fn scanStaticRootsActive(self: *ConcurrentCollector) Error!void {
        for (self.static_root_slots) |slot_address| {
            const slot: *const std.atomic.Value(u64) = @ptrFromInt(slot_address);
            const root: Handle = @bitCast(slot.load(.acquire));
            if (!root.isNull()) _ = try self.markHandleActive(root);
            _ = self.counters.static_roots_scanned.fetchAdd(1, .monotonic);
        }
    }

    /// Re-publishes a compiler-proven repeated write without advancing the
    /// refinement sequence. The reference store precedes this release store.
    /// A refiner either acquires it before scanning, or wins the claim first
    /// and then observes `.dirty` instead of cleaning the card on completion.
    fn republishDirtyCard(self: *ConcurrentCollector, destination: Handle) Error!bool {
        if (destination.isNull()) return false;
        const location = try self.liveLocation(destination);
        if (self.regions[location.region_id].kind.load(.acquire) != .old) return false;
        const offset: usize = @as(usize, location.offset_units) * runtime_value.object_alignment;
        if (!try self.heap.isObjectStart(location.region_id, offset)) return error.InvalidObject;
        const card = &self.regions[location.region_id].cards[offset / self.card_bytes];
        if (card.state.load(.acquire) != .dirty) return false;
        card.state.store(.dirty, .release);
        return true;
    }

    /// The sequentially-consistent operation credit closes the race between
    /// observing `.marking` and the termination CAS. An operation is either
    /// visible to the terminator or observes `.terminating` and retries.
    fn enterMarkOperation(self: *ConcurrentCollector) Error!void {
        _ = self.active_operations.fetchAdd(1, .seq_cst);
        switch (self.phase_value.load(.seq_cst)) {
            .marking => return,
            .preparing, .terminating => {
                self.leaveConcurrentOperation();
                return error.RetryBarrier;
            },
            .idle => {
                self.leaveConcurrentOperation();
                return error.WrongPhase;
            },
        }
    }

    fn leaveConcurrentOperation(self: *ConcurrentCollector) void {
        const old = self.active_operations.fetchSub(1, .seq_cst);
        std.debug.assert(old != 0);
    }

    fn liveLocation(self: *const ConcurrentCollector, handle: Handle) Error!runtime_value.LocationDescriptor {
        const location = try self.handles.inspect(handle);
        if (location.state != .live and location.state != .evacuating) return error.InvalidState;
        const index: usize = location.region_id;
        if (index >= self.regions.len) return error.InvalidRegion;
        return location;
    }

    const CardLocation = struct { region_id: u8, card_index: usize };

    fn cardLocation(self: *const ConcurrentCollector, global: usize) ?CardLocation {
        for (self.regions, 0..) |region, index| {
            if (global >= region.card_base and global - region.card_base < region.cards.len) {
                return .{ .region_id = @intCast(index), .card_index = global - region.card_base };
            }
        }
        return null;
    }
};

fn validateHandshakeMember(context: *anyopaque, member: *thread_registry.ThreadContext) bool {
    const collector: *ConcurrentCollector = @ptrCast(@alignCast(context));
    return collector.findBufferForContext(member) != null;
}

/// Collector-owned asynchronous root/SATB handshake. `advance` never waits:
/// each ready member is copied into stable mark work and released immediately,
/// while unready members continue toward their next poll independently.
pub const MarkHandshake = struct {
    collector: *ConcurrentCollector,
    registry_handshake: thread_registry.Handshake,
    mark_epoch: u64,
    owner: std.Thread.Id,
    completed: bool = false,

    pub fn advance(self: *MarkHandshake) Error!bool {
        if (std.Thread.getCurrentId() != self.owner) return error.WrongThread;
        if (self.completed) return error.InvalidState;
        var all_released = true;
        for (self.registry_handshake.members, 0..) |member, index| {
            const state = &self.collector.handshake_states[index];
            if (state.load(.acquire) == .released) continue;
            if (state.cmpxchgStrong(.pending, .feeding, .acq_rel, .acquire) != null) {
                all_released = false;
                continue;
            }
            if (!self.registry_handshake.isReady(member)) {
                state.store(.pending, .release);
                all_released = false;
                continue;
            }

            const buffer = self.collector.findBufferForContext(member) orelse {
                state.store(.pending, .release);
                return error.MissingSatbBuffer;
            };
            if (buffer.active.load(.acquire) or buffer.pending.load(.acquire) != 0) {
                state.store(.pending, .release);
                all_released = false;
                continue;
            }
            if (buffer.acknowledged_epoch.load(.acquire) != self.mark_epoch) {
                // A ready registry member cannot execute mutator barriers. An
                // empty owner buffer can therefore be acknowledged passively.
                buffer.acknowledged_epoch.store(self.mark_epoch, .release);
            }

            const roots = self.registry_handshake.snapshot(member) catch |err| {
                state.store(.pending, .release);
                return err;
            };
            for (roots) |root| {
                _ = self.collector.markHandle(root) catch |err| {
                    state.store(.pending, .release);
                    return err;
                };
            }
            self.registry_handshake.release(member) catch |err| {
                state.store(.pending, .release);
                return err;
            };
            state.store(.released, .release);
            _ = self.collector.counters.root_snapshots.fetchAdd(1, .monotonic);
            _ = self.collector.counters.roots_discovered.fetchAdd(roots.len, .monotonic);
        }

        if (!all_released) {
            for (self.registry_handshake.members, 0..) |_, index| {
                if (self.collector.handshake_states[index].load(.acquire) != .released) return false;
            }
        }
        try self.registry_handshake.finish();
        self.completed = true;
        self.collector.completed_handshake_epoch.store(self.mark_epoch, .release);
        self.collector.handshake_active.store(false, .release);
        _ = self.collector.counters.thread_handshakes.fetchAdd(1, .monotonic);
        return true;
    }
};

/// Owner-confined pre-write log. `record` must complete successfully before
/// the associated reference store is committed. Queue pressure and a mark
/// termination election are explicit retry conditions, never silent drops.
pub const SatbBuffer = struct {
    allocator: std.mem.Allocator,
    entries: []Handle,
    owner: std.Thread.Id,
    collector: ?*ConcurrentCollector = null,
    thread_context: ?*thread_registry.ThreadContext = null,
    slot_index: usize = 0,
    cursor: usize = 0,
    active: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    pending: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    acknowledged_epoch: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    last_reference_slot: usize = 0,
    last_reference_epoch: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, capacity: usize) Error!SatbBuffer {
        if (capacity == 0) return error.InvalidCapacity;
        return .{
            .allocator = allocator,
            .entries = try allocator.alloc(Handle, capacity),
            .owner = std.Thread.getCurrentId(),
        };
    }

    pub fn deinit(self: *SatbBuffer) Error!void {
        if (self.collector != null) return error.StillRegistered;
        self.allocator.free(self.entries);
        self.* = undefined;
    }

    pub fn record(self: *SatbBuffer, old_value: Handle) Error!void {
        if (std.Thread.getCurrentId() != self.owner) return error.WrongThread;
        const collector = self.collector orelse return error.Unregistered;
        _ = collector.active_operations.fetchAdd(1, .seq_cst);
        defer collector.leaveConcurrentOperation();
        self.active.store(true, .release);
        defer self.active.store(false, .release);

        switch (collector.phase_value.load(.seq_cst)) {
            .idle => return,
            .preparing, .terminating => return error.RetryBarrier,
            .marking => {},
        }
        try self.recordActive(collector, old_value);
    }

    /// Compiler-proven repeat stores still enter the phase operation credit.
    /// The physical slot load is skipped only if the first store populated the
    /// owner-local cache in this exact mark epoch. This closes the otherwise
    /// unsafe idle-to-marking transition between two compiled stores.
    fn recordReferenceSlot(self: *SatbBuffer, slot_address: usize, repeat_proven: bool) Error!void {
        if (std.Thread.getCurrentId() != self.owner) return error.WrongThread;
        const collector = self.collector orelse return error.Unregistered;
        _ = collector.active_operations.fetchAdd(1, .seq_cst);
        defer collector.leaveConcurrentOperation();
        self.active.store(true, .release);
        defer self.active.store(false, .release);

        switch (collector.phase_value.load(.seq_cst)) {
            .idle => return,
            .preparing, .terminating => return error.RetryBarrier,
            .marking => {},
        }
        const mark_epoch = collector.epoch_value.load(.acquire);
        if (repeat_proven and self.last_reference_epoch == mark_epoch and self.last_reference_slot == slot_address) {
            _ = collector.counters.satb_repeat_elisions.fetchAdd(1, .monotonic);
            return;
        }
        const slot: *const std.atomic.Value(u64) = @ptrFromInt(slot_address);
        const old_value: Handle = @bitCast(slot.load(.acquire));
        try self.recordActive(collector, old_value);
        self.last_reference_slot = slot_address;
        self.last_reference_epoch = mark_epoch;
    }

    fn recordActive(self: *SatbBuffer, collector: *ConcurrentCollector, old_value: Handle) Error!void {
        if (old_value.isNull()) return;
        if (self.cursor == self.entries.len) try self.flushEntries(collector);
        self.acknowledged_epoch.store(0, .release);
        self.entries[self.cursor] = old_value;
        self.cursor += 1;
        self.pending.store(self.cursor, .release);
    }

    pub fn flushForEpoch(self: *SatbBuffer, expected_epoch: u64) Error!void {
        if (std.Thread.getCurrentId() != self.owner) return error.WrongThread;
        const collector = self.collector orelse return error.Unregistered;
        _ = collector.active_operations.fetchAdd(1, .seq_cst);
        defer collector.leaveConcurrentOperation();
        self.active.store(true, .release);
        defer self.active.store(false, .release);
        const current_phase = collector.phase_value.load(.seq_cst);
        if (current_phase == .preparing or current_phase == .terminating) return error.RetryBarrier;
        if (current_phase != .marking) return error.WrongPhase;
        if (collector.epoch_value.load(.acquire) != expected_epoch) return error.WrongPhase;
        try self.flushEntries(collector);
        self.acknowledged_epoch.store(expected_epoch, .release);
    }

    pub fn pendingCount(self: *const SatbBuffer) usize {
        return self.pending.load(.acquire);
    }

    fn flushEntries(self: *SatbBuffer, collector: *ConcurrentCollector) Error!void {
        var published: usize = 0;
        while (published < self.cursor) : (published += 1) {
            if (!collector.queue.enqueue(self.entries[published])) break;
        }
        if (published != 0) {
            const remaining = self.cursor - published;
            std.mem.copyForwards(Handle, self.entries[0..remaining], self.entries[published..self.cursor]);
            self.cursor = remaining;
            self.pending.store(remaining, .release);
            _ = collector.counters.satb_published.fetchAdd(published, .monotonic);
        }
        if (self.cursor != 0) return error.SatbQueueFull;
        self.pending.store(0, .release);
    }
};

fn addCapacity(current: usize, added: usize) Error!usize {
    if (added > std.math.maxInt(usize) - current) return error.InvalidCapacity;
    return current + added;
}

fn validateLayoutSpecs(layouts: []const LayoutSpec) Error!usize {
    var offset_count: usize = 0;
    var previous_id: u32 = 0;
    for (layouts, 0..) |layout, index| {
        if (layout.id == 0 or (index != 0 and layout.id <= previous_id) or
            layout.minimum_size == 0 or
            !std.mem.isAligned(layout.minimum_size, runtime_value.object_alignment))
        {
            return error.InvalidLayout;
        }
        previous_id = layout.id;
        offset_count = try addCapacity(offset_count, layout.reference_offsets.len);
        for (layout.reference_offsets) |offset| {
            if (!std.mem.isAligned(offset, runtime_value.object_alignment) or
                offset > layout.minimum_size or
                @sizeOf(Handle) > layout.minimum_size - offset)
            {
                return error.InvalidLayout;
            }
            if (layout.trailing_references) |trailing| {
                if (offset >= trailing.offset) return error.InvalidLayout;
            }
        }
        if (layout.trailing_references) |trailing| {
            if (!std.mem.isAligned(trailing.offset, runtime_value.object_alignment) or
                trailing.offset > layout.minimum_size or
                trailing.stride < @sizeOf(Handle) or
                !std.mem.isAligned(trailing.stride, runtime_value.object_alignment))
            {
                return error.InvalidLayout;
            }
        }
    }
    return offset_count;
}

fn validateStaticRootSlots(slots: []const usize) Error!void {
    for (slots, 0..) |slot, index| {
        if (slot == 0 or !std.mem.isAligned(slot, @alignOf(std.atomic.Value(u64)))) {
            return error.UnalignedAddress;
        }
        if (index != 0 and slots[index - 1] >= slot) return error.InvalidLayout;
    }
}

fn advanceCardSequence(card: *CardEntry) Error!void {
    while (true) {
        const old = card.sequence.load(.acquire);
        if (old == std.math.maxInt(u64)) return error.CardSequenceExhausted;
        if (card.sequence.cmpxchgWeak(old, old + 1, .acq_rel, .acquire) == null) return;
    }
}

fn publishTestObject(heap: *ManagedHeap, handles: *HandleTable, tlab: *runtime_heap.ThreadAllocator, size: usize) !Handle {
    return publishTypedTestObject(heap, handles, tlab, size, 0);
}

fn publishTypedTestObject(
    heap: *ManagedHeap,
    handles: *HandleTable,
    tlab: *runtime_heap.ThreadAllocator,
    size: usize,
    layout_id: u32,
) !Handle {
    const reservation = try tlab.allocate(size, runtime_value.object_alignment);
    const handle = try handles.reserve(0, 0);
    errdefer handles.cancelReservation(handle) catch {};
    try heap.publishObjectWithLayout(reservation, handle, layout_id);
    return handle;
}

fn storeTestReference(handles: *HandleTable, object: Handle, offset: usize, value: Handle) !void {
    const payload = @intFromPtr(try handles.resolve(object));
    const slot: *std.atomic.Value(u64) = @ptrFromInt(payload + offset);
    slot.store(@bitCast(value), .release);
}

fn waitMarkHandshake(handshake: *MarkHandshake) !void {
    for (0..1_000_000) |_| {
        if (try handshake.advance()) return;
        std.atomic.spinLoopHint();
        std.Thread.yield() catch {};
    }
    return error.Timeout;
}

test "SATB values are marked before a cycle can terminate" {
    var storage: [1024]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 8, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 128);
    defer heap.deinit();
    var collector = try ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 4,
        .max_satb_buffers = 2,
        .card_bytes = 64,
    });
    defer collector.deinit() catch unreachable;
    var tlab = heap.threadAllocator();
    const root = try publishTestObject(&heap, &handles, &tlab, 24);
    const old_a = try publishTestObject(&heap, &handles, &tlab, 24);
    const old_b = try publishTestObject(&heap, &handles, &tlab, 24);
    var buffer = try SatbBuffer.init(std.testing.allocator, 2);
    defer buffer.deinit() catch unreachable;
    try collector.registerSatbBuffer(&buffer);
    defer collector.unregisterSatbBuffer(&buffer) catch unreachable;

    const epoch = try collector.beginMark();
    try std.testing.expect(try collector.markHandle(root));
    try std.testing.expect(!(try collector.markHandle(root)));
    try buffer.record(old_a);
    try buffer.record(old_b);
    try std.testing.expectError(error.BuffersNotFlushed, collector.tryFinishMark());
    try buffer.flushForEpoch(epoch);
    try std.testing.expectError(error.SatbNotDrained, collector.tryFinishMark());
    try std.testing.expectEqual(@as(usize, 2), try collector.drainSatb(16));
    try std.testing.expectEqual(@as(usize, 3), try collector.traceWork(16));
    try collector.tryFinishMark();

    try std.testing.expect(try collector.isMarked(root));
    try std.testing.expect(try collector.isMarked(old_a));
    try std.testing.expect(try collector.isMarked(old_b));
    try std.testing.expectEqual(Phase.idle, collector.phase());
    try std.testing.expectEqual(@as(u64, 3), collector.stats().first_marks);
}

test "SATB saturation retains the unpublished suffix" {
    var storage: [512]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 4, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 128);
    defer heap.deinit();
    var collector = try ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 1,
        .max_satb_buffers = 1,
        .card_bytes = 64,
    });
    defer collector.deinit() catch unreachable;
    var tlab = heap.threadAllocator();
    const first = try publishTestObject(&heap, &handles, &tlab, 24);
    const second = try publishTestObject(&heap, &handles, &tlab, 24);
    var buffer = try SatbBuffer.init(std.testing.allocator, 2);
    defer buffer.deinit() catch unreachable;
    try collector.registerSatbBuffer(&buffer);
    defer collector.unregisterSatbBuffer(&buffer) catch unreachable;

    const epoch = try collector.beginMark();
    try buffer.record(first);
    try buffer.record(second);
    try std.testing.expectError(error.SatbQueueFull, buffer.flushForEpoch(epoch));
    try std.testing.expectEqual(@as(usize, 1), buffer.pendingCount());
    try std.testing.expectEqual(@as(usize, 1), try collector.drainSatb(1));
    try buffer.flushForEpoch(epoch);
    try std.testing.expectEqual(@as(usize, 1), try collector.drainSatb(1));
    try std.testing.expectEqual(@as(usize, 2), try collector.traceWork(4));
    try collector.tryFinishMark();
    try std.testing.expect(try collector.isMarked(first));
    try std.testing.expect(try collector.isMarked(second));
}

test "epoch-validated repeat SATB skips only same-cycle slot loads" {
    var storage: [512]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 3, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 64);
    defer heap.deinit();
    var collector = try ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 2,
        .max_satb_buffers = 1,
        .card_bytes = 64,
    });
    defer collector.deinit() catch unreachable;
    var tlab = heap.threadAllocator();
    const first = try publishTestObject(&heap, &handles, &tlab, 8);
    const second = try publishTestObject(&heap, &handles, &tlab, 8);
    var buffer = try SatbBuffer.init(std.testing.allocator, 2);
    defer buffer.deinit() catch unreachable;
    try collector.registerSatbBuffer(&buffer);
    defer collector.unregisterSatbBuffer(&buffer) catch unreachable;
    var slot: std.atomic.Value(u64) align(runtime_value.object_alignment) = std.atomic.Value(u64).init(@bitCast(first));

    const first_epoch = try collector.beginMark();
    try collector.referenceStorePreWrite(&buffer, @intFromPtr(&slot), false);
    slot.store(@bitCast(second), .release);
    try collector.referenceStorePreWrite(&buffer, @intFromPtr(&slot), true);
    try std.testing.expectEqual(@as(usize, 1), buffer.pendingCount());
    try std.testing.expectEqual(@as(u64, 1), collector.stats().satb_repeat_elisions);
    try buffer.flushForEpoch(first_epoch);
    try std.testing.expectEqual(@as(usize, 1), try collector.drainSatb(2));
    try std.testing.expectEqual(@as(usize, 1), try collector.traceWork(2));
    try collector.tryFinishMark();

    // The compiler proof remains true, but an epoch transition invalidates the
    // owner-local cache and forces the current slot value to be logged.
    const second_epoch = try collector.beginMark();
    try collector.referenceStorePreWrite(&buffer, @intFromPtr(&slot), true);
    try std.testing.expectEqual(@as(usize, 1), buffer.pendingCount());
    try std.testing.expectEqual(@as(u64, 1), collector.stats().satb_repeat_elisions);
    try buffer.flushForEpoch(second_epoch);
    try std.testing.expectEqual(@as(usize, 1), try collector.drainSatb(2));
    try std.testing.expectEqual(@as(usize, 1), try collector.traceWork(2));
    try collector.tryFinishMark();
}

test "precise layout tracing ignores scalars and discovers reference graph" {
    const reference_offsets = [_]u32{ 0, 16 };
    const layouts = [_]LayoutSpec{.{
        .id = 7,
        .minimum_size = 32,
        .reference_offsets = &reference_offsets,
    }};
    var storage: [1024]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 8, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 128);
    defer heap.deinit();
    var collector = try ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 4,
        .max_satb_buffers = 1,
        .card_bytes = 64,
        .layouts = &layouts,
    });
    defer collector.deinit() catch unreachable;
    var tlab = heap.threadAllocator();
    const parent = try publishTypedTestObject(&heap, &handles, &tlab, 32, 7);
    const first_child = try publishTestObject(&heap, &handles, &tlab, 8);
    const second_child = try publishTestObject(&heap, &handles, &tlab, 8);
    try storeTestReference(&handles, parent, 0, first_child);
    try storeTestReference(&handles, parent, 16, second_child);
    const payload: [*]u8 = @ptrCast(try handles.resolve(parent));
    std.mem.writeInt(u64, payload[8..16], @bitCast(Handle{ .index = 6, .generation = 99 }), .little);

    _ = try collector.beginMark();
    try std.testing.expect(try collector.markHandle(parent));
    try std.testing.expectError(error.WorkNotDrained, collector.tryFinishMark());
    try std.testing.expectEqual(@as(usize, 3), try collector.traceWork(8));
    try collector.tryFinishMark();

    try std.testing.expect(try collector.isMarked(parent));
    try std.testing.expect(try collector.isMarked(first_child));
    try std.testing.expect(try collector.isMarked(second_child));
    try std.testing.expectEqual(@as(u64, 2), collector.stats().references_scanned);
}

test "parallel collectors trace a trailing reference array exactly once" {
    const child_count = 64;
    const layouts = [_]LayoutSpec{.{
        .id = 9,
        .minimum_size = 8,
        .trailing_references = .{ .offset = 0 },
    }};
    var storage: [8192]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, child_count + 1, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 1024);
    defer heap.deinit();
    var collector = try ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 8,
        .max_satb_buffers = 1,
        .card_bytes = 64,
        .layouts = &layouts,
    });
    defer collector.deinit() catch unreachable;
    var tlab = heap.threadAllocator();
    const parent = try publishTypedTestObject(&heap, &handles, &tlab, child_count * @sizeOf(Handle), 9);
    var children: [child_count]Handle = undefined;
    for (&children, 0..) |*child, index| {
        child.* = try publishTestObject(&heap, &handles, &tlab, 8);
        try storeTestReference(&handles, parent, index * @sizeOf(Handle), child.*);
    }

    _ = try collector.beginMark();
    try std.testing.expect(try collector.markHandle(parent));
    const worker_count = 4;
    var failed = std.atomic.Value(bool).init(false);
    const Worker = struct {
        collector: *ConcurrentCollector,
        target: u64,
        failed: *std.atomic.Value(bool),

        fn run(self: *@This()) void {
            while (self.collector.stats().objects_scanned < self.target) {
                const traced = self.collector.traceWork(4) catch {
                    self.failed.store(true, .release);
                    return;
                };
                if (traced == 0) std.atomic.spinLoopHint();
            }
        }
    };
    var workers: [worker_count]Worker = undefined;
    var threads: [worker_count]std.Thread = undefined;
    for (&workers, &threads) |*worker, *thread| {
        worker.* = .{
            .collector = &collector,
            .target = child_count + 1,
            .failed = &failed,
        };
        thread.* = try std.Thread.spawn(.{}, Worker.run, .{worker});
    }
    for (threads) |thread| thread.join();
    try std.testing.expect(!failed.load(.acquire));
    try collector.tryFinishMark();
    try std.testing.expectEqual(@as(u64, child_count + 1), collector.stats().objects_scanned);
    try std.testing.expectEqual(@as(u64, child_count), collector.stats().references_scanned);
    for (children) |child| try std.testing.expect(try collector.isMarked(child));
}

test "unknown layouts fail closed and cycle abort clears work" {
    var storage: [256]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 2, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 64);
    defer heap.deinit();
    var collector = try ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 2,
        .max_satb_buffers = 1,
        .card_bytes = 64,
    });
    defer collector.deinit() catch unreachable;
    var tlab = heap.threadAllocator();
    const object = try publishTypedTestObject(&heap, &handles, &tlab, 8, 404);

    _ = try collector.beginMark();
    _ = try collector.markHandle(object);
    try std.testing.expectError(error.UnknownLayout, collector.traceWork(1));
    try std.testing.expectError(error.WorkNotDrained, collector.tryFinishMark());
    try collector.abortMark();
    try std.testing.expectEqual(Phase.idle, collector.phase());
    _ = try collector.beginMark();
    try collector.tryFinishMark();
}

test "allocation publication is black and closes mark startup race" {
    var storage: [512]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 3, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 64);
    defer heap.deinit();
    var collector = try ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 2,
        .max_satb_buffers = 1,
        .card_bytes = 64,
    });
    defer collector.deinit() catch unreachable;
    var tlab = heap.threadAllocator();

    _ = collector.active_operations.fetchAdd(1, .seq_cst);
    try std.testing.expectError(error.BuffersBusy, collector.beginMark());
    try std.testing.expectEqual(Phase.idle, collector.phase());
    collector.leaveConcurrentOperation();

    _ = try collector.beginMark();
    const reservation = try tlab.allocate(8, 8);
    const handle = try handles.reserve(0, 0);
    try collector.publishAllocatedObject(reservation, handle, 0);
    try std.testing.expect(try collector.isMarked(handle));
    try std.testing.expectEqual(@as(u64, 1), collector.stats().allocations_blackened);
    try std.testing.expectEqual(@as(usize, 1), try collector.traceWork(2));
    try collector.tryFinishMark();

    const idle_reservation = try tlab.allocate(8, 8);
    const idle_handle = try handles.reserve(0, 0);
    try collector.publishAllocatedObject(idle_reservation, idle_handle, 0);
    try std.testing.expect(!(try collector.isMarked(idle_handle)));
}

test "termination reopens marking while a credited operation is active" {
    var storage: [256]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 1, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 64);
    defer heap.deinit();
    var collector = try ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 1,
        .max_satb_buffers = 1,
        .card_bytes = 64,
    });
    defer collector.deinit() catch unreachable;

    _ = try collector.beginMark();
    _ = collector.active_operations.fetchAdd(1, .seq_cst);
    try std.testing.expectError(error.BuffersBusy, collector.tryFinishMark());
    try std.testing.expectEqual(Phase.marking, collector.phase());
    collector.leaveConcurrentOperation();
    try collector.tryFinishMark();
    try std.testing.expectEqual(Phase.idle, collector.phase());
    try std.testing.expectEqual(@as(u64, 1), collector.stats().termination_retries);
}

test "sequence validated card completion preserves racing dirty writes" {
    var first_region: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    var second_region: [128]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{
        try runtime_value.Region.fromSlice(&first_region),
        try runtime_value.Region.fromSlice(&second_region),
    };
    var handles = try HandleTable.init(std.testing.allocator, 4, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 64);
    defer heap.deinit();
    var collector = try ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 2,
        .max_satb_buffers = 1,
        .card_bytes = 32,
    });
    defer collector.deinit() catch unreachable;
    var tlab = heap.threadAllocator();
    const old_object = try publishTestObject(&heap, &handles, &tlab, 48);
    const young_object = try publishTestObject(&heap, &handles, &tlab, 48);
    try std.testing.expectEqual(@as(u8, 0), (try handles.inspect(old_object)).region_id);
    try std.testing.expectEqual(@as(u8, 1), (try handles.inspect(young_object)).region_id);
    try collector.setRegionKind(0, .old);
    try collector.setRegionKind(1, .young);

    try std.testing.expect(try collector.dirtyCardForStore(old_object, young_object));
    const first_claim = collector.claimDirtyCard() orelse return error.TestExpectedEqual;
    try std.testing.expect(try collector.dirtyCardForStore(old_object, young_object));
    try collector.finishCardClaim(first_claim);
    try std.testing.expect(try collector.isCardDirty(old_object));
    const second_claim = collector.claimDirtyCard() orelse return error.TestExpectedEqual;
    try collector.finishCardClaim(second_claim);
    try std.testing.expect(!(try collector.isCardDirty(old_object)));

    var cached_destination: u64 = 0;
    try collector.referenceStorePostWrite(old_object, young_object, false, &cached_destination);
    try collector.referenceStorePostWrite(old_object, young_object, true, &cached_destination);
    try std.testing.expectEqual(@as(u64, 1), collector.stats().card_repeat_elisions);
    const coalesced_claim = collector.claimDirtyCard() orelse return error.TestExpectedEqual;
    try collector.finishCardClaim(coalesced_claim);
    try std.testing.expect(!(try collector.isCardDirty(old_object)));

    // A refiner that has already claimed the card disables coalescing. The
    // repeat barrier advances the sequence and republishes dirty, so stale
    // claim completion cannot erase the racing write.
    try collector.referenceStorePostWrite(old_object, young_object, false, &cached_destination);
    const racing_claim = collector.claimDirtyCard() orelse return error.TestExpectedEqual;
    try collector.referenceStorePostWrite(old_object, young_object, true, &cached_destination);
    try collector.finishCardClaim(racing_claim);
    try std.testing.expect(try collector.isCardDirty(old_object));
    try std.testing.expectEqual(@as(u64, 1), collector.stats().card_repeat_elisions);
    const final_claim = collector.claimDirtyCard() orelse return error.TestExpectedEqual;
    try collector.finishCardClaim(final_claim);
    try std.testing.expect(!(try collector.isCardDirty(old_object)));
}

test "SATB owner confinement rejects cross thread mutation" {
    var storage: [256]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 2, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 64);
    defer heap.deinit();
    var collector = try ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 2,
        .max_satb_buffers = 1,
        .card_bytes = 64,
    });
    defer collector.deinit() catch unreachable;
    var tlab = heap.threadAllocator();
    const object = try publishTestObject(&heap, &handles, &tlab, 24);
    var buffer = try SatbBuffer.init(std.testing.allocator, 2);
    defer buffer.deinit() catch unreachable;
    try collector.registerSatbBuffer(&buffer);
    defer collector.unregisterSatbBuffer(&buffer) catch unreachable;
    const epoch = try collector.beginMark();
    var rejected = std.atomic.Value(bool).init(false);

    const Worker = struct {
        buffer: *SatbBuffer,
        object: Handle,
        rejected: *std.atomic.Value(bool),
        fn run(self: *@This()) void {
            self.buffer.record(self.object) catch |err| {
                self.rejected.store(err == error.WrongThread, .release);
            };
        }
    };
    var worker = Worker{ .buffer = &buffer, .object = object, .rejected = &rejected };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    thread.join();
    try std.testing.expect(rejected.load(.acquire));
    try buffer.flushForEpoch(epoch);
    try collector.tryFinishMark();
}

test "concurrent owner buffers publish without locks or lost handles" {
    const worker_count = 4;
    var storage: [2048]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 8, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 128);
    defer heap.deinit();
    var collector = try ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 8,
        .max_satb_buffers = worker_count,
        .card_bytes = 64,
    });
    defer collector.deinit() catch unreachable;
    var tlab = heap.threadAllocator();
    var objects: [worker_count]Handle = undefined;
    for (&objects) |*object| object.* = try publishTestObject(&heap, &handles, &tlab, 24);

    var ready = std.atomic.Value(usize).init(0);
    var completed = std.atomic.Value(usize).init(0);
    var start = std.atomic.Value(bool).init(false);
    var release = std.atomic.Value(bool).init(false);
    var failed = std.atomic.Value(bool).init(false);
    var cycle_epoch = std.atomic.Value(u64).init(0);

    const Worker = struct {
        collector: *ConcurrentCollector,
        object: Handle,
        ready: *std.atomic.Value(usize),
        completed: *std.atomic.Value(usize),
        start: *std.atomic.Value(bool),
        release: *std.atomic.Value(bool),
        failed: *std.atomic.Value(bool),
        cycle_epoch: *std.atomic.Value(u64),

        fn run(self: *@This()) void {
            var buffer = SatbBuffer.init(std.testing.allocator, 2) catch {
                self.failed.store(true, .release);
                _ = self.ready.fetchAdd(1, .release);
                return;
            };
            defer buffer.deinit() catch self.failed.store(true, .release);
            while (true) {
                self.collector.registerSatbBuffer(&buffer) catch |err| switch (err) {
                    error.WrongPhase => {
                        std.atomic.spinLoopHint();
                        continue;
                    },
                    else => {
                        self.failed.store(true, .release);
                        _ = self.ready.fetchAdd(1, .release);
                        return;
                    },
                };
                break;
            }
            _ = self.ready.fetchAdd(1, .release);
            while (!self.start.load(.acquire)) std.atomic.spinLoopHint();

            const epoch = self.cycle_epoch.load(.acquire);
            buffer.record(self.object) catch self.failed.store(true, .release);
            buffer.flushForEpoch(epoch) catch self.failed.store(true, .release);
            _ = self.completed.fetchAdd(1, .release);
            while (!self.release.load(.acquire)) std.atomic.spinLoopHint();

            while (true) {
                self.collector.unregisterSatbBuffer(&buffer) catch |err| switch (err) {
                    error.WrongPhase => {
                        std.atomic.spinLoopHint();
                        continue;
                    },
                    else => self.failed.store(true, .release),
                };
                break;
            }
        }
    };

    var workers: [worker_count]Worker = undefined;
    var threads: [worker_count]std.Thread = undefined;
    for (&workers, &threads, objects) |*worker, *thread, object| {
        worker.* = .{
            .collector = &collector,
            .object = object,
            .ready = &ready,
            .completed = &completed,
            .start = &start,
            .release = &release,
            .failed = &failed,
            .cycle_epoch = &cycle_epoch,
        };
        thread.* = try std.Thread.spawn(.{}, Worker.run, .{worker});
    }
    while (ready.load(.acquire) != worker_count) std.atomic.spinLoopHint();
    try std.testing.expect(!failed.load(.acquire));
    cycle_epoch.store(try collector.beginMark(), .release);
    start.store(true, .release);
    while (completed.load(.acquire) != worker_count) std.atomic.spinLoopHint();
    try std.testing.expect(!failed.load(.acquire));
    try std.testing.expectEqual(@as(usize, worker_count), try collector.drainSatb(worker_count * 2));
    try std.testing.expectEqual(@as(usize, worker_count), try collector.traceWork(worker_count * 2));
    try collector.tryFinishMark();
    release.store(true, .release);
    for (threads) |thread| thread.join();
    try std.testing.expect(!failed.load(.acquire));
    for (objects) |object| try std.testing.expect(try collector.isMarked(object));
}

test "blocked thread is passively rooted and released without a global pause" {
    var storage: [512]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 2, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 64);
    defer heap.deinit();
    var collector = try ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 2,
        .max_satb_buffers = 1,
        .card_bytes = 64,
    });
    defer collector.deinit() catch unreachable;
    var registry = try thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try thread_registry.ThreadContext.init(std.testing.allocator, 1);
    defer context.deinit();
    var buffer = try SatbBuffer.init(std.testing.allocator, 2);
    defer buffer.deinit() catch unreachable;
    var tlab = heap.threadAllocator();
    var root = try publishTestObject(&heap, &handles, &tlab, 8);
    try context.addRoot(&root);
    try collector.registerThreadSatbBuffer(&buffer, &context);
    defer collector.unregisterSatbBuffer(&buffer) catch unreachable;
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    try collector.enterBlockedForMark(&registry, &context, &buffer);

    _ = try collector.beginMark();
    try std.testing.expectError(error.MarkHandshakeIncomplete, collector.tryFinishMark());
    var handshake = try collector.beginThreadHandshake(&registry);
    try waitMarkHandshake(&handshake);
    try std.testing.expectEqual(@as(usize, 1), try collector.traceWork(2));
    try collector.tryFinishMark();
    try registry.leaveBlocked(&context);
    try std.testing.expect(try collector.isMarked(root));
    try std.testing.expectEqual(@as(u64, 1), collector.stats().thread_handshakes);
}

test "final asynchronous handshake flushes racing SATB state" {
    var storage: [1024]u8 align(runtime_value.object_alignment) = @splat(0);
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(std.testing.allocator, 4, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(std.testing.allocator, &handles, 128);
    defer heap.deinit();
    var collector = try ConcurrentCollector.init(std.testing.allocator, &heap, &handles, .{
        .satb_queue_capacity = 4,
        .max_satb_buffers = 1,
        .card_bytes = 64,
    });
    defer collector.deinit() catch unreachable;
    var registry = try thread_registry.Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var tlab = heap.threadAllocator();
    const root = try publishTestObject(&heap, &handles, &tlab, 8);
    const overwritten = try publishTestObject(&heap, &handles, &tlab, 8);
    var command = std.atomic.Value(u8).init(0);
    var stage = std.atomic.Value(u8).init(0);
    var failed = std.atomic.Value(bool).init(false);

    const Worker = struct {
        collector: *ConcurrentCollector,
        registry: *thread_registry.Registry,
        root: Handle,
        overwritten: Handle,
        command: *std.atomic.Value(u8),
        stage: *std.atomic.Value(u8),
        failed: *std.atomic.Value(bool),

        fn run(self: *@This()) void {
            var context = thread_registry.ThreadContext.init(std.testing.allocator, 1) catch {
                self.failed.store(true, .release);
                return;
            };
            defer context.deinit();
            var buffer = SatbBuffer.init(std.testing.allocator, 2) catch {
                self.failed.store(true, .release);
                return;
            };
            defer buffer.deinit() catch self.failed.store(true, .release);
            var root_slot = self.root;
            context.addRoot(&root_slot) catch {
                self.failed.store(true, .release);
                return;
            };
            self.collector.registerThreadSatbBuffer(&buffer, &context) catch {
                self.failed.store(true, .release);
                return;
            };
            defer self.collector.unregisterSatbBuffer(&buffer) catch self.failed.store(true, .release);
            self.registry.register(&context) catch {
                self.failed.store(true, .release);
                return;
            };
            defer self.registry.unregister(&context) catch self.failed.store(true, .release);
            self.stage.store(1, .release);

            while (self.command.load(.acquire) < 1) std.atomic.spinLoopHint();
            _ = self.collector.pollThreadHandshake(self.registry, &context, &buffer) catch {
                self.failed.store(true, .release);
                return;
            };
            self.stage.store(2, .release);

            while (self.command.load(.acquire) < 2) std.atomic.spinLoopHint();
            buffer.record(self.overwritten) catch {
                self.failed.store(true, .release);
                return;
            };
            self.stage.store(3, .release);

            while (self.command.load(.acquire) < 3) std.atomic.spinLoopHint();
            _ = self.collector.pollThreadHandshake(self.registry, &context, &buffer) catch {
                self.failed.store(true, .release);
                return;
            };
            self.stage.store(4, .release);
            while (self.command.load(.acquire) < 4) std.atomic.spinLoopHint();
        }
    };
    var worker = Worker{
        .collector = &collector,
        .registry = &registry,
        .root = root,
        .overwritten = overwritten,
        .command = &command,
        .stage = &stage,
        .failed = &failed,
    };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    while (stage.load(.acquire) < 1) std.atomic.spinLoopHint();
    try std.testing.expect(!failed.load(.acquire));

    _ = try collector.beginMark();
    var initial = try collector.beginThreadHandshake(&registry);
    command.store(1, .release);
    try waitMarkHandshake(&initial);
    while (stage.load(.acquire) < 2) std.atomic.spinLoopHint();
    try std.testing.expectEqual(@as(usize, 1), try collector.traceWork(2));

    command.store(2, .release);
    while (stage.load(.acquire) < 3) std.atomic.spinLoopHint();
    try std.testing.expectError(error.BuffersNotFlushed, collector.tryFinishMark());
    var final_handshake = try collector.beginThreadHandshake(&registry);
    command.store(3, .release);
    try waitMarkHandshake(&final_handshake);
    while (stage.load(.acquire) < 4) std.atomic.spinLoopHint();
    try std.testing.expectEqual(@as(usize, 1), try collector.drainSatb(2));
    try std.testing.expectEqual(@as(usize, 1), try collector.traceWork(2));
    try collector.tryFinishMark();
    command.store(4, .release);
    thread.join();
    try std.testing.expect(!failed.load(.acquire));
    try std.testing.expect(try collector.isMarked(root));
    try std.testing.expect(try collector.isMarked(overwritten));
    try std.testing.expectEqual(@as(u64, 2), collector.stats().thread_handshakes);
}

fn allocationFailureProbe(allocator: std.mem.Allocator) !void {
    var storage: [256]u8 align(runtime_value.object_alignment) = undefined;
    const regions = [_]runtime_value.Region{try runtime_value.Region.fromSlice(&storage)};
    var handles = try HandleTable.init(allocator, 2, &regions);
    defer handles.deinit();
    var heap = try ManagedHeap.init(allocator, &handles, 64);
    defer heap.deinit();
    const offsets = [_]u32{0};
    const layouts = [_]LayoutSpec{.{
        .id = 1,
        .minimum_size = 8,
        .reference_offsets = &offsets,
    }};
    var static_root = std.atomic.Value(u64).init(@bitCast(Handle.none));
    const static_roots = [_]usize{@intFromPtr(&static_root)};
    var collector = try ConcurrentCollector.init(allocator, &heap, &handles, .{
        .satb_queue_capacity = 2,
        .max_satb_buffers = 2,
        .card_bytes = 64,
        .layouts = &layouts,
        .static_root_slots = &static_roots,
    });
    defer collector.deinit() catch unreachable;
    var buffer = try SatbBuffer.init(allocator, 2);
    defer buffer.deinit() catch unreachable;
}

test "collector metadata is leak free at every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureProbe, .{});
}

test "layout registry rejects ambiguous or unaligned metadata" {
    const duplicate = [_]LayoutSpec{
        .{ .id = 2, .minimum_size = 8 },
        .{ .id = 2, .minimum_size = 8 },
    };
    const bad_offsets = [_]u32{4};
    const unaligned = [_]LayoutSpec{.{
        .id = 3,
        .minimum_size = 8,
        .reference_offsets = &bad_offsets,
    }};
    try std.testing.expectError(error.InvalidLayout, validateLayoutSpecs(&duplicate));
    try std.testing.expectError(error.InvalidLayout, validateLayoutSpecs(&unaligned));
    try std.testing.expectError(
        error.InvalidLayout,
        validateLayoutSpecs(&.{.{ .id = 0, .minimum_size = 8 }}),
    );
}
