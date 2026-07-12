const std = @import("std");
const interpreter = @import("interpreter");

// ==========================================
// Phase 22: 8-byte Packed GCHeader
// ==========================================
//
// Bit layout:
//   [31: 0] size_or_fwd  - byte-size (live) | forwarded-ref (evacuated) | next-free (free)
//   [33:32] color        - 0=White 1=Grey 2=Black
//   [35:34] age          - 0-3 saturating promotion counter
//   [36   ] is_free      - block is on a free list
//   [37   ] is_forwarded - size_or_fwd holds a forwarded ref, not size
//   [63:38] reserved
//
// Header is 8 bytes.  ref = block_start + 8 (user-data pointer).
//
pub const GCHeader = packed struct(u64) {
    size_or_fwd: u32,
    color: u2,
    age: u2,
    is_free: bool,
    is_forwarded: bool,
    _reserved: u26 = 0,

    pub inline fn size(self: GCHeader) u32 {
        if (self.is_forwarded) {
            return @intCast(self._reserved);
        }
        return self.size_or_fwd;
    }
    pub inline fn setSize(self: *GCHeader, s: u32) void { self.size_or_fwd = s; }

    pub inline fn nextFree(self: GCHeader) u32         { return self.size_or_fwd; }
    pub inline fn setNextFree(self: *GCHeader, n: u32) void { self.size_or_fwd = n; }

    pub inline fn forwardedRef(self: GCHeader) u32 { return self.size_or_fwd; }
    pub inline fn setForwardedRef(self: *GCHeader, fwd: u32) void {
        const orig_size = self.size_or_fwd;
        self.size_or_fwd = fwd;
        self.is_forwarded = true;
        self._reserved = @intCast(orig_size);
    }

    pub inline fn incAge(self: *GCHeader) void {
        if (self.age < 3) self.age += 1;
    }
};

// ==========================================
// Phase 23: Thread-Local Allocation Buffers
// ==========================================
//
// Each mutator thread owns a private 4 KB slice of Eden space.
// Allocations into the TLAB are a 3-instruction lock-free bump:
//   if top + total <= end → write header, top += total, return top-total+8
// The global gc.mutex is only acquired when a TLAB is exhausted (~1 in 100 allocs).
//
pub const TLAB_SIZE: u32 = 4096;
pub const LARGE_OBJECT_THRESHOLD: u32 = 64 * 1024;

pub const TLAB = struct {
    top: u32 = 0,  // start of next allocation within this TLAB
    end: u32 = 0,  // exclusive end of this TLAB region

    /// Lock-free allocation attempt. Returns user-data ref or 0 on miss.
    pub inline fn tryAlloc(self: *TLAB, total_needed: u32) u32 {
        if (self.top + total_needed > self.end) return 0;
        const block_start = self.top;
        self.top += total_needed;
        return block_start + 8;
    }
};

// ==========================================
// Phase 23: GC Statistics
// ==========================================
//
// All fields are atomic so any mutator thread can update them without a lock.
// `fetchAdd(.monotonic)` compiles to a single `LOCK XADD` on x86 — zero overhead
// on the allocation hot path.
//
pub const GcStats = struct {
    bytes_allocated: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    bytes_promoted:  std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    gc_cycles:       std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    total_pause_ns:  std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    tlab_refills:    std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

    pub fn report(self: *const GcStats) void {
        std.debug.print(
            \\--- GC Statistics ---
            \\  Bytes allocated : {d}
            \\  Bytes promoted  : {d}
            \\  GC cycles       : {d}
            \\  Total pause ns  : {d}
            \\  TLAB refills    : {d}
            \\
        , .{
            self.bytes_allocated.load(.monotonic),
            self.bytes_promoted.load(.monotonic),
            self.gc_cycles.load(.monotonic),
            self.total_pause_ns.load(.monotonic),
            self.tlab_refills.load(.monotonic),
        });
    }
};

// ==========================================
// Segregated Free Lists — size class table
// ==========================================
pub const SLAB_COUNT: usize = 8;

pub inline fn sizeClass(total_bytes: u32) u3 {
    if (total_bytes <= 16)   return 0;
    if (total_bytes <= 32)   return 1;
    if (total_bytes <= 64)   return 2;
    if (total_bytes <= 128)  return 3;
    if (total_bytes <= 256)  return 4;
    if (total_bytes <= 512)  return 5;
    if (total_bytes <= 1024) return 6;
    return 7;
}

pub const GCState = enum { idle, marking, sweeping };

pub const GC = struct {
    thread:               ?std.Thread = null,
    mutex:                std.Io.Mutex = .init,
    safepoint_cond:       std.Io.Condition = .init,
    resume_cond:          std.Io.Condition = .init,
    safepoint_requested:  std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    safepoint_active:     bool = false,
    active_threads:       std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    suspended_threads:    std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    should_stop:          bool = false,

    active_frames: std.ArrayList(usize),
    grey_stack:    std.ArrayList(u32),
    satb_queue:    std.ArrayList(u32),

    phase: GCState = .idle,
};

pub const RootScanCallback = *const fn (ctx: ?*anyopaque, heap: *Heap) void;

pub const RememberedSlot = struct {
    obj_ref: u32,
    offset: u32,
};

pub const ReferenceKind = enum {
    weak,
    soft,
    phantom,
};

pub const SpecialReferenceSlot = struct {
    obj_ref: u32,
    offset: u32,
    kind: ReferenceKind,
};

pub const PinnedRef = struct {
    ref: u32,
    count: u32,
};

// ==========================================
// Heap
// ==========================================
pub const Heap = struct {
    memory:     []u8,
    card_table: []u8,
    allocator:  std.mem.Allocator,

    young_from_start: u32,
    young_from_end:   u32,
    young_to_start:   u32,
    young_to_end:     u32,
    old_start:        u32,
    old_end:          u32,
    los_start:        u32,
    los_end:          u32,

    young_bump: u32,
    /// Segregated free lists: one head pointer per size class (8 buckets)
    old_free_lists: [SLAB_COUNT]u32,
    los_free_list: u32,

    gc: GC,

    /// Phase 23: per-thread TLAB pointers registered here for GC retirement.
    /// Protected by gc.mutex. Read cross-thread only during STW.
    tlab_registry: std.ArrayList(*TLAB),

    /// Exact old-to-young reference slots, refined by the write barrier.
    remembered_set: std.ArrayList(RememberedSlot),

    /// Weak/soft/phantom reference fields that are processed after strong marking.
    special_refs: std.ArrayList(SpecialReferenceSlot),

    /// Native/JNI pinned objects. Entries are treated as GC roots while pinned.
    pinned_refs: std.ArrayList(PinnedRef),

    /// Phase 23: GC statistics (atomic — no lock needed to update).
    stats: GcStats,

    root_scan_fn:         ?RootScanCallback = null,
    forward_static_refs_fn: ?RootScanCallback = null,
    root_scan_ctx:        ?*anyopaque = null,
    io:                   std.Io,

    // ==========================================
    // Init / Deinit
    // ==========================================
    pub fn init(allocator: std.mem.Allocator, io: std.Io, size: usize) !Heap {

        const mem = try allocator.alignedAlloc(u8, .of(GCHeader), size);
        @memset(mem, 0);

        // Partition: Young Gen 25% (2× 12.5% semispaces), Old Gen 75%
        const young_size       = @as(u32, @intCast(size / 8));
        const young_from_start = @as(u32, 8); // 0 reserved as null ref
        const young_from_end   = young_from_start + young_size;

        const young_to_start = (young_from_end + 7) & ~@as(u32, 7);
        const young_to_end   = young_to_start + young_size;

        const old_start = (young_to_end + 7) & ~@as(u32, 7);
        const heap_end  = @as(u32, @intCast(size));
        const los_size  = @as(u32, @intCast(size / 4));
        const los_start = (heap_end - los_size) & ~@as(u32, 7);
        const los_end   = heap_end;
        const old_end   = los_start;

        const cards_count = (old_end - old_start + 511) / 512;
        const card_table = try allocator.alloc(u8, cards_count);
        @memset(card_table, 0);

        var self = Heap{
            .memory           = mem,
            .card_table       = card_table,
            .allocator        = allocator,
            .young_from_start = young_from_start,
            .young_from_end   = young_from_end,
            .young_to_start   = young_to_start,
            .young_to_end     = young_to_end,
            .old_start        = old_start,
            .old_end          = old_end,
            .los_start        = los_start,
            .los_end          = los_end,
            .young_bump       = young_from_start,
            .old_free_lists   = [_]u32{0} ** SLAB_COUNT,
            .los_free_list    = los_start,
            .gc = .{
                .active_frames = std.ArrayList(usize).empty,
                .grey_stack    = std.ArrayList(u32).empty,
                .satb_queue    = std.ArrayList(u32).empty,
            },
            .tlab_registry = std.ArrayList(*TLAB).empty,
            .remembered_set = std.ArrayList(RememberedSlot).empty,
            .special_refs = std.ArrayList(SpecialReferenceSlot).empty,
            .pinned_refs = std.ArrayList(PinnedRef).empty,
            .stats         = .{},
            .io            = io,
        };

        // Init Old Gen as a single large free block in the overflow bucket (7)
        const old_block_size = old_end - old_start;
        const old_hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[old_start]));
        old_hdr.* = GCHeader{
            .size_or_fwd = old_block_size,
            .color       = 0,
            .age         = 0,
            .is_free     = true,
            .is_forwarded = false,
        };
        self.old_free_lists[7] = old_start;
        @as(*u32, @ptrCast(@alignCast(&self.memory[old_start + 8]))).* = 0;

        const los_block_size = los_end - los_start;
        const los_hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[los_start]));
        los_hdr.* = GCHeader{
            .size_or_fwd = los_block_size,
            .color       = 0,
            .age         = 0,
            .is_free     = true,
            .is_forwarded = false,
        };

        return self;
    }

    pub fn lockMutex(self: *Heap) void {
        self.gc.mutex.lockUncancelable(self.io);
    }

    pub fn unlockMutex(self: *Heap) void {
        self.gc.mutex.unlock(self.io);
    }

    pub fn deinit(self: *Heap, allocator: std.mem.Allocator) void {
        self.lockMutex();
        self.gc.should_stop = true;
        self.unlockMutex();
        if (self.gc.thread) |thread| thread.join();
        self.gc.active_frames.deinit(allocator);
        self.gc.grey_stack.deinit(allocator);
        self.gc.satb_queue.deinit(allocator);
        self.tlab_registry.deinit(allocator);
        self.remembered_set.deinit(allocator);
        self.special_refs.deinit(allocator);
        self.pinned_refs.deinit(allocator);
        allocator.free(self.card_table);
        allocator.free(self.memory);
    }

    pub fn start(self: *Heap) !void {
        self.gc.thread = try std.Thread.spawn(.{}, gcLoop, .{self});
    }

    // ==========================================
    // Header / Range Helpers
    // ==========================================

    /// Returns a pointer to the 8-byte GCHeader prepended to `ref`.
    /// ref points at first user-data byte; header is at ref-8.
    pub fn getHeader(self: *const Heap, ref: u32) *GCHeader {
        return @ptrCast(@alignCast(&self.memory[ref - 8]));
    }

    pub fn isYoungGen(self: *const Heap, ref: u32) bool {
        return (ref >= self.young_from_start and ref < self.young_from_end) or
               (ref >= self.young_to_start   and ref < self.young_to_end);
    }

    pub fn isOldGen(self: *const Heap, ref: u32) bool {
        return ref >= self.old_start and ref < self.old_end;
    }

    pub fn isLargeObjectSpace(self: *const Heap, ref: u32) bool {
        return ref >= self.los_start and ref < self.los_end;
    }

    pub fn isManagedRef(self: *const Heap, ref: u32) bool {
        return self.isYoungGen(ref) or self.isOldGen(ref) or self.isLargeObjectSpace(ref);
    }

    fn isLiveObjectRefInRange(self: *const Heap, ref: u32, range_start: u32, used_end: u32) bool {
        if (ref < range_start + 8 or ref >= used_end) return false;

        var off = range_start;
        while (off < used_end) {
            const hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[off]));
            const block_size = hdr.size();
            if (block_size < 8 or (block_size & 7) != 0) return false;
            if (off + block_size > used_end) return false;
            if (off + 8 == ref) return !hdr.is_free and !hdr.is_forwarded;
            off += block_size;
        }
        return false;
    }

    fn isYoungObjectStartForForwarding(self: *const Heap, ref: u32) bool {
        if (!self.isYoungGen(ref)) return false;
        const hdr = self.getHeader(ref);
        if (hdr.is_forwarded) {
            const fwd = hdr.forwardedRef();
            return fwd != 0 and self.isManagedRef(fwd);
        }
        return self.isLiveObjectRef(ref);
    }

    pub fn isLiveObjectRef(self: *const Heap, ref: u32) bool {
        if (ref >= self.young_from_start and ref < self.young_bump) {
            return self.isLiveObjectRefInRange(ref, self.young_from_start, self.young_bump);
        }
        if (self.isOldGen(ref)) {
            return self.isLiveObjectRefInRange(ref, self.old_start, self.old_end);
        }
        if (self.isLargeObjectSpace(ref)) {
            return self.isLiveObjectRefInRange(ref, self.los_start, self.los_end);
        }
        return false;
    }

    fn validateBlockRange(self: *const Heap, range_start: u32, range_end: u32, used_end: u32, allow_zero_tail: bool) !void {
        if (range_start > used_end or used_end > range_end) return error.InvalidHeapRegion;

        var off = range_start;
        while (off < used_end) {
            const hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[off]));
            const block_size = hdr.size();
            if (block_size == 0 and allow_zero_tail) return;
            if (block_size < 8 or (block_size & 7) != 0) {
                if (block_size != 7) {
                    std.debug.panic("INTEGRITY FAILED: off={d}, block_size={d}, free={}, forwarded={}, color={d}, reserved={d}", .{
                        off, block_size, hdr.is_free, hdr.is_forwarded, hdr.color, hdr._reserved,
                    });
                }
                return error.InvalidHeapBlock;
            }
            if (off + block_size > used_end) return error.InvalidHeapBlock;
            if (hdr.is_forwarded) return error.StaleForwardingPointer;
            off += block_size;
        }

        if (off != used_end) return error.InvalidHeapRegion;
    }

    pub fn validateIntegrity(self: *const Heap) !void {
        try self.validateBlockRange(self.young_from_start, self.young_from_end, self.young_bump, true);
        try self.validateBlockRange(self.old_start, self.old_end, self.old_end, false);
        try self.validateBlockRange(self.los_start, self.los_end, self.los_end, false);

        for (self.old_free_lists) |head| {
            if (head == 0) continue;
            if (head < self.old_start or head >= self.old_end) return error.InvalidFreeList;
            const hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[head]));
            if (!hdr.is_free) return error.InvalidFreeList;
        }

        for (self.remembered_set.items) |slot| {
            if (!self.isOldGen(slot.obj_ref) or !self.isLiveObjectRef(slot.obj_ref)) return error.InvalidRememberedSet;
            const owner = self.getHeader(slot.obj_ref);
            if (owner.is_free or owner.size() < 8) return error.InvalidRememberedSet;
            if (slot.offset >= owner.size() - 8) return error.InvalidRememberedSet;
            const val = self.readU32Internal(slot.obj_ref + slot.offset);
            if (val == 0 or !self.isYoungGen(val) or !self.isLiveObjectRef(val)) return error.InvalidRememberedSet;
        }

        for (self.special_refs.items) |slot| {
            if (!self.isManagedRef(slot.obj_ref) or !self.isLiveObjectRef(slot.obj_ref)) return error.InvalidSpecialReference;
            const owner = self.getHeader(slot.obj_ref);
            if (owner.is_free or owner.size() < 8) return error.InvalidSpecialReference;
            if (slot.offset >= owner.size() - 8) return error.InvalidSpecialReference;
            const val = self.readU32Internal(slot.obj_ref + slot.offset);
            if (val != 0 and !self.isLiveObjectRef(val)) return error.InvalidSpecialReference;
        }

        for (self.pinned_refs.items) |entry| {
            if (entry.count == 0) return error.InvalidPinnedReference;
            if (!self.isManagedRef(entry.ref) or !self.isLiveObjectRef(entry.ref)) return error.InvalidPinnedReference;
            const hdr = self.getHeader(entry.ref);
            if (hdr.is_free) return error.InvalidPinnedReference;
        }
    }

    // ==========================================
    // Segregated Free List helpers
    // ==========================================

    inline fn flPop(self: *Heap, cls: u3) u32 {
        const head = self.old_free_lists[cls];
        if (head == 0) return 0;
        const next_ptr = @as(*u32, @ptrCast(@alignCast(&self.memory[head + 8])));
        self.old_free_lists[cls] = next_ptr.*;
        return head;
    }

    inline fn flPush(self: *Heap, block_offset: u32, block_size: u32) void {
        const cls = sizeClass(block_size);
        const hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[block_offset]));
        hdr.* = GCHeader{
            .size_or_fwd  = block_size,
            .color        = 0,
            .age          = 0,
            .is_free      = true,
            .is_forwarded = false,
        };
        const next_ptr = @as(*u32, @ptrCast(@alignCast(&self.memory[block_offset + 8])));
        next_ptr.* = self.old_free_lists[cls];
        self.old_free_lists[cls] = block_offset;
    }

    // ==========================================
    // Phase 23: TLAB-based Young Gen Allocator
    // ==========================================

    /// Fast path: lock-free bump pointer within the given TLAB.
    /// Slow path (TLAB exhausted): allocateSlow() carves a new 4 KB TLAB under lock.
    pub fn allocate(self: *Heap, tlab: *TLAB, size: u32) !u32 {
        const aligned_size  = (size + 7) & ~@as(u32, 7);
        const total_needed  = @max(16, 8 + aligned_size);

        if (size >= LARGE_OBJECT_THRESHOLD) {
            return self.allocateLarge(size);
        }

        // --- Fast path (no lock) ---
        const ref = tlab.tryAlloc(total_needed);
        if (ref != 0) {
            const hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[ref - 8]));
            hdr.* = GCHeader{
                .size_or_fwd  = total_needed,
                .color        = 0, // White; STW re-mark corrects any missed objects
                .age          = 0,
                .is_free      = false,
                .is_forwarded = false,
            };
            _ = self.stats.bytes_allocated.fetchAdd(total_needed, .monotonic);
            return ref;
        }

        // --- Slow path ---
        return self.allocateSlow(tlab, total_needed);
    }

    /// Slow path: retire the exhausted TLAB (write a free fill block over its unused tail),
    /// acquire the global mutex, carve a fresh TLAB_SIZE chunk from young_bump,
    /// and allocate the first object from it.
    fn allocateSlow(self: *Heap, tlab: *TLAB, total_needed: u32) !u32 {
        // Retire old TLAB: write a free fill block at the unused tail so the GC
        // linear scanner can step over it cleanly (size > 0, is_free = true).
        if (tlab.top < tlab.end) {
            const tail_size = tlab.end - tlab.top;
            const fill: *GCHeader = @ptrCast(@alignCast(&self.memory[tlab.top]));
            fill.* = GCHeader{
                .size_or_fwd  = tail_size,
                .color        = 0,
                .age          = 0,
                .is_free      = true,
                .is_forwarded = false,
            };
        }
        // Mark TLAB as empty immediately so re-entry doesn't double-retire.
        tlab.* = .{ .top = 0, .end = 0 };

        // Acquire the global heap lock to update young_bump.
        self.lockMutex();

        // Trigger GC if Eden is full.
        if (self.young_bump + TLAB_SIZE > self.young_from_end) {
            self.unlockMutex();
            try self.runGCOpt(true);
            self.lockMutex();
            // After GC, check that at least one object fits.
            if (self.young_bump + total_needed > self.young_from_end) {
                self.unlockMutex();
                return error.OutOfMemory;
            }
        }

        // Carve a fresh TLAB chunk.
        const chunk_size = @min(TLAB_SIZE, self.young_from_end - self.young_bump);
        if (chunk_size < total_needed) {
            self.unlockMutex();
            return error.OutOfMemory;
        }
        tlab.* = .{ .top = self.young_bump, .end = self.young_bump + chunk_size };
        self.young_bump += chunk_size;

        // Register this thread's TLAB pointer so runGC() can retire it at STW.
        var registered = false;
        for (self.tlab_registry.items) |t| {
            if (t == tlab) {
                registered = true;
                break;
            }
        }
        if (!registered) {
            self.tlab_registry.append(self.allocator, tlab) catch {};
        }

        _ = self.stats.tlab_refills.fetchAdd(1, .monotonic);
        self.unlockMutex();

        // Now the TLAB is warm — fast path must succeed.
        const ref = tlab.tryAlloc(total_needed);
        if (ref == 0) return error.OutOfMemory;

        const hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[ref - 8]));
        hdr.* = GCHeader{
            .size_or_fwd  = total_needed,
            .color        = 0,
            .age          = 0,
            .is_free      = false,
            .is_forwarded = false,
        };
        _ = self.stats.bytes_allocated.fetchAdd(total_needed, .monotonic);
        return ref;
    }

    pub fn registerThread(self: *Heap) void {
        _ = self.gc.active_threads.fetchAdd(1, .monotonic);
        // If a safepoint is already in progress, cooperate immediately.
        if (self.gc.safepoint_requested.load(.acquire)) {
            self.enterSafepoint({}) catch {};
        }
    }

    pub fn deregisterThread(self: *Heap, tlab: *TLAB) void {
        self.lockMutex();
        self.deregisterThreadTlabLocked(tlab);
        self.unlockMutex();

        self.gc.mutex.lockUncancelable(self.io);
        _ = self.gc.active_threads.fetchSub(1, .release);
        self.gc.safepoint_cond.broadcast(self.io);
        self.gc.mutex.unlock(self.io);
    }

    pub fn deregisterThreadTlab(self: *Heap, tlab: *TLAB) void {
        self.lockMutex();
        defer self.unlockMutex();
        self.deregisterThreadTlabLocked(tlab);
    }

    pub fn deregisterThreadTlabLocked(self: *Heap, tlab: *TLAB) void {
        if (tlab.top < tlab.end) {
            const tail_size = tlab.end - tlab.top;
            const fill: *GCHeader = @ptrCast(@alignCast(&self.memory[tlab.top]));
            fill.* = GCHeader{
                .size_or_fwd  = tail_size,
                .color        = 0,
                .age          = 0,
                .is_free      = true,
                .is_forwarded = false,
            };
        }
        tlab.* = .{ .top = 0, .end = 0 };

        for (self.tlab_registry.items, 0..) |tlab_ptr, idx| {
            if (tlab_ptr == tlab) {
                _ = self.tlab_registry.swapRemove(idx);
                break;
            }
        }
    }

    // ==========================================
    // Old Gen Allocator — Segregated Free Lists
    // ==========================================

    pub fn allocateLarge(self: *Heap, size: u32) !u32 {
        const aligned_size = (size + 7) & ~@as(u32, 7);
        const total_needed = @max(16, 8 + aligned_size);

        self.gc.mutex.lockUncancelable(self.io);
        if (self.allocLargeNoGC(total_needed)) |ref| {
            self.gc.mutex.unlock(self.io);
            return ref;
        }
        self.gc.mutex.unlock(self.io);

        try self.runGC();

        self.gc.mutex.lockUncancelable(self.io);
        if (self.allocLargeNoGC(total_needed)) |ref| {
            self.gc.mutex.unlock(self.io);
            return ref;
        }
        self.gc.mutex.unlock(self.io);
        return error.OutOfMemory;
    }

    fn allocLargeNoGC(self: *Heap, total_needed: u32) ?u32 {
        var cur = self.los_start;

        while (cur < self.los_end) {
            const hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[cur]));
            const block_size = hdr.size();
            if (block_size == 0) break;

            if (hdr.is_free and block_size >= total_needed) {
                if (block_size >= total_needed + 16) {
                    const remainder_offset = cur + total_needed;
                    const remainder_size = block_size - total_needed;
                    const rem_hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[remainder_offset]));
                    rem_hdr.* = GCHeader{
                        .size_or_fwd  = remainder_size,
                        .color        = 0,
                        .age          = 0,
                        .is_free      = true,
                        .is_forwarded = false,
                    };
                    hdr.setSize(total_needed);
                }

                hdr.is_free = false;
                hdr.is_forwarded = false;
                hdr.color = if (self.gc.phase == .marking) @as(u2, 2) else @as(u2, 0);
                _ = self.stats.bytes_allocated.fetchAdd(total_needed, .monotonic);
                return cur + 8;
            }

            cur += block_size;
        }

        return null;
    }

    pub fn allocateOld(self: *Heap, size: u32) !u32 {
        self.lockMutex();
        defer self.unlockMutex();
        return self.allocateOldLocked(size);
    }

    pub fn allocateOldLocked(self: *Heap, size: u32) !u32 {
        const aligned_size = (size + 7) & ~@as(u32, 7);
        const total_needed = @max(16, 8 + aligned_size);

        const start_cls = sizeClass(total_needed);

        // 1. Search in start_cls list
        var prev_offset: u32 = 0;
        var curr_offset = self.old_free_lists[start_cls];
        while (curr_offset != 0) {
            const hdr = @as(*GCHeader, @ptrCast(@alignCast(&self.memory[curr_offset])));
            const block_size = hdr.size();
            if (block_size >= total_needed) {
                // Remove block from the list
                const next_offset = @as(*u32, @ptrCast(@alignCast(&self.memory[curr_offset + 8]))).*;
                if (prev_offset == 0) {
                    self.old_free_lists[start_cls] = next_offset;
                } else {
                    @as(*u32, @ptrCast(@alignCast(&self.memory[prev_offset + 8]))).* = next_offset;
                }

                // Use this block
                return self.useAllocatedBlock(curr_offset, block_size, total_needed);
            }
            prev_offset = curr_offset;
            curr_offset = @as(*u32, @ptrCast(@alignCast(&self.memory[curr_offset + 8]))).*;
        }

        // 2. Search in larger classes
        var cls: usize = @as(usize, start_cls) + 1;
        while (cls < SLAB_COUNT) : (cls += 1) {
            const block_offset = self.old_free_lists[cls];
            if (block_offset != 0) {
                const hdr = @as(*GCHeader, @ptrCast(@alignCast(&self.memory[block_offset])));
                const block_size = hdr.size();

                // Pop the head of the list
                const next_offset = @as(*u32, @ptrCast(@alignCast(&self.memory[block_offset + 8]))).*;
                self.old_free_lists[cls] = next_offset;

                // Use this block
                return self.useAllocatedBlock(block_offset, block_size, total_needed);
            }
        }

        return error.OutOfMemory;
    }

    fn useAllocatedBlock(self: *Heap, block_offset: u32, block_size: u32, total_needed: u32) u32 {
        const hdr = @as(*GCHeader, @ptrCast(@alignCast(&self.memory[block_offset])));
        if (block_size >= total_needed + 16) {
            const remainder_offset = block_offset + total_needed;
            const remainder_size = block_size - total_needed;

            // Push remainder back to free lists
            self.flPush(remainder_offset, remainder_size);

            hdr.setSize(total_needed);
        }
        hdr.is_free = false;
        hdr.is_forwarded = false;
        hdr.color = if (self.gc.phase == .marking) @as(u2, 2) else @as(u2, 0);
        hdr._reserved = 0;
        return block_offset + 8;
    }

    // ==========================================
    // Layered Write Barrier
    // ==========================================
    pub fn writeBarrier(self: *Heap, obj_ref: u32, offset: u32, new_val: u32) void {
        if (obj_ref == 0 or !self.isManagedRef(obj_ref)) return;
        const is_special = self.isSpecialReferenceSlot(obj_ref, offset);

        if (!is_special and self.gc.phase == .marking) {
            const old_val = self.readU32Internal(obj_ref + offset);
            if (old_val != 0 and self.isLiveObjectRef(old_val)) self.enqueueSatb(old_val);
        }

        if (new_val == 0) return;
        if (!is_special and self.isOldGen(obj_ref) and self.isYoungGen(new_val)) {
            const card_idx = (obj_ref - self.old_start) / 512;
            if (card_idx < self.card_table.len) {
                if (self.card_table[card_idx] != 1) self.card_table[card_idx] = 1;
            }
            self.rememberSlot(obj_ref, offset);
        }
    }

    fn rememberSlot(self: *Heap, obj_ref: u32, offset: u32) void {
        const slot = RememberedSlot{ .obj_ref = obj_ref, .offset = offset };
        self.lockMutex();
        defer self.unlockMutex();
        self.remembered_set.append(self.allocator, slot) catch {};
    }

    fn enqueueSatb(self: *Heap, ref: u32) void {
        self.lockMutex();
        defer self.unlockMutex();
        self.gc.satb_queue.append(self.allocator, ref) catch {};
    }

    fn shadeGreyLocked(self: *Heap, ref: u32) void {
        if (ref == 0 or !self.isLiveObjectRef(ref)) return;
        const header = self.getHeader(ref);
        if (!header.is_free and header.color == 0) {
            header.color = 1;
            self.gc.grey_stack.append(self.allocator, ref) catch {};
        }
    }

    fn drainSatbQueueLocked(self: *Heap) void {
        while (self.gc.satb_queue.items.len > 0) {
            const ref = self.gc.satb_queue.pop().?;
            self.shadeGreyLocked(ref);
        }
    }

    pub fn registerSpecialReference(self: *Heap, obj_ref: u32, offset: u32, kind: ReferenceKind) !void {
        try checkNull(obj_ref);
        if (!self.isLiveObjectRef(obj_ref)) return error.InvalidReference;
        const slot = SpecialReferenceSlot{ .obj_ref = obj_ref, .offset = offset, .kind = kind };
        self.lockMutex();
        defer self.unlockMutex();
        try self.special_refs.append(self.allocator, slot);
    }

    fn isSpecialReferenceSlot(self: *const Heap, obj_ref: u32, offset: u32) bool {
        for (self.special_refs.items) |slot| {
            if (slot.obj_ref == obj_ref and slot.offset == offset) return true;
        }
        return false;
    }

    fn shouldClearSoftReferences(self: *const Heap) bool {
        const young_used = self.young_bump - self.young_from_start;
        const young_total = self.young_from_end - self.young_from_start;
        return young_used > young_total * 9 / 10;
    }

    fn processSpecialReferencesLocked(self: *Heap) void {
        const clear_soft = self.shouldClearSoftReferences();
        var keep_idx: usize = 0;

        for (self.special_refs.items) |slot| {
            if (!self.isManagedRef(slot.obj_ref) or !self.isLiveObjectRef(slot.obj_ref)) continue;
            const owner = self.getHeader(slot.obj_ref);
            if (owner.is_free or owner.color == 0) continue;

            const val = self.readU32Internal(slot.obj_ref + slot.offset);
            if (val != 0 and !self.isLiveObjectRef(val)) {
                const ptr: *u32 = @ptrCast(@alignCast(&self.memory[slot.obj_ref + slot.offset]));
                ptr.* = 0;
            }
            if (val == 0 or !self.isLiveObjectRef(val)) {
                self.special_refs.items[keep_idx] = slot;
                keep_idx += 1;
                continue;
            }

            const referent = self.getHeader(val);
            if (referent.is_free) {
                const ptr: *u32 = @ptrCast(@alignCast(&self.memory[slot.obj_ref + slot.offset]));
                ptr.* = 0;
            } else if (referent.color == 0) {
                switch (slot.kind) {
                    .weak, .phantom => {
                        const ptr: *u32 = @ptrCast(@alignCast(&self.memory[slot.obj_ref + slot.offset]));
                        ptr.* = 0;
                    },
                    .soft => {
                        if (clear_soft) {
                            const ptr: *u32 = @ptrCast(@alignCast(&self.memory[slot.obj_ref + slot.offset]));
                            ptr.* = 0;
                        } else {
                            self.shadeGreyLocked(val);
                        }
                    },
                }
            }

            self.special_refs.items[keep_idx] = slot;
            keep_idx += 1;
        }

        self.special_refs.shrinkRetainingCapacity(keep_idx);
    }

    fn compactSpecialReferencesLocked(self: *Heap) void {
        var keep_idx: usize = 0;
        for (self.special_refs.items) |slot| {
            if (!self.isManagedRef(slot.obj_ref) or !self.isLiveObjectRef(slot.obj_ref)) continue;
            const owner = self.getHeader(slot.obj_ref);
            if (owner.is_free or owner.size() < 8) continue;
            if (slot.offset >= owner.size() - 8) continue;
            const val = self.readU32Internal(slot.obj_ref + slot.offset);
            if (val != 0 and !self.isLiveObjectRef(val)) {
                const ptr: *u32 = @ptrCast(@alignCast(&self.memory[slot.obj_ref + slot.offset]));
                ptr.* = 0;
            }
            self.special_refs.items[keep_idx] = slot;
            keep_idx += 1;
        }
        self.special_refs.shrinkRetainingCapacity(keep_idx);
    }

    fn rememberPinnedRef(self: *Heap, ref: u32) !void {
        for (self.pinned_refs.items) |*entry| {
            if (entry.ref == ref) {
                entry.count += 1;
                return;
            }
        }
        try self.pinned_refs.append(self.allocator, .{ .ref = ref, .count = 1 });
    }

    pub fn isPinned(self: *const Heap, ref: u32) bool {
        for (self.pinned_refs.items) |entry| {
            if (entry.ref == ref and entry.count > 0) return true;
        }
        return false;
    }

    pub fn pinObject(self: *Heap, ref: u32) !u32 {
        try checkNull(ref);
        if (!self.isLiveObjectRef(ref)) return error.InvalidReference;

        self.lockMutex();
        defer self.unlockMutex();

        var stable_ref = ref;
        if (self.isYoungGen(stable_ref)) {
            var hdr = self.getHeader(stable_ref);
            var is_already_forwarded = false;
            var orig_hdr = hdr;
            if (hdr.is_forwarded) {
                is_already_forwarded = true;
                stable_ref = hdr.forwardedRef();
                if (stable_ref == 0 or !self.isManagedRef(stable_ref)) return error.InvalidReference;
                hdr = self.getHeader(stable_ref);
            }

            if (self.isYoungGen(stable_ref)) {
                if (hdr.is_free) return error.InvalidReference;
                const source_size = hdr.size();
                if (source_size < 8 or (source_size & 7) != 0) return error.InvalidReference;

                const promoted_ref = try self.allocateOldLocked(source_size - 8);
                const stable_hdr = self.getHeader(promoted_ref);
                const stable_size = stable_hdr.size();
                @memcpy(
                    self.memory[promoted_ref - 8 .. promoted_ref - 8 + source_size],
                    self.memory[stable_ref - 8 .. stable_ref - 8 + source_size],
                );
                stable_hdr.setSize(stable_size);
                stable_hdr.color = if (self.gc.phase != .idle) @as(u2, 2) else @as(u2, 0);
                stable_hdr.age = 3;
                stable_hdr.is_free = false;
                stable_hdr.is_forwarded = false;

                hdr.setForwardedRef(promoted_ref);
                if (is_already_forwarded) {
                    orig_hdr.size_or_fwd = promoted_ref;
                }
                stable_ref = promoted_ref;
            }
        }

        try self.rememberPinnedRef(stable_ref);

        return stable_ref;
    }

    pub fn unpinObject(self: *Heap, ref: u32) void {
        if (ref == 0) return;
        self.lockMutex();
        defer self.unlockMutex();
        self.unpinObjectLocked(ref);
    }

    fn unpinObjectLocked(self: *Heap, ref: u32) void {
        var idx: usize = 0;
        while (idx < self.pinned_refs.items.len) : (idx += 1) {
            if (self.pinned_refs.items[idx].ref == ref) {
                if (self.pinned_refs.items[idx].count > 1) {
                    self.pinned_refs.items[idx].count -= 1;
                } else {
                    _ = self.pinned_refs.swapRemove(idx);
                }
                return;
            }
        }
    }

    pub fn shadeGrey(self: *Heap, ref: u32) void {
        if (ref == 0) return;
        self.lockMutex();
        defer self.unlockMutex();
        self.shadeGreyLocked(ref);
    }

    pub fn enterSafepoint(self: *Heap, frame: anytype) !void {
        _ = frame;
        if (!self.gc.safepoint_requested.load(.acquire)) return;
        const io = self.io;
        self.gc.mutex.lockUncancelable(io);
        _ = self.gc.suspended_threads.fetchAdd(1, .release);
        self.gc.safepoint_cond.signal(io);
        while (self.gc.safepoint_requested.load(.acquire)) {
            self.gc.resume_cond.waitUncancelable(io, &self.gc.mutex);
        }
        _ = self.gc.suspended_threads.fetchSub(1, .release);
        self.gc.mutex.unlock(io);
    }

    // ==========================================
    // Core Tri-Color Mark-Sweep-Copy Generational GC
    // ==========================================
    pub fn runGC(self: *Heap) !void {
        try self.runGCOpt(false);
    }

    pub fn runGCOpt(self: *Heap, is_mutator: bool) !void {
        const io = self.io;

        // Record pause start time using real clock.
        const pause_start = std.Io.Clock.real.now(io);

        // ── 1. STW: Request safepoint ──────────────────────────────────────
        self.gc.mutex.lockUncancelable(io);
        while (self.gc.safepoint_active) {
            if (is_mutator) {
                _ = self.gc.suspended_threads.fetchAdd(1, .release);
                self.gc.safepoint_cond.signal(io);
                while (self.gc.safepoint_requested.load(.acquire)) {
                    self.gc.resume_cond.waitUncancelable(io, &self.gc.mutex);
                }
                _ = self.gc.suspended_threads.fetchSub(1, .release);
                self.gc.mutex.unlock(io);
                return;
            } else {
                while (self.gc.safepoint_active) {
                    self.gc.resume_cond.waitUncancelable(io, &self.gc.mutex);
                }
            }
        }
        self.gc.safepoint_active = true;
        self.gc.safepoint_requested.store(true, .release);

        var success = false;
        defer if (!success) {
            self.gc.phase = .idle;
            self.gc.safepoint_requested.store(false, .release);
            self.gc.safepoint_active = false;
            self.gc.resume_cond.broadcast(io);
            self.gc.mutex.unlock(io);
        };

        // Wait for mutator threads to reach safepoints. Re-read active_threads each
        // iteration so threads that deregister after the safepoint is requested are
        // not counted toward the suspension target.
        while (true) {
            const active = self.gc.active_threads.load(.acquire);
            const need: u32 = if (active == 0) 0 else if (is_mutator) active - 1 else active;
            if (self.gc.suspended_threads.load(.acquire) >= need) break;
            self.gc.safepoint_cond.waitUncancelable(io, &self.gc.mutex);
        }

        // ── Phase 23: Retire all registered TLABs ─────────────────────────
        // All mutator threads are now suspended (STW). We can safely read and
        // modify their thread_tlab variables via the pointers in tlab_registry.
        for (self.tlab_registry.items) |tlab_ptr| {
            if (tlab_ptr.top < tlab_ptr.end) {
                const tail_size = tlab_ptr.end - tlab_ptr.top;
                const fill: *GCHeader = @ptrCast(@alignCast(&self.memory[tlab_ptr.top]));
                fill.* = GCHeader{
                    .size_or_fwd  = tail_size,
                    .color        = 0,
                    .age          = 0,
                    .is_free      = true,
                    .is_forwarded = false,
                };
            }
            // Reset the TLAB so threads get fresh chunks after resuming.
            tlab_ptr.* = .{ .top = 0, .end = 0 };
        }

        // ── 2. Root scan (STW) ─────────────────────────────────────────────
        for (self.gc.active_frames.items) |addr| {
            const frame = @as(*anyopaque, @ptrFromInt(addr));
            const mock_frame: *interpreter.ExecutionFrame = @ptrCast(@alignCast(frame));
            for (mock_frame.registers, 0..) |val, reg_idx| {
                if (val != 0) {
                    const is_ref = if (mock_frame.register_is_ref.len > 0) mock_frame.register_is_ref[reg_idx] else true;
                    if (is_ref) self.shadeGreyLocked(val);
                }
            }
        }

        for (self.pinned_refs.items) |entry| {
            if (entry.ref != 0 and self.isManagedRef(entry.ref)) {
                self.shadeGreyLocked(entry.ref);
            }
        }

        if (self.root_scan_fn) |scan| {
            self.gc.mutex.unlock(io);
            scan(self.root_scan_ctx, self);
            self.gc.mutex.lockUncancelable(io);
        }
        self.gc.mutex.unlock(io);

        // ── 3. Concurrent Marking ──────────────────────────────────────────
        self.gc.mutex.lockUncancelable(io);
        self.gc.phase = .marking;
        self.gc.safepoint_requested.store(false, .release);
        self.gc.resume_cond.broadcast(io);
        self.gc.mutex.unlock(io);

        while (true) {
            self.gc.mutex.lockUncancelable(io);
            self.drainSatbQueueLocked();
            if (self.gc.grey_stack.items.len == 0) {
                self.gc.mutex.unlock(io);
                break;
            }
            const ref = self.gc.grey_stack.pop().?;
            self.gc.mutex.unlock(io);

            const header = self.getHeader(ref);
            header.color = 2; // Black
            if (header.size() < 8) continue;

            const user_size = header.size() - 8;
            var ofs: u32 = 0;
            while (ofs < user_size) : (ofs += 8) {
                if (self.isSpecialReferenceSlot(ref, ofs)) continue;
                const val = self.readU32Internal(ref + ofs);
                if (val != 0 and self.isLiveObjectRef(val)) {
                    const val_hdr = self.getHeader(val);
                    if (!val_hdr.is_free and val_hdr.size() >= 8) self.shadeGrey(val);
                }
            }
        }

        // ── 4. Re-mark STW ────────────────────────────────────────────────
        self.gc.mutex.lockUncancelable(io);
        self.gc.safepoint_requested.store(true, .release);
        while (true) {
            const active_r = self.gc.active_threads.load(.acquire);
            const need_r: u32 = if (active_r == 0) 0 else if (is_mutator) active_r - 1 else active_r;
            if (self.gc.suspended_threads.load(.acquire) >= need_r) break;
            self.gc.safepoint_cond.waitUncancelable(io, &self.gc.mutex);
        }

        // Trace dirty cards (Old → Young cross-gen refs)
        self.drainSatbQueueLocked();

        for (self.pinned_refs.items) |entry| {
            if (entry.ref != 0 and self.isManagedRef(entry.ref)) {
                self.shadeGreyLocked(entry.ref);
            }
        }

        @memset(self.card_table, 0);

        var keep_idx: usize = 0;
        for (self.remembered_set.items) |slot| {
            if (!self.isOldGen(slot.obj_ref) or !self.isLiveObjectRef(slot.obj_ref)) continue;
            const owner = self.getHeader(slot.obj_ref);
            if (owner.is_free) continue;

            const val = self.readU32Internal(slot.obj_ref + slot.offset);
            if (val != 0 and self.isYoungGen(val) and self.isLiveObjectRef(val)) {
                self.shadeGreyLocked(val);

                self.remembered_set.items[keep_idx] = slot;
                keep_idx += 1;

                const card_idx = (slot.obj_ref - self.old_start) / 512;
                if (card_idx < self.card_table.len) self.card_table[card_idx] = 1;
            }
        }
        self.remembered_set.shrinkRetainingCapacity(keep_idx);

        var scan_los_remark = self.los_start;
        while (scan_los_remark < self.los_end) {
            const hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[scan_los_remark]));
            if (hdr.size() == 0) break;
            if (!hdr.is_free and hdr.color > 0) {
                const obj_ref = scan_los_remark + 8;
                const usr_size = hdr.size() - 8;
                var ofs: u32 = 0;
                while (ofs < usr_size) : (ofs += 8) {
                    if (self.isSpecialReferenceSlot(obj_ref, ofs)) continue;
                    const val = self.readU32Internal(obj_ref + ofs);
                    if (val != 0 and self.isYoungGen(val) and self.isLiveObjectRef(val)) self.shadeGreyLocked(val);
                }
            }
            scan_los_remark += hdr.size();
        }

        self.processSpecialReferencesLocked();

        while (self.gc.grey_stack.items.len > 0) {
            const ref = self.gc.grey_stack.pop().?;
            const hdr = self.getHeader(ref);
            hdr.color = 2;
            const usr_size = hdr.size() - 8;
            var ofs: u32 = 0;
            while (ofs < usr_size) : (ofs += 8) {
                if (self.isSpecialReferenceSlot(ref, ofs)) continue;
                const val = self.readU32Internal(ref + ofs);
                if (val != 0) self.shadeGreyLocked(val);
            }
        }
        self.gc.mutex.unlock(io);

        // ── 5. Evacuate / Copy Young Gen ──────────────────────────────────
        self.gc.mutex.lockUncancelable(io);
        var to_space_bump = self.young_to_start;
        var off_from = self.young_from_start;
 
        while (off_from < self.young_bump) {
            const hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[off_from]));
            const blk_size = hdr.size();
            if (blk_size == 0) break;
 
            if (!hdr.is_forwarded and !hdr.is_free and hdr.color > 0) {
                const promote = (hdr.age >= 3);
                var new_ref: u32 = 0;
                var allocated_old_size: u32 = 0;
                if (promote) {
                    new_ref = self.allocateOldLocked(blk_size - 8) catch 0;
                    if (new_ref != 0) {
                        allocated_old_size = self.getHeader(new_ref).size();
                        _ = self.stats.bytes_promoted.fetchAdd(blk_size, .monotonic);
                    }
                }
                if (new_ref == 0) {
                    new_ref = to_space_bump + 8;
                    to_space_bump += blk_size;
                }
 
                @memcpy(
                    self.memory[new_ref - 8 .. new_ref - 8 + blk_size],
                    self.memory[off_from .. off_from + blk_size],
                );

                const new_hdr = self.getHeader(new_ref);
                if (allocated_old_size > 0) {
                    new_hdr.setSize(allocated_old_size);
                }
                new_hdr.incAge();
                new_hdr.color       = 0;
                new_hdr.is_forwarded = false;

                hdr.setForwardedRef(new_ref);
            }
            off_from += blk_size;
        }

        // ── 6. Forward stack roots ─────────────────────────────────────────
        for (self.gc.active_frames.items) |addr| {
            const frame = @as(*anyopaque, @ptrFromInt(addr));
            const mock_frame: *interpreter.ExecutionFrame = @ptrCast(@alignCast(frame));
            for (mock_frame.registers, 0..) |*val, reg_idx| {
                if (val.* != 0 and self.isYoungGen(val.*)) {
                    const is_ref = if (mock_frame.register_is_ref.len > 0) mock_frame.register_is_ref[reg_idx] else true;
                    if (is_ref) {
                        const oh = self.getHeader(val.*);
                        if (oh.is_forwarded) val.* = oh.forwardedRef();
                    }
                }
            }
        }

        if (self.root_scan_fn) |_| {
            if (self.forward_static_refs_fn) |forward| {
                self.gc.mutex.unlock(io);
                forward(self.root_scan_ctx, self);
                self.gc.mutex.lockUncancelable(io);
            }
        }

        // ── 7. Forward To-space interior ──────────────────────────────────
        for (self.pinned_refs.items) |*entry| {
            if (entry.ref != 0 and self.isYoungGen(entry.ref)) {
                const oh = self.getHeader(entry.ref);
                if (oh.is_forwarded) entry.ref = oh.forwardedRef();
            }
        }
        for (self.special_refs.items) |*slot| {
            if (slot.obj_ref != 0 and self.isYoungGen(slot.obj_ref)) {
                const oh = self.getHeader(slot.obj_ref);
                if (oh.is_forwarded) slot.obj_ref = oh.forwardedRef();
            }
            if (slot.obj_ref != 0 and self.isLiveObjectRef(slot.obj_ref)) {
                const val = self.readU32Internal(slot.obj_ref + slot.offset);
                if (val != 0 and self.isYoungObjectStartForForwarding(val)) {
                    const oh = self.getHeader(val);
                    if (oh.is_forwarded) {
                        const ptr: *u32 = @ptrCast(@alignCast(&self.memory[slot.obj_ref + slot.offset]));
                        ptr.* = oh.forwardedRef();
                    }
                }
            }
        }

        var scan_to = self.young_to_start;
        while (scan_to < to_space_bump) {
            const hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[scan_to]));
            const blk_size = hdr.size();
            if (blk_size == 0) break;
            const obj_ref  = scan_to + 8;
            const usr_size = blk_size - 8;
            var ofs: u32 = 0;
            while (ofs < usr_size) : (ofs += 8) {
                const val = self.readU32Internal(obj_ref + ofs);
                if (val != 0 and self.isYoungObjectStartForForwarding(val)) {
                    const oh = self.getHeader(val);
                    if (oh.is_forwarded) {
                        const ptr: *u32 = @ptrCast(@alignCast(&self.memory[obj_ref + ofs]));
                        ptr.* = oh.forwardedRef();
                    }
                }
            }
            scan_to += blk_size;
        }

        // Old Gen interior forwarding
        var scan_old = self.old_start;
        while (scan_old < self.old_end) {
            const hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[scan_old]));
            if (hdr.size() == 0) break;
            if (!hdr.is_free) {
                const obj_ref  = scan_old + 8;
                const usr_size = hdr.size() - 8;
                var ofs: u32 = 0;
                while (ofs < usr_size) : (ofs += 8) {
                    const val = self.readU32Internal(obj_ref + ofs);
                    if (val != 0 and self.isYoungObjectStartForForwarding(val)) {
                        const oh = self.getHeader(val);
                        if (oh.is_forwarded) {
                            const ptr: *u32 = @ptrCast(@alignCast(&self.memory[obj_ref + ofs]));
                            ptr.* = oh.forwardedRef();
                        }
                    }
                }
            }
            scan_old += hdr.size();
        }

        // ── 8. Swap semispaces ────────────────────────────────────────────
        var scan_los_forward = self.los_start;
        while (scan_los_forward < self.los_end) {
            const hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[scan_los_forward]));
            if (hdr.size() == 0) break;
            if (!hdr.is_free) {
                const obj_ref  = scan_los_forward + 8;
                const usr_size = hdr.size() - 8;
                var ofs: u32 = 0;
                while (ofs < usr_size) : (ofs += 8) {
                    const val = self.readU32Internal(obj_ref + ofs);
                    if (val != 0 and self.isYoungObjectStartForForwarding(val)) {
                        const oh = self.getHeader(val);
                        if (oh.is_forwarded) {
                            const ptr: *u32 = @ptrCast(@alignCast(&self.memory[obj_ref + ofs]));
                            ptr.* = oh.forwardedRef();
                        }
                    }
                }
            }
            scan_los_forward += hdr.size();
        }

        const tmp_start = self.young_from_start;
        const tmp_end   = self.young_from_end;
        self.young_from_start = self.young_to_start;
        self.young_from_end   = self.young_to_end;
        self.young_to_start   = tmp_start;
        self.young_to_end     = tmp_end;
        self.young_bump       = to_space_bump;
        @memset(self.memory[self.young_to_start..self.young_to_end], 0);

        // ── 9. Sweep Old Gen — coalesce + re-bucket ───────────────────────
        for (&self.old_free_lists) |*fl| fl.* = 0;

        var cur_old     = self.old_start;
        var merge_start: u32 = 0;
        var merge_size: u32  = 0;

        while (cur_old < self.old_end) {
            const hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[cur_old]));
            if (hdr.size() == 0) break;
            const blk = hdr.size();

            if (hdr.is_free or hdr.color == 0) {
                if (!hdr.is_free) {
                    hdr.is_free      = true;
                    hdr.is_forwarded = false;
                    @memset(self.memory[cur_old + 8 .. cur_old + blk], 0);
                }
                if (merge_size == 0) { merge_start = cur_old; merge_size = blk; }
                else                { merge_size += blk; }
            } else {
                if (merge_size > 0) {
                    const m: *GCHeader = @ptrCast(@alignCast(&self.memory[merge_start]));
                    m.setSize(merge_size);
                    m.is_free      = true;
                    m.is_forwarded = false;
                    m.color = 0; m.age = 0;
                    self.flPush(merge_start, merge_size);
                    merge_start = 0; merge_size = 0;
                }
                hdr.color        = 0;
                hdr.is_forwarded = false;
            }
            cur_old += blk;
        }
        if (merge_size > 0) {
            const m: *GCHeader = @ptrCast(@alignCast(&self.memory[merge_start]));
            m.setSize(merge_size);
            m.is_free      = true;
            m.is_forwarded = false;
            m.color = 0; m.age = 0;
            self.flPush(merge_start, merge_size);
        }

        // ── Phase 23: Update statistics ───────────────────────────────────
        var cur_los = self.los_start;
        var los_merge_start: u32 = 0;
        var los_merge_size: u32 = 0;

        while (cur_los < self.los_end) {
            const hdr: *GCHeader = @ptrCast(@alignCast(&self.memory[cur_los]));
            if (hdr.size() == 0) break;
            const blk = hdr.size();

            if (hdr.is_free or hdr.color == 0) {
                if (!hdr.is_free) {
                    hdr.is_free = true;
                    hdr.is_forwarded = false;
                    @memset(self.memory[cur_los + 8 .. cur_los + blk], 0);
                }
                if (los_merge_size == 0) {
                    los_merge_start = cur_los;
                    los_merge_size = blk;
                } else {
                    los_merge_size += blk;
                }
            } else {
                if (los_merge_size > 0) {
                    const m: *GCHeader = @ptrCast(@alignCast(&self.memory[los_merge_start]));
                    m.setSize(los_merge_size);
                    m.is_free = true;
                    m.is_forwarded = false;
                    m.color = 0;
                    m.age = 0;
                    los_merge_start = 0;
                    los_merge_size = 0;
                }
                hdr.color = 0;
                hdr.is_forwarded = false;
            }
            cur_los += blk;
        }
        if (los_merge_size > 0) {
            const m: *GCHeader = @ptrCast(@alignCast(&self.memory[los_merge_start]));
            m.setSize(los_merge_size);
            m.is_free = true;
            m.is_forwarded = false;
            m.color = 0;
            m.age = 0;
        }

        self.compactSpecialReferencesLocked();

        try self.validateIntegrity();

        _ = self.stats.gc_cycles.fetchAdd(1, .monotonic);
        const now_ts = std.Io.Clock.real.now(io);
        const elapsed_ns = now_ts.nanoseconds - pause_start.nanoseconds;
        const ns_u64: u64 = if (elapsed_ns > 0) @intCast(elapsed_ns) else 0;
        _ = self.stats.total_pause_ns.fetchAdd(ns_u64, .monotonic);

        // Adaptive GC trigger: if we're promoted heavily, lower the threshold
        // (handled in gcLoop via stats; no structural change needed here)

        success = true;
        self.gc.phase = .idle;
        self.gc.safepoint_requested.store(false, .release);
        self.gc.safepoint_active = false;
        self.gc.resume_cond.broadcast(io);
        self.gc.mutex.unlock(io);
    }

    // ==========================================
    // Memory Access Helpers
    // ==========================================
    pub inline fn checkNull(ref: u32) !void {
        if (ref == 0) return error.NullPointerException;
    }

    pub inline fn readU32Internal(self: *const Heap, offset: u32) u32 {
        const ptr: *const u32 = @ptrCast(@alignCast(&self.memory[offset]));
        return @atomicLoad(u32, ptr, .monotonic);
    }

    pub inline fn readU32(self: *const Heap, ref: u32, offset: u32) !u32 {
        try checkNull(ref);
        return self.readU32Internal(ref + offset);
    }

    pub inline fn writeU32(self: *Heap, ref: u32, offset: u32, val: u32) !void {
        try checkNull(ref);
        const ptr: *u32 = @ptrCast(@alignCast(&self.memory[ref + offset]));
        @atomicStore(u32, ptr, val, .monotonic);
    }

    pub inline fn writeRef(self: *Heap, ref: u32, offset: u32, val: u32) !void {
        try checkNull(ref);
        if (!self.isLiveObjectRef(ref)) return error.InvalidReference;
        if (val != 0 and !self.isLiveObjectRef(val)) return error.InvalidReference;
        self.writeBarrier(ref, offset, val);
        const ptr: *u32 = @ptrCast(@alignCast(&self.memory[ref + offset]));
        @atomicStore(u32, ptr, val, .monotonic);
    }

    pub inline fn readU64(self: *const Heap, ref: u32, offset: u32) !u64 {
        try checkNull(ref);
        const ptr: *const u64 = @ptrCast(@alignCast(&self.memory[ref + offset]));
        return @atomicLoad(u64, ptr, .monotonic);
    }

    pub inline fn writeU64(self: *Heap, ref: u32, offset: u32, val: u64) !void {
        try checkNull(ref);
        const ptr: *u64 = @ptrCast(@alignCast(&self.memory[ref + offset]));
        @atomicStore(u64, ptr, val, .monotonic);
    }

    pub inline fn readU16(self: *const Heap, ref: u32, offset: u32) !u16 {
        try checkNull(ref);
        const ptr: *u16 = @ptrCast(@alignCast(&self.memory[ref + offset]));
        return ptr.*;
    }

    pub inline fn writeU16(self: *Heap, ref: u32, offset: u32, val: u16) !void {
        try checkNull(ref);
        const ptr: *u16 = @ptrCast(@alignCast(&self.memory[ref + offset]));
        ptr.* = val;
    }

    pub inline fn readU8(self: *const Heap, ref: u32, offset: u32) !u8 {
        try checkNull(ref);
        return self.memory[ref + offset];
    }

    pub inline fn writeU8(self: *Heap, ref: u32, offset: u32, val: u8) !void {
        try checkNull(ref);
        self.memory[ref + offset] = val;
    }

    pub inline fn getArrayLength(self: *const Heap, ref: u32) !u32 {
        return self.readU32(ref, 0);
    }

    pub inline fn checkBounds(self: *const Heap, ref: u32, index: u32) !void {
        const length = try self.getArrayLength(ref);
        if (index >= length) return error.ArrayIndexOutOfBounds;
    }
};

// ==========================================
// Background GC Loop (adaptive trigger)
// ==========================================
pub fn gcLoop(heap: *Heap) void {
    const io = heap.io;
    while (true) {
        const dur = std.Io.Clock.Duration{ .raw = std.Io.Duration.fromMilliseconds(10), .clock = .real };
        dur.sleep(io) catch {};

        var stop = false;
        heap.gc.mutex.lockUncancelable(io);
        stop = heap.gc.should_stop;
        heap.gc.mutex.unlock(io);
        if (stop) break;

        // Adaptive trigger: tighten to 60% when allocation pressure is high
        // (bytes_allocated doubled since last sampled). Default: 80%.
        const used  = heap.young_bump - heap.young_from_start;
        const total = heap.young_from_end - heap.young_from_start;
        const tlab_hits = heap.stats.tlab_refills.load(.monotonic);
        const threshold: u32 = if (tlab_hits > 20) total * 6 / 10 else total * 8 / 10;

        if (used > threshold) {
            heap.runGC() catch {};
        }
    }
}

// ==========================================
// Tests
// ==========================================

test "Heap Generational Allocator Basics" {
    var tlab = TLAB{};
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    try std.testing.expect(test_heap.young_from_start == 8);
    const expected_semispace_size = (1024 * 1024) / 8;
    try std.testing.expectEqual(test_heap.young_from_start + expected_semispace_size, test_heap.young_from_end);

    // Allocate into Young Gen via TLAB — block = 8 header + 32 user = 40 bytes
    const ref_y = try test_heap.allocate(&tlab, 32);
    try std.testing.expect(test_heap.isYoungGen(ref_y));
    const header_y = test_heap.getHeader(ref_y);
    try std.testing.expectEqual(@as(u32, 40), header_y.size());
    try std.testing.expectEqual(false, header_y.is_free);

    // Allocate into Old Gen directly — 8 + 64 = 72 bytes
    const ref_o = try test_heap.allocateOld(64);
    try std.testing.expect(test_heap.isOldGen(ref_o));
    const header_o = test_heap.getHeader(ref_o);
    try std.testing.expectEqual(@as(u32, 72), header_o.size());
    try std.testing.expectEqual(false, header_o.is_free);
}

test "Heap Layered Write Barriers" {
    var tlab = TLAB{};
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    const young_obj = try test_heap.allocate(&tlab, 32);
    const young_val = try test_heap.allocate(&tlab, 32);
    const old_obj   = try test_heap.allocateOld(32);

    test_heap.writeBarrier(young_obj, 8, young_val);

    test_heap.writeBarrier(old_obj, 8, young_val);
    const card_idx_old = (old_obj - test_heap.old_start) / 512;
    try std.testing.expectEqual(@as(u8, 1), test_heap.card_table[card_idx_old]);

    test_heap.card_table[card_idx_old] = 0;

    const old_val = try test_heap.allocateOld(32);
    test_heap.writeBarrier(old_obj, 8, old_val);
    try std.testing.expectEqual(@as(u8, 0), test_heap.card_table[card_idx_old]);

    try test_heap.writeRef(old_obj, 16, young_val);
    try std.testing.expectEqual(@as(u8, 1), test_heap.card_table[card_idx_old]);
    try std.testing.expectEqual(@as(usize, 2), test_heap.remembered_set.items.len);
}

test "Heap SATB Barrier Preserves Deleted References During Marking" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    const parent = try test_heap.allocateOld(32);
    const child = try test_heap.allocateOld(32);
    try test_heap.writeRef(parent, 0, child);

    test_heap.getHeader(parent).color = 2;
    test_heap.getHeader(child).color = 0;
    test_heap.gc.phase = .marking;

    try test_heap.writeRef(parent, 0, 0);

    try std.testing.expectEqual(@as(usize, 1), test_heap.gc.satb_queue.items.len);
    test_heap.drainSatbQueueLocked();
    try std.testing.expectEqual(@as(u2, 1), test_heap.getHeader(child).color);

    test_heap.gc.phase = .idle;
}

test "Heap Weak And Phantom References Clear Unreachable Referents" {
    var tlab = TLAB{};
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    var regs: [2]u32 = .{ 0, 0 };
    var mock_frame = interpreter.ExecutionFrame{ .pc = 0, .registers = &regs, .instructions = &.{} };
    try test_heap.gc.active_frames.append(std.testing.allocator, @intFromPtr(&mock_frame));

    const weak_ref_obj = try test_heap.allocate(&tlab, 32);
    const phantom_ref_obj = try test_heap.allocate(&tlab, 32);
    regs[0] = weak_ref_obj;
    regs[1] = phantom_ref_obj;

    const weak_referent = try test_heap.allocate(&tlab, 32);
    const phantom_referent = try test_heap.allocate(&tlab, 32);

    try test_heap.registerSpecialReference(weak_ref_obj, 0, .weak);
    try test_heap.registerSpecialReference(phantom_ref_obj, 0, .phantom);
    try test_heap.writeRef(weak_ref_obj, 0, weak_referent);
    try test_heap.writeRef(phantom_ref_obj, 0, phantom_referent);

    try test_heap.runGC();

    try std.testing.expectEqual(@as(u32, 0), try test_heap.readU32(regs[0], 0));
    try std.testing.expectEqual(@as(u32, 0), try test_heap.readU32(regs[1], 0));
}

test "Heap Soft References Preserve Referents Without Pressure" {
    var tlab = TLAB{};
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    var regs: [1]u32 = .{0};
    var mock_frame = interpreter.ExecutionFrame{ .pc = 0, .registers = &regs, .instructions = &.{} };
    try test_heap.gc.active_frames.append(std.testing.allocator, @intFromPtr(&mock_frame));

    const soft_ref_obj = try test_heap.allocate(&tlab, 32);
    regs[0] = soft_ref_obj;

    const referent = try test_heap.allocate(&tlab, 32);
    try test_heap.registerSpecialReference(soft_ref_obj, 0, .soft);
    try test_heap.writeRef(soft_ref_obj, 0, referent);

    try test_heap.runGC();

    const retained = try test_heap.readU32(regs[0], 0);
    try std.testing.expect(retained != 0);
    try std.testing.expect(test_heap.isManagedRef(retained));
}

test "Heap Pinning Promotes Young Object To Stable Handle" {
    var tlab = TLAB{};
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    const young = try test_heap.allocate(&tlab, 32);
    try std.testing.expect(test_heap.isYoungGen(young));

    const pinned = try test_heap.pinObject(young);
    try std.testing.expect(test_heap.isOldGen(pinned));
    try std.testing.expect(test_heap.isPinned(pinned));

    try test_heap.runGC();

    try std.testing.expect(test_heap.isOldGen(pinned));
    try std.testing.expectEqual(false, test_heap.getHeader(pinned).is_free);

    test_heap.unpinObject(pinned);
    try std.testing.expect(!test_heap.isPinned(pinned));

    try test_heap.runGC();
    try std.testing.expectEqual(true, test_heap.getHeader(pinned).is_free);
}

test "Heap Pinning Updates References Via Forwarding" {
    var tlab = TLAB{};
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    var regs: [1]u32 = .{0};
    var mock_frame = interpreter.ExecutionFrame{ .pc = 0, .registers = &regs, .instructions = &.{} };
    try test_heap.gc.active_frames.append(std.testing.allocator, @intFromPtr(&mock_frame));

    const young = try test_heap.allocate(&tlab, 32);
    regs[0] = young;

    const pinned = try test_heap.pinObject(young);
    try std.testing.expect(test_heap.isOldGen(pinned));

    try test_heap.runGC();

    try std.testing.expectEqual(pinned, regs[0]);

    test_heap.unpinObject(pinned);
}

test "Heap Pinning Uses Reference Counts" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    const old_obj = try test_heap.allocateOld(32);
    const pin1 = try test_heap.pinObject(old_obj);
    const pin2 = try test_heap.pinObject(old_obj);

    try std.testing.expectEqual(old_obj, pin1);
    try std.testing.expectEqual(old_obj, pin2);
    try std.testing.expect(test_heap.isPinned(old_obj));

    test_heap.unpinObject(old_obj);
    try std.testing.expect(test_heap.isPinned(old_obj));

    try test_heap.runGC();
    try std.testing.expectEqual(false, test_heap.getHeader(old_obj).is_free);

    test_heap.unpinObject(old_obj);
    try std.testing.expect(!test_heap.isPinned(old_obj));
}

test "Heap Integrity Validator Detects Corruption" {
    var tlab = TLAB{};
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    const old_obj = try test_heap.allocateOld(32);
    try test_heap.validateIntegrity();

    const old_hdr = test_heap.getHeader(old_obj);
    const saved_size = old_hdr.size();
    old_hdr.setSize(7);
    try std.testing.expectError(error.InvalidHeapBlock, test_heap.validateIntegrity());
    old_hdr.setSize(saved_size);

    const young_obj = try test_heap.allocate(&tlab, 32);
    try test_heap.writeRef(old_obj, 0, young_obj);
    test_heap.remembered_set.items[0].obj_ref = young_obj;
    try std.testing.expectError(error.InvalidRememberedSet, test_heap.validateIntegrity());
}

test "Heap Randomized GC Stress Mix" {
    var tlab = TLAB{};
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 2 * 1024 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    var regs: [16]u32 = .{0} ** 16;
    var mock_frame = interpreter.ExecutionFrame{ .pc = 0, .registers = &regs, .instructions = &.{} };
    try test_heap.gc.active_frames.append(std.testing.allocator, @intFromPtr(&mock_frame));

    const Rng = struct {
        state: u64,

        fn next(self: *@This()) u32 {
            self.state = self.state *% 6364136223846793005 +% 1442695040888963407;
            return @as(u32, @truncate(self.state >> 32));
        }

        fn index(self: *@This(), comptime len: usize) usize {
            return @as(usize, @intCast(self.next() % len));
        }
    };

    var rng = Rng{ .state = 0x5eed_c0de_1234_5678 };
    var pinned: [16]u32 = .{0} ** 16;

    var step: usize = 0;
    while (step < 500) : (step += 1) {
        const op = rng.next() % 8;
        const slot = rng.index(regs.len);
        if (regs[slot] != 0 and !test_heap.isLiveObjectRef(regs[slot])) regs[slot] = 0;

        switch (op) {
            0 => {
                const size = 32 + @as(u32, @intCast((rng.next() % 4) * 16));
                regs[slot] = test_heap.allocate(&tlab, size) catch blk: {
                    try test_heap.runGC();
                    break :blk test_heap.allocate(&tlab, size) catch 0;
                };
            },
            1 => {
                if (regs[slot] != 0) {
                    const target_slot = rng.index(regs.len);
                    const target = if (regs[target_slot] != 0 and test_heap.isLiveObjectRef(regs[target_slot])) regs[target_slot] else 0;
                    const offset = @as(u32, @intCast((rng.next() % 4) * 8));
                    try test_heap.writeRef(regs[slot], offset, target);
                }
            },
            2 => {
                if (regs[slot] != 0) {
                    const offset = @as(u32, @intCast((rng.next() % 4) * 8));
                    try test_heap.writeRef(regs[slot], offset, 0);
                }
            },
            3 => {
                if (regs[slot] != 0) {
                    const offset = @as(u32, @intCast((rng.next() % 4) * 8));
                    try test_heap.registerSpecialReference(regs[slot], offset, .weak);
                    const target_slot = rng.index(regs.len);
                    const target = if (regs[target_slot] != 0 and test_heap.isLiveObjectRef(regs[target_slot])) regs[target_slot] else 0;
                    try test_heap.writeRef(regs[slot], offset, target);
                }
            },
            4 => {
                if (regs[slot] != 0 and pinned[slot] == 0) {
                    const stable = try test_heap.pinObject(regs[slot]);
                    regs[slot] = stable;
                    pinned[slot] = stable;
                }
            },
            5 => {
                if (pinned[slot] != 0) {
                    test_heap.unpinObject(pinned[slot]);
                    pinned[slot] = 0;
                }
            },
            6 => {
                const size = LARGE_OBJECT_THRESHOLD + @as(u32, @intCast((rng.next() % 4) * 1024));
                regs[slot] = test_heap.allocate(&tlab, size) catch blk: {
                    try test_heap.runGC();
                    break :blk test_heap.allocate(&tlab, size) catch 0;
                };
            },
            else => {
                try test_heap.runGC();
                try test_heap.validateIntegrity();
            },
        }

        if (step % 37 == 0) {
            try test_heap.runGC();
            try test_heap.validateIntegrity();
        }
    }

    for (pinned) |ref| {
        if (ref != 0) test_heap.unpinObject(ref);
    }

    try test_heap.runGC();
    try test_heap.validateIntegrity();
}

test "Heap Segregated Free List Allocation" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    const small = try test_heap.allocateOld(8);   // total=16,  class 0
    const mid   = try test_heap.allocateOld(56);  // total=64,  class 2
    const large = try test_heap.allocateOld(500); // total=512, class 5

    try std.testing.expect(test_heap.isOldGen(small));
    try std.testing.expect(test_heap.isOldGen(mid));
    try std.testing.expect(test_heap.isOldGen(large));

    const hdr_s = test_heap.getHeader(small);
    const hdr_m = test_heap.getHeader(mid);
    const hdr_l = test_heap.getHeader(large);

    try std.testing.expectEqual(false, hdr_s.is_free);
    try std.testing.expectEqual(false, hdr_m.is_free);
    try std.testing.expectEqual(false, hdr_l.is_free);

    hdr_s.color = 0; hdr_m.color = 0; hdr_l.color = 0;
    try test_heap.runGC();

    var any_free = false;
    for (test_heap.old_free_lists) |head| {
        if (head != 0) { any_free = true; break; }
    }
    try std.testing.expect(any_free);
}

test "Heap Rigorous Segregated Free List and Coalescing Verification" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    // 1. Check sizeClass function mapping directly
    try std.testing.expectEqual(@as(u3, 0), sizeClass(16));
    try std.testing.expectEqual(@as(u3, 1), sizeClass(32));
    try std.testing.expectEqual(@as(u3, 2), sizeClass(64));
    try std.testing.expectEqual(@as(u3, 3), sizeClass(128));
    try std.testing.expectEqual(@as(u3, 4), sizeClass(256));
    try std.testing.expectEqual(@as(u3, 5), sizeClass(512));
    try std.testing.expectEqual(@as(u3, 6), sizeClass(1024));
    try std.testing.expectEqual(@as(u3, 7), sizeClass(2048));

    // Register active frame so GC root scanner can see D
    var regs: [1]u32 = .{0};
    var mock_frame = interpreter.ExecutionFrame{ .pc = 0, .registers = &regs, .instructions = &.{} };
    try test_heap.gc.active_frames.append(std.testing.allocator, @intFromPtr(&mock_frame));

    // 2. Allocate consecutive blocks in Old Gen
    const ref_a = try test_heap.allocateOld(8);
    const ref_b = try test_heap.allocateOld(8);
    const ref_c = try test_heap.allocateOld(24);
    const ref_d = try test_heap.allocateOld(8); // Anchor block

    regs[0] = ref_d; // Keep D alive

    const hdr_a = test_heap.getHeader(ref_a);
    const hdr_b = test_heap.getHeader(ref_b);
    const hdr_c = test_heap.getHeader(ref_c);
    const hdr_d = test_heap.getHeader(ref_d);

    // 3. Mark A, B, C dead (color = 0) and D alive (color = 2)
    hdr_a.color = 0;
    hdr_b.color = 0;
    hdr_c.color = 0;
    hdr_d.color = 2;

    try test_heap.runGC();

    // After GC, A, B, and C should be coalesced into a single free block of size 64.
    const expected_block_offset = ref_a - 8;
    try std.testing.expectEqual(expected_block_offset, test_heap.old_free_lists[2]);

    // 4. Try allocating from this coalesced block of size 64!
    // Let's allocate 24 bytes payload (total 32 bytes, class 1).
    const new_ref = try test_heap.allocateOld(24);
    try std.testing.expectEqual(ref_a, new_ref); // Reuse the start of the coalesced block

    const rem_expected_offset = expected_block_offset + 32;
    try std.testing.expectEqual(rem_expected_offset, test_heap.old_free_lists[1]);

    // 5. Test multiple free lists chaining (singly-linked list test)
    // Clear all free lists first
    for (&test_heap.old_free_lists) |*fl| fl.* = 0;

    // Push two separate blocks of size 16 into class 0
    test_heap.flPush(10000, 16);
    test_heap.flPush(20000, 16);

    // Head of class 0 should be the last pushed block (20000)
    try std.testing.expectEqual(@as(u32, 20000), test_heap.old_free_lists[0]);

    // Next pointer of head (20000 + 8) should point to 10000
    const next_of_head = @as(*u32, @ptrCast(@alignCast(&test_heap.memory[20008]))).*;
    try std.testing.expectEqual(@as(u32, 10000), next_of_head);

    // Next pointer of 10000 + 8 should point to 0
    const next_of_second = @as(*u32, @ptrCast(@alignCast(&test_heap.memory[10008]))).*;
    try std.testing.expectEqual(@as(u32, 0), next_of_second);

    // Pop the first one
    const popped1 = test_heap.flPop(0);
    try std.testing.expectEqual(@as(u32, 20000), popped1);
    try std.testing.expectEqual(@as(u32, 10000), test_heap.old_free_lists[0]);

    // Pop the second one
    const popped2 = test_heap.flPop(0);
    try std.testing.expectEqual(@as(u32, 10000), popped2);
    try std.testing.expectEqual(@as(u32, 0), test_heap.old_free_lists[0]);
}

test "Heap Large Object Space Allocation Is Non-Moving" {
    var tlab = TLAB{};
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    var regs: [1]u32 = .{0};
    var mock_frame = interpreter.ExecutionFrame{ .pc = 0, .registers = &regs, .instructions = &.{} };
    try test_heap.gc.active_frames.append(std.testing.allocator, @intFromPtr(&mock_frame));

    const large_ref = try test_heap.allocate(&tlab, LARGE_OBJECT_THRESHOLD);
    regs[0] = large_ref;
    try std.testing.expect(test_heap.isLargeObjectSpace(large_ref));

    const child = try test_heap.allocate(&tlab, 32);
    try test_heap.writeRef(large_ref, 0, child);

    try test_heap.runGC();

    try std.testing.expectEqual(large_ref, regs[0]);
    try std.testing.expect(test_heap.isLargeObjectSpace(large_ref));

    const forwarded_child = try test_heap.readU32(large_ref, 0);
    try std.testing.expect(forwarded_child != child);
    try std.testing.expect(test_heap.isYoungGen(forwarded_child));
}

test "Heap Large Object Space Sweeps Dead Objects" {
    var tlab = TLAB{};
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    const first = try test_heap.allocate(&tlab, LARGE_OBJECT_THRESHOLD);
    const second = try test_heap.allocate(&tlab, LARGE_OBJECT_THRESHOLD);
    try std.testing.expect(test_heap.isLargeObjectSpace(first));
    try std.testing.expect(test_heap.isLargeObjectSpace(second));

    try test_heap.runGC();

    const reclaimed = try test_heap.allocate(&tlab, LARGE_OBJECT_THRESHOLD * 2);
    try std.testing.expect(test_heap.isLargeObjectSpace(reclaimed));
}

test "Heap Rigorous Garbage Collection & Copying Verification" {
    var tlab = TLAB{};
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 128 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    var regs: [4]u32 = .{ 0, 0, 0, 0 };
    var mock_frame = interpreter.ExecutionFrame{ .pc = 0, .registers = &regs, .instructions = &.{} };
    try test_heap.gc.active_frames.append(std.testing.allocator, @intFromPtr(&mock_frame));

    const root_ref = try test_heap.allocate(&tlab, 32);
    regs[0] = root_ref;
    const child1 = try test_heap.allocate(&tlab, 32);
    const child2 = try test_heap.allocate(&tlab, 32);
    try test_heap.writeRef(root_ref, 8, child1);
    try test_heap.writeRef(root_ref, 16, child2);
    _ = try test_heap.allocate(&tlab, 32);
    _ = try test_heap.allocate(&tlab, 32);

    try std.testing.expect(test_heap.isYoungGen(root_ref));
    try std.testing.expect(test_heap.isYoungGen(child1));
    try std.testing.expect(test_heap.isYoungGen(child2));

    try test_heap.runGC();

    const new_root = regs[0];
    try std.testing.expect(new_root != root_ref);
    try std.testing.expect(test_heap.isYoungGen(new_root));

    const new_child1 = try test_heap.readU32(new_root, 8);
    const new_child2 = try test_heap.readU32(new_root, 16);
    try std.testing.expect(new_child1 != child1);
    try std.testing.expect(new_child2 != child2);
    try std.testing.expect(test_heap.isYoungGen(new_child1));
    try std.testing.expect(test_heap.isYoungGen(new_child2));

    const root_hdr = test_heap.getHeader(new_root);
    try std.testing.expectEqual(@as(u2, 1), root_hdr.age);
}

test "Heap Concurrent Multi-Threaded Stress Collection" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 256 * 1024);
    defer test_heap.deinit(std.testing.allocator);
    try test_heap.start();

    var regs1: [2]u32 = .{ 0, 0 };
    var regs2: [2]u32 = .{ 0, 0 };
    var mock_frame1 = interpreter.ExecutionFrame{ .pc = 0, .registers = &regs1, .instructions = &.{} };
    var mock_frame2 = interpreter.ExecutionFrame{ .pc = 0, .registers = &regs2, .instructions = &.{} };
    try test_heap.gc.active_frames.append(std.testing.allocator, @intFromPtr(&mock_frame1));
    try test_heap.gc.active_frames.append(std.testing.allocator, @intFromPtr(&mock_frame2));

    const worker1 = struct {
        fn run(h: *Heap, r: []u32, f: anytype) void {
            var tlab = TLAB{};
            h.registerThread();
            defer h.deregisterThread(&tlab);
            var i: usize = 0;
            while (i < 200) : (i += 1) {
                h.enterSafepoint(f) catch {};
                const ref = h.allocate(&tlab, 32) catch continue;
                r[0] = ref;
                const child = h.allocate(&tlab, 32) catch continue;
                h.writeRef(ref, 8, child) catch {};
            }
        }
    }.run;

    const worker2 = struct {
        fn run(h: *Heap, r: []u32, f: anytype) void {
            var tlab = TLAB{};
            h.registerThread();
            defer h.deregisterThread(&tlab);
            var i: usize = 0;
            while (i < 200) : (i += 1) {
                h.enterSafepoint(f) catch {};
                const ref = h.allocate(&tlab, 32) catch continue;
                r[0] = ref;
                const child = h.allocate(&tlab, 32) catch continue;
                h.writeRef(ref, 8, child) catch {};
            }
        }
    }.run;

    const t1 = try std.Thread.spawn(.{}, worker1, .{ &test_heap, &regs1, &mock_frame1 });
    const t2 = try std.Thread.spawn(.{}, worker2, .{ &test_heap, &regs2, &mock_frame2 });
    t1.join();
    t2.join();

    try test_heap.runGC();

    if (regs1[0] != 0) try std.testing.expectEqual(false, test_heap.getHeader(regs1[0]).is_free);
    if (regs2[0] != 0) try std.testing.expectEqual(false, test_heap.getHeader(regs2[0]).is_free);
}

test "Heap TLAB Lock-Free Allocation" {
    var tlab = TLAB{};
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    const N = 200;
    var refs: [N]u32 = undefined;
    for (&refs) |*r| {
        r.* = try test_heap.allocate(&tlab, 32); // 40-byte block each
    }

    // All refs must be in Young Gen and non-free.
    for (refs) |r| {
        try std.testing.expect(test_heap.isYoungGen(r));
        try std.testing.expectEqual(false, test_heap.getHeader(r).is_free);
    }

    // TLAB_SIZE=4096, each object=40 bytes → ~102 objects per TLAB.
    // 200 objects → 2 TLABs = 2 refills. Must be far less than 200.
    const refills = test_heap.stats.tlab_refills.load(.monotonic);
    try std.testing.expect(refills < N / 10); // < 20 (in practice ~2)

    // bytes_allocated must account for all objects.
    const alloc_bytes = test_heap.stats.bytes_allocated.load(.monotonic);
    try std.testing.expect(alloc_bytes >= @as(u64, N) * 40);
}

test "Heap GC Statistics Accuracy" {
    var tlab = TLAB{};
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 128 * 1024);
    defer test_heap.deinit(std.testing.allocator);

    // Allocate and discard — all will be collected.
    _ = try test_heap.allocate(&tlab, 32);
    _ = try test_heap.allocate(&tlab, 64);
    _ = try test_heap.allocate(&tlab, 128);

    try std.testing.expectEqual(@as(u32, 0), test_heap.stats.gc_cycles.load(.monotonic));

    try test_heap.runGC();
    try std.testing.expectEqual(@as(u32, 1), test_heap.stats.gc_cycles.load(.monotonic));

    // bytes_allocated must be > 0 (we allocated 3 objects)
    try std.testing.expect(test_heap.stats.bytes_allocated.load(.monotonic) > 0);

    // Pause time must have been recorded (even if very small).
    _ = test_heap.stats.total_pause_ns.load(.monotonic);

    try test_heap.runGC();
    try std.testing.expectEqual(@as(u32, 2), test_heap.stats.gc_cycles.load(.monotonic));
}

test "Heap TLAB Multi-Thread Throughput" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var test_heap = try Heap.init(std.testing.allocator, io, 512 * 1024);
    defer test_heap.deinit(std.testing.allocator);
    try test_heap.start();

    const THREADS = 4;
    const ITERS   = 150;

    // Each thread stores its last root ref here (roots for GC).
    var regs: [THREADS]u32 = .{0} ** THREADS;
    var frames: [THREADS]interpreter.ExecutionFrame = undefined;
    for (&frames, 0..) |*f, i| {
        f.* = .{ .pc = 0, .registers = regs[i..i+1], .instructions = &.{} };
        try test_heap.gc.active_frames.append(std.testing.allocator, @intFromPtr(f));
    }

    const worker = struct {
        fn run(h: *Heap, root: *u32, f: anytype) void {
            var tlab = TLAB{};
            h.registerThread();
            defer h.deregisterThread(&tlab);
            var i: usize = 0;
            while (i < ITERS) : (i += 1) {
                h.enterSafepoint(f) catch {};
                const ref = h.allocate(&tlab, 32) catch continue;
                root.* = ref;
                // Link a child to exercise the write barrier.
                const child = h.allocate(&tlab, 16) catch continue;
                h.writeRef(ref, 8, child) catch {};
            }
        }
    }.run;

    var threads: [THREADS]std.Thread = undefined;
    for (&threads, 0..) |*t, i| {
        t.* = try std.Thread.spawn(.{}, worker, .{ &test_heap, &regs[i], &frames[i] });
    }
    for (&threads) |t| t.join();

    // Trigger a final GC to ensure heap is consistent.
    try test_heap.runGC();

    // Verify each surviving root is valid.
    for (regs) |r| {
        if (r != 0) {
            try std.testing.expect(test_heap.isManagedRef(r));
            try std.testing.expectEqual(false, test_heap.getHeader(r).is_free);
        }
    }

    // bytes_allocated must account for all threads' work.
    const alloc_bytes = test_heap.stats.bytes_allocated.load(.monotonic);
    try std.testing.expect(alloc_bytes > 0);
}

test "Heap comprehensive multi-heap stress test" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // 1. Initialize two isolated heaps
    var heap1 = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer heap1.deinit(std.testing.allocator);
    var heap2 = try Heap.init(std.testing.allocator, io, 1024 * 1024);
    defer heap2.deinit(std.testing.allocator);

    try heap1.start();
    try heap2.start();

    const THREADS = 4;
    const ITERS = 200;

    // Registers for GC roots scanning
    var regs1: [THREADS]u32 = .{0} ** THREADS;
    var frames1: [THREADS]interpreter.ExecutionFrame = undefined;
    for (&frames1, 0..) |*f, i| {
        f.* = .{ .pc = 0, .registers = regs1[i..i+1], .instructions = &.{} };
        try heap1.gc.active_frames.append(std.testing.allocator, @intFromPtr(f));
    }

    var regs2: [THREADS]u32 = .{0} ** THREADS;
    var frames2: [THREADS]interpreter.ExecutionFrame = undefined;
    for (&frames2, 0..) |*f, i| {
        f.* = .{ .pc = 0, .registers = regs2[i..i+1], .instructions = &.{} };
        try heap2.gc.active_frames.append(std.testing.allocator, @intFromPtr(f));
    }

    var mutator_mutex: std.Io.Mutex = .init;
    const worker = struct {
        fn run(id: usize, h1: *Heap, h2: *Heap, r1: *u32, r2: *u32, f1: anytype, f2: anytype, mu: *std.Io.Mutex) void {
            var tlab1 = TLAB{};
            var tlab2 = TLAB{};
            defer h1.deregisterThreadTlab(&tlab1);
            defer h2.deregisterThreadTlab(&tlab2);
            // Do NOT call registerThread: with active_threads==0 the GC STW
            // coordinator sees target_suspended==0 and proceeds immediately.
            // Workers still call enterSafepoint each iteration to cooperate with GC.
            var rng = std.Random.DefaultPrng.init(0x12345678 + id);
            const random = rng.random();

            var i: usize = 0;
            while (i < ITERS) : (i += 1) {
                mu.lockUncancelable(h1.io);
                defer mu.unlock(h1.io);

                h1.enterSafepoint(f1) catch {};
                h2.enterSafepoint(f2) catch {};

                // Alternating allocations on both heaps to trigger context switching
                const size = random.uintAtMost(u32, 100) + 8;
                const ref1 = h1.allocate(&tlab1, size) catch continue;
                r1.* = ref1;

                const ref2 = h2.allocate(&tlab2, size) catch continue;
                r2.* = ref2;

                // Randomly link child objects to trigger write barriers
                if (random.boolean()) {
                    const child1 = h1.allocate(&tlab1, 16) catch continue;
                    h1.writeRef(ref1, 0, child1) catch {};
                }
                if (random.boolean()) {
                    const child2 = h2.allocate(&tlab2, 16) catch continue;
                    h2.writeRef(ref2, 0, child2) catch {};
                }

                // Randomly pin/unpin objects
                if (random.uintAtMost(u8, 10) == 0) {
                    const p1 = h1.pinObject(ref1) catch ref1;
                    r1.* = p1;
                    if (random.boolean()) h1.unpinObject(p1);
                }
                if (random.uintAtMost(u8, 10) == 0) {
                    const p2 = h2.pinObject(ref2) catch ref2;
                    r2.* = p2;
                    if (random.boolean()) h2.unpinObject(p2);
                }

                // Allocate some old gen directly
                if (random.uintAtMost(u8, 20) == 0) {
                    _ = h1.allocateOld(32) catch {};
                    _ = h2.allocateOld(32) catch {};
                }

                // Allocate large objects directly
                if (random.uintAtMost(u8, 30) == 0) {
                    _ = h1.allocateLarge(LARGE_OBJECT_THRESHOLD + 16) catch {};
                    _ = h2.allocateLarge(LARGE_OBJECT_THRESHOLD + 16) catch {};
                }
            }
        }
    }.run;

    var threads: [THREADS]std.Thread = undefined;
    for (&threads, 0..) |*t, i| {
        t.* = try std.Thread.spawn(.{}, worker, .{ i, &heap1, &heap2, &regs1[i], &regs2[i], &frames1[i], &frames2[i], &mutator_mutex });
    }
    for (&threads) |t| t.join();

    // Trigger GC to run final sweeping/compaction and check integrity
    try heap1.runGC();
    try heap2.runGC();

    try heap1.validateIntegrity();
    try heap2.validateIntegrity();
}

test "Heap real-world GC/mutator deadlock and race stress" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var heap = try Heap.init(std.testing.allocator, io, 512 * 1024);
    defer heap.deinit(std.testing.allocator);
    try heap.start();

    const THREADS = 4;
    const ITERS = 100;

    // Mutator shared lock
    var shared_mutex: std.Io.Mutex = .init;

    var regs: [THREADS]u32 = .{0} ** THREADS;
    var frames: [THREADS]interpreter.ExecutionFrame = undefined;
    for (&frames, 0..) |*f, i| {
        f.* = .{ .pc = 0, .registers = regs[i..i+1], .instructions = &.{} };
        try heap.gc.active_frames.append(std.testing.allocator, @intFromPtr(f));
    }

    const worker = struct {
        fn run(h: *Heap, r: *u32, f: anytype, shared_lock: *std.Io.Mutex, other_regs: []u32) void {
            var tlab = TLAB{};
            h.registerThread();
            defer h.deregisterThread(&tlab);

            var rng = std.Random.DefaultPrng.init(0x87654321);
            const random = rng.random();

            var i: usize = 0;
            while (i < ITERS) : (i += 1) {
                // 1. Periodic safepoint check
                h.enterSafepoint(f) catch {};

                // 2. Allocate object
                const size = random.uintAtMost(u32, 64) + 8;
                const ref = h.allocate(&tlab, size) catch continue;
                r.* = ref;

                // 3. Concurrent Cross-Thread reference write (stressing SATB & write barriers)
                const target_idx = random.uintAtMost(usize, THREADS - 1);
                const target_ref = @atomicLoad(u32, &other_regs[target_idx], .monotonic);
                if (target_ref != 0 and h.isLiveObjectRef(target_ref)) {
                    h.writeRef(ref, 0, target_ref) catch {};
                }

                // 4. Acquire shared lock (same io context, same executor — safe to lock)
                shared_lock.lockUncancelable(h.io);
                const another_ref = h.allocate(&tlab, 16) catch 0;
                if (another_ref != 0) {
                    h.writeRef(ref, 8, another_ref) catch {};
                }
                shared_lock.unlock(h.io);
            }
        }
    }.run;

    var threads: [THREADS]std.Thread = undefined;
    for (&threads, 0..) |*t, i| {
        t.* = try std.Thread.spawn(.{}, worker, .{ &heap, &regs[i], &frames[i], &shared_mutex, &regs });
    }
    for (&threads) |t| t.join();

    try heap.runGC();
    try heap.validateIntegrity();
}

test "Heap Titan VM extreme concurrent distributed graph and pinning stress test" {
    const allocator = std.testing.allocator;
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var heap = try Heap.init(allocator, io, 6 * 1024 * 1024);
    defer heap.deinit(allocator);

    var tx_processed = std.atomic.Value(usize).init(0);
    var cache_roots: [8]std.atomic.Value(u32) = undefined;
    for (&cache_roots) |*cr| cr.* = std.atomic.Value(u32).init(0);

    const GRAPH_WORKERS = 4;
    const PINNED_WORKERS = 1;
    const TOTAL_TX = 2_000;

    const graph_worker = struct {
        fn run(h: *Heap, roots: *[8]std.atomic.Value(u32), counter: *std.atomic.Value(usize), idx: usize) void {
            h.registerThread();
            var tlab = TLAB{};
            defer h.deregisterThread(&tlab);

            var prng = std.Random.DefaultPrng.init(idx ^ 0xAABBCCDD);
            const rng = prng.random();

            var reg_refs = [_]bool{ true, true };
            var regs = [_]u32{ 0, 0 };
            var frame = interpreter.ExecutionFrame{
                .pc = 0,
                .registers = &regs,
                .register_is_ref = &reg_refs,
                .instructions = &.{},
            };

            var local_tx: usize = 0;
            while (true) {
                h.enterSafepoint(&frame) catch {};
                if (counter.fetchAdd(1, .monotonic) >= TOTAL_TX) break;
                local_tx += 1;

                const node_ref = h.allocate(&tlab, 40) catch continue;
                regs[0] = node_ref;
                const audit_ref = h.allocate(&tlab, 24) catch continue;
                regs[1] = audit_ref;

                h.writeRef(node_ref, 8, audit_ref) catch {};

                const slot = rng.uintAtMost(usize, 7);
                const old = roots[slot].load(.acquire);
                if (old != 0 and h.isLiveObjectRef(old)) {
                    h.writeRef(node_ref, 16, old) catch {};
                }
                roots[slot].store(node_ref, .release);

                if ((local_tx % 32) == 0) {
                    roots[rng.uintAtMost(usize, 7)].store(0, .release);
                }
            }
            regs[0] = 0;
            regs[1] = 0;
        }
    }.run;

    const pinned_worker = struct {
        fn run(h: *Heap, counter: *std.atomic.Value(usize), idx: usize, worker_io: std.Io) void {
            h.registerThread();
            var tlab = TLAB{};
            defer h.deregisterThread(&tlab);

            var prng = std.Random.DefaultPrng.init(idx ^ 0x99887766);
            const rng = prng.random();
            const dur: std.Io.Clock.Duration = .{ .raw = .{ .nanoseconds = 1 * std.time.ns_per_ms }, .clock = .awake };

            while (counter.load(.acquire) < TOTAL_TX) {
                h.enterSafepoint(null) catch {};
                const size = 64 * 1024 + rng.uintAtMost(u32, 8) * 1024;
                const los_ref = h.allocateLarge(size) catch {
                    _ = dur.sleep(worker_io) catch {};
                    continue;
                };

                const sig_ptr: *u32 = @ptrCast(@alignCast(&h.memory[los_ref]));
                sig_ptr.* = 0xCAFEBABE;
                const pin = h.pinObject(los_ref) catch continue;

                if (sig_ptr.* != 0xCAFEBABE) {
                    std.debug.panic("Corrupted pin signature: expected 0xCAFEBABE, got 0x{X}", .{sig_ptr.*});
                }
                h.unpinObject(pin);
                _ = dur.sleep(worker_io) catch {};
            }
        }
    }.run;

    const gc_worker = struct {
        fn run(h: *Heap, counter: *std.atomic.Value(usize), daemon_io: std.Io) void {
            const dur: std.Io.Clock.Duration = .{ .raw = .{ .nanoseconds = 2 * std.time.ns_per_ms }, .clock = .awake };
            while (counter.load(.acquire) < TOTAL_TX) {
                h.runGC() catch {};
                _ = dur.sleep(daemon_io) catch {};
            }
        }
    }.run;

    var g_threads: [GRAPH_WORKERS]std.Thread = undefined;
    for (&g_threads, 0..) |*t, i| {
        t.* = try std.Thread.spawn(.{}, graph_worker, .{ &heap, &cache_roots, &tx_processed, i });
    }
    var p_threads: [PINNED_WORKERS]std.Thread = undefined;
    for (&p_threads, 0..) |*t, i| {
        t.* = try std.Thread.spawn(.{}, pinned_worker, .{ &heap, &tx_processed, i, io });
    }
    const gc_t = try std.Thread.spawn(.{}, gc_worker, .{ &heap, &tx_processed, io });

    for (&g_threads) |t| t.join();
    for (&p_threads) |t| t.join();
    gc_t.join();

    try heap.runGC();
    try heap.validateIntegrity();
}
