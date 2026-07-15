//! Stable VM-thread registration and asynchronous per-thread root handshakes.

const std = @import("std");
const Handle = @import("runtime_value").Handle;

pub const ThreadState = enum(u8) {
    detached,
    running,
    publishing,
    blocked,
    passive_publishing,
    exiting,
};

pub const Error = error{
    AlreadyRegistered,
    HandshakeInProgress,
    InvalidState,
    MemberCapacityExceeded,
    MemberBufferTooSmall,
    MemberRejected,
    NotRegistered,
    RootCapacityExceeded,
    Shutdown,
    SnapshotNotReady,
    ThreadsStillRegistered,
    EpochExhausted,
};

pub const MemberValidator = struct {
    context: *anyopaque,
    validate: *const fn (*anyopaque, *ThreadContext) bool,
};

pub const ThreadContext = struct {
    allocator: std.mem.Allocator,
    state: std.atomic.Value(ThreadState),
    observed_epoch: std.atomic.Value(u64),
    root_ack_epoch: std.atomic.Value(u64),
    release_epoch: std.atomic.Value(u64),

    /// Mutator-owned while running; immutable while blocked/publishing.
    root_slots: std.ArrayList(*const Handle),
    snapshots: [2][]Handle,
    snapshot_counts: [2]std.atomic.Value(usize),
    published_snapshot: std.atomic.Value(u8),

    wait_mutex: std.Io.Mutex,
    wait_condition: std.Io.Condition,

    pub fn init(
        allocator: std.mem.Allocator,
        max_roots: usize,
    ) std.mem.Allocator.Error!ThreadContext {
        var root_slots = try std.ArrayList(*const Handle).initCapacity(allocator, max_roots);
        errdefer root_slots.deinit(allocator);
        const first = try allocator.alloc(Handle, max_roots);
        errdefer allocator.free(first);
        const second = try allocator.alloc(Handle, max_roots);
        errdefer allocator.free(second);

        return .{
            .allocator = allocator,
            .state = std.atomic.Value(ThreadState).init(.detached),
            .observed_epoch = std.atomic.Value(u64).init(0),
            .root_ack_epoch = std.atomic.Value(u64).init(0),
            .release_epoch = std.atomic.Value(u64).init(0),
            .root_slots = root_slots,
            .snapshots = .{ first, second },
            .snapshot_counts = .{
                std.atomic.Value(usize).init(0),
                std.atomic.Value(usize).init(0),
            },
            .published_snapshot = std.atomic.Value(u8).init(0),
            .wait_mutex = .init,
            .wait_condition = .init,
        };
    }

    pub fn deinit(self: *ThreadContext) void {
        std.debug.assert(self.state.load(.acquire) == .detached);
        self.root_slots.deinit(self.allocator);
        self.allocator.free(self.snapshots[0]);
        self.allocator.free(self.snapshots[1]);
        self.* = undefined;
    }

    /// Root registration is owner-only and allocation-free after init.
    pub fn addRoot(self: *ThreadContext, slot: *const Handle) Error!void {
        const state = self.state.load(.acquire);
        if (state != .detached and state != .running) return error.InvalidState;
        self.root_slots.appendBounded(slot) catch return error.RootCapacityExceeded;
    }

    pub fn clearRoots(self: *ThreadContext) Error!void {
        const state = self.state.load(.acquire);
        if (state != .detached and state != .running) return error.InvalidState;
        self.root_slots.clearRetainingCapacity();
    }

    /// Opens an allocation-free, owner-thread root scope. Slow runtime paths
    /// use this before polling so a register-only handle becomes visible to a
    /// concurrent collector. Scopes are strictly nested and must not escape
    /// the owner thread.
    pub fn beginRootScope(self: *ThreadContext) Error!RootScope {
        const current = self.state.load(.acquire);
        if (current != .detached and current != .running) return error.InvalidState;
        return .{ .context = self, .mark = self.root_slots.items.len };
    }

    pub fn isRunning(self: *const ThreadContext) bool {
        return self.state.load(.acquire) == .running;
    }

    pub fn observedEpoch(self: *const ThreadContext) u64 {
        return self.observed_epoch.load(.acquire);
    }

    pub fn rootCount(self: *const ThreadContext) usize {
        return self.root_slots.items.len;
    }

    fn publishRoots(self: *ThreadContext, epoch: u64) void {
        const previous = self.published_snapshot.load(.monotonic);
        const target: u8 = previous ^ 1;
        const snapshot_buffer = self.snapshots[target];
        std.debug.assert(self.root_slots.items.len <= snapshot_buffer.len);
        for (self.root_slots.items, 0..) |slot, index| snapshot_buffer[index] = slot.*;

        self.snapshot_counts[target].store(self.root_slots.items.len, .monotonic);
        self.published_snapshot.store(target, .monotonic);
        self.observed_epoch.store(epoch, .monotonic);
        // Releases all root copies and metadata above to the collector.
        self.root_ack_epoch.store(epoch, .release);
    }

    fn snapshot(self: *const ThreadContext, epoch: u64) Error![]const Handle {
        if (self.root_ack_epoch.load(.acquire) != epoch) return error.SnapshotNotReady;
        const index = self.published_snapshot.load(.monotonic);
        const count = self.snapshot_counts[index].load(.monotonic);
        return self.snapshots[index][0..count];
    }

    fn waitUntilReleased(self: *ThreadContext, io: std.Io, epoch: u64) Error!void {
        self.wait_mutex.lockUncancelable(io);
        defer self.wait_mutex.unlock(io);
        while (self.release_epoch.load(.acquire) < epoch) {
            self.wait_condition.waitUncancelable(io, &self.wait_mutex);
        }
    }
};

pub const RootScope = struct {
    context: *ThreadContext,
    mark: usize,
    added: usize = 0,
    active: bool = true,

    pub fn add(self: *RootScope, slot: *const Handle) Error!void {
        if (!self.active) return error.InvalidState;
        try self.context.addRoot(slot);
        self.added += 1;
    }

    pub fn deinit(self: *RootScope) void {
        if (!self.active) return;
        std.debug.assert(self.context.root_slots.items.len == self.mark + self.added);
        self.context.root_slots.shrinkRetainingCapacity(self.mark);
        self.active = false;
    }
};

pub const Registry = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    membership_mutex: std.Io.Mutex,
    members: std.ArrayList(*ThreadContext),
    request_epoch: std.atomic.Value(u64),
    active_epoch: std.atomic.Value(u64),
    shutdown: std.atomic.Value(bool),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        max_threads: usize,
    ) std.mem.Allocator.Error!Registry {
        return .{
            .allocator = allocator,
            .io = io,
            .membership_mutex = .init,
            .members = try std.ArrayList(*ThreadContext).initCapacity(allocator, max_threads),
            .request_epoch = std.atomic.Value(u64).init(0),
            .active_epoch = std.atomic.Value(u64).init(0),
            .shutdown = std.atomic.Value(bool).init(false),
        };
    }

    pub fn deinit(self: *Registry) Error!void {
        self.membership_mutex.lockUncancelable(self.io);
        if (self.members.items.len != 0) {
            self.membership_mutex.unlock(self.io);
            return error.ThreadsStillRegistered;
        }
        self.membership_mutex.unlock(self.io);
        self.members.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn register(self: *Registry, context: *ThreadContext) Error!void {
        self.membership_mutex.lockUncancelable(self.io);
        defer self.membership_mutex.unlock(self.io);
        if (self.shutdown.load(.acquire)) return error.Shutdown;
        if (self.active_epoch.load(.acquire) != 0) return error.HandshakeInProgress;
        if (context.state.load(.acquire) != .detached) return error.AlreadyRegistered;
        for (self.members.items) |member| {
            if (member == context) return error.AlreadyRegistered;
        }

        const epoch = self.request_epoch.load(.acquire);
        context.observed_epoch.store(epoch, .monotonic);
        context.root_ack_epoch.store(epoch, .monotonic);
        context.release_epoch.store(epoch, .monotonic);
        self.members.appendBounded(context) catch return error.MemberCapacityExceeded;
        context.state.store(.running, .release);
    }

    pub fn unregister(self: *Registry, context: *ThreadContext) Error!void {
        self.membership_mutex.lockUncancelable(self.io);
        defer self.membership_mutex.unlock(self.io);
        if (self.active_epoch.load(.acquire) != 0) return error.HandshakeInProgress;

        var found: ?usize = null;
        for (self.members.items, 0..) |member, index| {
            if (member == context) {
                found = index;
                break;
            }
        }
        const index = found orelse return error.NotRegistered;
        const state = context.state.load(.acquire);
        if (state != .running and state != .blocked) return error.InvalidState;
        context.state.store(.exiting, .release);
        _ = self.members.swapRemove(index);
        context.state.store(.detached, .release);
    }

    /// Copies a stable membership set into caller-owned storage, publishes the
    /// new request epoch, and passively snapshots already-blocked threads.
    pub fn beginHandshake(self: *Registry, storage: []*ThreadContext) Error!Handshake {
        return self.beginValidatedHandshake(storage, null);
    }

    /// Runs an optional subsystem validator while membership is stable and
    /// before publishing a request epoch. Rejection leaves registry state
    /// unchanged, so callers cannot strand mutators in a partial handshake.
    pub fn beginValidatedHandshake(
        self: *Registry,
        storage: []*ThreadContext,
        validator: ?MemberValidator,
    ) Error!Handshake {
        self.membership_mutex.lockUncancelable(self.io);
        if (self.shutdown.load(.acquire)) {
            self.membership_mutex.unlock(self.io);
            return error.Shutdown;
        }
        if (self.active_epoch.load(.acquire) != 0) {
            self.membership_mutex.unlock(self.io);
            return error.HandshakeInProgress;
        }
        if (storage.len < self.members.items.len) {
            self.membership_mutex.unlock(self.io);
            return error.MemberBufferTooSmall;
        }
        if (validator) |member_validator| {
            for (self.members.items) |member| {
                if (!member_validator.validate(member_validator.context, member)) {
                    self.membership_mutex.unlock(self.io);
                    return error.MemberRejected;
                }
            }
        }

        const member_count = self.members.items.len;
        @memcpy(storage[0..member_count], self.members.items);
        const previous_epoch = self.request_epoch.load(.acquire);
        if (previous_epoch == std.math.maxInt(u64)) {
            self.membership_mutex.unlock(self.io);
            return error.EpochExhausted;
        }
        const epoch = previous_epoch + 1;
        self.request_epoch.store(epoch, .release);
        self.active_epoch.store(epoch, .release);
        self.membership_mutex.unlock(self.io);

        var handshake = Handshake{
            .registry = self,
            .epoch = epoch,
            .members = storage[0..member_count],
            .finished = false,
        };
        for (handshake.members) |member| handshake.tryPublishPassive(member);
        return handshake;
    }

    /// Mutator poll. The fast path is one acquire load and one comparison.
    pub fn poll(self: *Registry, context: *ThreadContext) Error!bool {
        const requested = self.request_epoch.load(.acquire);
        if (context.observed_epoch.load(.monotonic) == requested) {
            std.debug.assert(context.state.load(.monotonic) == .running);
            return false;
        }
        if (self.shutdown.load(.acquire)) return error.Shutdown;

        if (context.state.cmpxchgStrong(
            .running,
            .publishing,
            .acq_rel,
            .acquire,
        ) != null) return error.InvalidState;

        context.publishRoots(requested);
        try context.waitUntilReleased(self.io, requested);
        std.debug.assert(context.state.load(.acquire) == .running);
        return true;
    }

    /// Makes roots stable before a potentially long external/blocking call.
    pub fn enterBlocked(self: *Registry, context: *ThreadContext) Error!void {
        while (true) {
            _ = try self.poll(context);
            if (context.state.cmpxchgStrong(
                .running,
                .blocked,
                .acq_rel,
                .acquire,
            ) != null) return error.InvalidState;

            const requested = self.request_epoch.load(.acquire);
            if (context.observed_epoch.load(.acquire) == requested) return;

            // A handshake raced the transition. Return to running and poll, or
            // wait if the collector already claimed the passive snapshot.
            if (context.state.cmpxchgStrong(
                .blocked,
                .running,
                .acq_rel,
                .acquire,
            ) == null) continue;

            try context.waitUntilReleased(self.io, requested);
            std.debug.assert(context.state.load(.acquire) == .blocked);
            return;
        }
    }

    pub fn leaveBlocked(self: *Registry, context: *ThreadContext) Error!void {
        while (true) {
            if (context.state.cmpxchgStrong(
                .blocked,
                .running,
                .acq_rel,
                .acquire,
            ) == null) return;

            if (context.state.load(.acquire) != .passive_publishing) return error.InvalidState;
            const epoch = context.root_ack_epoch.load(.acquire);
            try context.waitUntilReleased(self.io, epoch);
        }
    }

    pub fn requestEpoch(self: *const Registry) u64 {
        return self.request_epoch.load(.acquire);
    }

    /// Immutable registration bound used to preallocate collector handshake
    /// storage. Runtime slow paths never grow this list.
    pub fn memberCapacity(self: *const Registry) usize {
        return self.members.capacity;
    }
};

pub const Handshake = struct {
    registry: *Registry,
    epoch: u64,
    members: []*ThreadContext,
    finished: bool,

    fn tryPublishPassive(self: *Handshake, context: *ThreadContext) void {
        if (context.state.cmpxchgStrong(
            .blocked,
            .passive_publishing,
            .acq_rel,
            .acquire,
        ) == null) {
            context.publishRoots(self.epoch);
        }
    }

    pub fn isReady(self: *const Handshake, context: *const ThreadContext) bool {
        if (!self.contains(context)) return false;
        return context.root_ack_epoch.load(.acquire) == self.epoch;
    }

    pub fn snapshot(self: *const Handshake, context: *const ThreadContext) Error![]const Handle {
        if (!self.contains(context)) return error.NotRegistered;
        return context.snapshot(self.epoch);
    }

    pub fn release(self: *Handshake, context: *ThreadContext) Error!void {
        if (!self.contains(context)) return error.NotRegistered;
        if (context.root_ack_epoch.load(.acquire) != self.epoch) return error.SnapshotNotReady;

        context.wait_mutex.lockUncancelable(self.registry.io);
        defer context.wait_mutex.unlock(self.registry.io);
        switch (context.state.load(.acquire)) {
            .publishing => context.state.store(.running, .release),
            .passive_publishing => context.state.store(.blocked, .release),
            else => return error.InvalidState,
        }
        context.release_epoch.store(self.epoch, .release);
        context.wait_condition.broadcast(self.registry.io);
    }

    pub fn finish(self: *Handshake) Error!void {
        if (self.finished) return error.InvalidState;
        for (self.members) |member| {
            if (member.release_epoch.load(.acquire) < self.epoch) return error.SnapshotNotReady;
        }

        self.registry.membership_mutex.lockUncancelable(self.registry.io);
        defer self.registry.membership_mutex.unlock(self.registry.io);
        if (self.registry.active_epoch.load(.acquire) != self.epoch) return error.InvalidState;
        self.registry.active_epoch.store(0, .release);
        self.finished = true;
    }

    fn contains(self: *const Handshake, context: *const ThreadContext) bool {
        for (self.members) |member| if (member == context) return true;
        return false;
    }
};

fn waitReady(handshake: *const Handshake, context: *const ThreadContext) !void {
    for (0..1_000_000) |_| {
        if (handshake.isReady(context)) return;
        std.atomic.spinLoopHint();
        std.Thread.yield() catch {};
    }
    return error.Timeout;
}

test "running mutator publishes exact roots and resumes independently" {
    var registry = try Registry.init(std.testing.allocator, std.testing.io, 2);
    defer registry.deinit() catch unreachable;
    var context = try ThreadContext.init(std.testing.allocator, 4);
    defer context.deinit();

    var first = Handle{ .index = 11, .generation = 2 };
    var second = Handle{ .index = 12, .generation = 3 };
    try context.addRoot(&first);
    try context.addRoot(&second);
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;

    const Worker = struct {
        registry: *Registry,
        context: *ThreadContext,
        completed: *std.atomic.Value(bool),

        fn run(self: *@This()) void {
            while (self.registry.requestEpoch() == 0) std.atomic.spinLoopHint();
            _ = self.registry.poll(self.context) catch return;
            self.completed.store(true, .release);
        }
    };
    var completed = std.atomic.Value(bool).init(false);
    var worker = Worker{ .registry = &registry, .context = &context, .completed = &completed };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});

    var members: [2]*ThreadContext = undefined;
    var handshake = try registry.beginHandshake(&members);
    try waitReady(&handshake, &context);
    const snapshot = try handshake.snapshot(&context);
    try std.testing.expectEqualSlices(Handle, &.{ first, second }, snapshot);
    try handshake.release(&context);
    try handshake.finish();
    thread.join();
    try std.testing.expect(completed.load(.acquire));
}

test "blocked mutator is passively snapshotted without a global pause" {
    var registry = try Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try ThreadContext.init(std.testing.allocator, 2);
    defer context.deinit();

    var root = Handle{ .index = 99, .generation = 7 };
    try context.addRoot(&root);
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;
    try registry.enterBlocked(&context);

    var members: [1]*ThreadContext = undefined;
    var handshake = try registry.beginHandshake(&members);
    try std.testing.expect(handshake.isReady(&context));
    try std.testing.expectEqualSlices(Handle, &.{root}, try handshake.snapshot(&context));
    try handshake.release(&context);
    try handshake.finish();
    try registry.leaveBlocked(&context);
}

test "root capacity and overlapping handshakes fail without state corruption" {
    var registry = try Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try ThreadContext.init(std.testing.allocator, 1);
    defer context.deinit();

    var root = Handle{ .index = 1, .generation = 1 };
    try context.addRoot(&root);
    try std.testing.expectError(error.RootCapacityExceeded, context.addRoot(&root));
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;

    try registry.enterBlocked(&context);
    var members: [1]*ThreadContext = undefined;
    var first = try registry.beginHandshake(&members);
    try std.testing.expectError(error.HandshakeInProgress, registry.beginHandshake(&members));
    try first.release(&context);
    try first.finish();
    try registry.leaveBlocked(&context);
}

test "validated handshake rejection publishes no epoch" {
    var registry = try Registry.init(std.testing.allocator, std.testing.io, 1);
    defer registry.deinit() catch unreachable;
    var context = try ThreadContext.init(std.testing.allocator, 0);
    defer context.deinit();
    try registry.register(&context);
    defer registry.unregister(&context) catch unreachable;

    const Reject = struct {
        fn member(_: *anyopaque, _: *ThreadContext) bool {
            return false;
        }
    };
    var validation_context: u8 = 0;
    var members: [1]*ThreadContext = undefined;
    try std.testing.expectError(
        error.MemberRejected,
        registry.beginValidatedHandshake(&members, .{
            .context = @ptrCast(&validation_context),
            .validate = Reject.member,
        }),
    );
    try std.testing.expectEqual(@as(u64, 0), registry.requestEpoch());
    try std.testing.expect(context.isRunning());
}

fn allocationFailureContextInit(allocator: std.mem.Allocator) !void {
    var context = try ThreadContext.init(allocator, 8);
    defer context.deinit();
}

fn allocationFailureRegistryInit(allocator: std.mem.Allocator) !void {
    var registry = try Registry.init(allocator, std.testing.io, 8);
    defer registry.deinit() catch unreachable;
}

test "thread metadata initialization is leak-free at every allocation failure" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        allocationFailureContextInit,
        .{},
    );
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        allocationFailureRegistryInit,
        .{},
    );
}
