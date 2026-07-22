//! Immutable synchronized-method monitor metadata.
//!
//! Instance methods name a logical parameter containing the receiver Handle.
//! Static methods name a collector-registered root slot containing the class
//! object Handle. The table is borrowed, sorted, and allocation-free at lookup.

const std = @import("std");
const runtime_gc = @import("runtime_gc");
const runtime_value = @import("runtime_value");

const Handle = runtime_value.Handle;

pub const Target = union(enum) {
    instance_parameter: u16,
    static_root_slot: usize,
};

pub const Entry = struct {
    method_id: u32,
    target: Target,
};

pub const Error = runtime_value.Error || error{
    InvalidSynchronizationMetadata,
};

pub const Table = struct {
    entries: []const Entry,
    collector: *const runtime_gc.ConcurrentCollector,

    pub fn init(entries: []const Entry, collector: *const runtime_gc.ConcurrentCollector) Error!Table {
        for (entries, 0..) |entry, index| {
            if (index != 0 and entries[index - 1].method_id >= entry.method_id) {
                return error.InvalidSynchronizationMetadata;
            }
            switch (entry.target) {
                .instance_parameter => {},
                .static_root_slot => |address| {
                    if (!std.mem.isAligned(address, @alignOf(std.atomic.Value(u64))) or
                        !collector.isStaticRootSlot(address))
                    {
                        return error.InvalidSynchronizationMetadata;
                    }
                    _ = try loadStaticHandle(collector, address);
                },
            }
        }
        return .{ .entries = entries, .collector = collector };
    }

    pub fn find(self: *const Table, method_id: u32) ?Entry {
        var low: usize = 0;
        var high = self.entries.len;
        while (low < high) {
            const middle = low + (high - low) / 2;
            const candidate = self.entries[middle];
            if (candidate.method_id < method_id) {
                low = middle + 1;
            } else if (candidate.method_id > method_id) {
                high = middle;
            } else return candidate;
        }
        return null;
    }

    pub fn loadStatic(self: *const Table, entry: Entry) Error!Handle {
        return switch (entry.target) {
            .static_root_slot => |address| loadStaticHandle(self.collector, address),
            .instance_parameter => error.InvalidSynchronizationMetadata,
        };
    }

    pub fn collectorDomain(self: *const Table) *const runtime_gc.ConcurrentCollector {
        return self.collector;
    }
};

fn loadStaticHandle(collector: *const runtime_gc.ConcurrentCollector, address: usize) Error!Handle {
    const slot: *const std.atomic.Value(u64) = @ptrFromInt(address);
    const handle: Handle = @bitCast(slot.load(.acquire));
    if (handle.isNull()) return error.InvalidSynchronizationMetadata;
    const location = try collector.handleTable().inspect(handle);
    switch (location.state) {
        .live, .evacuating => return handle,
        else => return error.InvalidSynchronizationMetadata,
    }
}
