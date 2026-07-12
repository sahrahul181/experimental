//! Machine-code byte buffer.
//!
//! Target encoders use this module to emit bytes, bind labels, and register
//! relocations for branches or absolute addresses. It deliberately avoids
//! owning executable memory; the next layer can copy finalized bytes into a
//! platform-specific executable allocation.

const std = @import("std");

pub const LabelId = u32;
pub const INVALID_LABEL: LabelId = std.math.maxInt(LabelId);

pub const Error = error{
    BadLabel,
    BadRelocation,
    DuplicateLabel,
    OutOfMemory,
    RelocationOutOfRange,
    UnboundLabel,
};

pub const RelocKind = enum(u8) {
    rel8,
    rel16,
    rel32,
    abs32,
    abs64,

    pub inline fn width(self: RelocKind) u8 {
        return switch (self) {
            .rel8 => 1,
            .rel16 => 2,
            .rel32, .abs32 => 4,
            .abs64 => 8,
        };
    }

    pub inline fn relative(self: RelocKind) bool {
        return switch (self) {
            .rel8, .rel16, .rel32 => true,
            .abs32, .abs64 => false,
        };
    }
};

pub const Label = struct {
    offset: u32 = 0,
    bound: bool = false,
};

pub const Relocation = struct {
    offset: u32,
    label: LabelId,
    kind: RelocKind,
    addend: i64 = 0,
};

pub const Stats = struct {
    bytes: u32 = 0,
    labels: u32 = 0,
    bound_labels: u32 = 0,
    relocations: u32 = 0,
    patched: u32 = 0,
    align_padding: u32 = 0,
};

pub const Buffer = struct {
    allocator: std.mem.Allocator,
    bytes: std.ArrayList(u8) = .empty,
    labels: std.ArrayList(Label) = .empty,
    relocs: std.ArrayList(Relocation) = .empty,
    patched: bool = false,
    stats: Stats = .{},

    pub fn init(allocator: std.mem.Allocator) Buffer {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Buffer) void {
        self.relocs.deinit(self.allocator);
        self.labels.deinit(self.allocator);
        self.bytes.deinit(self.allocator);
        self.* = undefined;
    }

    pub inline fn len(self: *const Buffer) u32 {
        return @intCast(self.bytes.items.len);
    }

    pub inline fn slice(self: *const Buffer) []const u8 {
        return self.bytes.items;
    }

    pub fn newLabel(self: *Buffer) Error!LabelId {
        const id: LabelId = @intCast(self.labels.items.len);
        try self.labels.append(self.allocator, .{});
        self.stats.labels += 1;
        return id;
    }

    pub fn bindLabel(self: *Buffer, label: LabelId) Error!void {
        if (label >= self.labels.items.len) return error.BadLabel;
        if (self.labels.items[label].bound) return error.DuplicateLabel;
        self.labels.items[label] = .{ .offset = self.len(), .bound = true };
        self.stats.bound_labels += 1;
        self.patched = false;
    }

    pub fn emitU8(self: *Buffer, value: u8) Error!void {
        try self.bytes.append(self.allocator, value);
        self.stats.bytes = self.len();
        self.patched = false;
    }

    pub fn emitU16(self: *Buffer, value: u16) Error!void {
        const start = self.bytes.items.len;
        try self.bytes.appendNTimes(self.allocator, 0, 2);
        std.mem.writeInt(u16, self.bytes.items[start..][0..2], value, .little);
        self.stats.bytes = self.len();
        self.patched = false;
    }

    pub fn emitU32(self: *Buffer, value: u32) Error!void {
        const start = self.bytes.items.len;
        try self.bytes.appendNTimes(self.allocator, 0, 4);
        std.mem.writeInt(u32, self.bytes.items[start..][0..4], value, .little);
        self.stats.bytes = self.len();
        self.patched = false;
    }

    pub fn emitU64(self: *Buffer, value: u64) Error!void {
        const start = self.bytes.items.len;
        try self.bytes.appendNTimes(self.allocator, 0, 8);
        std.mem.writeInt(u64, self.bytes.items[start..][0..8], value, .little);
        self.stats.bytes = self.len();
        self.patched = false;
    }

    pub fn emitBytes(self: *Buffer, data: []const u8) Error!void {
        try self.bytes.appendSlice(self.allocator, data);
        self.stats.bytes = self.len();
        self.patched = false;
    }

    pub fn alignTo(self: *Buffer, alignment: u32, fill: u8) Error!void {
        if (alignment == 0 or !std.math.isPowerOfTwo(alignment)) return error.BadRelocation;
        const mask = alignment - 1;
        const padding = (alignment - (self.len() & mask)) & mask;
        try self.bytes.appendNTimes(self.allocator, fill, padding);
        self.stats.align_padding += padding;
        self.stats.bytes = self.len();
        self.patched = false;
    }

    pub fn reserve(self: *Buffer, count: u32, fill: u8) Error!u32 {
        const offset = self.len();
        try self.bytes.appendNTimes(self.allocator, fill, count);
        self.stats.bytes = self.len();
        self.patched = false;
        return offset;
    }

    pub fn reloc(self: *Buffer, label: LabelId, kind: RelocKind, addend: i64) Error!u32 {
        if (label >= self.labels.items.len) return error.BadLabel;
        const offset = try self.reserve(kind.width(), 0);
        try self.relocs.append(self.allocator, .{
            .offset = offset,
            .label = label,
            .kind = kind,
            .addend = addend,
        });
        self.stats.relocations += 1;
        return offset;
    }

    pub fn patchAll(self: *Buffer) Error!void {
        for (self.relocs.items) |rel| try self.patchOne(rel);
        self.stats.patched = self.stats.relocations;
        self.patched = true;
    }

    pub fn verify(self: *Buffer) Error!void {
        for (self.relocs.items) |rel| {
            if (rel.label >= self.labels.items.len) return error.BadLabel;
            if (rel.offset + rel.kind.width() > self.bytes.items.len) return error.BadRelocation;
            _ = try self.relocationValue(rel);
        }
    }

    pub fn finalize(self: *Buffer) Error![]u8 {
        try self.verify();
        try self.patchAll();
        return try self.allocator.dupe(u8, self.bytes.items);
    }

    pub fn reset(self: *Buffer) void {
        self.bytes.clearRetainingCapacity();
        self.labels.clearRetainingCapacity();
        self.relocs.clearRetainingCapacity();
        self.patched = false;
        self.stats = .{};
    }

    pub fn print(self: *const Buffer, writer: anytype) !void {
        try writer.print(
            "code_buffer bytes={d} labels={d} bound={d} relocs={d} patched={d} align_padding={d}\n",
            .{
                self.bytes.items.len,
                self.labels.items.len,
                self.stats.bound_labels,
                self.relocs.items.len,
                self.stats.patched,
                self.stats.align_padding,
            },
        );
        try writer.print("labels:\n", .{});
        for (self.labels.items, 0..) |label, i| {
            if (label.bound) {
                try writer.print("  L{d} @{d}\n", .{ i, label.offset });
            } else {
                try writer.print("  L{d} <unbound>\n", .{i});
            }
        }
        try writer.print("relocations:\n", .{});
        for (self.relocs.items) |rel| {
            try writer.print("  @{d} {s} L{d} addend={d}\n", .{ rel.offset, @tagName(rel.kind), rel.label, rel.addend });
        }
        try writer.print("bytes:", .{});
        for (self.bytes.items) |byte| try writer.print(" {x:0>2}", .{byte});
        try writer.print("\n", .{});
    }

    fn relocationValue(self: *const Buffer, rel: Relocation) Error!i64 {
        const label = self.labels.items[rel.label];
        if (!label.bound) return error.UnboundLabel;

        var value: i64 = @intCast(label.offset);
        value += rel.addend;
        if (rel.kind.relative()) value -= @as(i64, @intCast(rel.offset + rel.kind.width()));

        switch (rel.kind) {
            .rel8 => if (value < std.math.minInt(i8) or value > std.math.maxInt(i8)) return error.RelocationOutOfRange,
            .rel16 => if (value < std.math.minInt(i16) or value > std.math.maxInt(i16)) return error.RelocationOutOfRange,
            .rel32 => if (value < std.math.minInt(i32) or value > std.math.maxInt(i32)) return error.RelocationOutOfRange,
            .abs32 => if (value < 0 or value > std.math.maxInt(u32)) return error.RelocationOutOfRange,
            .abs64 => if (value < 0) return error.RelocationOutOfRange,
        }
        return value;
    }

    fn patchOne(self: *Buffer, rel: Relocation) Error!void {
        const value = try self.relocationValue(rel);
        const start: usize = rel.offset;
        switch (rel.kind) {
            .rel8 => self.bytes.items[start] = @bitCast(@as(i8, @intCast(value))),
            .rel16 => std.mem.writeInt(i16, self.bytes.items[start..][0..2], @intCast(value), .little),
            .rel32 => std.mem.writeInt(i32, self.bytes.items[start..][0..4], @intCast(value), .little),
            .abs32 => std.mem.writeInt(u32, self.bytes.items[start..][0..4], @intCast(value), .little),
            .abs64 => std.mem.writeInt(u64, self.bytes.items[start..][0..8], @intCast(value), .little),
        }
    }
};

test "code_buffer emits integers and alignment padding" {
    var buf = Buffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.emitU8(0xaa);
    try buf.emitU16(0x1122);
    try buf.alignTo(8, 0xcc);
    try std.testing.expectEqual(@as(u32, 8), buf.len());
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0xaa, 0x22, 0x11, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc }, buf.slice());
    try std.testing.expectEqual(@as(u32, 5), buf.stats.align_padding);
}

test "code_buffer patches forward and backward relative labels" {
    var buf = Buffer.init(std.testing.allocator);
    defer buf.deinit();

    const start = try buf.newLabel();
    const end = try buf.newLabel();
    try buf.bindLabel(start);
    try buf.emitU8(0xeb);
    _ = try buf.reloc(end, .rel8, 0);
    try buf.emitU8(0x90);
    try buf.bindLabel(end);
    try buf.emitU8(0xeb);
    _ = try buf.reloc(start, .rel8, 0);
    try buf.patchAll();

    try std.testing.expectEqualSlices(u8, &[_]u8{ 0xeb, 0x01, 0x90, 0xeb, 0xfb }, buf.slice());
}

test "code_buffer patches absolute relocations" {
    var buf = Buffer.init(std.testing.allocator);
    defer buf.deinit();

    const label = try buf.newLabel();
    _ = try buf.reloc(label, .abs32, 4);
    try buf.emitBytes(&[_]u8{ 1, 2, 3, 4 });
    try buf.bindLabel(label);
    try buf.patchAll();

    try std.testing.expectEqual(@as(u32, 12), std.mem.readInt(u32, buf.slice()[0..4], .little));
}

test "code_buffer rejects unbound and out of range relocations" {
    var unbound = Buffer.init(std.testing.allocator);
    defer unbound.deinit();
    const missing = try unbound.newLabel();
    _ = try unbound.reloc(missing, .rel8, 0);
    try std.testing.expectError(error.UnboundLabel, unbound.verify());

    var far = Buffer.init(std.testing.allocator);
    defer far.deinit();
    const target = try far.newLabel();
    _ = try far.reloc(target, .rel8, 0);
    _ = try far.reserve(200, 0x90);
    try far.bindLabel(target);
    try std.testing.expectError(error.RelocationOutOfRange, far.verify());
}

test "code_buffer finalize returns patched owned bytes" {
    var buf = Buffer.init(std.testing.allocator);
    defer buf.deinit();

    const label = try buf.newLabel();
    try buf.emitU8(0xeb);
    _ = try buf.reloc(label, .rel8, 0);
    try buf.bindLabel(label);

    const out = try buf.finalize();
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0xeb, 0x00 }, out);
    try std.testing.expect(buf.patched);
}

test "code_buffer print helper emits stable summary" {
    var buf = Buffer.init(std.testing.allocator);
    defer buf.deinit();

    const label = try buf.newLabel();
    try buf.bindLabel(label);
    try buf.emitU32(0x12345678);

    var storage: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&storage);
    try buf.print(&stream);
    const output = stream.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "code_buffer bytes=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "L0 @0") != null);
}
