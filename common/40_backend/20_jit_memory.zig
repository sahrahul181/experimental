//! JIT executable memory/code cache.
//!
//! This layer owns page-backed memory for finalized machine bytes. It copies
//! code while writable, then flips the region to read+execute so the eventual
//! native backend can expose callable function handles without leaking platform
//! memory details into the optimizer.

const std = @import("std");
const register_encoder = @import("register_encoder");
const optimizer = @import("optimizer");
const Instruction = @import("instructions").Instruction;

pub const Error = error{
    EmptyCode,
    InvalidCode,
    InvalidAlignment,
    OutOfMemory,
    ProtectFailed,
    UnknownAllocation,
};

pub const Permission = enum(u8) {
    writable,
    executable,
    released,
};

pub const Stats = struct {
    functions: u32 = 0,
    code_bytes: u32 = 0,
    reserved_bytes: u32 = 0,
    executable_bytes: u32 = 0,
};

pub const Allocation = struct {
    allocator: std.mem.Allocator,
    memory: []align(std.heap.page_size_min) u8,
    code_len: u32,
    permission: Permission,

    pub fn deinit(self: *Allocation) void {
        self.release() catch {};
    }

    pub fn release(self: *Allocation) Error!void {
        if (self.permission == .released) return;
        try makeWritable(self.memory);
        self.allocator.free(self.memory);
        self.memory = &.{};
        self.code_len = 0;
        self.permission = .released;
    }

    pub inline fn bytes(self: *const Allocation) []const u8 {
        return self.memory[0..self.code_len];
    }

    pub inline fn entryAddress(self: *const Allocation) usize {
        return @intFromPtr(self.memory.ptr);
    }

    pub fn typedEntry(self: *const Allocation, comptime Fn: type) *const Fn {
        return @ptrCast(@alignCast(self.memory.ptr));
    }

    pub fn verify(self: *const Allocation) Error!void {
        if (self.permission == .released) return error.EmptyCode;
        if (self.code_len == 0 or self.code_len > self.memory.len) return error.EmptyCode;
        if (!std.mem.isAligned(@intFromPtr(self.memory.ptr), std.heap.page_size_min)) return error.InvalidAlignment;
        if (self.permission != .executable) return error.ProtectFailed;
    }
};

pub const Cache = struct {
    allocator: std.mem.Allocator,
    functions: std.ArrayList(*Allocation) = .empty,
    stats: Stats = .{},

    pub fn init(allocator: std.mem.Allocator) Cache {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Cache) void {
        for (self.functions.items) |function| {
            function.deinit();
            self.allocator.destroy(function);
        }
        self.functions.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn addBytes(self: *Cache, code: []const u8) Error!*Allocation {
        if (code.len == 0) return error.EmptyCode;

        const reserved_len = alignToPage(code.len);
        var memory = try self.allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), reserved_len);
        var executable = false;
        errdefer {
            var can_free = true;
            if (executable) makeWritable(memory) catch {
                can_free = false;
            };
            if (can_free) self.allocator.free(memory);
        }

        @memcpy(memory[0..code.len], code);
        if (reserved_len > code.len) @memset(memory[code.len..], 0xcc);
        try makeExecutable(memory);
        executable = true;

        const allocation = try self.allocator.create(Allocation);
        errdefer self.allocator.destroy(allocation);

        allocation.* = Allocation{
            .allocator = self.allocator,
            .memory = memory,
            .code_len = @intCast(code.len),
            .permission = .executable,
        };
        try self.functions.append(self.allocator, allocation);

        self.stats.functions += 1;
        self.stats.code_bytes += @intCast(code.len);
        self.stats.reserved_bytes += @intCast(reserved_len);
        self.stats.executable_bytes += @intCast(reserved_len);
        return allocation;
    }

    pub fn addEncoded(self: *Cache, encoded: *register_encoder.EncodedFunction) Error!*Allocation {
        const code = encoded.finalize() catch return error.InvalidCode;
        defer encoded.allocator.free(code);
        return try self.addBytes(code);
    }

    /// Release one allocation only after an external code-epoch manager has
    /// proven that no native reader can still execute it. Cache ownership is
    /// removed before page permissions are changed or memory is freed.
    pub fn release(self: *Cache, target: *Allocation) Error!void {
        for (self.functions.items, 0..) |allocation, index| {
            if (allocation != target) continue;
            const code_len = allocation.code_len;
            const reserved_len = allocation.memory.len;
            try allocation.release();
            _ = self.functions.swapRemove(index);
            self.stats.functions -= 1;
            self.stats.code_bytes -= code_len;
            self.stats.reserved_bytes -= @intCast(reserved_len);
            self.stats.executable_bytes -= @intCast(reserved_len);
            self.allocator.destroy(allocation);
            return;
        }
        return error.UnknownAllocation;
    }

    pub fn verify(self: *const Cache) Error!void {
        for (self.functions.items) |function| try function.verify();
    }

    pub fn print(self: *const Cache, writer: anytype) !void {
        try writer.print(
            "jit_memory functions={d} code_bytes={d} reserved_bytes={d} executable_bytes={d}\n",
            .{ self.stats.functions, self.stats.code_bytes, self.stats.reserved_bytes, self.stats.executable_bytes },
        );
        for (self.functions.items, 0..) |function, i| {
            try writer.print(
                "  fn{d} addr=0x{x} code={d} reserved={d} perm={s}\n",
                .{ i, function.entryAddress(), function.code_len, function.memory.len, @tagName(function.permission) },
            );
        }
    }
};

fn alignToPage(len: usize) usize {
    const page = std.heap.pageSize();
    return std.mem.alignForward(usize, len, page);
}

fn makeWritable(memory: []align(std.heap.page_size_min) u8) Error!void {
    std.process.protectMemory(memory, .{ .read = true, .write = true }) catch return error.ProtectFailed;
}

fn makeExecutable(memory: []align(std.heap.page_size_min) u8) Error!void {
    std.process.protectMemory(memory, .{ .read = true, .execute = true }) catch return error.ProtectFailed;
}

test "jit_memory copies bytes into executable allocation" {
    var cache = Cache.init(std.testing.allocator);
    defer cache.deinit();

    const code = [_]u8{ 0x90, 0x90, 0xc3 };
    const allocation = try cache.addBytes(&code);

    try allocation.verify();
    try std.testing.expectEqualSlices(u8, &code, allocation.bytes());
    try std.testing.expect(allocation.memory.len >= code.len);
    try std.testing.expectEqual(Permission.executable, allocation.permission);
}

test "jit_memory rejects empty code" {
    var cache = Cache.init(std.testing.allocator);
    defer cache.deinit();

    try std.testing.expectError(error.EmptyCode, cache.addBytes(&.{}));
}

test "jit_memory loads finalized register encoder output" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .return_void,
    };
    var optimized = try optimizer.optimize(std.testing.allocator, &insts, &.{}, .{});
    defer optimized.deinit();

    var encoded = try register_encoder.encode(std.testing.allocator, &optimized.machine);
    defer encoded.deinit();

    var cache = Cache.init(std.testing.allocator);
    defer cache.deinit();

    const allocation = try cache.addEncoded(&encoded);
    try allocation.verify();
    try cache.verify();
    try std.testing.expect(allocation.bytes().len > 0);
    try std.testing.expectEqual(@as(u32, 1), cache.stats.functions);
}

test "jit_memory deinit releases allocations" {
    var cache = Cache.init(std.testing.allocator);
    const code = [_]u8{ 1, 2, 3, 4 };
    _ = try cache.addBytes(&code);
    cache.deinit();
}

test "jit_memory releases one epoch-safe allocation" {
    var cache = Cache.init(std.testing.allocator);
    defer cache.deinit();
    const first = try cache.addBytes(&.{ 0x90, 0xc3 });
    const second = try cache.addBytes(&.{0xc3});
    try cache.release(first);
    try std.testing.expectEqual(@as(u32, 1), cache.stats.functions);
    try std.testing.expectEqual(@as(u32, 1), cache.stats.code_bytes);
    try second.verify();
}

test "jit_memory print helper emits stable summary" {
    var cache = Cache.init(std.testing.allocator);
    defer cache.deinit();

    const code = [_]u8{ 0x90, 0xc3 };
    _ = try cache.addBytes(&code);

    var storage: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&storage);
    try cache.print(&stream);
    const output = stream.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "jit_memory functions=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "perm=executable") != null);
}
