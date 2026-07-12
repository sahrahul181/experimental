const std = @import("std");
const parser = @import("parser");
const print = @import("print");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try init.minimal.args.iterateAllocator(allocator);
    defer args.deinit();

    _ = args.next(); // skip executable name

    const file_path = args.next() orelse {
        std.debug.print("Usage: dex_printer <path_to_dex_file>\n", .{});
        std.process.exit(1);
    };

    const max_size: std.Io.Limit = @enumFromInt(100 * 1024 * 1024); // 100 MB max
    const bytes = std.Io.Dir.cwd().readFileAlloc(init.io, file_path, allocator, max_size) catch |err| {
        std.debug.print("Failed to read file {s}: {s}\n", .{ file_path, @errorName(err) });
        std.process.exit(1);
    };
    defer allocator.free(bytes);

    // Arena for the DEX parser structures
    const arena = init.arena.allocator();

    const dex = parser.parse(arena, bytes) catch |err| {
        std.debug.print("Failed to parse DEX file: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };

    var out_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writerStreaming(init.io, &out_buf);

    try print.printDex(&stdout_w.interface, dex, allocator);
    try stdout_w.flush();
}
