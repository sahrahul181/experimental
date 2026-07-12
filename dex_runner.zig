const std = @import("std");
const parser = @import("parser");
const interpreter = @import("interpreter");

// Embed the core DEX directly into the CLI binary
const core_dex_bytes = @embedFile("core.dex");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try init.minimal.args.iterateAllocator(allocator);
    defer args.deinit();

    _ = args.next(); // skip executable name

    const user_dex_path = args.next() orelse {
        printUsage();
        std.process.exit(1);
    };

    const next_arg = args.next() orelse {
        printUsage();
        std.process.exit(1);
    };

    var class_name: []const u8 = undefined;
    var method_name: []const u8 = undefined;

    // Check if the argument is in the form class_name->method_name
    if (std.mem.indexOf(u8, next_arg, "->")) |arrow_idx| {
        class_name = next_arg[0..arrow_idx];
        method_name = next_arg[arrow_idx + 2..];
    } else {
        class_name = next_arg;
        method_name = args.next() orelse {
            std.debug.print("Error: Missing method name.\n", .{});
            printUsage();
            std.process.exit(1);
        };
    }

    const max_size: std.Io.Limit = @enumFromInt(100 * 1024 * 1024); // 100 MB max
    
    // Arena for parser allocations
    const arena = init.arena.allocator();

    var multidex = parser.MultiDex.init(allocator);
    defer multidex.deinit();

    // Load embedded core DEX
    _ = multidex.loadDexBytes(arena, core_dex_bytes) catch |err| {
        std.debug.print("Failed to parse embedded core DEX: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };

    // Load user DEX
    const user_bytes = std.Io.Dir.cwd().readFileAlloc(init.io, user_dex_path, allocator, max_size) catch |err| {
        std.debug.print("Failed to read user DEX {s}: {s}\n", .{ user_dex_path, @errorName(err) });
        std.process.exit(1);
    };
    defer allocator.free(user_bytes);
    _ = multidex.loadDexBytes(arena, user_bytes) catch |err| {
        std.debug.print("Failed to parse user DEX: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };

    // Find target method
    const resolved_method = multidex.findMethod(class_name, method_name) orelse {
        std.debug.print("Method not found: {s}->{s}\n", .{ class_name, method_name });
        std.process.exit(1);
    };

    std.debug.print("[DexRunner] Resolved method: {s}\n", .{resolved_method.method.name});
    std.debug.print("  registers_size: {d}, ins_size: {d}, outs_size: {d}, static: {}\n", .{
        resolved_method.method.registers_size,
        resolved_method.method.ins_size,
        resolved_method.method.outs_size,
        resolved_method.method.is_static,
    });

    const insts = multidex.decodeMethod(allocator, resolved_method) catch |err| {
        std.debug.print("Failed to decode method: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
    defer {
        parser.freeDecodedInstructions(allocator, insts);
        allocator.free(insts);
    }

    const reg_count = resolved_method.method.registers_size;
    const ins_count = resolved_method.method.ins_size;
    const regs = try allocator.alloc(u32, reg_count);
    @memset(regs, 0);
    defer allocator.free(regs);

    // Read remaining arguments to fill method registers
    var arg_idx: usize = 0;
    while (args.next()) |arg_str| {
        if (arg_idx < ins_count) {
            const val = std.fmt.parseInt(i32, arg_str, 10) catch blk: {
                std.debug.print("Warning: Could not parse argument '{s}' as integer, using 0\n", .{arg_str});
                break :blk 0;
            };
            regs[reg_count - ins_count + arg_idx] = @bitCast(val);
            arg_idx += 1;
        } else {
            std.debug.print("Warning: Extra argument '{s}' ignored (method only accepts {d} parameters)\n", .{ arg_str, ins_count });
        }
    }

    if (arg_idx < ins_count) {
        std.debug.print("Warning: Only provided {d} of {d} expected parameters (others initialized to 0)\n", .{ arg_idx, ins_count });
    }

    var frame = interpreter.ExecutionFrame{
        .pc = 0,
        .registers = regs,
        .instructions = insts,
    };

    std.debug.print("[DexRunner] Starting execution...\n", .{});
    const result = interpreter.execute(&frame) catch |err| {
        std.debug.print("\n[DexRunner] Execution failed with error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };

    std.debug.print("\n[DexRunner] Execution successful!\n", .{});
    std.debug.print("  Return Type: {s}\n", .{@tagName(result.kind)});
    switch (result.kind) {
        .void => {},
        .single => std.debug.print("  Value (32-bit): {d} (0x{x})\n", .{ @as(i32, @bitCast(result.value32)), result.value32 }),
        .wide => std.debug.print("  Value (64-bit): {d} (0x{x})\n", .{ @as(i64, @bitCast(result.value64)), result.value64 }),
        .object => std.debug.print("  Object Ref: 0x{x}\n", .{result.value32}),
    }
}

fn printUsage() void {
    std.debug.print("Usage: dex_runner <user_dex> <class_name->method_name> [args...]\n", .{});
    std.debug.print("   or: dex_runner <user_dex> <class_name> <method_name> [args...]\n", .{});
    std.debug.print("\nExamples:\n", .{});
    std.debug.print("  zig build run-runner -- user_test.dex com/user/UserTest->add 10 20\n", .{});
    std.debug.print("  zig build run-runner -- user_test.dex com/user/UserTest->fib 40\n", .{});
}
