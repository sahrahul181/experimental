const std = @import("std");
const parser = @import("parser");
const jit_compiler = @import("jit_compiler");
const runtime_code_manager = @import("runtime_code_manager");

// Embed the core DEX directly into the CLI binary
const core_dex_bytes = @embedFile("core.dex");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try init.minimal.args.iterateAllocator(allocator);
    defer args.deinit();

    _ = args.next(); // skip executable name

    var user_dex_path = args.next() orelse {
        printUsage();
        std.process.exit(1);
    };

    if (std.mem.eql(u8, user_dex_path, "--")) {
        user_dex_path = args.next() orelse {
            printUsage();
            std.process.exit(1);
        };
    }

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

    std.debug.print("[DexJitRunner] Resolved method: {s}\n", .{resolved_method.method.name});
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

    // Parse up to 4 arguments passed on command-line according to shorty signature
    const sig = resolved_method.method.signature;
    var args_list = [4]i64{ 0, 0, 0, 0 };
    var arg_slot: usize = 0;
    var param_idx: usize = 0;
    while (args.next()) |arg_str| {
        if (arg_slot < 4) {
            const param_type = if (1 + param_idx < sig.len) sig[1 + param_idx] else 'I';
            switch (param_type) {
                'D' => {
                    if (std.fmt.parseFloat(f64, arg_str)) |val| {
                        args_list[arg_slot] = @bitCast(val);
                    } else |_| {
                        std.debug.print("Warning: Could not parse '{s}' as double, using 0.0\n", .{arg_str});
                    }
                    arg_slot += 2; // double occupies 2 Dalvik slots (v15 and v16)
                },
                'J' => {
                    if (std.fmt.parseInt(i64, arg_str, 10)) |val| {
                        args_list[arg_slot] = val;
                    } else |_| {
                        std.debug.print("Warning: Could not parse '{s}' as long, using 0\n", .{arg_str});
                    }
                    arg_slot += 2; // long occupies 2 Dalvik slots
                },
                'F' => {
                    if (std.fmt.parseFloat(f32, arg_str)) |val| {
                        const u_val: u32 = @bitCast(val);
                        args_list[arg_slot] = @as(i64, u_val);
                    } else |_| {
                        std.debug.print("Warning: Could not parse '{s}' as float, using 0.0\n", .{arg_str});
                    }
                    arg_slot += 1;
                },
                else => {
                    if (std.fmt.parseInt(i32, arg_str, 10)) |val| {
                        const u_val: u32 = @bitCast(val);
                        args_list[arg_slot] = @as(i64, u_val);
                    } else |_| {
                        std.debug.print("Warning: Could not parse '{s}' as int, using 0\n", .{arg_str});
                    }
                    arg_slot += 1;
                },
            }
            param_idx += 1;
        } else {
            std.debug.print("Warning: Extra argument '{s}' ignored (CLI currently supports up to 4 direct arguments)\n", .{arg_str});
        }
    }

    if (arg_slot < ins_count) {
        std.debug.print("Warning: Only provided {d} of {d} expected parameter slots (others initialized to 0)\n", .{ arg_slot, ins_count });
    }

    std.debug.print("[DexJitRunner] Compiling method directly to x86-64 machine code...\n", .{});
    var manager = try runtime_code_manager.Manager.init(allocator, 16, 64, 64);
    defer manager.deinit() catch {};

    var compiler = jit_compiler.Compiler.init();
    _ = compiler.compileAndPublish(allocator, &manager, 1, insts, .{
        .register_count = reg_count,
        .parameter_count = ins_count,
    }) catch |err| {
        std.debug.print("\n[DexJitRunner] Direct JIT compilation failed: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };

    var reader = try manager.registerReader();
    defer reader.deinit();
    var lease = try manager.enter(&reader, 1);
    defer lease.deinit();

    std.debug.print("[DexJitRunner] Starting JIT execution (entry address: 0x{x})...\n", .{lease.entryAddress()});

    // Check return type from signature (either shorty like "II" or full descriptor like "(I)I")
    var return_type_desc: []const u8 = "V";
    if (std.mem.lastIndexOfScalar(u8, resolved_method.method.signature, ')')) |r_idx| {
        if (r_idx + 1 < resolved_method.method.signature.len) {
            return_type_desc = resolved_method.method.signature[r_idx + 1 ..];
        }
    } else if (resolved_method.method.signature.len > 0) {
        return_type_desc = resolved_method.method.signature[0..1];
    }

    if (std.mem.eql(u8, return_type_desc, "D")) {
        const Fn = fn (i64, i64, i64, i64) callconv(.c) f64;
        const exec = lease.typedEntry(Fn);
        const res_f64 = exec(args_list[0], args_list[1], args_list[2], args_list[3]);
        std.debug.print("\n[DexJitRunner] Execution successful!\n", .{});
        std.debug.print("  Return Type: wide (double)\n", .{});
        std.debug.print("  Value (double): {d} (0x{x})\n", .{ res_f64, @as(u64, @bitCast(res_f64)) });
    } else if (std.mem.eql(u8, return_type_desc, "F")) {
        const Fn = fn (i64, i64, i64, i64) callconv(.c) f32;
        const exec = lease.typedEntry(Fn);
        const res_f32 = exec(args_list[0], args_list[1], args_list[2], args_list[3]);
        std.debug.print("\n[DexJitRunner] Execution successful!\n", .{});
        std.debug.print("  Return Type: single (float)\n", .{});
        std.debug.print("  Value (float): {d} (0x{x})\n", .{ res_f32, @as(u32, @bitCast(res_f32)) });
    } else {
        const Fn = fn (i64, i64, i64, i64) callconv(.c) i64;
        const exec = lease.typedEntry(Fn);
        const res_i64 = exec(args_list[0], args_list[1], args_list[2], args_list[3]);

        std.debug.print("\n[DexJitRunner] Execution successful!\n", .{});
        if (std.mem.eql(u8, return_type_desc, "V")) {
            std.debug.print("  Return Type: void\n", .{});
        } else if (std.mem.eql(u8, return_type_desc, "J")) {
            std.debug.print("  Return Type: wide (long)\n", .{});
            std.debug.print("  Value (64-bit): {d} (0x{x})\n", .{ res_i64, @as(u64, @bitCast(res_i64)) });
        } else if (std.mem.startsWith(u8, return_type_desc, "L") or std.mem.startsWith(u8, return_type_desc, "[")) {
            std.debug.print("  Return Type: object\n", .{});
            const obj_ref: u32 = @truncate(@as(u64, @bitCast(res_i64)));
            std.debug.print("  Object Ref: 0x{x}\n", .{obj_ref});
        } else {
            std.debug.print("  Return Type: single ({s})\n", .{return_type_desc});
            const val32: i32 = @truncate(res_i64);
            std.debug.print("  Value (32-bit): {d} (0x{x})\n", .{ val32, @as(u32, @bitCast(val32)) });
        }
    }
}

fn printUsage() void {
    std.debug.print("Usage: dex_jit_runner <user_dex> <class_name->method_name> [args...]\n", .{});
    std.debug.print("   or: dex_jit_runner <user_dex> <class_name> <method_name> [args...]\n", .{});
    std.debug.print("\nExamples:\n", .{});
    std.debug.print("  zig build run-jit-runner -- user_test.dex com/user/UserTest->add 10 20\n", .{});
    std.debug.print("  zig build run-jit-runner -- user_test.dex com/user/UserTest->fib 40\n", .{});
}
