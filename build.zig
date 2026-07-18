const std = @import("std");

fn addModuleTest(b: *std.Build, test_step: *std.Build.Step, module: *std.Build.Module, filters: []const []const u8) void {
    const tests = b.addTest(.{ .root_module = module, .filters = filters });
    const run_tests = b.addRunArtifact(tests);
    test_step.dependOn(&run_tests.step);
}

fn addModuleTestSerial(
    b: *std.Build,
    test_step: *std.Build.Step,
    module: *std.Build.Module,
    filters: []const []const u8,
    previous: ?*std.Build.Step,
) *std.Build.Step {
    const tests = b.addTest(.{ .root_module = module, .filters = filters });
    if (previous) |dependency| tests.step.dependOn(dependency);
    const run_tests = b.addRunArtifact(tests);
    test_step.dependOn(&run_tests.step);
    return &run_tests.step;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize_mode = b.standardOptimizeOption(.{});
    const test_filters = b.option([]const []const u8, "test-filter", "Skip tests that do not match filter") orelse &.{};

    const instructions = b.addModule("instructions", .{ .root_source_file = b.path("common/00_frontend/00_instructions.zig"), .target = target, .optimize = optimize_mode });
    const parser = b.addModule("parser", .{ .root_source_file = b.path("common/00_frontend/10_parser.zig"), .target = target, .optimize = optimize_mode });
    const print = b.addModule("print", .{ .root_source_file = b.path("common/00_frontend/20_print.zig"), .target = target, .optimize = optimize_mode });
    const interpreter = b.addModule("interpreter", .{ .root_source_file = b.path("common/00_frontend/30_interpreter.zig"), .target = target, .optimize = optimize_mode });
    // const heap = b.addModule("heap", .{ .root_source_file = b.path("common/00_frontend/40_heap.zig"), .target = target, .optimize = optimize_mode });
    // const java_thread = b.addModule("java_thread", .{ .root_source_file = b.path("common/00_frontend/50_java_thread.zig"), .target = target, .optimize = optimize_mode });

    const cfg = b.addModule("cfg", .{ .root_source_file = b.path("common/10_cfg/00_cfg.zig"), .target = target, .optimize = optimize_mode });
    const dominator = b.addModule("dominator", .{ .root_source_file = b.path("common/10_cfg/10_dominator.zig"), .target = target, .optimize = optimize_mode });
    const cfg_phase = b.addModule("cfg_phase", .{ .root_source_file = b.path("common/10_cfg/20_cfg_phase.zig"), .target = target, .optimize = optimize_mode });
    const cfg_rewrite = b.addModule("cfg_rewrite", .{ .root_source_file = b.path("common/10_cfg/30_cfg_rewrite.zig"), .target = target, .optimize = optimize_mode });

    const ssa = b.addModule("ssa", .{ .root_source_file = b.path("common/20_ssa_opt/analysis/00_ssa.zig"), .target = target, .optimize = optimize_mode });
    const typedir = b.addModule("typedir", .{ .root_source_file = b.path("common/20_ssa_opt/analysis/10_typedir.zig"), .target = target, .optimize = optimize_mode });
    const ssa_phase = b.addModule("ssa_phase", .{ .root_source_file = b.path("common/20_ssa_opt/opt/00_ssa_phase.zig"), .target = target, .optimize = optimize_mode });
    const typed_ir = b.addModule("typed_ir", .{ .root_source_file = b.path("common/20_ssa_opt/opt/10_typed_ir.zig"), .target = target, .optimize = optimize_mode });
    const loop_phase = b.addModule("loop_phase", .{ .root_source_file = b.path("common/20_ssa_opt/opt/20_loop_phase.zig"), .target = target, .optimize = optimize_mode });
    const memory_phase = b.addModule("memory_phase", .{ .root_source_file = b.path("common/20_ssa_opt/opt/30_memory_phase.zig"), .target = target, .optimize = optimize_mode });
    const barrier_phase = b.addModule("barrier_phase", .{ .root_source_file = b.path("common/20_ssa_opt/opt/40_barrier_phase.zig"), .target = target, .optimize = optimize_mode });
    const optimizer = b.addModule("optimizer", .{ .root_source_file = b.path("common/20_ssa_opt/90_optimizer.zig"), .target = target, .optimize = optimize_mode });

    const lowering = b.addModule("lowering", .{ .root_source_file = b.path("common/30_lowering/00_lowering.zig"), .target = target, .optimize = optimize_mode });
    const machine_bridge = b.addModule("machine_bridge", .{ .root_source_file = b.path("common/30_lowering/10_machine_bridge.zig"), .target = target, .optimize = optimize_mode });
    const derived_verify = b.addModule("derived_verify", .{ .root_source_file = b.path("common/30_lowering/20_derived_verify.zig"), .target = target, .optimize = optimize_mode });

    const code_buffer = b.addModule("code_buffer", .{ .root_source_file = b.path("common/40_backend/00_code_buffer.zig"), .target = target, .optimize = optimize_mode });
    const register_encoder = b.addModule("register_encoder", .{ .root_source_file = b.path("common/40_backend/10_register_encoder.zig"), .target = target, .optimize = optimize_mode });
    const jit_memory = b.addModule("jit_memory", .{ .root_source_file = b.path("common/40_backend/20_jit_memory.zig"), .target = target, .optimize = optimize_mode });
    const jit_compiler = b.addModule("jit_compiler", .{ .root_source_file = b.path("common/40_backend/25_jit_compiler.zig"), .target = target, .optimize = optimize_mode });
    const x64_encoder = b.addModule("x64_encoder", .{ .root_source_file = b.path("common/40_backend/30_x64_encoder.zig"), .target = target, .optimize = optimize_mode });
    const x64_register_encoder = b.addModule("x64_register_encoder", .{ .root_source_file = b.path("common/40_backend/40_x64_register_encoder.zig"), .target = target, .optimize = optimize_mode });
    const x64_runtime_shim = b.addModule("x64_runtime_shim", .{ .root_source_file = b.path("common/40_backend/50_x64_runtime_shim.zig"), .target = target, .optimize = optimize_mode });

    const liveness = b.addModule("liveness", .{ .root_source_file = b.path("common/50_regalloc/00_liveness.zig"), .target = target, .optimize = optimize_mode });
    const intervals = b.addModule("intervals", .{ .root_source_file = b.path("common/50_regalloc/10_intervals.zig"), .target = target, .optimize = optimize_mode });
    const linear_scan = b.addModule("linear_scan", .{ .root_source_file = b.path("common/50_regalloc/20_linear_scan.zig"), .target = target, .optimize = optimize_mode });
    const spill_rewrite = b.addModule("spill_rewrite", .{ .root_source_file = b.path("common/50_regalloc/30_spill_rewrite.zig"), .target = target, .optimize = optimize_mode });
    const regalloc = b.addModule("regalloc", .{ .root_source_file = b.path("common/50_regalloc/40_allocator.zig"), .target = target, .optimize = optimize_mode });
    const post_derived_verify = b.addModule("post_derived_verify", .{ .root_source_file = b.path("common/50_regalloc/50_derived_verify.zig"), .target = target, .optimize = optimize_mode });

    const runtime_value = b.addModule("runtime_value", .{ .root_source_file = b.path("common/60_runtime/00_value.zig"), .target = target, .optimize = optimize_mode });
    const runtime_thread_registry = b.addModule("runtime_thread_registry", .{ .root_source_file = b.path("common/60_runtime/10_thread_registry.zig"), .target = target, .optimize = optimize_mode });
    const runtime_heap = b.addModule("runtime_heap", .{ .root_source_file = b.path("common/60_runtime/20_heap.zig"), .target = target, .optimize = optimize_mode });
    const runtime_gc = b.addModule("runtime_gc", .{ .root_source_file = b.path("common/60_runtime/25_gc.zig"), .target = target, .optimize = optimize_mode });
    const runtime_stack_map = b.addModule("runtime_stack_map", .{ .root_source_file = b.path("common/60_runtime/30_stack_map.zig"), .target = target, .optimize = optimize_mode });
    const runtime_jit = b.addModule("runtime_jit", .{ .root_source_file = b.path("common/60_runtime/40_jit_runtime.zig"), .target = target, .optimize = optimize_mode });
    const runtime_interpreter = b.addModule("runtime_interpreter", .{ .root_source_file = b.path("common/60_runtime/45_interpreter_runtime.zig"), .target = target, .optimize = optimize_mode });
    const runtime_code_manager = b.addModule("runtime_code_manager", .{ .root_source_file = b.path("common/60_runtime/50_code_manager.zig"), .target = target, .optimize = optimize_mode });
    const runtime_deopt = b.addModule("runtime_deopt", .{ .root_source_file = b.path("common/60_runtime/55_deoptimization.zig"), .target = target, .optimize = optimize_mode });

    const scheduler = b.addModule("scheduler", .{ .root_source_file = b.path("src/engine/scheduler.zig"), .target = target, .optimize = optimize_mode });
    const lock = b.addModule("lock", .{ .root_source_file = b.path("src/engine/lock.zig"), .target = target, .optimize = optimize_mode });
    const immix = b.addModule("immix", .{ .root_source_file = b.path("src/db/immix.zig"), .target = target, .optimize = optimize_mode });
    const storage = b.addModule("storage", .{ .root_source_file = b.path("src/db/storage.zig"), .target = target, .optimize = optimize_mode });
    const actor_runtime = b.addModule("actor_runtime", .{ .root_source_file = b.path("src/actor/runtime.zig"), .target = target, .optimize = optimize_mode });
    const network_simulator = b.addModule("network_simulator", .{ .root_source_file = b.path("src/network/simulator.zig"), .target = target, .optimize = optimize_mode });
    const two_phase = b.addModule("two_phase", .{ .root_source_file = b.path("src/consensus/two_phase.zig"), .target = target, .optimize = optimize_mode });

    parser.addImport("instructions", instructions);
    print.addImport("parser", parser);
    print.addImport("instructions", instructions);
    interpreter.addImport("instructions", instructions);
    // java_thread.addImport("interpreter", interpreter);
    // java_thread.addImport("heap", heap);
    // heap.addImport("interpreter", interpreter);

    cfg.addImport("instructions", instructions);
    dominator.addImport("cfg", cfg);
    dominator.addImport("instructions", instructions);
    cfg_phase.addImport("cfg", cfg);
    cfg_phase.addImport("dominator", dominator);
    cfg_phase.addImport("instructions", instructions);
    cfg_rewrite.addImport("cfg", cfg);
    cfg_rewrite.addImport("cfg_phase", cfg_phase);
    cfg_rewrite.addImport("instructions", instructions);

    ssa.addImport("cfg", cfg);
    ssa.addImport("dominator", dominator);
    ssa.addImport("instructions", instructions);
    ssa_phase.addImport("cfg", cfg);
    ssa_phase.addImport("dominator", dominator);
    ssa_phase.addImport("ssa", ssa);
    ssa_phase.addImport("instructions", instructions);
    typedir.addImport("cfg", cfg);
    typedir.addImport("dominator", dominator);
    typedir.addImport("ssa", ssa);
    typedir.addImport("instructions", instructions);
    typed_ir.addImport("cfg", cfg);
    typed_ir.addImport("dominator", dominator);
    typed_ir.addImport("ssa", ssa);
    typed_ir.addImport("typedir", typedir);
    typed_ir.addImport("optimizer", optimizer);
    typed_ir.addImport("instructions", instructions);
    loop_phase.addImport("cfg", cfg);
    loop_phase.addImport("dominator", dominator);
    loop_phase.addImport("ssa", ssa);
    loop_phase.addImport("ssa_phase", ssa_phase);
    loop_phase.addImport("instructions", instructions);
    memory_phase.addImport("cfg", cfg);
    memory_phase.addImport("dominator", dominator);
    memory_phase.addImport("ssa", ssa);
    memory_phase.addImport("typedir", typedir);
    memory_phase.addImport("instructions", instructions);
    barrier_phase.addImport("cfg", cfg);
    barrier_phase.addImport("dominator", dominator);
    barrier_phase.addImport("ssa", ssa);
    barrier_phase.addImport("typedir", typedir);
    barrier_phase.addImport("loop_phase", loop_phase);
    barrier_phase.addImport("instructions", instructions);

    lowering.addImport("cfg", cfg);
    lowering.addImport("dominator", dominator);
    lowering.addImport("ssa", ssa);
    lowering.addImport("ssa_phase", ssa_phase);
    lowering.addImport("typedir", typedir);
    lowering.addImport("typed_ir", typed_ir);
    lowering.addImport("optimizer", optimizer);
    lowering.addImport("memory_phase", memory_phase);
    lowering.addImport("barrier_phase", barrier_phase);
    lowering.addImport("instructions", instructions);
    machine_bridge.addImport("cfg", cfg);
    machine_bridge.addImport("barrier_phase", barrier_phase);
    machine_bridge.addImport("dominator", dominator);
    machine_bridge.addImport("lowering", lowering);
    machine_bridge.addImport("memory_phase", memory_phase);
    machine_bridge.addImport("ssa", ssa);
    machine_bridge.addImport("ssa_phase", ssa_phase);
    machine_bridge.addImport("typed_ir", typed_ir);
    machine_bridge.addImport("typedir", typedir);
    machine_bridge.addImport("instructions", instructions);
    derived_verify.addImport("barrier_phase", barrier_phase);
    derived_verify.addImport("machine_bridge", machine_bridge);

    optimizer.addImport("cfg", cfg);
    optimizer.addImport("cfg_rewrite", cfg_rewrite);
    optimizer.addImport("dominator", dominator);
    optimizer.addImport("ssa", ssa);
    optimizer.addImport("ssa_phase", ssa_phase);
    optimizer.addImport("typedir", typedir);
    optimizer.addImport("typed_ir", typed_ir);
    optimizer.addImport("loop_phase", loop_phase);
    optimizer.addImport("memory_phase", memory_phase);
    optimizer.addImport("barrier_phase", barrier_phase);
    optimizer.addImport("lowering", lowering);
    optimizer.addImport("machine_bridge", machine_bridge);
    optimizer.addImport("derived_verify", derived_verify);
    optimizer.addImport("instructions", instructions);

    register_encoder.addImport("cfg", cfg);
    register_encoder.addImport("code_buffer", code_buffer);
    register_encoder.addImport("machine_bridge", machine_bridge);
    register_encoder.addImport("optimizer", optimizer);
    register_encoder.addImport("typedir", typedir);
    register_encoder.addImport("instructions", instructions);
    jit_memory.addImport("register_encoder", register_encoder);
    jit_memory.addImport("optimizer", optimizer);
    jit_memory.addImport("instructions", instructions);
    jit_compiler.addImport("code_buffer", code_buffer);
    jit_compiler.addImport("instructions", instructions);
    jit_compiler.addImport("jit_memory", jit_memory);
    jit_compiler.addImport("runtime_code_manager", runtime_code_manager);
    jit_compiler.addImport("runtime_deopt", runtime_deopt);
    jit_compiler.addImport("runtime_stack_map", runtime_stack_map);
    x64_encoder.addImport("cfg", cfg);
    x64_encoder.addImport("code_buffer", code_buffer);
    x64_encoder.addImport("jit_memory", jit_memory);
    x64_encoder.addImport("machine_bridge", machine_bridge);
    x64_encoder.addImport("optimizer", optimizer);
    x64_encoder.addImport("instructions", instructions);
    x64_register_encoder.addImport("cfg", cfg);
    x64_register_encoder.addImport("code_buffer", code_buffer);
    x64_register_encoder.addImport("jit_memory", jit_memory);
    x64_register_encoder.addImport("machine_bridge", machine_bridge);
    x64_register_encoder.addImport("optimizer", optimizer);
    x64_register_encoder.addImport("regalloc", regalloc);
    x64_register_encoder.addImport("runtime_stack_map", runtime_stack_map);
    x64_register_encoder.addImport("runtime_jit", runtime_jit);
    x64_register_encoder.addImport("runtime_deopt", runtime_deopt);
    x64_register_encoder.addImport("runtime_value", runtime_value);
    x64_register_encoder.addImport("instructions", instructions);

    liveness.addImport("intervals", intervals);
    liveness.addImport("optimizer", optimizer);
    liveness.addImport("instructions", instructions);
    intervals.addImport("machine_bridge", machine_bridge);
    intervals.addImport("typedir", typedir);
    intervals.addImport("optimizer", optimizer);
    intervals.addImport("instructions", instructions);
    linear_scan.addImport("intervals", intervals);
    linear_scan.addImport("machine_bridge", machine_bridge);
    linear_scan.addImport("typedir", typedir);
    linear_scan.addImport("optimizer", optimizer);
    linear_scan.addImport("instructions", instructions);
    spill_rewrite.addImport("linear_scan", linear_scan);
    spill_rewrite.addImport("machine_bridge", machine_bridge);
    spill_rewrite.addImport("typedir", typedir);
    spill_rewrite.addImport("optimizer", optimizer);
    spill_rewrite.addImport("instructions", instructions);
    post_derived_verify.addImport("derived_verify", derived_verify);
    post_derived_verify.addImport("intervals", intervals);
    post_derived_verify.addImport("linear_scan", linear_scan);
    post_derived_verify.addImport("spill_rewrite", spill_rewrite);
    post_derived_verify.addImport("machine_bridge", machine_bridge);
    post_derived_verify.addImport("optimizer", optimizer);
    post_derived_verify.addImport("instructions", instructions);
    regalloc.addImport("linear_scan", linear_scan);
    regalloc.addImport("spill_rewrite", spill_rewrite);
    regalloc.addImport("post_derived_verify", post_derived_verify);
    regalloc.addImport("machine_bridge", machine_bridge);
    regalloc.addImport("optimizer", optimizer);
    regalloc.addImport("instructions", instructions);

    runtime_thread_registry.addImport("runtime_value", runtime_value);
    runtime_heap.addImport("runtime_value", runtime_value);
    runtime_heap.addImport("runtime_thread_registry", runtime_thread_registry);
    runtime_gc.addImport("runtime_value", runtime_value);
    runtime_gc.addImport("runtime_heap", runtime_heap);
    runtime_gc.addImport("runtime_thread_registry", runtime_thread_registry);
    runtime_stack_map.addImport("runtime_value", runtime_value);
    runtime_jit.addImport("runtime_value", runtime_value);
    runtime_jit.addImport("runtime_gc", runtime_gc);
    runtime_jit.addImport("runtime_stack_map", runtime_stack_map);
    runtime_jit.addImport("runtime_thread_registry", runtime_thread_registry);
    runtime_jit.addImport("runtime_code_manager", runtime_code_manager);
    runtime_jit.addImport("runtime_deopt", runtime_deopt);
    runtime_code_manager.addImport("jit_memory", jit_memory);
    runtime_deopt.addImport("interpreter", interpreter);
    runtime_deopt.addImport("runtime_value", runtime_value);
    runtime_deopt.addImport("runtime_stack_map", runtime_stack_map);
    runtime_interpreter.addImport("interpreter", interpreter);
    runtime_interpreter.addImport("runtime_gc", runtime_gc);
    runtime_interpreter.addImport("runtime_heap", runtime_heap);
    runtime_interpreter.addImport("runtime_value", runtime_value);

    x64_runtime_shim.addImport("code_buffer", code_buffer);
    x64_runtime_shim.addImport("interpreter", interpreter);
    x64_runtime_shim.addImport("jit_memory", jit_memory);
    x64_runtime_shim.addImport("optimizer", optimizer);
    x64_runtime_shim.addImport("runtime_gc", runtime_gc);
    x64_runtime_shim.addImport("runtime_heap", runtime_heap);
    x64_runtime_shim.addImport("runtime_jit", runtime_jit);
    x64_runtime_shim.addImport("runtime_code_manager", runtime_code_manager);
    x64_runtime_shim.addImport("runtime_deopt", runtime_deopt);
    x64_runtime_shim.addImport("runtime_stack_map", runtime_stack_map);
    x64_runtime_shim.addImport("runtime_thread_registry", runtime_thread_registry);
    x64_runtime_shim.addImport("runtime_value", runtime_value);
    x64_runtime_shim.addImport("x64_register_encoder", x64_register_encoder);
    x64_runtime_shim.addImport("instructions", instructions);

    storage.addImport("lock", lock);
    actor_runtime.addImport("scheduler", scheduler);
    actor_runtime.addImport("immix", immix);
    network_simulator.addImport("scheduler", scheduler);
    two_phase.addImport("actor_runtime", actor_runtime);
    two_phase.addImport("storage", storage);
    two_phase.addImport("scheduler", scheduler);
    two_phase.addImport("immix", immix);
    two_phase.addImport("network_simulator", network_simulator);

    const public_modules = [_]struct { name: []const u8, module: *std.Build.Module }{
        .{ .name = "instructions", .module = instructions },
        .{ .name = "parser", .module = parser },
        .{ .name = "print", .module = print },
        .{ .name = "interpreter", .module = interpreter },
        // .{ .name = "heap", .module = heap },
        // .{ .name = "java_thread", .module = java_thread },
        .{ .name = "cfg", .module = cfg },
        .{ .name = "cfg_phase", .module = cfg_phase },
        .{ .name = "cfg_rewrite", .module = cfg_rewrite },
        .{ .name = "dominator", .module = dominator },
        .{ .name = "ssa", .module = ssa },
        .{ .name = "ssa_phase", .module = ssa_phase },
        .{ .name = "optimizer", .module = optimizer },
        .{ .name = "typedir", .module = typedir },
        .{ .name = "typed_ir", .module = typed_ir },
        .{ .name = "loop_phase", .module = loop_phase },
        .{ .name = "memory_phase", .module = memory_phase },
        .{ .name = "barrier_phase", .module = barrier_phase },
        .{ .name = "lowering", .module = lowering },
        .{ .name = "machine_bridge", .module = machine_bridge },
        .{ .name = "derived_verify", .module = derived_verify },
        .{ .name = "code_buffer", .module = code_buffer },
        .{ .name = "register_encoder", .module = register_encoder },
        .{ .name = "jit_memory", .module = jit_memory },
        .{ .name = "jit_compiler", .module = jit_compiler },
        .{ .name = "x64_encoder", .module = x64_encoder },
        .{ .name = "x64_register_encoder", .module = x64_register_encoder },
        .{ .name = "x64_runtime_shim", .module = x64_runtime_shim },
        .{ .name = "liveness", .module = liveness },
        .{ .name = "intervals", .module = intervals },
        .{ .name = "linear_scan", .module = linear_scan },
        .{ .name = "spill_rewrite", .module = spill_rewrite },
        .{ .name = "regalloc", .module = regalloc },
        .{ .name = "post_derived_verify", .module = post_derived_verify },
        .{ .name = "runtime_value", .module = runtime_value },
        .{ .name = "runtime_thread_registry", .module = runtime_thread_registry },
        .{ .name = "runtime_heap", .module = runtime_heap },
        .{ .name = "runtime_gc", .module = runtime_gc },
        .{ .name = "runtime_stack_map", .module = runtime_stack_map },
        .{ .name = "runtime_jit", .module = runtime_jit },
        .{ .name = "runtime_interpreter", .module = runtime_interpreter },
        .{ .name = "runtime_code_manager", .module = runtime_code_manager },
        .{ .name = "runtime_deopt", .module = runtime_deopt },
        .{ .name = "scheduler", .module = scheduler },
        .{ .name = "lock", .module = lock },
        .{ .name = "immix", .module = immix },
        .{ .name = "storage", .module = storage },
        .{ .name = "actor_runtime", .module = actor_runtime },
        .{ .name = "network_simulator", .module = network_simulator },
        .{ .name = "two_phase", .module = two_phase },
    };

    const root_mod = b.addModule("Thread_Test", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize_mode,
    });
    for (public_modules) |entry| root_mod.addImport(entry.name, entry.module);

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize_mode,
    });
    exe_mod.addImport("Thread_Test", root_mod);
    for (public_modules) |entry| exe_mod.addImport(entry.name, entry.module);

    const exe = b.addExecutable(.{ .name = "Thread_Test", .root_module = exe_mod });
    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    run_step.dependOn(&run_cmd.step);

    const dex_mod = b.createModule(.{
        .root_source_file = b.path("dex_printer.zig"),
        .target = target,
        .optimize = optimize_mode,
    });
    dex_mod.addImport("parser", parser);
    dex_mod.addImport("print", print);
    const dex_cli = b.addExecutable(.{ .name = "dex_printer", .root_module = dex_mod });
    b.installArtifact(dex_cli);

    const run_dex_cli = b.addRunArtifact(dex_cli);
    run_dex_cli.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_dex_cli.addArgs(args);
    const run_dex_step = b.step("run-dex", "Run the DEX printer CLI (usage: zig build run-dex -- <path>)");
    run_dex_step.dependOn(&run_dex_cli.step);

    const runner_mod = b.createModule(.{
        .root_source_file = b.path("dex_runner.zig"),
        .target = target,
        .optimize = optimize_mode,
    });
    runner_mod.addImport("parser", parser);
    runner_mod.addImport("interpreter", interpreter);
    const runner_cli = b.addExecutable(.{ .name = "dex_runner", .root_module = runner_mod });
    b.installArtifact(runner_cli);

    const run_runner_cli = b.addRunArtifact(runner_cli);
    run_runner_cli.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_runner_cli.addArgs(args);
    const run_runner_step = b.step("run-runner", "Run the DEX runner CLI (usage: zig build run-runner -- <user_dex> <class_name> <method_name> [args...])");
    run_runner_step.dependOn(&run_runner_cli.step);

    const jit_runner_mod = b.createModule(.{
        .root_source_file = b.path("dex_jit_runner.zig"),
        .target = target,
        .optimize = optimize_mode,
    });
    jit_runner_mod.addImport("parser", parser);
    jit_runner_mod.addImport("jit_compiler", jit_compiler);
    jit_runner_mod.addImport("runtime_code_manager", runtime_code_manager);
    const jit_runner_cli = b.addExecutable(.{ .name = "dex_jit_runner", .root_module = jit_runner_mod });
    b.installArtifact(jit_runner_cli);

    const run_jit_runner_cli = b.addRunArtifact(jit_runner_cli);
    run_jit_runner_cli.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_jit_runner_cli.addArgs(args);
    const run_jit_runner_step = b.step("run-jit-runner", "Run the DEX JIT runner CLI (usage: zig build run-jit-runner -- <user_dex> <class_name> <method_name> [args...])");
    run_jit_runner_step.dependOn(&run_jit_runner_cli.step);

    const test_step = b.step("test", "Run tests");
    addModuleTest(b, test_step, root_mod, test_filters);
    addModuleTest(b, test_step, exe.root_module, test_filters);
    for (public_modules) |entry| addModuleTest(b, test_step, entry.module, test_filters);

    // Runtime/compiler validation intentionally excludes the unrelated legacy
    // modules rooted under src/. Keep this boundary stable for common/ work.
    const runtime_test_step = b.step("test-runtime", "Run common interpreter/JIT/runtime tests without src modules");
    const src_module_count = 7;
    var previous_runtime_test: ?*std.Build.Step = null;
    for (public_modules[0 .. public_modules.len - src_module_count]) |entry| {
        previous_runtime_test = addModuleTestSerial(
            b,
            runtime_test_step,
            entry.module,
            test_filters,
            previous_runtime_test,
        );
    }
}
