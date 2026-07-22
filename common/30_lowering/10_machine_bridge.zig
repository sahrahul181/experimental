//! Register-machine bridge.
//!
//! This pass converts backend-facing lowered IR into a compact register-machine
//! stream. SSA values are kept as virtual registers, while phi nodes are removed
//! into explicit parallel edge moves that a concrete machine-code emitter can
//! resolve at block boundaries.

const std = @import("std");
const cfg = @import("cfg");
const dom = @import("dominator");
const lowering = @import("lowering");
const memory_phase = @import("memory_phase");
const ssa = @import("ssa");
const ssa_phase = @import("ssa_phase");
const typed_ir = @import("typed_ir");
const typedir = @import("typedir");
const barrier_phase = @import("barrier_phase");
const instmod = @import("instructions");
const Instruction = instmod.Instruction;

pub const Error = error{
    InvalidLowering,
    OutOfMemory,
};

pub const VerifyError = error{
    BadBlock,
    BadEdgeMove,
    BadInstruction,
    BadRegister,
    InvalidLowering,
    OutOfMemory,
    UndefinedRegister,
};

pub const RegId = u32;
pub const INVALID_REG: RegId = std.math.maxInt(RegId);
pub const FloatOperation = lowering.FloatOperation;
pub const ValueType = typedir.Type;

pub const Opcode = enum(u8) {
    mov,
    const_i32,
    const_i64,
    add_i32,
    sub_i32,
    mul_i32,
    div_i32,
    rem_i32,
    and_i32,
    or_i32,
    xor_i32,
    add_i64,
    sub_i64,
    mul_i64,
    div_i64,
    rem_i64,
    f32_op,
    f64_op,
    check_null,
    check_bounds,
    resolve_handle,
    loop_epoch_guard,
    array_load,
    array_store,
    field_load,
    field_store,
    array_load_ptr,
    array_store_ptr,
    field_load_ptr,
    field_store_ptr,
    satb_pre_write,
    card_mark,
    static_satb_pre_write,
    static_root_post_write,
    static_load,
    static_store,
    call_direct,
    call_static,
    call_virtual,
    call_quick,
    monitor_enter,
    monitor_exit,
    jump,
    branch,
    switch_,
    ret,
    throw_,
    memory_barrier,
    unsupported,
};

pub const Flags = packed struct {
    null_check_elided: bool = false,
    bounds_check_elided: bool = false,
    forwarded: bool = false,
    cse: bool = false,
};

pub const Condition = enum(u8) {
    eq,
    ne,
    lt,
    ge,
    gt,
    le,
};

pub const Inst = struct {
    opcode: Opcode,
    pc: ?u32 = null,
    defs: []RegId = &.{},
    uses: []RegId = &.{},
    target: ?cfg.BlockId = null,
    false_target: ?cfg.BlockId = null,
    condition: ?Condition = null,
    imm: i64 = 0,
    float_op: ?FloatOperation = null,
    field_idx: ?u32 = null,
    address: ?RegId = null,
    state_handle: ?RegId = null,
    reloc_token: ?u32 = null,
    resolve_id: ?u32 = null,
    guard_site_id: ?u32 = null,
    exception_site_id: ?u32 = null,
    monitor_site_id: ?u32 = null,
    pre_write: barrier_phase.PreWriteBarrier = .none,
    post_write: barrier_phase.PostWriteBarrier = .none,
    flags: Flags = .{},
};

pub const Block = struct {
    id: cfg.BlockId,
    insts: []Inst,
};

pub const Move = struct {
    dst: RegId,
    src: RegId,
    ty: typedir.Type,
};

pub const EdgeMoves = struct {
    from: cfg.BlockId,
    to: cfg.BlockId,
    moves: []Move,
};

pub const Stats = struct {
    blocks: u32 = 0,
    instructions: u32 = 0,
    edge_moves: u32 = 0,
    constants: u32 = 0,
    branches: u32 = 0,
    calls: u32 = 0,
    checks: u32 = 0,
    resolves: u32 = 0,
    loop_epoch_guards: u32 = 0,
    bounds_exception_sites: u32 = 0,
    monitor_sites: u32 = 0,
    pointer_accesses: u32 = 0,
    forwarded: u32 = 0,
    cse: u32 = 0,
};

pub const CoverageStats = struct {
    reachable_blocks: u32 = 0,
    cfg_edges: u32 = 0,
    join_blocks: u32 = 0,
    loop_edges: u32 = 0,
    phi_edge_moves: u32 = 0,
    safepoint_sites: u32 = 0,
};

pub const Function = struct {
    allocator: std.mem.Allocator,
    source: *const lowering.Function,
    blocks: []Block,
    edges: []EdgeMoves,
    reg_types: []typedir.Type,
    runtime_values: []lowering.RuntimeValueClass,
    value_kinds: []ssa.ValueKind,
    successors: [][]cfg.BlockId,
    stats: Stats,

    pub fn deinit(self: *Function) void {
        for (self.successors) |successors| self.allocator.free(successors);
        self.allocator.free(self.successors);
        self.allocator.free(self.value_kinds);
        for (self.edges) |edge| self.allocator.free(edge.moves);
        self.allocator.free(self.edges);
        for (self.blocks) |block| {
            for (block.insts) |inst| {
                self.allocator.free(inst.defs);
                self.allocator.free(inst.uses);
            }
            self.allocator.free(block.insts);
        }
        self.allocator.free(self.blocks);
        self.allocator.free(self.reg_types);
        self.allocator.free(self.runtime_values);
        self.* = undefined;
    }

    /// Proves that machine blocks are an exact reachable projection of the
    /// source CFG and that every logical safepoint id is present exactly once.
    /// Root-map construction relies on this proof before solving multi-block
    /// liveness, especially at phi joins and loop backedges.
    pub fn verifyCoverage(self: *const Function, allocator: std.mem.Allocator) VerifyError!CoverageStats {
        const graph = self.source.source.graph;
        if (self.blocks.len == 0 or graph.entry >= self.blocks.len or
            graph.blocks.len != self.blocks.len or self.successors.len != self.blocks.len or
            self.source.source.blocks.len != self.blocks.len) return error.BadBlock;

        const reachable = try allocator.alloc(bool, self.blocks.len);
        defer allocator.free(reachable);
        @memset(reachable, false);
        const work = try allocator.alloc(cfg.BlockId, self.blocks.len);
        defer allocator.free(work);
        var head: usize = 0;
        var tail: usize = 0;
        work[tail] = graph.entry;
        reachable[graph.entry] = true;
        tail += 1;

        var stats: CoverageStats = .{};
        while (head < tail) : (head += 1) {
            const block_id = work[head];
            if (block_id >= self.blocks.len) return error.BadBlock;
            stats.reachable_blocks += 1;
            const machine_successors = self.successors[block_id];
            const graph_block = graph.blocks[block_id];
            if (!std.mem.eql(cfg.BlockId, machine_successors, graph_block.successors)) return error.BadBlock;
            if (graph_block.predecessors.len > 1) stats.join_blocks += 1;
            for (machine_successors, 0..) |successor, successor_index| {
                if (successor >= self.blocks.len) return error.BadBlock;
                for (machine_successors[0..successor_index]) |previous| {
                    if (previous == successor) return error.BadBlock;
                }
                stats.cfg_edges += 1;
                if (self.source.source.tree.dominates(successor, block_id)) stats.loop_edges += 1;
                if (!reachable[successor]) {
                    if (tail >= work.len) return error.BadBlock;
                    reachable[successor] = true;
                    work[tail] = successor;
                    tail += 1;
                }
            }
        }
        for (reachable) |is_reachable| if (!is_reachable) return error.BadBlock;

        for (self.blocks) |block| {
            if (block.id >= self.blocks.len or block.id != graph.blocks[block.id].id) return error.BadBlock;
            for (block.insts) |inst| {
                if (inst.target) |target| if (!hasLocalSuccessor(self.successors, block.id, target)) return error.BadInstruction;
                if (inst.false_target) |target| if (!hasLocalSuccessor(self.successors, block.id, target)) return error.BadInstruction;
            }
        }

        for (graph.blocks) |from_block| {
            for (from_block.successors) |to| {
                var expected_moves: usize = 0;
                for (self.source.source.blocks[to].phis) |phi| {
                    for (phi.incoming) |incoming| {
                        if (incoming.pred == from_block.id) expected_moves += 1;
                    }
                }
                var matching_edges: usize = 0;
                for (self.edges) |edge| {
                    if (edge.from != from_block.id or edge.to != to) continue;
                    matching_edges += 1;
                    if (edge.moves.len != expected_moves) return error.BadEdgeMove;
                    for (edge.moves, 0..) |move, move_index| {
                        if (move.dst >= self.reg_types.len or move.src >= self.reg_types.len or
                            move.ty != self.reg_types[move.dst]) return error.BadEdgeMove;
                        for (edge.moves[0..move_index]) |previous| {
                            if (previous.dst == move.dst) return error.BadEdgeMove;
                        }
                        var found = false;
                        for (self.source.source.blocks[to].phis) |phi| {
                            if (phi.dest != move.dst) continue;
                            for (phi.incoming) |incoming| {
                                if (incoming.pred == from_block.id and incoming.value == move.src) {
                                    found = true;
                                    break;
                                }
                            }
                            if (found) break;
                        }
                        if (!found) return error.BadEdgeMove;
                    }
                    stats.phi_edge_moves += @intCast(edge.moves.len);
                }
                if (matching_edges != @intFromBool(expected_moves != 0)) return error.BadEdgeMove;
            }
        }
        for (self.edges) |edge| {
            if (!hasLocalSuccessor(self.successors, edge.from, edge.to)) return error.BadEdgeMove;
        }

        const resolve_and_guards = std.math.add(u32, self.stats.resolves, self.stats.loop_epoch_guards) catch return error.BadInstruction;
        const monitors_end = std.math.add(u32, resolve_and_guards, self.stats.monitor_sites) catch return error.BadInstruction;
        const total_sites = std.math.add(u32, monitors_end, self.stats.bounds_exception_sites) catch return error.BadInstruction;
        const seen_sites = try allocator.alloc(bool, total_sites);
        defer allocator.free(seen_sites);
        @memset(seen_sites, false);
        for (self.blocks) |block| {
            for (block.insts) |inst| {
                const site = if (inst.opcode == .resolve_handle)
                    inst.resolve_id
                else if (inst.opcode == .loop_epoch_guard)
                    inst.guard_site_id
                else if (inst.opcode == .check_bounds and inst.exception_site_id != null)
                    inst.exception_site_id
                else if (inst.opcode == .monitor_enter or inst.opcode == .monitor_exit)
                    inst.monitor_site_id
                else
                    null;
                if (site) |site_id| {
                    if (site_id >= total_sites or seen_sites[site_id]) return error.BadInstruction;
                    if (inst.opcode == .resolve_handle and site_id >= self.stats.resolves) return error.BadInstruction;
                    if (inst.opcode == .loop_epoch_guard and
                        (site_id < self.stats.resolves or site_id >= resolve_and_guards)) return error.BadInstruction;
                    if ((inst.opcode == .monitor_enter or inst.opcode == .monitor_exit) and
                        (site_id < resolve_and_guards or site_id >= monitors_end)) return error.BadInstruction;
                    if (inst.opcode == .check_bounds and site_id < monitors_end) return error.BadInstruction;
                    seen_sites[site_id] = true;
                    stats.safepoint_sites += 1;
                }
            }
        }
        for (seen_sites) |seen| if (!seen) return error.BadInstruction;
        if (stats.safepoint_sites != total_sites) return error.BadInstruction;
        return stats;
    }

    pub fn verify(self: *const Function) VerifyError!void {
        _ = try self.verifyCoverage(self.allocator);
        if (self.reg_types.len != self.runtime_values.len or self.value_kinds.len != self.source.source.values.len or self.successors.len != self.blocks.len) return error.InvalidLowering;

        var defined = try self.allocator.alloc(bool, self.reg_types.len);
        defer self.allocator.free(defined);
        @memset(defined, false);
        for (self.value_kinds, 0..) |kind, value_id| {
            if (kind == .parameter or kind == .phi) defined[value_id] = true;
        }
        for (self.edges) |edge| {
            for (edge.moves) |move| {
                if (move.dst >= self.reg_types.len or move.src >= self.reg_types.len) return error.BadRegister;
                defined[move.dst] = true;
            }
        }
        for (self.blocks) |block| {
            for (block.insts) |inst| {
                for (inst.defs) |def| {
                    if (def >= self.reg_types.len) return error.BadRegister;
                    defined[def] = true;
                }
            }
        }

        for (self.edges) |edge| {
            if (edge.from >= self.blocks.len or edge.to >= self.blocks.len) return error.BadEdgeMove;
            if (!hasLocalSuccessor(self.successors, edge.from, edge.to)) return error.BadEdgeMove;
            for (edge.moves) |move| {
                if (move.dst >= self.reg_types.len or move.src >= self.reg_types.len) return error.BadRegister;
                if (!defined[move.src]) return error.UndefinedRegister;
            }
        }

        const barriers = self.source.barriers;
        const monitor_site_base: u32 = if (barriers) |plan|
            std.math.add(u32, @intCast(plan.resolves.len), plan.stats.loop_epoch_guards) catch return error.BadInstruction
        else
            0;
        const exception_site_base = std.math.add(u32, monitor_site_base, self.stats.monitor_sites) catch return error.BadInstruction;
        const loop_guard_count: u32 = if (barriers) |plan| plan.stats.loop_epoch_guards else 0;
        const loop_guard_seen = try self.allocator.alloc(bool, loop_guard_count);
        defer self.allocator.free(loop_guard_seen);
        @memset(loop_guard_seen, false);
        var verified_exception_sites: u32 = 0;
        var verified_monitor_sites: u32 = 0;
        for (self.blocks, 0..) |block, i| {
            if (block.id != i) return error.BadBlock;
            for (block.insts) |inst| {
                for (inst.uses) |use| {
                    if (use >= self.reg_types.len) return error.BadRegister;
                    if (!defined[use]) return error.UndefinedRegister;
                }
                if (inst.address) |address| {
                    if (address >= self.reg_types.len) return error.BadRegister;
                    if (!defined[address]) return error.UndefinedRegister;
                }
                if (inst.state_handle) |handle| {
                    if (handle >= self.reg_types.len) return error.BadRegister;
                    if (!defined[handle]) return error.UndefinedRegister;
                }
                for (inst.defs) |def| if (def >= self.reg_types.len) return error.BadRegister;
                for (inst.defs) |def| switch (self.runtime_values[def]) {
                    .dalvik => {},
                    .derived_ptr => if (inst.opcode != .resolve_handle) return error.BadInstruction,
                };
                switch (inst.opcode) {
                    .jump => if (inst.target == null) return error.BadInstruction,
                    .branch => if (inst.target == null or inst.false_target == null or inst.uses.len == 0 or inst.condition == null) return error.BadInstruction,
                    .field_load, .field_store, .static_load, .static_store, .static_satb_pre_write, .static_root_post_write => if (inst.field_idx == null) return error.BadInstruction,
                    .resolve_handle => try self.verifyResolve(inst),
                    .loop_epoch_guard => {
                        const guard_id = try self.verifyLoopEpochGuard(block.id, inst);
                        if (guard_id >= loop_guard_seen.len or loop_guard_seen[guard_id]) return error.BadInstruction;
                        loop_guard_seen[guard_id] = true;
                    },
                    .check_bounds, .field_load_ptr, .field_store_ptr, .array_load_ptr, .array_store_ptr, .satb_pre_write, .card_mark => if (inst.address != null) try self.verifyAddress(inst),
                    .call_direct, .call_static, .call_virtual, .call_quick => if (inst.address != null) try self.verifyAddress(inst),
                    .monitor_enter, .monitor_exit => {
                        if (inst.address != null or inst.state_handle == null or inst.reloc_token != null or inst.resolve_id != null or
                            !self.isGcRoot(inst.state_handle.?)) return error.BadInstruction;
                    },
                    else => {},
                }
                if ((inst.opcode == .f32_op or inst.opcode == .f64_op) != (inst.float_op != null)) return error.BadInstruction;
                for (inst.uses) |use| switch (self.runtime_values[use]) {
                    .dalvik => {},
                    .derived_ptr => return error.BadInstruction,
                };
                const permits_address = switch (inst.opcode) {
                    .field_load_ptr,
                    .field_store_ptr,
                    .array_load_ptr,
                    .array_store_ptr,
                    .check_bounds,
                    .satb_pre_write,
                    .card_mark,
                    .call_direct,
                    .call_static,
                    .call_virtual,
                    .call_quick,
                    .loop_epoch_guard,
                    => true,
                    else => false,
                };
                if (inst.address != null and !permits_address) return error.BadInstruction;
                if (inst.address == null and inst.state_handle != null and inst.opcode != .resolve_handle and
                    inst.opcode != .monitor_enter and inst.opcode != .monitor_exit) return error.BadInstruction;
                if (inst.address == null and inst.opcode != .resolve_handle and (inst.reloc_token != null or inst.resolve_id != null)) return error.BadInstruction;
                if ((inst.opcode == .loop_epoch_guard) != (inst.guard_site_id != null)) return error.BadInstruction;
                const is_monitor = inst.opcode == .monitor_enter or inst.opcode == .monitor_exit;
                if (is_monitor != (inst.monitor_site_id != null)) return error.BadInstruction;
                if (is_monitor) {
                    const expected = std.math.add(u32, monitor_site_base, verified_monitor_sites) catch return error.BadInstruction;
                    if (inst.monitor_site_id.? != expected or inst.defs.len != 0 or inst.uses.len != 1 or
                        inst.state_handle == null or inst.uses[0] != inst.state_handle.?) return error.BadInstruction;
                    verified_monitor_sites += 1;
                }
                const is_mapped_bounds = inst.opcode == .check_bounds and inst.address != null;
                if (is_mapped_bounds != (inst.exception_site_id != null)) return error.BadInstruction;
                if (is_mapped_bounds) {
                    const expected = std.math.add(u32, exception_site_base, verified_exception_sites) catch return error.BadInstruction;
                    if (inst.exception_site_id.? != expected or inst.uses.len != 1) return error.BadInstruction;
                    verified_exception_sites += 1;
                }
                const is_pre_write = inst.opcode == .satb_pre_write or inst.opcode == .static_satb_pre_write;
                const is_post_write = inst.opcode == .card_mark or inst.opcode == .static_root_post_write;
                if (is_pre_write != (inst.pre_write != .none)) return error.BadInstruction;
                if (is_post_write != (inst.post_write != .none)) return error.BadInstruction;
                if (inst.opcode == .satb_pre_write) {
                    if (inst.field_idx == null and inst.uses.len != 1) return error.BadInstruction;
                    if (inst.field_idx != null and inst.uses.len != 0) return error.BadInstruction;
                }
                if (inst.opcode == .static_satb_pre_write and (inst.uses.len != 0 or inst.pre_write != .satb_guarded)) return error.BadInstruction;
                if (inst.opcode == .static_root_post_write and (inst.uses.len != 1 or inst.post_write != .root_guarded)) return error.BadInstruction;
            }
        }
        for (loop_guard_seen) |seen| if (!seen) return error.BadInstruction;
        if (loop_guard_count != self.stats.loop_epoch_guards or loop_guard_count != self.source.stats.loop_epoch_guards) {
            return error.BadInstruction;
        }
        if (verified_exception_sites != self.stats.bounds_exception_sites or
            verified_exception_sites != self.source.stats.bounds_exception_sites) return error.BadInstruction;
        if (verified_monitor_sites != self.stats.monitor_sites or
            verified_monitor_sites != self.source.stats.monitor_sites) return error.BadInstruction;
    }

    fn verifyResolve(self: *const Function, inst: Inst) VerifyError!void {
        if (inst.defs.len != 1 or inst.uses.len != 1 or inst.address != null or inst.state_handle == null or inst.reloc_token == null or inst.resolve_id == null) return error.BadInstruction;
        if (inst.state_handle.? != inst.uses[0]) return error.BadInstruction;
        if (!self.isGcRoot(inst.state_handle.?)) return error.BadInstruction;
        switch (self.runtime_values[inst.defs[0]]) {
            .derived_ptr => |ptr| if (ptr.handle != inst.uses[0] or ptr.token != inst.reloc_token.? or ptr.resolve != inst.resolve_id.?) return error.BadInstruction,
            .dalvik => return error.BadInstruction,
        }
    }

    fn verifyAddress(self: *const Function, inst: Inst) VerifyError!void {
        if ((inst.opcode == .field_load_ptr or inst.opcode == .field_store_ptr) and inst.field_idx == null) return error.BadInstruction;
        const address = inst.address orelse return error.BadInstruction;
        const handle = inst.state_handle orelse return error.BadInstruction;
        const token = inst.reloc_token orelse return error.BadInstruction;
        const resolve_id = inst.resolve_id orelse return error.BadInstruction;
        if (!self.isGcRoot(handle)) return error.BadInstruction;
        switch (self.runtime_values[address]) {
            .derived_ptr => |ptr| {
                if (ptr.token != token or ptr.resolve != resolve_id) return error.BadInstruction;
                switch (self.runtime_values[handle]) {
                    .dalvik => |value| if (value.value != ptr.handle) return error.BadInstruction,
                    .derived_ptr => return error.BadInstruction,
                }
            },
            .dalvik => return error.BadInstruction,
        }
    }

    fn verifyLoopEpochGuard(self: *const Function, block_id: cfg.BlockId, inst: Inst) VerifyError!u32 {
        if (inst.defs.len != 0 or inst.uses.len != 0 or inst.field_idx != null or
            inst.pre_write != .none or inst.post_write != .none) return error.BadInstruction;
        try self.verifyAddress(inst);
        const lowered = self.source;
        const barriers = lowered.barriers orelse return error.BadInstruction;
        const resolve_id = inst.resolve_id orelse return error.BadInstruction;
        if (resolve_id >= barriers.resolves.len) return error.BadInstruction;
        const resolve = barriers.resolves[resolve_id];
        if (resolve.handle != inst.state_handle.? or resolve.token != inst.reloc_token.?) return error.BadInstruction;
        const site_id = inst.guard_site_id.?;
        if (site_id < barriers.resolves.len) return error.BadInstruction;
        const guard_id: u32 = @intCast(site_id - barriers.resolves.len);
        if (guard_id >= barriers.stats.loop_epoch_guards) return error.BadInstruction;
        for (barriers.loop_reuses) |reuse| {
            if (reuse.resolve != resolve_id or guard_id < reuse.guard_start) continue;
            const offset = guard_id - reuse.guard_start;
            if (offset >= reuse.latches.len) continue;
            if (reuse.latches[offset] != block_id) return error.BadInstruction;
            return guard_id;
        }
        return error.BadInstruction;
    }

    pub fn isGcRoot(self: *const Function, reg: RegId) bool {
        if (reg >= self.runtime_values.len) return false;
        return self.runtime_values[reg].isGcRoot();
    }

    pub fn print(self: *const Function, writer: anytype) !void {
        try writer.print(
            "machine_bridge blocks={d} regs={d} insts={d} edge_moves={d} consts={d} branches={d} calls={d} checks={d} resolves={d} loop_guards={d} bounds_exceptions={d} ptr_accesses={d} forwarded={d} cse={d}\n",
            .{
                self.stats.blocks,
                self.reg_types.len,
                self.stats.instructions,
                self.stats.edge_moves,
                self.stats.constants,
                self.stats.branches,
                self.stats.calls,
                self.stats.checks,
                self.stats.resolves,
                self.stats.loop_epoch_guards,
                self.stats.bounds_exception_sites,
                self.stats.pointer_accesses,
                self.stats.forwarded,
                self.stats.cse,
            },
        );
        for (self.edges) |edge| {
            try writer.print("edge b{d}->b{d}", .{ edge.from, edge.to });
            for (edge.moves) |move| try writer.print(" r{d}:{s}=r{d}", .{ move.dst, @tagName(move.ty), move.src });
            try writer.print("\n", .{});
        }
        for (self.blocks) |block| {
            try writer.print("b{d}\n", .{block.id});
            for (block.insts) |inst| {
                try writer.print("  {s}", .{@tagName(inst.opcode)});
                if (inst.pc) |pc| try writer.print(" pc{d}", .{pc});
                if (inst.defs.len != 0) {
                    try writer.print(" defs:", .{});
                    for (inst.defs) |def| try writer.print(" r{d}:{s}", .{ def, @tagName(self.reg_types[def]) });
                }
                if (inst.uses.len != 0) {
                    try writer.print(" uses:", .{});
                    for (inst.uses) |use| try writer.print(" r{d}", .{use});
                }
                if (inst.target) |target| try writer.print(" target=b{d}", .{target});
                if (inst.false_target) |target| try writer.print(" false=b{d}", .{target});
                if (inst.condition) |condition| try writer.print(" cond={s}", .{@tagName(condition)});
                if (inst.field_idx) |field| try writer.print(" field={d}", .{field});
                if (inst.address) |address| try writer.print(" address=r{d}", .{address});
                if (inst.state_handle) |handle| try writer.print(" state=r{d}", .{handle});
                if (inst.reloc_token) |token| try writer.print(" token={d}", .{token});
                if (inst.resolve_id) |resolve| try writer.print(" resolve={d}", .{resolve});
                if (inst.guard_site_id) |site| try writer.print(" guard_site={d}", .{site});
                if (inst.exception_site_id) |site| try writer.print(" exception_site={d}", .{site});
                if (inst.imm != 0 or inst.opcode == .const_i32 or inst.opcode == .const_i64) try writer.print(" imm={d}", .{inst.imm});
                if (inst.flags.null_check_elided) try writer.print(" null_elided", .{});
                if (inst.flags.bounds_check_elided) try writer.print(" bounds_elided", .{});
                if (inst.flags.forwarded) try writer.print(" forwarded", .{});
                if (inst.flags.cse) try writer.print(" cse", .{});
                try writer.print("\n", .{});
            }
        }
    }
};

fn hasSuccessor(graph: *const cfg.Graph, from: cfg.BlockId, to: cfg.BlockId) bool {
    if (from >= graph.blocks.len) return false;
    for (graph.blocks[from].successors) |succ| if (succ == to) return true;
    return false;
}

fn hasLocalSuccessor(successors: []const []const cfg.BlockId, from: cfg.BlockId, to: cfg.BlockId) bool {
    if (from >= successors.len) return false;
    for (successors[from]) |succ| if (succ == to) return true;
    return false;
}

fn mapOpcode(kind: lowering.Kind) ?Opcode {
    return switch (kind) {
        .phi => null,
        .copy => .mov,
        .const_i32 => .const_i32,
        .const_i64 => .const_i64,
        .add_i32 => .add_i32,
        .sub_i32 => .sub_i32,
        .mul_i32 => .mul_i32,
        .div_i32 => .div_i32,
        .rem_i32 => .rem_i32,
        .and_i32 => .and_i32,
        .or_i32 => .or_i32,
        .xor_i32 => .xor_i32,
        .add_i64 => .add_i64,
        .sub_i64 => .sub_i64,
        .mul_i64 => .mul_i64,
        .div_i64 => .div_i64,
        .rem_i64 => .rem_i64,
        .f32_op => .f32_op,
        .f64_op => .f64_op,
        .check_null => .check_null,
        .check_bounds => .check_bounds,
        .resolve_handle => .resolve_handle,
        .loop_epoch_guard => .loop_epoch_guard,
        .array_load => .array_load,
        .array_store => .array_store,
        .field_load => .field_load,
        .field_store => .field_store,
        .array_load_ptr => .array_load_ptr,
        .array_store_ptr => .array_store_ptr,
        .field_load_ptr => .field_load_ptr,
        .field_store_ptr => .field_store_ptr,
        .satb_pre_write => .satb_pre_write,
        .card_mark => .card_mark,
        .static_satb_pre_write => .static_satb_pre_write,
        .static_root_post_write => .static_root_post_write,
        .static_load => .static_load,
        .static_store => .static_store,
        .call_direct => .call_direct,
        .call_static => .call_static,
        .call_virtual => .call_virtual,
        .call_quick => .call_quick,
        .monitor_enter => .monitor_enter,
        .monitor_exit => .monitor_exit,
        .branch => .jump,
        .cond_branch => .branch,
        .switch_ => .switch_,
        .ret => .ret,
        .throw_ => .throw_,
        .memory_barrier => .memory_barrier,
        .unsupported => .unsupported,
    };
}

fn toRegs(allocator: std.mem.Allocator, values: []const lowering.ValueId) ![]RegId {
    const regs = try allocator.alloc(RegId, values.len);
    for (values, 0..) |value, i| regs[i] = value;
    return regs;
}

const OwnedRegs = struct {
    defs: []RegId,
    uses: []RegId,
};

fn ownRegs(
    allocator: std.mem.Allocator,
    defs: []const lowering.ValueId,
    uses: []const lowering.ValueId,
) !OwnedRegs {
    const owned_defs = try toRegs(allocator, defs);
    errdefer allocator.free(owned_defs);
    return .{
        .defs = owned_defs,
        .uses = try toRegs(allocator, uses),
    };
}

fn appendInst(allocator: std.mem.Allocator, list: *std.ArrayList(Inst), inst: Inst) !void {
    list.append(allocator, inst) catch |err| {
        allocator.free(inst.defs);
        allocator.free(inst.uses);
        return err;
    };
}

fn flagsFrom(inst: lowering.Inst) Flags {
    return .{
        .null_check_elided = inst.flags.null_check_elided,
        .bounds_check_elided = inst.flags.bounds_check_elided,
        .forwarded = inst.flags.forwarded,
        .cse = inst.flags.cse,
    };
}

fn conditionFromInst(inst: instmod.Instruction) ?Condition {
    return switch (inst) {
        .if_eq, .if_eqz => .eq,
        .if_ne, .if_nez => .ne,
        .if_lt, .if_ltz => .lt,
        .if_ge, .if_gez => .ge,
        .if_gt, .if_gtz => .gt,
        .if_le, .if_lez => .le,
        else => null,
    };
}

fn conditionForLowered(source: *const lowering.Function, lowered: lowering.Inst) ?Condition {
    const pc = lowered.pc orelse return null;
    if (pc >= source.source.pc_to_op.len) return null;
    const ref = source.source.pc_to_op[pc] orelse return null;
    if (ref.block >= source.source.blocks.len) return null;
    const block = source.source.blocks[ref.block];
    if (ref.index >= block.ops.len) return null;
    return conditionFromInst(block.ops[ref.index].inst);
}

fn updateStats(inst: Inst, stats: *Stats) void {
    stats.instructions += 1;
    switch (inst.opcode) {
        .const_i32, .const_i64 => stats.constants += 1,
        .jump, .branch, .switch_ => stats.branches += 1,
        .call_direct, .call_static, .call_virtual, .call_quick => stats.calls += 1,
        .monitor_enter, .monitor_exit => stats.monitor_sites += 1,
        .check_null => stats.checks += 1,
        .check_bounds => {
            stats.checks += 1;
            if (inst.exception_site_id != null) stats.bounds_exception_sites += 1;
        },
        .resolve_handle => stats.resolves += 1,
        .loop_epoch_guard => stats.loop_epoch_guards += 1,
        .array_load_ptr, .array_store_ptr, .field_load_ptr, .field_store_ptr => stats.pointer_accesses += 1,
        else => {},
    }
    if (inst.flags.forwarded) stats.forwarded += 1;
    if (inst.flags.cse) stats.cse += 1;
}

fn setMachineType(reg_types: []typedir.Type, reg: RegId, ty: typedir.Type) Error!void {
    if (reg >= reg_types.len) return error.InvalidLowering;
    const current = reg_types[reg];
    if (current == ty) return;
    if (current == .unknown or
        (current == .int and ty == .float) or
        (current == .long and ty == .double))
    {
        reg_types[reg] = ty;
        return;
    }
    return error.InvalidLowering;
}

/// Dalvik constants and incoming parameters are physically untyped bit
/// patterns. Once scalar arithmetic consumes them, specialize their machine
/// register class before interval construction so GP and XMM values can never
/// share an incorrectly classified allocation.
fn refineScalarXmmTypes(reg_types: []typedir.Type, source: *const lowering.Function) Error!void {
    for (source.blocks) |block| for (block.insts) |inst| {
        const operation = inst.float_op orelse continue;
        const ty: typedir.Type = switch (inst.kind) {
            .f32_op => .float,
            .f64_op => .double,
            else => return error.InvalidLowering,
        };
        switch (operation) {
            .add, .sub, .mul, .div, .rem, .neg => {
                for (inst.defs) |reg| try setMachineType(reg_types, reg, ty);
                for (inst.uses) |reg| try setMachineType(reg_types, reg, ty);
            },
            .compare_l, .compare_g => {
                for (inst.defs) |reg| try setMachineType(reg_types, reg, .int);
                for (inst.uses) |reg| try setMachineType(reg_types, reg, ty);
            },
            .int_to_float => {
                for (inst.defs) |reg| try setMachineType(reg_types, reg, .float);
                for (inst.uses) |reg| try setMachineType(reg_types, reg, .int);
            },
            .int_to_double => {
                for (inst.defs) |reg| try setMachineType(reg_types, reg, .double);
                for (inst.uses) |reg| try setMachineType(reg_types, reg, .int);
            },
            .long_to_float => {
                for (inst.defs) |reg| try setMachineType(reg_types, reg, .float);
                for (inst.uses) |reg| try setMachineType(reg_types, reg, .long);
            },
            .long_to_double => {
                for (inst.defs) |reg| try setMachineType(reg_types, reg, .double);
                for (inst.uses) |reg| try setMachineType(reg_types, reg, .long);
            },
            .float_to_int => {
                for (inst.defs) |reg| try setMachineType(reg_types, reg, .int);
                for (inst.uses) |reg| try setMachineType(reg_types, reg, .float);
            },
            .float_to_long => {
                for (inst.defs) |reg| try setMachineType(reg_types, reg, .long);
                for (inst.uses) |reg| try setMachineType(reg_types, reg, .float);
            },
            .float_to_double => {
                for (inst.defs) |reg| try setMachineType(reg_types, reg, .double);
                for (inst.uses) |reg| try setMachineType(reg_types, reg, .float);
            },
            .double_to_int => {
                for (inst.defs) |reg| try setMachineType(reg_types, reg, .int);
                for (inst.uses) |reg| try setMachineType(reg_types, reg, .double);
            },
            .double_to_long => {
                for (inst.defs) |reg| try setMachineType(reg_types, reg, .long);
                for (inst.uses) |reg| try setMachineType(reg_types, reg, .double);
            },
            .double_to_float => {
                for (inst.defs) |reg| try setMachineType(reg_types, reg, .float);
                for (inst.uses) |reg| try setMachineType(reg_types, reg, .double);
            },
        }
    };
}

pub fn build(allocator: std.mem.Allocator, source: *const lowering.Function) Error!Function {
    source.verify() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidLowering,
    };

    const reg_types = try allocator.dupe(typedir.Type, source.value_types);
    errdefer allocator.free(reg_types);
    try refineScalarXmmTypes(reg_types, source);
    const runtime_values = try allocator.dupe(lowering.RuntimeValueClass, source.runtime_values);
    errdefer allocator.free(runtime_values);

    const value_kinds = try allocator.alloc(ssa.ValueKind, source.source.values.len);
    errdefer allocator.free(value_kinds);
    for (source.source.values, 0..) |value, i| value_kinds[i] = value.kind;

    const successors = try allocator.alloc([]cfg.BlockId, source.source.graph.blocks.len);
    errdefer allocator.free(successors);
    var built_successors: usize = 0;
    errdefer {
        for (successors[0..built_successors]) |succs| allocator.free(succs);
    }
    for (source.source.graph.blocks, 0..) |block, i| {
        successors[i] = try allocator.dupe(cfg.BlockId, block.successors);
        built_successors += 1;
    }

    var edge_lists = try allocator.alloc(std.ArrayList(Move), source.blocks.len * source.blocks.len);
    defer allocator.free(edge_lists);
    for (edge_lists) |*list| list.* = .empty;
    defer {
        for (edge_lists) |*list| list.deinit(allocator);
    }

    for (source.source.blocks) |block| {
        for (block.phis) |phi| {
            for (phi.incoming) |incoming| {
                const index = incoming.pred * source.blocks.len + block.id;
                try edge_lists[index].append(allocator, .{
                    .dst = phi.dest,
                    .src = incoming.value,
                    .ty = if (phi.dest < reg_types.len) reg_types[phi.dest] else .unknown,
                });
            }
        }
    }

    var edge_count: usize = 0;
    var edge_move_count: u32 = 0;
    for (edge_lists, 0..) |list, i| {
        if (list.items.len == 0) continue;
        const from: cfg.BlockId = @intCast(i / source.blocks.len);
        const to: cfg.BlockId = @intCast(i % source.blocks.len);
        if (!hasSuccessor(source.source.graph, from, to)) return error.InvalidLowering;
        edge_count += 1;
        edge_move_count += @intCast(list.items.len);
    }

    const edges = try allocator.alloc(EdgeMoves, edge_count);
    var built_edges: usize = 0;
    errdefer {
        for (edges[0..built_edges]) |edge| allocator.free(edge.moves);
        allocator.free(edges);
    }
    var edge_i: usize = 0;
    for (edge_lists, 0..) |list, i| {
        if (list.items.len == 0) continue;
        edges[edge_i] = .{
            .from = @intCast(i / source.blocks.len),
            .to = @intCast(i % source.blocks.len),
            .moves = try allocator.dupe(Move, list.items),
        };
        edge_i += 1;
        built_edges += 1;
    }

    const blocks = try allocator.alloc(Block, source.blocks.len);
    errdefer allocator.free(blocks);
    var built_blocks: usize = 0;
    errdefer {
        for (blocks[0..built_blocks]) |block| {
            for (block.insts) |inst| {
                allocator.free(inst.defs);
                allocator.free(inst.uses);
            }
            allocator.free(block.insts);
        }
    }

    var stats: Stats = .{ .blocks = @intCast(source.blocks.len), .edge_moves = edge_move_count };
    for (source.blocks, 0..) |block, block_i| {
        var list: std.ArrayList(Inst) = .empty;
        errdefer {
            for (list.items) |inst| {
                allocator.free(inst.defs);
                allocator.free(inst.uses);
            }
            list.deinit(allocator);
        }

        for (block.insts) |lowered| {
            const opcode = mapOpcode(lowered.kind) orelse continue;
            const regs = try ownRegs(allocator, lowered.defs, lowered.uses);
            const inst = Inst{
                .opcode = opcode,
                .pc = lowered.pc,
                .defs = regs.defs,
                .uses = regs.uses,
                .target = lowered.target,
                .false_target = lowered.false_target,
                .condition = if (opcode == .branch) conditionForLowered(source, lowered) else null,
                .imm = lowered.imm,
                .float_op = lowered.float_op,
                .field_idx = lowered.field_idx,
                .address = lowered.address,
                .state_handle = lowered.state_handle,
                .reloc_token = lowered.reloc_token,
                .resolve_id = lowered.resolve_id,
                .guard_site_id = lowered.guard_site_id,
                .exception_site_id = lowered.exception_site_id,
                .monitor_site_id = lowered.monitor_site_id,
                .pre_write = lowered.pre_write,
                .post_write = lowered.post_write,
                .flags = flagsFrom(lowered),
            };
            try appendInst(allocator, &list, inst);
            updateStats(inst, &stats);
        }

        blocks[block_i] = .{ .id = @intCast(block_i), .insts = try list.toOwnedSlice(allocator) };
        built_blocks += 1;
    }

    return .{
        .allocator = allocator,
        .source = source,
        .blocks = blocks,
        .edges = edges,
        .reg_types = reg_types,
        .runtime_values = runtime_values,
        .value_kinds = value_kinds,
        .successors = successors,
        .stats = stats,
    };
}

const TestPipeline = struct {
    graph: cfg.Graph,
    tree: dom.Tree,
    function: ssa.Function,
    facts: ssa_phase.Result,
    types: typedir.Function,
    typed: typed_ir.Function,
    memory: memory_phase.Result,
    lowered: lowering.Function,

    fn deinit(self: *TestPipeline) void {
        self.lowered.deinit();
        self.memory.deinit();
        self.typed.deinit();
        self.types.deinit();
        self.facts.deinit();
        self.function.deinit();
        self.tree.deinit();
        self.graph.deinit();
    }
};

fn initTestPipeline(allocator: std.mem.Allocator, insts: []const Instruction, pipeline: *TestPipeline) !void {
    pipeline.graph = try cfg.build(allocator, insts);
    errdefer pipeline.graph.deinit();
    pipeline.tree = try dom.build(allocator, &pipeline.graph);
    errdefer pipeline.tree.deinit();
    pipeline.function = try ssa.build(allocator, &pipeline.graph, &pipeline.tree);
    errdefer pipeline.function.deinit();
    pipeline.facts = try ssa_phase.run(allocator, &pipeline.function);
    errdefer pipeline.facts.deinit();
    pipeline.types = try typedir.build(allocator, &pipeline.function);
    errdefer pipeline.types.deinit();
    pipeline.typed = try typed_ir.build(allocator, &pipeline.function, &pipeline.types, null);
    errdefer pipeline.typed.deinit();
    pipeline.memory = try memory_phase.run(allocator, &pipeline.function, &pipeline.types);
    errdefer pipeline.memory.deinit();
    pipeline.lowered = try lowering.build(allocator, .{
        .function = &pipeline.function,
        .types = &pipeline.types,
        .typed = &pipeline.typed,
        .ssa_facts = &pipeline.facts,
        .memory = &pipeline.memory,
    });
}

test "machine_bridge converts lowered ir into register machine instructions" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var pipeline: TestPipeline = undefined;
    try initTestPipeline(std.testing.allocator, &insts, &pipeline);
    defer pipeline.deinit();

    var machine = try build(std.testing.allocator, &pipeline.lowered);
    defer machine.deinit();

    try machine.verify();
    try std.testing.expect(machine.stats.instructions > 0);
    try std.testing.expect(machine.stats.constants >= 1);
}

test "machine_bridge lowers phi nodes into parallel edge moves" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .if_eqz = .{ .src = 0, .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 10 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 20 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var pipeline: TestPipeline = undefined;
    try initTestPipeline(std.testing.allocator, &insts, &pipeline);
    defer pipeline.deinit();

    var machine = try build(std.testing.allocator, &pipeline.lowered);
    defer machine.deinit();

    try machine.verify();
    try std.testing.expect(machine.stats.edge_moves >= 2);
    for (machine.blocks) |block| {
        for (block.insts) |inst| try std.testing.expect(inst.opcode != .unsupported or inst.defs.len != 0 or inst.uses.len != 0);
    }
}

test "machine_bridge preserves branch targets and memory flags" {
    const insts = [_]Instruction{
        .{ .new_instance = .{ .dest = 0, .type_idx = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 7 } },
        .{ .iput = .{ .field_idx = 10, .dest_or_src = 1, .obj = 0 } },
        .{ .iget = .{ .field_idx = 10, .dest_or_src = 2, .obj = 0 } },
        .{ .if_eqz = .{ .src = 2, .offset = 2 } },
        .return_void,
        .{ .return_ = .{ .src = 2 } },
    };
    var pipeline: TestPipeline = undefined;
    try initTestPipeline(std.testing.allocator, &insts, &pipeline);
    defer pipeline.deinit();

    var machine = try build(std.testing.allocator, &pipeline.lowered);
    defer machine.deinit();

    try machine.verify();
    try std.testing.expect(machine.stats.branches >= 1);
    try std.testing.expect(machine.stats.forwarded >= 1);
}

test "machine_bridge print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .return_void,
    };
    var pipeline: TestPipeline = undefined;
    try initTestPipeline(std.testing.allocator, &insts, &pipeline);
    defer pipeline.deinit();

    var machine = try build(std.testing.allocator, &pipeline.lowered);
    defer machine.deinit();

    var buf: [1024]u8 = undefined;
    var stream = std.Io.Writer.fixed(&buf);
    try machine.print(&stream);
    const output = stream.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "machine_bridge blocks=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "ret") != null);
}

test "machine coverage proves joins loop backedges and complete phi edges" {
    const join_insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .if_eqz = .{ .src = 0, .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 10 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 20 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var join_pipeline: TestPipeline = undefined;
    try initTestPipeline(std.testing.allocator, &join_insts, &join_pipeline);
    defer join_pipeline.deinit();
    var join_machine = try build(std.testing.allocator, &join_pipeline.lowered);
    defer join_machine.deinit();
    const join_coverage = try join_machine.verifyCoverage(std.testing.allocator);
    try std.testing.expect(join_coverage.join_blocks >= 1);
    try std.testing.expect(join_coverage.phi_edge_moves >= 2);
    try std.testing.expectEqual(@as(u32, @intCast(join_pipeline.graph.blocks.len)), join_coverage.reachable_blocks);

    const loop_insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 0 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .const_ = .{ .dest = 2, .value = 3 } },
        .{ .add_int = .{ .dest = 0, .src1 = 0, .src2 = 1 } },
        .{ .if_lt = .{ .src1 = 0, .src2 = 2, .offset = -1 } },
        .{ .return_ = .{ .src = 0 } },
    };
    var loop_pipeline: TestPipeline = undefined;
    try initTestPipeline(std.testing.allocator, &loop_insts, &loop_pipeline);
    defer loop_pipeline.deinit();
    var loop_machine = try build(std.testing.allocator, &loop_pipeline.lowered);
    defer loop_machine.deinit();
    const loop_coverage = try loop_machine.verifyCoverage(std.testing.allocator);
    try std.testing.expect(loop_coverage.loop_edges >= 1);
    try std.testing.expect(loop_coverage.join_blocks >= 1);
    try std.testing.expect(loop_coverage.phi_edge_moves >= 2);
}

test "machine coverage rejects CFG phi and safepoint holes" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .if_eqz = .{ .src = 0, .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 10 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 20 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var pipeline: TestPipeline = undefined;
    try initTestPipeline(std.testing.allocator, &insts, &pipeline);
    defer pipeline.deinit();
    var machine = try build(std.testing.allocator, &pipeline.lowered);
    defer machine.deinit();
    _ = try machine.verifyCoverage(std.testing.allocator);

    const entry = pipeline.graph.entry;
    const saved_successor = machine.successors[entry][0];
    machine.successors[entry][0] = entry;
    try std.testing.expectError(error.BadBlock, machine.verifyCoverage(std.testing.allocator));
    machine.successors[entry][0] = saved_successor;

    try std.testing.expect(machine.edges.len != 0 and machine.edges[0].moves.len != 0);
    const saved_source = machine.edges[0].moves[0].src;
    machine.edges[0].moves[0].src = @intCast(machine.reg_types.len);
    try std.testing.expectError(error.BadEdgeMove, machine.verifyCoverage(std.testing.allocator));
    machine.edges[0].moves[0].src = saved_source;

    machine.stats.resolves += 1;
    try std.testing.expectError(error.BadInstruction, machine.verifyCoverage(std.testing.allocator));
    machine.stats.resolves -= 1;
    _ = try machine.verifyCoverage(std.testing.allocator);
}

fn coverageAllocationFailureProbe(allocator: std.mem.Allocator) !void {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 0 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .const_ = .{ .dest = 2, .value = 2 } },
        .{ .add_int = .{ .dest = 0, .src1 = 0, .src2 = 1 } },
        .{ .if_lt = .{ .src1 = 0, .src2 = 2, .offset = -1 } },
        .{ .return_ = .{ .src = 0 } },
    };
    var pipeline: TestPipeline = undefined;
    try initTestPipeline(allocator, &insts, &pipeline);
    defer pipeline.deinit();
    var machine = try build(allocator, &pipeline.lowered);
    defer machine.deinit();
    _ = try machine.verifyCoverage(allocator);
}

test "machine CFG coverage is leak free at every allocation failure" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        coverageAllocationFailureProbe,
        .{},
    );
}
