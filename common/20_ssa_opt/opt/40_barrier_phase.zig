//! Relocation-aware object-access and GC-barrier planning.
//!
//! Handles remain the canonical SSA identity. This pass assigns relocation
//! tokens and performs dominator-scoped value numbering for temporary resolved
//! addresses. It records an immutable plan; it does not rewrite SSA or expose
//! raw pointers to the collector.

const std = @import("std");
const cfg = @import("cfg");
const dom = @import("dominator");
const ssa = @import("ssa");
const typedir = @import("typedir");
const loop_phase = @import("loop_phase");
const instmod = @import("instructions");
const Instruction = instmod.Instruction;

pub const RelocTokenId = u32;
pub const ResolveId = u32;
pub const INVALID_TOKEN: RelocTokenId = std.math.maxInt(RelocTokenId);

pub const Error = error{
    InvalidInput,
    InvalidPlan,
    TooManyResolves,
    TooManyTokens,
    OutOfMemory,
};

pub const CallEffects = packed struct(u8) {
    may_safepoint: bool = true,
    may_relocate: bool = true,
    may_read_heap: bool = true,
    may_write_heap: bool = true,
    _reserved: u4 = 0,
};

pub const Options = struct {
    /// Method targets in this list are trusted runtime/compiler leaf calls.
    /// They preserve the current relocation token but may still alias heap
    /// memory. Unknown, native, and unresolved calls remain conservative.
    no_safepoint_leaf_targets: []const u32 = &.{},
    /// Optional natural-loop analysis enables strictly proven preheader
    /// placement. The result must describe this exact SSA function/tree.
    loops: ?*const loop_phase.Result = null,
};

pub const PreWriteBarrier = enum(u2) {
    none,
    satb,
    satb_guarded,
    /// The same exact field slot was already processed in this straight-line
    /// alias region. Runtime may skip the slot load only when its cached mark
    /// epoch still matches the active collector epoch.
    satb_repeat_guarded,
};

pub const PostWriteBarrier = enum(u3) {
    none,
    card,
    card_guarded,
    /// The previous reference store used the same canonical destination. The
    /// runtime may coalesce only while the destination card is still dirty.
    card_repeat_guarded,
    /// Static reference slots are precise roots. They require active-mark
    /// publication, never a remembered-set card for a fabricated object.
    root_guarded,
};

pub const ResolveUse = union(enum) {
    none,
    define: ResolveId,
    reuse: ResolveId,
};

pub const Effects = struct {
    relocation_kill: bool = false,
    heap_alias_kill: bool = false,
    may_safepoint: bool = false,
    pre_write: PreWriteBarrier = .none,
    post_write: PostWriteBarrier = .none,
};

pub const OpPlan = struct {
    token_in: RelocTokenId = INVALID_TOKEN,
    token_out: RelocTokenId = INVALID_TOKEN,
    base_handle: ?ssa.ValueId = null,
    resolve: ResolveUse = .none,
    relocation_kill: bool = false,
    heap_alias_kill: bool = false,
    may_safepoint: bool = false,
    pre_write: PreWriteBarrier = .none,
    post_write: PostWriteBarrier = .none,
};

pub const Resolve = struct {
    handle: ssa.ValueId,
    token: RelocTokenId,
    defining_op: ssa.OpRef,
    placement_block: cfg.BlockId,
    hoisted: bool = false,
    loop_header: ?cfg.BlockId = null,
    /// Backedge that must compare the thread's acknowledged relocation epoch
    /// with the current request epoch before another iteration can reuse this
    /// derived address.
    loop_latch: ?cfg.BlockId = null,
    /// Dense id among guarded loop polling sites. Runtime stack-map ids place
    /// these after ordinary resolve ids.
    guard_id: ?u32 = null,
};

pub const Stats = struct {
    tokens: u32 = 0,
    resolves_inserted: u32 = 0,
    resolves_reused: u32 = 0,
    relocation_kills: u32 = 0,
    safepoints: u32 = 0,
    heap_alias_kills: u32 = 0,
    satb_barriers: u32 = 0,
    satb_repeat_barriers: u32 = 0,
    card_barriers: u32 = 0,
    card_repeat_barriers: u32 = 0,
    loop_resolves_hoisted: u32 = 0,
    loop_epoch_guards: u32 = 0,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    function: *const ssa.Function,
    types: *const typedir.Function,
    loops: ?*const loop_phase.Result,
    ops: [][]OpPlan,
    resolves: []Resolve,
    canonical_handles: []ssa.ValueId,
    block_in: []RelocTokenId,
    block_out: []RelocTokenId,
    token_count: u32,
    stats: Stats,

    pub fn deinit(self: *Result) void {
        self.allocator.free(self.block_out);
        self.allocator.free(self.block_in);
        self.allocator.free(self.canonical_handles);
        self.allocator.free(self.resolves);
        for (self.ops) |plans| self.allocator.free(plans);
        self.allocator.free(self.ops);
        self.* = undefined;
    }

    pub fn verify(self: *const Result) Error!void {
        self.function.verify() catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidInput,
        };
        self.types.verify() catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidInput,
        };
        if (self.types.source != self.function) return error.InvalidInput;
        if (self.loops) |loops| {
            if (loops.function != self.function or loops.tree != self.function.tree) return error.InvalidInput;
        }
        if (self.ops.len != self.function.blocks.len or
            self.block_in.len != self.function.blocks.len or
            self.block_out.len != self.function.blocks.len or
            self.canonical_handles.len != self.function.values.len or
            self.token_count == 0)
        {
            return error.InvalidPlan;
        }

        const defined = try self.allocator.alloc(bool, self.resolves.len);
        defer self.allocator.free(defined);
        @memset(defined, false);

        var inserted: u32 = 0;
        var reused: u32 = 0;
        var kills: u32 = 0;
        var safepoints: u32 = 0;
        var alias_kills: u32 = 0;
        var satb_barriers: u32 = 0;
        var satb_repeats: u32 = 0;
        var card_barriers: u32 = 0;
        var card_repeats: u32 = 0;
        var loop_hoists: u32 = 0;
        var loop_guards: u32 = 0;

        for (self.resolves, 0..) |resolve, resolve_index| {
            if (!resolve.hoisted) {
                if (resolve.loop_header != null or resolve.loop_latch != null or resolve.guard_id != null) return error.InvalidPlan;
                continue;
            }
            const header = resolve.loop_header orelse return error.InvalidPlan;
            const latch = resolve.loop_latch orelse return error.InvalidPlan;
            const guard_id = resolve.guard_id orelse return error.InvalidPlan;
            if (guard_id != loop_guards) return error.InvalidPlan;
            const analysis = self.loops orelse return error.InvalidPlan;
            var matched_loop: ?loop_phase.Loop = null;
            for (analysis.loops) |loop| {
                if (loop.header == header and loop.latch == latch and loop.preheader == resolve.placement_block) {
                    matched_loop = loop;
                    break;
                }
            }
            const loop = matched_loop orelse return error.InvalidPlan;
            if (resolve.defining_op.block != header or resolve.defining_op.index != 0 or
                !self.function.tree.dominates(resolve.placement_block, header) or
                hasOutgoingException(self.function, header)) return error.InvalidPlan;
            const successors = self.function.graph.blocks[resolve.placement_block].successors;
            if (successors.len != 1 or successors[0] != header) return error.InvalidPlan;
            if (self.function.blocks[header].ops.len == 0) return error.InvalidPlan;
            const op = self.function.blocks[header].ops[0];
            const raw_handle = rawBaseHandle(op) orelse return error.InvalidPlan;
            if (raw_handle >= self.canonical_handles.len or self.canonical_handles[raw_handle] != resolve.handle or
                !valueAvailableAtPlacement(self.function, resolve.handle, resolve.placement_block) or
                loopContainsBlock(loop, self.function.values[resolve.handle].block)) return error.InvalidPlan;
            const defining_plan = self.ops[header][0];
            if (defining_plan.token_in != resolve.token or defining_plan.base_handle != resolve.handle) return error.InvalidPlan;
            switch (defining_plan.resolve) {
                .reuse => |id| if (id != resolve_index) return error.InvalidPlan,
                else => return error.InvalidPlan,
            }
            for (loop.blocks) |block_id| {
                for (self.ops[block_id]) |plan| {
                    if (plan.relocation_kill or plan.may_safepoint) return error.InvalidPlan;
                }
            }
            defined[resolve_index] = true;
            inserted += 1;
            loop_hoists += 1;
            loop_guards += 1;
        }

        for (self.function.blocks, 0..) |block, block_index| {
            if (self.ops[block_index].len != block.ops.len) return error.InvalidPlan;
            var token = self.block_in[block_index];
            var previous_reference_slot: ?ReferenceFieldSlot = null;
            var previous_card_handle: ?ssa.ValueId = null;
            if (token >= self.token_count) return error.InvalidPlan;

            for (block.ops, 0..) |op, op_index| {
                const plan = self.ops[block_index][op_index];
                if (plan.token_in != token or plan.token_in >= self.token_count or plan.token_out >= self.token_count) {
                    return error.InvalidPlan;
                }
                if (plan.may_safepoint and !plan.relocation_kill) return error.InvalidPlan;
                if (plan.relocation_kill) {
                    if (plan.token_out == plan.token_in) return error.InvalidPlan;
                    kills += 1;
                } else if (plan.token_out != plan.token_in) {
                    return error.InvalidPlan;
                }
                if (plan.may_safepoint) safepoints += 1;
                if (plan.heap_alias_kill) alias_kills += 1;
                token = plan.token_out;

                const effects = classify(op.inst, .{});
                const reference_slot = try referenceFieldSlot(op, self.canonical_handles);
                const expected_pre: PreWriteBarrier = if (effects.pre_write == .none)
                    .none
                else if (reference_slot) |slot|
                    if (previous_reference_slot != null and previous_reference_slot.?.eql(slot))
                        .satb_repeat_guarded
                    else
                        .satb_guarded
                else
                    effects.pre_write;
                const expected_post: PostWriteBarrier = if (effects.post_write == .none)
                    .none
                else if (reference_slot) |slot|
                    if (previous_card_handle != null and previous_card_handle.? == slot.handle)
                        .card_repeat_guarded
                    else
                        .card_guarded
                else
                    effects.post_write;
                if (plan.pre_write != expected_pre or plan.post_write != expected_post) return error.InvalidPlan;
                if (plan.pre_write != .none) satb_barriers += 1;
                if (plan.pre_write == .satb_repeat_guarded) satb_repeats += 1;
                if (plan.post_write != .none) card_barriers += 1;
                if (plan.post_write == .card_repeat_guarded) card_repeats += 1;
                if (reference_slot) |slot| {
                    previous_reference_slot = slot;
                    previous_card_handle = slot.handle;
                } else if (plan.heap_alias_kill) {
                    previous_reference_slot = null;
                    previous_card_handle = null;
                }

                const expected_raw_base = rawBaseHandle(op);
                if (expected_raw_base == null) {
                    if (plan.base_handle != null or plan.resolve != .none) return error.InvalidPlan;
                    continue;
                }
                const raw_base = expected_raw_base.?;
                if (raw_base >= self.canonical_handles.len) return error.InvalidPlan;
                const expected_base = self.canonical_handles[raw_base];
                if (plan.base_handle == null or plan.base_handle.? != expected_base) return error.InvalidPlan;
                if (!handleTypeCompatible(self.types, expected_base)) return error.InvalidPlan;

                switch (plan.resolve) {
                    .none => return error.InvalidPlan,
                    .define => |id| {
                        if (id >= self.resolves.len or defined[id]) return error.InvalidPlan;
                        const resolve = self.resolves[id];
                        if (resolve.hoisted) return error.InvalidPlan;
                        if (resolve.handle != expected_base or resolve.token != plan.token_in or
                            resolve.defining_op.block != block.id or resolve.defining_op.index != op_index or
                            resolve.placement_block != block.id)
                        {
                            return error.InvalidPlan;
                        }
                        defined[id] = true;
                        inserted += 1;
                    },
                    .reuse => |id| {
                        if (id >= self.resolves.len) return error.InvalidPlan;
                        const resolve = self.resolves[id];
                        if (resolve.handle != expected_base or resolve.token != plan.token_in) return error.InvalidPlan;
                        if (!self.function.tree.dominates(resolve.placement_block, block.id)) return error.InvalidPlan;
                        if (resolve.placement_block == block.id and resolve.defining_op.index >= op_index) return error.InvalidPlan;
                        reused += 1;
                    },
                }
            }
            if (self.block_out[block_index] != token) return error.InvalidPlan;
        }

        for (defined) |is_defined| if (!is_defined) return error.InvalidPlan;
        if (inserted != self.stats.resolves_inserted or reused != self.stats.resolves_reused or
            kills != self.stats.relocation_kills or safepoints != self.stats.safepoints or
            alias_kills != self.stats.heap_alias_kills or satb_barriers != self.stats.satb_barriers or
            satb_repeats != self.stats.satb_repeat_barriers or card_barriers != self.stats.card_barriers or
            card_repeats != self.stats.card_repeat_barriers or
            loop_hoists != self.stats.loop_resolves_hoisted or loop_guards != self.stats.loop_epoch_guards or
            inserted != self.resolves.len or self.stats.tokens != self.token_count)
        {
            return error.InvalidPlan;
        }
    }

    pub fn print(self: *const Result, writer: anytype) !void {
        try writer.print(
            "barrier_phase tokens={d} resolves={d} reused={d} loop_hoists={d} loop_guards={d} relocation_kills={d} safepoints={d} alias_kills={d} satb={d} satb_repeats={d} cards={d} card_repeats={d}\n",
            .{ self.stats.tokens, self.stats.resolves_inserted, self.stats.resolves_reused, self.stats.loop_resolves_hoisted, self.stats.loop_epoch_guards, self.stats.relocation_kills, self.stats.safepoints, self.stats.heap_alias_kills, self.stats.satb_barriers, self.stats.satb_repeat_barriers, self.stats.card_barriers, self.stats.card_repeat_barriers },
        );
        for (self.function.graph.rpo) |block_id| {
            try writer.print("b{d} token={d}->{d}\n", .{ block_id, self.block_in[block_id], self.block_out[block_id] });
            for (self.function.blocks[block_id].ops, 0..) |op, i| {
                const plan = self.ops[block_id][i];
                try writer.print(
                    "  pc{d} {s} token={d}->{d} relocate={} safepoint={} alias={}",
                    .{ op.pc, @tagName(op.inst), plan.token_in, plan.token_out, plan.relocation_kill, plan.may_safepoint, plan.heap_alias_kill },
                );
                if (plan.base_handle) |handle| try writer.print(" base=v{d}", .{handle});
                switch (plan.resolve) {
                    .none => {},
                    .define => |id| try writer.print(" resolve=r{d}", .{id}),
                    .reuse => |id| try writer.print(" reuse=r{d}", .{id}),
                }
                if (plan.pre_write != .none) try writer.print(" pre={s}", .{@tagName(plan.pre_write)});
                if (plan.post_write != .none) try writer.print(" post={s}", .{@tagName(plan.post_write)});
                try writer.print("\n", .{});
            }
        }
    }
};

const ResolveKey = struct {
    handle: ssa.ValueId,
    token: RelocTokenId,
};

const ReferenceFieldSlot = struct {
    handle: ssa.ValueId,
    field_idx: u32,
    quick: bool,

    fn eql(self: ReferenceFieldSlot, other: ReferenceFieldSlot) bool {
        return self.handle == other.handle and self.field_idx == other.field_idx and self.quick == other.quick;
    }
};

const WalkFrame = struct {
    block: cfg.BlockId,
    next_child: usize = 0,
    undo_mark: usize = 0,
    entered: bool = false,
};

const HoistCandidate = struct {
    key: ResolveKey,
    defining_op: ssa.OpRef,
    placement_block: cfg.BlockId,
    loop_header: cfg.BlockId,
    loop_latch: cfg.BlockId,
};

fn loopContainsBlock(loop: loop_phase.Loop, block: cfg.BlockId) bool {
    for (loop.blocks) |candidate| if (candidate == block) return true;
    return false;
}

fn hasOutgoingException(function: *const ssa.Function, block: cfg.BlockId) bool {
    for (function.graph.edges) |edge| {
        if (edge.from == block and edge.kind == .exception) return true;
    }
    return false;
}

fn valueAvailableAtPlacement(function: *const ssa.Function, value: ssa.ValueId, placement: cfg.BlockId) bool {
    if (value >= function.values.len) return false;
    const owner = function.values[value];
    return owner.kind == .parameter or function.tree.dominates(owner.block, placement);
}

fn appendHoistCandidate(
    list: *std.ArrayList(HoistCandidate),
    allocator: std.mem.Allocator,
    candidate: HoistCandidate,
) !void {
    for (list.items) |existing| {
        if (existing.key.handle == candidate.key.handle and existing.key.token == candidate.key.token and
            existing.placement_block == candidate.placement_block) return;
    }
    try list.append(allocator, candidate);
}

fn buildHoistCandidates(
    allocator: std.mem.Allocator,
    function: *const ssa.Function,
    canonical: []const ssa.ValueId,
    plans: []const []const OpPlan,
    loops: ?*const loop_phase.Result,
) Error![]HoistCandidate {
    const analysis = loops orelse return allocator.alloc(HoistCandidate, 0);
    if (analysis.function != function or analysis.tree != function.tree) return error.InvalidInput;
    var candidates: std.ArrayList(HoistCandidate) = .empty;
    errdefer candidates.deinit(allocator);

    for (analysis.loops) |loop| {
        const preheader = loop.preheader orelse continue;
        if (preheader >= function.blocks.len or loop.header >= function.blocks.len or
            !function.tree.dominates(preheader, loop.header)) continue;
        const successors = function.graph.blocks[preheader].successors;
        if (successors.len != 1 or successors[0] != loop.header) continue;
        if (hasOutgoingException(function, loop.header)) continue;

        var relocation_safe = true;
        for (loop.blocks) |block_id| {
            if (block_id >= plans.len) return error.InvalidInput;
            for (plans[block_id]) |plan| {
                if (plan.relocation_kill or plan.may_safepoint) {
                    relocation_safe = false;
                    break;
                }
            }
            if (!relocation_safe) break;
        }
        if (!relocation_safe) continue;

        // Resolution can throw. Restrict this first slice to an access that
        // was already the first header operation, preserving zero-trip and
        // exception/side-effect ordering when moved to the preheader tail.
        const header_ops = function.blocks[loop.header].ops;
        if (header_ops.len == 0) continue;
        const raw_handle = rawBaseHandle(header_ops[0]) orelse continue;
        if (raw_handle >= canonical.len) return error.InvalidInput;
        const handle = canonical[raw_handle];
        if (!valueAvailableAtPlacement(function, handle, preheader) or
            loopContainsBlock(loop, function.values[handle].block)) continue;
        const plan = plans[loop.header][0];
        try appendHoistCandidate(&candidates, allocator, .{
            .key = .{ .handle = handle, .token = plan.token_in },
            .defining_op = .{ .block = loop.header, .index = 0 },
            .placement_block = preheader,
            .loop_header = loop.header,
            .loop_latch = loop.latch,
        });
    }
    return candidates.toOwnedSlice(allocator);
}

fn isTrustedLeaf(options: Options, invoke: *const instmod.Invoke) bool {
    if (invoke.native_target != null) return false;
    const target = invoke.call_target orelse return false;
    for (options.no_safepoint_leaf_targets) |leaf| {
        if (leaf == target) return true;
    }
    return false;
}

fn callEffects(options: Options, invoke: *const instmod.Invoke) CallEffects {
    if (isTrustedLeaf(options, invoke)) {
        return .{ .may_safepoint = false, .may_relocate = false };
    }
    return .{};
}

fn classify(inst: Instruction, options: Options) Effects {
    var effects: Effects = .{};
    switch (inst) {
        .invoke, .invoke_virtual_quick, .invoke_super_quick => |invoke| {
            const call = callEffects(options, invoke);
            effects.may_safepoint = call.may_safepoint;
            effects.relocation_kill = call.may_relocate;
            effects.heap_alias_kill = call.may_write_heap or call.may_read_heap;
        },
        .new_instance,
        .new_array,
        .filled_new_array,
        .const_string,
        .const_class,
        .const_method_handle,
        .const_method_type,
        .monitor_enter,
        .monitor_exit,
        .check_cast,
        .instance_of,
        .fill_array_data,
        .throw_,
        => {
            effects.may_safepoint = true;
            effects.relocation_kill = true;
            effects.heap_alias_kill = true;
        },
        .iput_object, .iput_object_quick, .aput_object => {
            effects.heap_alias_kill = true;
            effects.pre_write = .satb_guarded;
            effects.post_write = .card_guarded;
        },
        .sput_object => {
            effects.heap_alias_kill = true;
            effects.pre_write = .satb_guarded;
            effects.post_write = .root_guarded;
        },
        .iput,
        .iput_wide,
        .iput_boolean,
        .iput_byte,
        .iput_char,
        .iput_short,
        .iput_quick,
        .iput_wide_quick,
        .aput,
        .aput_wide,
        .aput_boolean,
        .aput_byte,
        .aput_char,
        .aput_short,
        .sput,
        .sput_wide,
        .sput_boolean,
        .sput_byte,
        .sput_char,
        .sput_short,
        => effects.heap_alias_kill = true,
        else => {},
    }
    return effects;
}

fn rawBaseHandle(op: ssa.Operation) ?ssa.ValueId {
    return switch (op.inst) {
        .iget,
        .iget_wide,
        .iget_object,
        .iget_boolean,
        .iget_byte,
        .iget_char,
        .iget_short,
        .iget_quick,
        .iget_wide_quick,
        .iget_object_quick,
        .aget,
        .aget_wide,
        .aget_object,
        .aget_boolean,
        .aget_byte,
        .aget_char,
        .aget_short,
        .monitor_enter,
        .monitor_exit,
        .check_cast,
        .instance_of,
        .array_length,
        .fill_array_data,
        => if (op.uses.len >= 1) op.uses[0] else null,

        .iput,
        .iput_wide,
        .iput_object,
        .iput_boolean,
        .iput_byte,
        .iput_char,
        .iput_short,
        .iput_quick,
        .iput_wide_quick,
        .iput_object_quick,
        => if (op.uses.len >= 2) op.uses[op.uses.len - 1] else null,

        .aput,
        .aput_wide,
        .aput_object,
        .aput_boolean,
        .aput_byte,
        .aput_char,
        .aput_short,
        => if (op.uses.len >= 3) op.uses[op.uses.len - 2] else null,

        .invoke, .invoke_virtual_quick, .invoke_super_quick => |invoke| switch (invoke.kind) {
            .static, .custom => null,
            else => if (op.uses.len >= 1) op.uses[0] else null,
        },
        else => null,
    };
}

fn referenceFieldSlot(op: ssa.Operation, canonical: []const ssa.ValueId) Error!?ReferenceFieldSlot {
    var field_idx: u32 = undefined;
    const quick = switch (op.inst) {
        .iput_object => |field| quick: {
            field_idx = field.field_idx;
            break :quick false;
        },
        .iput_object_quick => |field| quick: {
            field_idx = field.field_idx;
            break :quick true;
        },
        else => return null,
    };
    const raw_handle = rawBaseHandle(op) orelse return error.InvalidInput;
    if (raw_handle >= canonical.len) return error.InvalidInput;
    return .{
        .handle = canonical[raw_handle],
        .field_idx = field_idx,
        .quick = quick,
    };
}

fn handleTypeCompatible(types: *const typedir.Function, value: ssa.ValueId) bool {
    const ty = types.typeOf(value) orelse return false;
    return ty == .object or ty == .unknown;
}

fn buildCanonicalHandles(allocator: std.mem.Allocator, function: *const ssa.Function) ![]ssa.ValueId {
    const canonical = try allocator.alloc(ssa.ValueId, function.values.len);
    for (canonical, 0..) |*value, i| value.* = @intCast(i);

    for (function.graph.rpo) |block_id| {
        for (function.blocks[block_id].ops) |op| {
            switch (op.inst) {
                .move_object => if (op.uses.len == 1 and op.defs.len == 1) {
                    canonical[op.defs[0]] = canonical[op.uses[0]];
                },
                else => {},
            }
        }
    }
    return canonical;
}

fn hasIncomingException(function: *const ssa.Function, block: cfg.BlockId) bool {
    for (function.graph.edges) |edge| {
        if (edge.to == block and edge.kind == .exception) return true;
    }
    return false;
}

fn nextToken(next: *u64) Error!RelocTokenId {
    if (next.* > std.math.maxInt(RelocTokenId)) return error.TooManyTokens;
    const token: RelocTokenId = @intCast(next.*);
    next.* += 1;
    return token;
}

fn entryToken(
    function: *const ssa.Function,
    block: cfg.BlockId,
    block_out: []const RelocTokenId,
    merge_tokens: []const RelocTokenId,
) RelocTokenId {
    if (block == function.graph.entry) return 0;
    if (hasIncomingException(function, block)) return merge_tokens[block];

    const predecessors = function.graph.blocks[block].predecessors;
    if (predecessors.len == 0) return merge_tokens[block];
    const common = block_out[predecessors[0]];
    if (common == INVALID_TOKEN) return merge_tokens[block];
    for (predecessors[1..]) |pred| {
        if (block_out[pred] != common) return merge_tokens[block];
    }
    return common;
}

fn processResolves(
    allocator: std.mem.Allocator,
    function: *const ssa.Function,
    canonical: []const ssa.ValueId,
    plans: [][]OpPlan,
    hoists: []const HoistCandidate,
    resolves: *std.ArrayList(Resolve),
    stats: *Stats,
) Error!void {
    var active = std.AutoHashMap(ResolveKey, ResolveId).init(allocator);
    defer active.deinit();
    var undo: std.ArrayList(ResolveKey) = .empty;
    defer undo.deinit(allocator);
    var stack: std.ArrayList(WalkFrame) = .empty;
    defer stack.deinit(allocator);

    try stack.append(allocator, .{ .block = function.graph.entry });
    while (stack.items.len != 0) {
        const frame_index = stack.items.len - 1;
        if (!stack.items[frame_index].entered) {
            stack.items[frame_index].entered = true;
            stack.items[frame_index].undo_mark = undo.items.len;
            const block_id = stack.items[frame_index].block;
            const block = function.blocks[block_id];
            for (block.ops, 0..) |op, op_index| {
                const raw_base = rawBaseHandle(op) orelse continue;
                if (raw_base >= canonical.len) return error.InvalidInput;
                const handle = canonical[raw_base];
                const key = ResolveKey{ .handle = handle, .token = plans[block_id][op_index].token_in };
                plans[block_id][op_index].base_handle = handle;
                if (active.get(key)) |id| {
                    plans[block_id][op_index].resolve = .{ .reuse = id };
                    stats.resolves_reused += 1;
                    continue;
                }
                if (resolves.items.len > std.math.maxInt(ResolveId)) return error.TooManyResolves;
                const id: ResolveId = @intCast(resolves.items.len);
                try resolves.append(allocator, .{
                    .handle = handle,
                    .token = key.token,
                    .defining_op = .{ .block = block_id, .index = @intCast(op_index) },
                    .placement_block = block_id,
                });
                try active.put(key, id);
                try undo.append(allocator, key);
                plans[block_id][op_index].resolve = .{ .define = id };
                stats.resolves_inserted += 1;
            }
            // Synthetic preheader definitions become active only after every
            // source operation in that block. Lowering inserts them at the
            // same semantic tail, immediately before control transfer.
            for (hoists) |candidate| {
                if (candidate.placement_block != block_id or active.contains(candidate.key)) continue;
                if (resolves.items.len > std.math.maxInt(ResolveId)) return error.TooManyResolves;
                const id: ResolveId = @intCast(resolves.items.len);
                try resolves.append(allocator, .{
                    .handle = candidate.key.handle,
                    .token = candidate.key.token,
                    .defining_op = candidate.defining_op,
                    .placement_block = candidate.placement_block,
                    .hoisted = true,
                    .loop_header = candidate.loop_header,
                    .loop_latch = candidate.loop_latch,
                    .guard_id = stats.loop_epoch_guards,
                });
                try active.put(candidate.key, id);
                try undo.append(allocator, candidate.key);
                stats.resolves_inserted += 1;
                stats.loop_resolves_hoisted += 1;
                stats.loop_epoch_guards += 1;
            }
        }

        const block_id = stack.items[frame_index].block;
        const children = function.tree.children[block_id];
        if (stack.items[frame_index].next_child < children.len) {
            const child = children[stack.items[frame_index].next_child];
            stack.items[frame_index].next_child += 1;
            try stack.append(allocator, .{ .block = child });
            continue;
        }

        const undo_mark = stack.items[frame_index].undo_mark;
        while (undo.items.len > undo_mark) {
            const key = undo.pop().?;
            if (!active.remove(key)) return error.InvalidPlan;
        }
        _ = stack.pop();
    }
}

pub fn run(
    allocator: std.mem.Allocator,
    function: *const ssa.Function,
    types: *const typedir.Function,
) Error!Result {
    return runWithOptions(allocator, function, types, .{});
}

pub fn runWithOptions(
    allocator: std.mem.Allocator,
    function: *const ssa.Function,
    types: *const typedir.Function,
    options: Options,
) Error!Result {
    function.verify() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidInput,
    };
    types.verify() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidInput,
    };
    if (types.source != function or function.tree != types.source.tree) return error.InvalidInput;

    const plans = try allocator.alloc([]OpPlan, function.blocks.len);
    errdefer allocator.free(plans);
    var built_plans: usize = 0;
    errdefer for (plans[0..built_plans]) |block_plans| allocator.free(block_plans);
    for (function.blocks, 0..) |block, i| {
        plans[i] = try allocator.alloc(OpPlan, block.ops.len);
        @memset(plans[i], .{});
        built_plans += 1;
    }

    const canonical = try buildCanonicalHandles(allocator, function);
    errdefer allocator.free(canonical);
    const block_in = try allocator.alloc(RelocTokenId, function.blocks.len);
    errdefer allocator.free(block_in);
    @memset(block_in, INVALID_TOKEN);
    const block_out = try allocator.alloc(RelocTokenId, function.blocks.len);
    errdefer allocator.free(block_out);
    @memset(block_out, INVALID_TOKEN);
    const merge_tokens = try allocator.alloc(RelocTokenId, function.blocks.len);
    defer allocator.free(merge_tokens);

    var next_token: u64 = 1;
    for (merge_tokens, 0..) |*token, block_index| {
        token.* = if (block_index == function.graph.entry) 0 else try nextToken(&next_token);
    }

    var stats: Stats = .{};
    for (function.blocks) |block| {
        var previous_reference_slot: ?ReferenceFieldSlot = null;
        var previous_card_handle: ?ssa.ValueId = null;
        for (block.ops, 0..) |op, i| {
            const effects = classify(op.inst, options);
            plans[block.id][i].relocation_kill = effects.relocation_kill;
            plans[block.id][i].heap_alias_kill = effects.heap_alias_kill;
            plans[block.id][i].may_safepoint = effects.may_safepoint;
            const reference_slot = try referenceFieldSlot(op, canonical);
            plans[block.id][i].pre_write = if (effects.pre_write != .none and reference_slot != null and
                previous_reference_slot != null and previous_reference_slot.?.eql(reference_slot.?))
                .satb_repeat_guarded
            else
                effects.pre_write;
            plans[block.id][i].post_write = if (effects.post_write != .none and reference_slot != null and
                previous_card_handle != null and previous_card_handle.? == reference_slot.?.handle)
                .card_repeat_guarded
            else
                effects.post_write;
            if (effects.relocation_kill) {
                plans[block.id][i].token_out = try nextToken(&next_token);
                stats.relocation_kills += 1;
            }
            if (effects.may_safepoint) stats.safepoints += 1;
            if (effects.heap_alias_kill) stats.heap_alias_kills += 1;
            if (plans[block.id][i].pre_write != .none) stats.satb_barriers += 1;
            if (plans[block.id][i].pre_write == .satb_repeat_guarded) stats.satb_repeat_barriers += 1;
            if (plans[block.id][i].post_write != .none) stats.card_barriers += 1;
            if (plans[block.id][i].post_write == .card_repeat_guarded) stats.card_repeat_barriers += 1;
            if (reference_slot) |slot| {
                previous_reference_slot = slot;
                previous_card_handle = slot.handle;
            } else if (effects.heap_alias_kill) {
                previous_reference_slot = null;
                previous_card_handle = null;
            }
        }
    }

    for (function.graph.rpo) |block_id| {
        var token = entryToken(function, block_id, block_out, merge_tokens);
        block_in[block_id] = token;
        for (plans[block_id]) |*plan| {
            plan.token_in = token;
            if (plan.relocation_kill) {
                token = plan.token_out;
            } else {
                plan.token_out = token;
            }
        }
        block_out[block_id] = token;
    }

    if (next_token > @as(u64, std.math.maxInt(u32)) + 1) return error.TooManyTokens;
    const token_count: u32 = @intCast(next_token);
    stats.tokens = token_count;

    const hoists = try buildHoistCandidates(allocator, function, canonical, plans, options.loops);
    defer allocator.free(hoists);

    var resolve_list: std.ArrayList(Resolve) = .empty;
    defer resolve_list.deinit(allocator);
    try processResolves(allocator, function, canonical, plans, hoists, &resolve_list, &stats);
    const resolves = try resolve_list.toOwnedSlice(allocator);
    errdefer allocator.free(resolves);

    var result = Result{
        .allocator = allocator,
        .function = function,
        .types = types,
        .loops = options.loops,
        .ops = plans,
        .resolves = resolves,
        .canonical_handles = canonical,
        .block_in = block_in,
        .block_out = block_out,
        .token_count = token_count,
        .stats = stats,
    };
    try result.verify();
    return result;
}

fn buildPipeline(
    insts: []const Instruction,
    graph: *cfg.Graph,
    tree: *dom.Tree,
    function: *ssa.Function,
    types: *typedir.Function,
) !void {
    graph.* = try cfg.build(std.testing.allocator, insts);
    errdefer graph.deinit();
    tree.* = try dom.build(std.testing.allocator, graph);
    errdefer tree.deinit();
    function.* = try ssa.build(std.testing.allocator, graph, tree);
    errdefer function.deinit();
    types.* = try typedir.build(std.testing.allocator, function);
}

test "barrier_phase reuses one resolution through field accesses and object copies" {
    const insts = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .move_object = .{ .dest = 1, .src = 0 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 2, .obj = 0 } },
        .{ .iget = .{ .field_idx = 2, .dest_or_src = 3, .obj = 1 } },
        .{ .iput = .{ .field_idx = 3, .dest_or_src = 2, .obj = 0 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function, &types);
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function, &types);
    defer result.deinit();

    try std.testing.expectEqual(@as(u32, 1), result.stats.resolves_inserted);
    try std.testing.expectEqual(@as(u32, 2), result.stats.resolves_reused);
    try std.testing.expect(result.ops[graph.entry][4].heap_alias_kill);
    try result.verify();
}

test "barrier_phase reuses into dominated child but not sibling branch" {
    const dominated = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .{ .if_eqz = .{ .src = 1, .offset = 2 } },
        .{ .iget = .{ .field_idx = 2, .dest_or_src = 2, .obj = 0 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    try buildPipeline(&dominated, &graph, &tree, &function, &types);
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function, &types);
    defer result.deinit();
    try std.testing.expectEqual(@as(u32, 1), result.stats.resolves_inserted);
    try std.testing.expectEqual(@as(u32, 1), result.stats.resolves_reused);

    const siblings = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .if_eqz = .{ .src = 4, .offset = 3 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .iget = .{ .field_idx = 2, .dest_or_src = 2, .obj = 0 } },
        .return_void,
    };
    var graph2: cfg.Graph = undefined;
    var tree2: dom.Tree = undefined;
    var function2: ssa.Function = undefined;
    var types2: typedir.Function = undefined;
    try buildPipeline(&siblings, &graph2, &tree2, &function2, &types2);
    defer types2.deinit();
    defer function2.deinit();
    defer tree2.deinit();
    defer graph2.deinit();
    var result2 = try run(std.testing.allocator, &function2, &types2);
    defer result2.deinit();
    try std.testing.expectEqual(@as(u32, 2), result2.stats.resolves_inserted);
    try std.testing.expectEqual(@as(u32, 0), result2.stats.resolves_reused);
}

test "barrier_phase call kills unless target is a verified leaf" {
    const args = [_]u16{0};
    var invoke = instmod.Invoke{
        .class_name = "LTest;",
        .method_name = "leaf",
        .signature = "()V",
        .args = &args,
        .dest = null,
        .kind = .virtual,
        .call_target = 17,
    };
    const insts = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .{ .invoke = &invoke },
        .{ .iget = .{ .field_idx = 2, .dest_or_src = 2, .obj = 0 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function, &types);
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();

    var conservative = try run(std.testing.allocator, &function, &types);
    defer conservative.deinit();
    try std.testing.expect(conservative.ops[graph.entry][2].relocation_kill);
    try std.testing.expectEqual(@as(u32, 2), conservative.stats.resolves_inserted);

    const leaves = [_]u32{17};
    var leaf = try runWithOptions(std.testing.allocator, &function, &types, .{
        .no_safepoint_leaf_targets = &leaves,
    });
    defer leaf.deinit();
    try std.testing.expect(!leaf.ops[graph.entry][2].relocation_kill);
    try std.testing.expect(!leaf.ops[graph.entry][2].may_safepoint);
    try std.testing.expectEqual(@as(u32, 1), leaf.stats.resolves_inserted);
    try std.testing.expectEqual(@as(u32, 2), leaf.stats.resolves_reused);
}

test "barrier_phase marks reference writes independently from relocation" {
    const insts = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .const_string = .{ .dest = 1, .index = 2 } },
        .{ .iput_object = .{ .field_idx = 4, .dest_or_src = 1, .obj = 0 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function, &types);
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function, &types);
    defer result.deinit();

    const store = result.ops[graph.entry][2];
    try std.testing.expectEqual(PreWriteBarrier.satb_guarded, store.pre_write);
    try std.testing.expectEqual(PostWriteBarrier.card_guarded, store.post_write);
    try std.testing.expect(!store.relocation_kill);
    try std.testing.expect(store.heap_alias_kill);
}

test "barrier_phase proves same-slot SATB repeats and rejects alias gaps" {
    const repeated = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .const_string = .{ .dest = 1, .index = 2 } },
        .{ .const_string = .{ .dest = 2, .index = 3 } },
        .{ .iput_object = .{ .field_idx = 4, .dest_or_src = 1, .obj = 0 } },
        .{ .iget = .{ .field_idx = 7, .dest_or_src = 3, .obj = 0 } },
        .{ .iput_object = .{ .field_idx = 4, .dest_or_src = 2, .obj = 0 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    try buildPipeline(&repeated, &graph, &tree, &function, &types);
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function, &types);
    defer result.deinit();

    try std.testing.expectEqual(PreWriteBarrier.satb_guarded, result.ops[graph.entry][3].pre_write);
    try std.testing.expectEqual(PreWriteBarrier.satb_repeat_guarded, result.ops[graph.entry][5].pre_write);
    try std.testing.expectEqual(PostWriteBarrier.card_repeat_guarded, result.ops[graph.entry][5].post_write);
    try std.testing.expectEqual(@as(u32, 2), result.stats.satb_barriers);
    try std.testing.expectEqual(@as(u32, 1), result.stats.satb_repeat_barriers);
    try std.testing.expectEqual(@as(u32, 1), result.stats.card_repeat_barriers);

    const alias_gap = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .const_string = .{ .dest = 1, .index = 2 } },
        .{ .iput_object = .{ .field_idx = 4, .dest_or_src = 1, .obj = 0 } },
        .{ .iput = .{ .field_idx = 9, .dest_or_src = 3, .obj = 0 } },
        .{ .iput_object = .{ .field_idx = 4, .dest_or_src = 1, .obj = 0 } },
        .return_void,
    };
    var graph2: cfg.Graph = undefined;
    var tree2: dom.Tree = undefined;
    var function2: ssa.Function = undefined;
    var types2: typedir.Function = undefined;
    try buildPipeline(&alias_gap, &graph2, &tree2, &function2, &types2);
    defer types2.deinit();
    defer function2.deinit();
    defer tree2.deinit();
    defer graph2.deinit();
    var conservative = try run(std.testing.allocator, &function2, &types2);
    defer conservative.deinit();
    try std.testing.expectEqual(PreWriteBarrier.satb_guarded, conservative.ops[graph2.entry][4].pre_write);
    try std.testing.expectEqual(PostWriteBarrier.card_guarded, conservative.ops[graph2.entry][4].post_write);
    try std.testing.expectEqual(@as(u32, 0), conservative.stats.satb_repeat_barriers);
    try std.testing.expectEqual(@as(u32, 0), conservative.stats.card_repeat_barriers);
}

fn allocationFailureProbe(
    allocator: std.mem.Allocator,
    function: *const ssa.Function,
    types: *const typedir.Function,
) !void {
    var result = try run(allocator, function, types);
    defer result.deinit();
    try result.verify();
}

test "barrier_phase construction is leak-free at every allocation failure" {
    const insts = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .{ .iget = .{ .field_idx = 2, .dest_or_src = 2, .obj = 0 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function, &types);
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();

    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        allocationFailureProbe,
        .{ &function, &types },
    );
}

test "barrier_phase print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .const_string = .{ .dest = 0, .index = 1 } },
        .{ .iget = .{ .field_idx = 1, .dest_or_src = 1, .obj = 0 } },
        .return_void,
    };
    var graph: cfg.Graph = undefined;
    var tree: dom.Tree = undefined;
    var function: ssa.Function = undefined;
    var types: typedir.Function = undefined;
    try buildPipeline(&insts, &graph, &tree, &function, &types);
    defer types.deinit();
    defer function.deinit();
    defer tree.deinit();
    defer graph.deinit();
    var result = try run(std.testing.allocator, &function, &types);
    defer result.deinit();

    var storage: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&storage);
    try result.print(&writer);
    const output = writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, output, "barrier_phase tokens=") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "resolve=r0") != null);
}
