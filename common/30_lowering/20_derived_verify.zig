//! Post-lowering proof for temporary heap addresses.
//!
//! This verifier is deliberately independent from register allocation. It
//! proves that every machine address is defined by the matching handle resolve,
//! that the definition dominates the use, and that the immutable barrier plan
//! still authorizes the handle/token/resolve relationship at the original SSA
//! operation. Derived pointers never participate in phi edge moves or ordinary
//! value operands.

const std = @import("std");
const barrier_phase = @import("barrier_phase");
const machine = @import("machine_bridge");

pub const Error = error{
    AddressDefinitionMissing,
    AddressDefinitionNotDominating,
    AddressDefinedTwice,
    BadCanonicalHandle,
    BadInstructionOrder,
    BadResolveDefinition,
    DerivedEdgeMove,
    InvalidMachine,
    InvalidPlan,
    OutOfMemory,
};

pub const DefSite = struct {
    block: u32,
    instruction: u32,
};

pub const Stats = struct {
    resolve_definitions: u32 = 0,
    address_uses: u32 = 0,
    canonical_state_uses: u32 = 0,
    safepoints_checked: u32 = 0,
    relocation_kills_checked: u32 = 0,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    source: *const machine.Function,
    definitions: []?DefSite,
    stats: Stats,

    pub fn deinit(self: *Result) void {
        self.allocator.free(self.definitions);
        self.* = undefined;
    }

    pub fn verify(self: *const Result) Error!void {
        const rebuilt = try self.allocator.alloc(?DefSite, self.definitions.len);
        defer self.allocator.free(rebuilt);
        @memset(rebuilt, null);
        try verifyWithDefinitions(self.source, rebuilt, null);
        for (rebuilt, self.definitions) |actual, expected| {
            if (actual == null or expected == null) {
                if (actual != null or expected != null) return error.BadResolveDefinition;
                continue;
            }
            if (actual.?.block != expected.?.block or actual.?.instruction != expected.?.instruction) return error.BadResolveDefinition;
        }
    }

    pub fn print(self: *const Result, writer: anytype) !void {
        try writer.print(
            "derived_verify resolves={d} address_uses={d} state_uses={d} safepoints={d} relocation_kills={d}\n",
            .{
                self.stats.resolve_definitions,
                self.stats.address_uses,
                self.stats.canonical_state_uses,
                self.stats.safepoints_checked,
                self.stats.relocation_kills_checked,
            },
        );
    }
};

const PlanRef = struct {
    block: u32,
    operation: u32,
    plan: barrier_phase.OpPlan,
};

fn planFor(source: *const machine.Function, block_id: u32, inst: machine.Inst) Error!PlanRef {
    const barriers = source.source.barriers orelse return error.InvalidPlan;
    const pc = inst.pc orelse return error.InvalidPlan;
    if (pc >= source.source.source.pc_to_op.len) return error.InvalidPlan;
    const op_ref = source.source.source.pc_to_op[pc] orelse return error.InvalidPlan;
    if (op_ref.block != block_id or op_ref.block >= barriers.ops.len or op_ref.index >= barriers.ops[op_ref.block].len) return error.InvalidPlan;
    return .{
        .block = op_ref.block,
        .operation = op_ref.index,
        .plan = barriers.ops[op_ref.block][op_ref.index],
    };
}

fn resolveFromPlan(plan: barrier_phase.OpPlan) ?barrier_phase.ResolveId {
    return switch (plan.resolve) {
        .none => null,
        .define => |id| id,
        .reuse => |resolve| resolve,
    };
}

fn verifyDefinition(
    source: *const machine.Function,
    definitions: []?DefSite,
    block_id: u32,
    instruction_index: usize,
    inst: machine.Inst,
) Error!void {
    if (inst.defs.len != 1) return error.BadResolveDefinition;
    const address = inst.defs[0];
    if (address >= definitions.len) return error.BadResolveDefinition;
    const ptr = switch (source.runtime_values[address]) {
        .derived_ptr => |ptr| ptr,
        .dalvik => return error.BadResolveDefinition,
    };
    if (definitions[address] != null) return error.AddressDefinedTwice;
    const resolve_id = inst.resolve_id orelse return error.BadResolveDefinition;
    const barriers = source.source.barriers orelse return error.InvalidPlan;
    if (resolve_id >= barriers.resolves.len or ptr.resolve != resolve_id or
        inst.reloc_token != ptr.token or inst.state_handle != ptr.handle or
        inst.uses.len != 1 or inst.uses[0] != ptr.handle) return error.BadResolveDefinition;
    const resolve = barriers.resolves[resolve_id];
    if (resolve.handle != ptr.handle or resolve.token != ptr.token) return error.BadResolveDefinition;
    if (resolve.hoisted) {
        if (resolve.placement_block != block_id) return error.BadResolveDefinition;
        definitions[address] = .{ .block = block_id, .instruction = @intCast(instruction_index) };
        return;
    }
    const plan_ref = try planFor(source, block_id, inst);
    const expected_resolve = switch (plan_ref.plan.resolve) {
        .define => |resolve_id_value| resolve_id_value,
        else => return error.BadResolveDefinition,
    };
    if (expected_resolve != ptr.resolve or plan_ref.plan.token_in != ptr.token or plan_ref.plan.base_handle != ptr.handle) return error.BadResolveDefinition;
    if (ptr.resolve >= barriers.resolves.len) return error.BadResolveDefinition;
    if (resolve.defining_op.block != block_id or resolve.defining_op.index != plan_ref.operation) return error.BadResolveDefinition;
    definitions[address] = .{ .block = block_id, .instruction = @intCast(instruction_index) };
}

fn verifyAddressUse(
    source: *const machine.Function,
    definitions: []const ?DefSite,
    block_id: u32,
    instruction_index: usize,
    inst: machine.Inst,
) Error!void {
    const address = inst.address orelse return;
    if (address >= definitions.len) return error.InvalidMachine;
    const ptr = switch (source.runtime_values[address]) {
        .derived_ptr => |ptr| ptr,
        .dalvik => return error.InvalidMachine,
    };
    const definition = definitions[address] orelse return error.AddressDefinitionMissing;
    if (!source.source.source.tree.dominates(definition.block, block_id)) return error.AddressDefinitionNotDominating;
    if (definition.block == block_id and definition.instruction >= instruction_index) return error.AddressDefinitionNotDominating;

    const plan_ref = try planFor(source, block_id, inst);
    const expected_resolve = resolveFromPlan(plan_ref.plan) orelse return error.InvalidPlan;
    if (expected_resolve != ptr.resolve or plan_ref.plan.token_in != ptr.token or plan_ref.plan.base_handle != ptr.handle) return error.InvalidPlan;
    if (inst.resolve_id != ptr.resolve or inst.reloc_token != ptr.token) return error.InvalidPlan;

    const state = inst.state_handle orelse return error.BadCanonicalHandle;
    if (!source.isGcRoot(state)) return error.BadCanonicalHandle;
    switch (source.runtime_values[state]) {
        .dalvik => |value| if (value.value != ptr.handle) return error.BadCanonicalHandle,
        .derived_ptr => return error.BadCanonicalHandle,
    }
}

fn verifyWithDefinitions(
    source: *const machine.Function,
    definitions: []?DefSite,
    stats_out: ?*Stats,
) Error!void {
    source.verify() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidMachine,
    };
    const barriers = source.source.barriers orelse return error.InvalidPlan;
    if (definitions.len != source.runtime_values.len) return error.InvalidMachine;

    for (source.edges) |edge| {
        for (edge.moves) |move| {
            switch (source.runtime_values[move.src]) {
                .derived_ptr => return error.DerivedEdgeMove,
                .dalvik => {},
            }
            switch (source.runtime_values[move.dst]) {
                .derived_ptr => return error.DerivedEdgeMove,
                .dalvik => {},
            }
        }
    }

    var stats: Stats = .{};
    for (barriers.ops) |plans| {
        for (plans) |plan| {
            if (plan.may_safepoint) stats.safepoints_checked += 1;
            if (plan.relocation_kill) stats.relocation_kills_checked += 1;
        }
    }

    for (source.blocks) |block| {
        var last_operation: ?u32 = null;
        for (block.insts, 0..) |inst, instruction_index| {
            if (inst.pc != null) {
                const plan_ref = planFor(source, block.id, inst) catch |err| switch (err) {
                    error.InvalidPlan => null,
                    else => return err,
                };
                if (plan_ref) |ref| {
                    if (last_operation) |last| if (ref.operation < last) return error.BadInstructionOrder;
                    last_operation = ref.operation;
                }
            }
            if (inst.opcode == .resolve_handle) {
                try verifyDefinition(source, definitions, block.id, instruction_index, inst);
                stats.resolve_definitions += 1;
            }
        }
    }

    for (source.blocks) |block| {
        for (block.insts, 0..) |inst, instruction_index| {
            if (inst.address != null) {
                try verifyAddressUse(source, definitions, block.id, instruction_index, inst);
                stats.address_uses += 1;
                stats.canonical_state_uses += 1;
            }
        }
    }
    if (stats_out) |out| out.* = stats;
}

pub fn run(allocator: std.mem.Allocator, source: *const machine.Function) Error!Result {
    const definitions = try allocator.alloc(?DefSite, source.runtime_values.len);
    errdefer allocator.free(definitions);
    @memset(definitions, null);
    var stats: Stats = .{};
    try verifyWithDefinitions(source, definitions, &stats);
    return .{
        .allocator = allocator,
        .source = source,
        .definitions = definitions,
        .stats = stats,
    };
}
