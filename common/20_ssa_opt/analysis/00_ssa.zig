//! SSA translation for Dalvik register bytecode.
//!
//! The translator consumes a pruned, RPO-ordered CFG plus its dominator tree,
//! places phi nodes from dominance frontiers, then performs classic stack-based
//! renaming over the dominator tree. Values are tracked per Dalvik register
//! slot, including the second slot of wide values.

const std = @import("std");
const cfg = @import("cfg");
const dom = @import("dominator");
const instmod = @import("instructions");
const Instruction = instmod.Instruction;

pub const ValueId = u32;
pub const INVALID_VALUE: ValueId = std.math.maxInt(ValueId);

pub const Error = error{
    InvalidGraph,
    OutOfMemory,
};

pub const VerifyError = error{
    BadPhiIncoming,
    BadPhiPredecessor,
    BadPhiRegister,
    BadUseDominance,
    BadValue,
    BadValueOwner,
    InvalidGraph,
    MissingOperationMap,
    OutOfMemory,
};

pub const ValueKind = enum(u8) {
    parameter,
    phi,
    instruction,
};

pub const Value = struct {
    id: ValueId,
    reg: u16,
    kind: ValueKind,
    block: cfg.BlockId,
    pc: ?u32,
};

pub const Incoming = struct {
    pred: cfg.BlockId,
    value: ValueId,
};

pub const Phi = struct {
    reg: u16,
    dest: ValueId,
    incoming: []Incoming,
};

pub const Operation = struct {
    pc: u32,
    inst: Instruction,
    uses: []ValueId,
    defs: []ValueId,
};

pub const OpRef = struct {
    block: cfg.BlockId,
    index: u32,
};

pub const Block = struct {
    id: cfg.BlockId,
    phis: []Phi,
    ops: []Operation,
};

const MutablePhi = struct {
    reg: u16,
    dest: ValueId = INVALID_VALUE,
    incoming: std.ArrayList(Incoming) = .empty,
};

pub const Function = struct {
    allocator: std.mem.Allocator,
    graph: *const cfg.Graph,
    tree: *const dom.Tree,
    register_count: u16,
    values: []Value,
    blocks: []Block,
    pc_to_op: []?OpRef,

    pub fn deinit(self: *Function) void {
        self.allocator.free(self.pc_to_op);
        for (self.blocks) |ssa_block| {
            for (ssa_block.phis) |phi| self.allocator.free(phi.incoming);
            for (ssa_block.ops) |op| {
                self.allocator.free(op.uses);
                self.allocator.free(op.defs);
            }
            self.allocator.free(ssa_block.phis);
            self.allocator.free(ssa_block.ops);
        }
        self.allocator.free(self.blocks);
        self.allocator.free(self.values);
        self.* = undefined;
    }

    pub inline fn block(self: *const Function, id: cfg.BlockId) ?*const Block {
        if (id >= self.blocks.len) return null;
        return &self.blocks[id];
    }

    fn valueDominatesBlock(self: *const Function, value_id: ValueId, block_id: cfg.BlockId) VerifyError!void {
        if (value_id >= self.values.len) return error.BadValue;
        const value = self.values[value_id];
        if (value.id != value_id or value.reg >= self.register_count) return error.BadValue;
        switch (value.kind) {
            .parameter => {},
            .phi, .instruction => if (!self.tree.dominates(value.block, block_id)) return error.BadUseDominance,
        }
    }

    fn valueDominatesEdge(self: *const Function, value_id: ValueId, pred: cfg.BlockId) VerifyError!void {
        if (value_id >= self.values.len) return error.BadValue;
        const value = self.values[value_id];
        if (value.id != value_id or value.reg >= self.register_count) return error.BadValue;
        switch (value.kind) {
            .parameter => {},
            .phi, .instruction => if (!self.tree.dominates(value.block, pred)) return error.BadUseDominance,
        }
    }

    fn isPredecessor(self: *const Function, block_id: cfg.BlockId, pred: cfg.BlockId) bool {
        if (block_id >= self.graph.blocks.len) return false;
        for (self.graph.blocks[block_id].predecessors) |candidate| {
            if (candidate == pred) return true;
        }
        return false;
    }

    pub fn verify(self: *const Function) VerifyError!void {
        if (self.blocks.len != self.graph.blocks.len or self.tree.idom.len != self.graph.blocks.len) return error.InvalidGraph;
        if (self.pc_to_op.len != self.graph.instructions.len) return error.InvalidGraph;

        var uses: std.ArrayList(u16) = .empty;
        defer uses.deinit(self.allocator);
        var defs: std.ArrayList(u16) = .empty;
        defer defs.deinit(self.allocator);

        for (self.blocks, 0..) |ssa_block, block_index| {
            const block_id: cfg.BlockId = @intCast(block_index);
            if (ssa_block.id != block_id) return error.InvalidGraph;

            for (ssa_block.phis) |phi| {
                if (phi.reg >= self.register_count or phi.dest >= self.values.len) return error.BadValue;
                const dest = self.values[phi.dest];
                if (dest.id != phi.dest or dest.kind != .phi or dest.block != block_id or dest.pc != null) return error.BadValueOwner;
                if (dest.reg != phi.reg) return error.BadPhiRegister;

                const preds = self.graph.blocks[block_id].predecessors;
                if (phi.incoming.len != preds.len) return error.BadPhiIncoming;
                for (phi.incoming, 0..) |incoming, i| {
                    if (!self.isPredecessor(block_id, incoming.pred)) return error.BadPhiPredecessor;
                    for (phi.incoming[0..i]) |prior| {
                        if (prior.pred == incoming.pred) return error.BadPhiIncoming;
                    }
                    if (incoming.value >= self.values.len) return error.BadValue;
                    if (self.values[incoming.value].reg != phi.reg) return error.BadPhiRegister;
                    try self.valueDominatesEdge(incoming.value, incoming.pred);
                }
            }

            for (ssa_block.ops, 0..) |op, op_index| {
                if (op.pc >= self.pc_to_op.len) return error.InvalidGraph;
                const mapped = self.pc_to_op[op.pc] orelse return error.MissingOperationMap;
                if (mapped.block != block_id or mapped.index != op_index) return error.MissingOperationMap;

                uses.clearRetainingCapacity();
                defs.clearRetainingCapacity();
                try collectUsesDefs(self.allocator, op.inst, &uses, &defs);
                if (op.uses.len != uses.items.len or op.defs.len != defs.items.len) return error.BadValueOwner;

                for (op.uses, 0..) |use, i| {
                    if (use >= self.values.len) return error.BadValue;
                    if (self.values[use].reg != uses.items[i]) return error.BadValueOwner;
                    try self.valueDominatesBlock(use, block_id);
                }

                for (op.defs, 0..) |def, i| {
                    if (def >= self.values.len) return error.BadValue;
                    const value = self.values[def];
                    if (value.id != def or value.kind != .instruction or value.block != block_id) return error.BadValueOwner;
                    if (value.pc == null or value.pc.? != op.pc or value.reg != defs.items[i]) return error.BadValueOwner;
                }
            }
        }
    }

    pub fn print(self: *const Function, writer: anytype) !void {
        try writer.print("ssa blocks={d} values={d} registers={d} entry=b{d}\n", .{
            self.blocks.len,
            self.values.len,
            self.register_count,
            self.graph.entry,
        });

        for (self.graph.rpo) |block_id| {
            const out = self.blocks[block_id];
            const source = self.graph.blocks[block_id];
            try writer.print("b{d} pc=[{d},{d})\n", .{ block_id, source.start, source.end });
            for (out.phis) |phi| {
                try writer.print("  v{d} = phi r{d}(", .{ phi.dest, phi.reg });
                for (phi.incoming, 0..) |incoming, i| {
                    if (i != 0) try writer.print(", ", .{});
                    try writer.print("b{d}:v{d}", .{ incoming.pred, incoming.value });
                }
                try writer.print(")\n", .{});
            }
            for (out.ops) |op| {
                try writer.print("  pc{d} {s}", .{ op.pc, @tagName(op.inst) });
                try writer.print(" uses:", .{});
                if (op.uses.len == 0) {
                    try writer.print(" <none>", .{});
                } else {
                    for (op.uses) |value| try writer.print(" v{d}", .{value});
                }
                try writer.print(" defs:", .{});
                if (op.defs.len == 0) {
                    try writer.print(" <none>", .{});
                } else {
                    for (op.defs) |value| try writer.print(" v{d}", .{value});
                }
                try writer.print("\n", .{});
            }
        }
    }
};

fn appendReg(list: *std.ArrayList(u16), allocator: std.mem.Allocator, reg: u16) !void {
    try list.append(allocator, reg);
}

fn appendWide(list: *std.ArrayList(u16), allocator: std.mem.Allocator, reg: u16) !void {
    try list.append(allocator, reg);
    if (reg != std.math.maxInt(u16)) try list.append(allocator, reg + 1);
}

fn appendInvokeArgs(list: *std.ArrayList(u16), allocator: std.mem.Allocator, invoke: *const instmod.Invoke) !void {
    for (invoke.args) |reg| try appendReg(list, allocator, reg);
}

fn collectUsesDefs(allocator: std.mem.Allocator, inst: Instruction, uses: *std.ArrayList(u16), defs: *std.ArrayList(u16)) !void {
    switch (inst) {
        .nop,
        .goto_,
        .return_void,
        => {},

        .move, .move_object => |op| {
            try appendReg(uses, allocator, op.src);
            try appendReg(defs, allocator, op.dest);
        },
        .move_wide => |op| {
            try appendWide(uses, allocator, op.src);
            try appendWide(defs, allocator, op.dest);
        },
        .move_result, .move_result_object, .move_exception => |op| try appendReg(defs, allocator, op.dest),
        .move_result_wide => |op| try appendWide(defs, allocator, op.dest),

        .return_, .return_object => |op| try appendReg(uses, allocator, op.src),
        .throw_ => |op| try appendReg(uses, allocator, op.src),
        .return_wide => |op| try appendWide(uses, allocator, op.src),

        .const_ => |op| try appendReg(defs, allocator, op.dest),
        .const_wide => |op| try appendWide(defs, allocator, op.dest),
        .const_string, .const_method_handle, .const_method_type => |op| try appendReg(defs, allocator, op.dest),
        .const_class, .new_instance => |op| try appendReg(defs, allocator, op.dest),

        .monitor_enter => |op| try appendReg(uses, allocator, op.src),
        .monitor_exit => |op| try appendReg(uses, allocator, op.src),
        .check_cast => |op| try appendReg(uses, allocator, op.src),
        .instance_of => |op| {
            try appendReg(uses, allocator, op.src);
            try appendReg(defs, allocator, op.dest);
        },
        .array_length => |op| {
            try appendReg(uses, allocator, op.array);
            try appendReg(defs, allocator, op.dest);
        },
        .new_array => |op| {
            try appendReg(uses, allocator, op.size);
            try appendReg(defs, allocator, op.dest);
        },
        .filled_new_array => |op| if (op.payload) |payload| {
            for (payload.args) |reg| try appendReg(uses, allocator, reg);
        },
        .fill_array_data => |op| try appendReg(uses, allocator, op.array),

        .packed_switch, .sparse_switch => |op| try appendReg(uses, allocator, op.src),
        .if_eq, .if_ne, .if_lt, .if_ge, .if_gt, .if_le => |op| {
            try appendReg(uses, allocator, op.src1);
            try appendReg(uses, allocator, op.src2);
        },
        .if_eqz, .if_nez, .if_ltz, .if_gez, .if_gtz, .if_lez => |op| try appendReg(uses, allocator, op.src),

        .cmpl_float, .cmpg_float, .cmp_long => |op| {
            try appendReg(uses, allocator, op.src1);
            try appendReg(uses, allocator, op.src2);
            try appendReg(defs, allocator, op.dest);
        },
        .cmpl_double, .cmpg_double => |op| {
            try appendWide(uses, allocator, op.src1);
            try appendWide(uses, allocator, op.src2);
            try appendReg(defs, allocator, op.dest);
        },

        .aget, .aget_object, .aget_boolean, .aget_byte, .aget_char, .aget_short => |op| {
            try appendReg(uses, allocator, op.array);
            try appendReg(uses, allocator, op.index);
            try appendReg(defs, allocator, op.dest_or_src);
        },
        .aget_wide => |op| {
            try appendReg(uses, allocator, op.array);
            try appendReg(uses, allocator, op.index);
            try appendWide(defs, allocator, op.dest_or_src);
        },
        .aput, .aput_object, .aput_boolean, .aput_byte, .aput_char, .aput_short => |op| {
            try appendReg(uses, allocator, op.dest_or_src);
            try appendReg(uses, allocator, op.array);
            try appendReg(uses, allocator, op.index);
        },
        .aput_wide => |op| {
            try appendWide(uses, allocator, op.dest_or_src);
            try appendReg(uses, allocator, op.array);
            try appendReg(uses, allocator, op.index);
        },

        .iget, .iget_object, .iget_boolean, .iget_byte, .iget_char, .iget_short, .iget_quick, .iget_object_quick => |op| {
            try appendReg(uses, allocator, op.obj);
            try appendReg(defs, allocator, op.dest_or_src);
        },
        .iget_wide, .iget_wide_quick => |op| {
            try appendReg(uses, allocator, op.obj);
            try appendWide(defs, allocator, op.dest_or_src);
        },
        .iput, .iput_object, .iput_boolean, .iput_byte, .iput_char, .iput_short, .iput_quick, .iput_object_quick => |op| {
            try appendReg(uses, allocator, op.dest_or_src);
            try appendReg(uses, allocator, op.obj);
        },
        .iput_wide, .iput_wide_quick => |op| {
            try appendWide(uses, allocator, op.dest_or_src);
            try appendReg(uses, allocator, op.obj);
        },

        .sget, .sget_object, .sget_boolean, .sget_byte, .sget_char, .sget_short => |op| try appendReg(defs, allocator, op.dest_or_src),
        .sget_wide => |op| try appendWide(defs, allocator, op.dest_or_src),
        .sput, .sput_object, .sput_boolean, .sput_byte, .sput_char, .sput_short => |op| try appendReg(uses, allocator, op.dest_or_src),
        .sput_wide => |op| try appendWide(uses, allocator, op.dest_or_src),

        .invoke, .invoke_virtual_quick, .invoke_super_quick => |invoke| {
            try appendInvokeArgs(uses, allocator, invoke);
            if (invoke.dest) |dest| try appendReg(defs, allocator, dest);
        },

        .neg_int,
        .not_int,
        .neg_float,
        .int_to_float,
        .int_to_double,
        .float_to_int,
        .float_to_double,
        .double_to_int,
        .double_to_float,
        .int_to_byte,
        .int_to_char,
        .int_to_short,
        => |op| {
            try appendReg(uses, allocator, op.src);
            try appendReg(defs, allocator, op.dest);
        },
        .int_to_long, .float_to_long => |op| {
            try appendReg(uses, allocator, op.src);
            try appendWide(defs, allocator, op.dest);
        },
        .neg_long,
        .not_long,
        .neg_double,
        .long_to_int,
        .long_to_float,
        .long_to_double,
        .double_to_long,
        => |op| {
            try appendWide(uses, allocator, op.src);
            if (inst == .long_to_int or inst == .long_to_float) {
                try appendReg(defs, allocator, op.dest);
            } else {
                try appendWide(defs, allocator, op.dest);
            }
        },

        .add_int,
        .sub_int,
        .mul_int,
        .div_int,
        .rem_int,
        .and_int,
        .or_int,
        .xor_int,
        .shl_int,
        .shr_int,
        .ushr_int,
        .add_float,
        .sub_float,
        .mul_float,
        .div_float,
        .rem_float,
        => |op| {
            try appendReg(uses, allocator, op.src1);
            try appendReg(uses, allocator, op.src2);
            try appendReg(defs, allocator, op.dest);
        },
        .add_long,
        .sub_long,
        .mul_long,
        .div_long,
        .rem_long,
        .and_long,
        .or_long,
        .xor_long,
        .add_double,
        .sub_double,
        .mul_double,
        .div_double,
        .rem_double,
        => |op| {
            try appendWide(uses, allocator, op.src1);
            try appendWide(uses, allocator, op.src2);
            try appendWide(defs, allocator, op.dest);
        },
        .shl_long, .shr_long, .ushr_long => |op| {
            try appendWide(uses, allocator, op.src1);
            try appendReg(uses, allocator, op.src2);
            try appendWide(defs, allocator, op.dest);
        },

        .add_int_lit16,
        .rsub_int_lit16,
        .mul_int_lit16,
        .div_int_lit16,
        .rem_int_lit16,
        .and_int_lit16,
        .or_int_lit16,
        .xor_int_lit16,
        => |op| {
            try appendReg(uses, allocator, op.src);
            try appendReg(defs, allocator, op.dest);
        },
        .add_int_lit8,
        .rsub_int_lit8,
        .mul_int_lit8,
        .div_int_lit8,
        .rem_int_lit8,
        .and_int_lit8,
        .or_int_lit8,
        .xor_int_lit8,
        .shl_int_lit8,
        .shr_int_lit8,
        .ushr_int_lit8,
        => |op| {
            try appendReg(uses, allocator, op.src);
            try appendReg(defs, allocator, op.dest);
        },
    }
}

fn maxRegister(allocator: std.mem.Allocator, graph: *const cfg.Graph) !u16 {
    var uses: std.ArrayList(u16) = .empty;
    defer uses.deinit(allocator);
    var defs: std.ArrayList(u16) = .empty;
    defer defs.deinit(allocator);

    var max_reg: u16 = 0;
    for (graph.instructions) |inst| {
        uses.clearRetainingCapacity();
        defs.clearRetainingCapacity();
        try collectUsesDefs(allocator, inst, &uses, &defs);
        for (uses.items) |reg| max_reg = @max(max_reg, reg);
        for (defs.items) |reg| max_reg = @max(max_reg, reg);
    }
    return if (max_reg == std.math.maxInt(u16)) max_reg else max_reg + 1;
}

fn appendUniqueBlock(list: *std.ArrayList(cfg.BlockId), allocator: std.mem.Allocator, id: cfg.BlockId) !void {
    for (list.items) |existing| if (existing == id) return;
    try list.append(allocator, id);
}

fn newValue(values: *std.ArrayList(Value), allocator: std.mem.Allocator, reg: u16, kind: ValueKind, block: cfg.BlockId, pc: ?u32) !ValueId {
    const id: ValueId = @intCast(values.items.len);
    try values.append(allocator, .{ .id = id, .reg = reg, .kind = kind, .block = block, .pc = pc });
    return id;
}

fn top(stack: []std.ArrayList(ValueId), reg: u16) ValueId {
    const items = stack[reg].items;
    return items[items.len - 1];
}

const RenameContext = struct {
    allocator: std.mem.Allocator,
    graph: *const cfg.Graph,
    tree: *const dom.Tree,
    values: *std.ArrayList(Value),
    mutable_phis: []std.ArrayList(MutablePhi),
    op_lists: []std.ArrayList(Operation),
    stacks: []std.ArrayList(ValueId),
    pc_to_op: []?OpRef,
};

fn renameBlock(ctx: *RenameContext, block_id: cfg.BlockId) !void {
    var pushed: std.ArrayList(u16) = .empty;
    defer pushed.deinit(ctx.allocator);

    for (ctx.mutable_phis[block_id].items) |*phi| {
        phi.dest = try newValue(ctx.values, ctx.allocator, phi.reg, .phi, block_id, null);
        try ctx.stacks[phi.reg].append(ctx.allocator, phi.dest);
        try pushed.append(ctx.allocator, phi.reg);
    }

    const block = ctx.graph.blocks[block_id];
    var uses: std.ArrayList(u16) = .empty;
    defer uses.deinit(ctx.allocator);
    var defs: std.ArrayList(u16) = .empty;
    defer defs.deinit(ctx.allocator);

    var pc = block.start;
    while (pc < block.end) : (pc += 1) {
        const inst = ctx.graph.instructions[pc];
        uses.clearRetainingCapacity();
        defs.clearRetainingCapacity();
        try collectUsesDefs(ctx.allocator, inst, &uses, &defs);

        const use_values = try ctx.allocator.alloc(ValueId, uses.items.len);
        errdefer ctx.allocator.free(use_values);
        for (uses.items, 0..) |reg, i| use_values[i] = top(ctx.stacks, reg);

        const def_values = try ctx.allocator.alloc(ValueId, defs.items.len);
        errdefer ctx.allocator.free(def_values);
        for (defs.items, 0..) |reg, i| {
            const value = try newValue(ctx.values, ctx.allocator, reg, .instruction, block_id, pc);
            def_values[i] = value;
            try ctx.stacks[reg].append(ctx.allocator, value);
            try pushed.append(ctx.allocator, reg);
        }

        const op_index: u32 = @intCast(ctx.op_lists[block_id].items.len);
        try ctx.op_lists[block_id].append(ctx.allocator, .{
            .pc = pc,
            .inst = inst,
            .uses = use_values,
            .defs = def_values,
        });
        ctx.pc_to_op[pc] = .{ .block = block_id, .index = op_index };
    }

    for (block.successors) |succ| {
        for (ctx.mutable_phis[succ].items) |*phi| {
            try phi.incoming.append(ctx.allocator, .{ .pred = block_id, .value = top(ctx.stacks, phi.reg) });
        }
    }

    for (ctx.tree.children[block_id]) |child| try renameBlock(ctx, child);

    var i = pushed.items.len;
    while (i > 0) {
        i -= 1;
        const reg = pushed.items[i];
        _ = ctx.stacks[reg].pop();
    }
}

fn placePhis(
    allocator: std.mem.Allocator,
    graph: *const cfg.Graph,
    tree: *const dom.Tree,
    register_count: u16,
    mutable_phis: []std.ArrayList(MutablePhi),
) !void {
    const block_count = graph.blocks.len;
    var defsites = try allocator.alloc(std.ArrayList(cfg.BlockId), register_count);
    defer {
        for (defsites) |*list| list.deinit(allocator);
        allocator.free(defsites);
    }
    for (defsites) |*list| list.* = .empty;

    var uses: std.ArrayList(u16) = .empty;
    defer uses.deinit(allocator);
    var defs: std.ArrayList(u16) = .empty;
    defer defs.deinit(allocator);

    for (graph.blocks) |block| {
        var pc = block.start;
        while (pc < block.end) : (pc += 1) {
            uses.clearRetainingCapacity();
            defs.clearRetainingCapacity();
            try collectUsesDefs(allocator, graph.instructions[pc], &uses, &defs);
            for (defs.items) |reg| try appendUniqueBlock(&defsites[reg], allocator, block.id);
        }
    }

    const has_phi = try allocator.alloc(bool, block_count * register_count);
    defer allocator.free(has_phi);
    @memset(has_phi, false);

    var work: std.ArrayList(cfg.BlockId) = .empty;
    defer work.deinit(allocator);
    var queued = try allocator.alloc(bool, block_count);
    defer allocator.free(queued);

    for (0..register_count) |reg_usize| {
        const reg: u16 = @intCast(reg_usize);
        if (defsites[reg].items.len == 0) continue;

        work.clearRetainingCapacity();
        @memset(queued, false);
        for (defsites[reg].items) |block_id| {
            try work.append(allocator, block_id);
            queued[block_id] = true;
        }

        var cursor: usize = 0;
        while (cursor < work.items.len) : (cursor += 1) {
            const block_id = work.items[cursor];
            for (tree.frontier[block_id]) |frontier_block| {
                const slot = @as(usize, frontier_block) * register_count + reg;
                if (has_phi[slot]) continue;
                has_phi[slot] = true;
                try mutable_phis[frontier_block].append(allocator, .{ .reg = reg });
                if (!queued[frontier_block]) {
                    try work.append(allocator, frontier_block);
                    queued[frontier_block] = true;
                }
            }
        }
    }
}

pub fn build(allocator: std.mem.Allocator, graph: *const cfg.Graph, tree: *const dom.Tree) Error!Function {
    if (graph.blocks.len != tree.idom.len or graph.instructions.len != graph.inst_to_block.len) return error.InvalidGraph;

    const register_count = try maxRegister(allocator, graph);
    const block_count = graph.blocks.len;

    const mutable_phis = try allocator.alloc(std.ArrayList(MutablePhi), block_count);
    defer allocator.free(mutable_phis);
    for (mutable_phis) |*list| list.* = .empty;
    defer {
        for (mutable_phis) |*list| {
            for (list.items) |*phi| phi.incoming.deinit(allocator);
            list.deinit(allocator);
        }
    }

    try placePhis(allocator, graph, tree, register_count, mutable_phis);

    var op_lists = try allocator.alloc(std.ArrayList(Operation), block_count);
    defer allocator.free(op_lists);
    for (op_lists) |*list| list.* = .empty;
    var op_lists_owned = false;
    defer if (!op_lists_owned) {
        for (op_lists) |*list| {
            for (list.items) |op| {
                allocator.free(op.uses);
                allocator.free(op.defs);
            }
            list.deinit(allocator);
        }
    };

    var stacks = try allocator.alloc(std.ArrayList(ValueId), register_count);
    defer allocator.free(stacks);
    for (stacks) |*stack| stack.* = .empty;
    defer {
        for (stacks) |*stack| stack.deinit(allocator);
    }

    var values: std.ArrayList(Value) = .empty;
    var values_owned = false;
    defer if (!values_owned) values.deinit(allocator);

    for (0..register_count) |reg_usize| {
        const reg: u16 = @intCast(reg_usize);
        const value = try newValue(&values, allocator, reg, .parameter, graph.entry, null);
        try stacks[reg].append(allocator, value);
    }

    const pc_to_op = try allocator.alloc(?OpRef, graph.instructions.len);
    errdefer allocator.free(pc_to_op);
    @memset(pc_to_op, null);

    var ctx = RenameContext{
        .allocator = allocator,
        .graph = graph,
        .tree = tree,
        .values = &values,
        .mutable_phis = mutable_phis,
        .op_lists = op_lists,
        .stacks = stacks,
        .pc_to_op = pc_to_op,
    };
    try renameBlock(&ctx, graph.entry);

    const blocks = try allocator.alloc(Block, block_count);
    errdefer allocator.free(blocks);
    var blocks_built: usize = 0;
    errdefer {
        for (blocks[0..blocks_built]) |block| {
            for (block.phis) |phi| allocator.free(phi.incoming);
            for (block.ops) |op| {
                allocator.free(op.uses);
                allocator.free(op.defs);
            }
            allocator.free(block.phis);
            allocator.free(block.ops);
        }
    }

    for (0..block_count) |block_id| {
        const phi_slice = try allocator.alloc(Phi, mutable_phis[block_id].items.len);
        var phis_built: usize = 0;
        errdefer {
            for (phi_slice[0..phis_built]) |phi| allocator.free(phi.incoming);
            allocator.free(phi_slice);
        }
        for (mutable_phis[block_id].items, 0..) |*phi, i| {
            phi_slice[i] = .{
                .reg = phi.reg,
                .dest = phi.dest,
                .incoming = try phi.incoming.toOwnedSlice(allocator),
            };
            phis_built += 1;
        }

        const op_slice = try op_lists[block_id].toOwnedSlice(allocator);
        blocks[block_id] = .{
            .id = @intCast(block_id),
            .phis = phi_slice,
            .ops = op_slice,
        };
        blocks_built += 1;
    }
    op_lists_owned = true;

    const value_slice = try values.toOwnedSlice(allocator);
    values_owned = true;

    return .{
        .allocator = allocator,
        .graph = graph,
        .tree = tree,
        .register_count = register_count,
        .values = value_slice,
        .blocks = blocks,
        .pc_to_op = pc_to_op,
    };
}

test "ssa renames straight-line uses and defs" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .add_int = .{ .dest = 2, .src1 = 0, .src2 = 1 } },
        .{ .return_ = .{ .src = 2 } },
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var ssa = try build(std.testing.allocator, &graph, &tree);
    defer ssa.deinit();

    const block = ssa.blocks[graph.entry];
    try std.testing.expectEqual(@as(usize, 4), block.ops.len);
    try std.testing.expectEqual(@as(usize, 0), block.ops[0].uses.len);
    try std.testing.expectEqual(@as(usize, 1), block.ops[0].defs.len);
    try std.testing.expectEqual(block.ops[0].defs[0], block.ops[2].uses[0]);
    try std.testing.expectEqual(block.ops[1].defs[0], block.ops[2].uses[1]);
    try std.testing.expectEqual(block.ops[2].defs[0], block.ops[3].uses[0]);
}

test "ssa inserts phi at diamond join" {
    const insts = [_]Instruction{
        .{ .if_eqz = .{ .src = 0, .offset = 3 } },
        .{ .const_ = .{ .dest = 1, .value = 1 } },
        .{ .goto_ = .{ .offset = 2 } },
        .{ .const_ = .{ .dest = 1, .value = 2 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var ssa = try build(std.testing.allocator, &graph, &tree);
    defer ssa.deinit();

    const join = graph.blockForPc(4).?.id;
    const join_block = ssa.blocks[join];
    try std.testing.expectEqual(@as(usize, 1), join_block.phis.len);
    try std.testing.expectEqual(@as(u16, 1), join_block.phis[0].reg);
    try std.testing.expectEqual(@as(usize, 2), join_block.phis[0].incoming.len);
    try std.testing.expectEqual(join_block.phis[0].dest, join_block.ops[0].uses[0]);
}

test "ssa inserts loop header phi" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 0 } },
        .{ .if_eqz = .{ .src = 0, .offset = 3 } },
        .{ .add_int_lit8 = .{ .dest = 0, .src = 0, .lit = 1 } },
        .{ .goto_ = .{ .offset = -2 } },
        .{ .return_ = .{ .src = 0 } },
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var ssa = try build(std.testing.allocator, &graph, &tree);
    defer ssa.deinit();

    const header = graph.blockForPc(1).?.id;
    const header_block = ssa.blocks[header];
    try std.testing.expectEqual(@as(usize, 1), header_block.phis.len);
    try std.testing.expectEqual(@as(u16, 0), header_block.phis[0].reg);
    try std.testing.expectEqual(@as(usize, 2), header_block.phis[0].incoming.len);
    try std.testing.expectEqual(header_block.phis[0].dest, header_block.ops[0].uses[0]);
}

test "ssa tracks wide register slots" {
    const insts = [_]Instruction{
        .{ .const_wide = .{ .dest = 0, .value = 1 } },
        .{ .const_wide = .{ .dest = 2, .value = 2 } },
        .{ .add_long = .{ .dest = 4, .src1 = 0, .src2 = 2 } },
        .{ .return_wide = .{ .src = 4 } },
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var ssa = try build(std.testing.allocator, &graph, &tree);
    defer ssa.deinit();

    const block = ssa.blocks[graph.entry];
    try std.testing.expectEqual(@as(usize, 2), block.ops[0].defs.len);
    try std.testing.expectEqual(@as(usize, 4), block.ops[2].uses.len);
    try std.testing.expectEqual(@as(usize, 2), block.ops[2].defs.len);
    try std.testing.expectEqual(block.ops[2].defs[0], block.ops[3].uses[0]);
    try std.testing.expectEqual(block.ops[2].defs[1], block.ops[3].uses[1]);
}

test "ssa creates parameter values for live-in registers" {
    const insts = [_]Instruction{
        .{ .add_int_lit8 = .{ .dest = 1, .src = 0, .lit = 7 } },
        .{ .return_ = .{ .src = 1 } },
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var ssa = try build(std.testing.allocator, &graph, &tree);
    defer ssa.deinit();

    const block = ssa.blocks[graph.entry];
    try std.testing.expectEqual(@as(ValueKind, .parameter), ssa.values[block.ops[0].uses[0]].kind);
    try std.testing.expectEqual(@as(u16, 0), ssa.values[block.ops[0].uses[0]].reg);
}

test "ssa print helper emits stable summary" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 1 } },
        .{ .return_ = .{ .src = 0 } },
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var ssa = try build(std.testing.allocator, &graph, &tree);
    defer ssa.deinit();

    var buf: [512]u8 = undefined;
    var stream = std.Io.Writer.fixed(&buf);
    try ssa.print(&stream);
    const output = stream.buffered();

    try std.testing.expect(std.mem.indexOf(u8, output, "ssa blocks=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "pc0 const_") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "uses:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "defs:") != null);
}

test "ssa round-trip dominance invariant holds for branches loops and wide values" {
    const insts = [_]Instruction{
        .{ .const_ = .{ .dest = 0, .value = 0 } },
        .{ .const_wide = .{ .dest = 2, .value = 10 } },
        .{ .if_eqz = .{ .src = 0, .offset = 4 } },
        .{ .add_int_lit8 = .{ .dest = 0, .src = 0, .lit = 1 } },
        .{ .add_long = .{ .dest = 2, .src1 = 2, .src2 = 2 } },
        .{ .goto_ = .{ .offset = 3 } },
        .{ .const_ = .{ .dest = 0, .value = 7 } },
        .{ .const_wide = .{ .dest = 2, .value = 20 } },
        .{ .if_ltz = .{ .src = 0, .offset = -5 } },
        .{ .return_wide = .{ .src = 2 } },
    };
    var graph = try cfg.build(std.testing.allocator, &insts);
    defer graph.deinit();
    var tree = try dom.build(std.testing.allocator, &graph);
    defer tree.deinit();
    var ssa = try build(std.testing.allocator, &graph, &tree);
    defer ssa.deinit();

    try ssa.verify();

    const join = graph.blockForPc(8).?.id;
    try std.testing.expect(ssa.blocks[join].phis.len >= 3);
}
