//! DEX pretty-printer: dumps every class, method, and decoded instruction.

const std = @import("std");
const parser = @import("parser");
const instmod = @import("instructions");
const Instruction = instmod.Instruction;
const InvokeKind = instmod.InvokeKind;

pub fn printDex(writer: anytype, dex: parser.DexFile, allocator: std.mem.Allocator) !void {
    for (dex.classes.items) |class| {
        try writer.print("class {s}\n", .{class.name});
        for (class.methods.items) |method| {
            try writer.print("  method {s}  regs={d} ins={d} outs={d} static={}\n", .{
                method.name,
                method.registers_size,
                method.ins_size,
                method.outs_size,
                method.is_static,
            });
            const insts = dex.decodeMethod(allocator, method) catch |err| {
                try writer.print("    <decode error: {s}>\n", .{@errorName(err)});
                continue;
            };
            defer {
                parser.freeDecodedInstructions(allocator, insts);
                allocator.free(insts);
            }
            for (insts, 0..) |inst, idx| {
                try writer.print("    [{d:>4}] ", .{idx});
                try printInstruction(writer, inst);
                try writer.writeByte('\n');
            }
        }
    }
}

pub fn printInstruction(writer: anytype, inst: Instruction) !void {
    switch (inst) {
        .nop => try writer.writeAll("nop"),
        .move => |v| try writer.print("move v{d}, v{d}", .{ v.dest, v.src }),
        .move_wide => |v| try writer.print("move-wide v{d}, v{d}", .{ v.dest, v.src }),
        .move_object => |v| try writer.print("move-object v{d}, v{d}", .{ v.dest, v.src }),
        .move_result => |v| try writer.print("move-result v{d}", .{v.dest}),
        .move_result_wide => |v| try writer.print("move-result-wide v{d}", .{v.dest}),
        .move_result_object => |v| try writer.print("move-result-object v{d}", .{v.dest}),
        .move_exception => |v| try writer.print("move-exception v{d}", .{v.dest}),

        .return_void => try writer.writeAll("return-void"),
        .return_ => |v| try writer.print("return v{d}", .{v.src}),
        .return_wide => |v| try writer.print("return-wide v{d}", .{v.src}),
        .return_object => |v| try writer.print("return-object v{d}", .{v.src}),

        .const_ => |v| try writer.print("const v{d}, #{d}", .{ v.dest, v.value }),
        .const_wide => |v| try writer.print("const-wide v{d}, #{d}", .{ v.dest, v.value }),
        .const_string => |v| try writer.print("const-string v{d}, string@{d}", .{ v.dest, v.index }),
        .const_class => |v| try writer.print("const-class v{d}, type@{d}", .{ v.dest, v.type_idx }),
        .const_method_handle => |v| try writer.print("const-method-handle v{d}, handle@{d}", .{ v.dest, v.index }),
        .const_method_type => |v| try writer.print("const-method-type v{d}, type@{d}", .{ v.dest, v.index }),

        .monitor_enter => |v| try writer.print("monitor-enter v{d}", .{v.src}),
        .monitor_exit => |v| try writer.print("monitor-exit v{d}", .{v.src}),

        .check_cast => |v| try writer.print("check-cast v{d}, type@{d}", .{ v.src, v.type_idx }),
        .instance_of => |v| try writer.print("instance-of v{d}, v{d}, type@{d}", .{ v.dest, v.src, v.type_idx }),

        .array_length => |v| try writer.print("array-length v{d}, v{d}", .{ v.dest, v.array }),
        .new_instance => |v| try writer.print("new-instance v{d}, type@{d}", .{ v.dest, v.type_idx }),
        .new_array => |v| try writer.print("new-array v{d}, v{d}, type@{d}", .{ v.dest, v.size, v.type_idx }),
        .filled_new_array => |v| {
            const count = if (v.payload) |p| p.args.len else 0;
            try writer.print("filled-new-array {{{d} args}}, type@{d}", .{ count, v.type_idx });
        },
        .fill_array_data => |v| {
            try writer.print("fill-array-data v{d}, +{d}", .{ v.array, v.payload_offset });
            if (v.payload) |p| {
                try writer.print(" // [", .{});
                for (p.data, 0..) |d, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.print("{d}", .{d});
                    if (i >= 3) {
                        try writer.print("... ({d} total)", .{p.data.len});
                        break;
                    }
                }
                try writer.writeAll("]");
            }
        },

        .throw_ => |v| try writer.print("throw v{d}", .{v.src}),

        .goto_ => |v| try writer.print("goto {d}", .{v.offset}),
        .packed_switch => |v| {
            try writer.print("packed-switch v{d}, +{d}", .{ v.src, v.payload_offset });
            if (v.payload) |p| {
                try writer.print(" // {d} targets", .{p.targets.len});
            }
        },
        .sparse_switch => |v| {
            try writer.print("sparse-switch v{d}, +{d}", .{ v.src, v.payload_offset });
            if (v.payload) |p| {
                try writer.print(" // {d} targets", .{p.targets.len});
            }
        },

        .cmpl_float => |v| try writer.print("cmpl-float v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .cmpg_float => |v| try writer.print("cmpg-float v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .cmpl_double => |v| try writer.print("cmpl-double v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .cmpg_double => |v| try writer.print("cmpg-double v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .cmp_long => |v| try writer.print("cmp-long v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),

        .if_eq => |v| try writer.print("if-eq v{d}, v{d}, {d}", .{ v.src1, v.src2, v.offset }),
        .if_ne => |v| try writer.print("if-ne v{d}, v{d}, {d}", .{ v.src1, v.src2, v.offset }),
        .if_lt => |v| try writer.print("if-lt v{d}, v{d}, {d}", .{ v.src1, v.src2, v.offset }),
        .if_ge => |v| try writer.print("if-ge v{d}, v{d}, {d}", .{ v.src1, v.src2, v.offset }),
        .if_gt => |v| try writer.print("if-gt v{d}, v{d}, {d}", .{ v.src1, v.src2, v.offset }),
        .if_le => |v| try writer.print("if-le v{d}, v{d}, {d}", .{ v.src1, v.src2, v.offset }),
        .if_eqz => |v| try writer.print("if-eqz v{d}, {d}", .{ v.src, v.offset }),
        .if_nez => |v| try writer.print("if-nez v{d}, {d}", .{ v.src, v.offset }),
        .if_ltz => |v| try writer.print("if-ltz v{d}, {d}", .{ v.src, v.offset }),
        .if_gez => |v| try writer.print("if-gez v{d}, {d}", .{ v.src, v.offset }),
        .if_gtz => |v| try writer.print("if-gtz v{d}, {d}", .{ v.src, v.offset }),
        .if_lez => |v| try writer.print("if-lez v{d}, {d}", .{ v.src, v.offset }),

        .aget => |v| try writer.print("aget v{d}, v{d}[v{d}]", .{ v.dest_or_src, v.array, v.index }),
        .aget_wide => |v| try writer.print("aget-wide v{d}, v{d}[v{d}]", .{ v.dest_or_src, v.array, v.index }),
        .aget_object => |v| try writer.print("aget-object v{d}, v{d}[v{d}]", .{ v.dest_or_src, v.array, v.index }),
        .aget_boolean => |v| try writer.print("aget-boolean v{d}, v{d}[v{d}]", .{ v.dest_or_src, v.array, v.index }),
        .aget_byte => |v| try writer.print("aget-byte v{d}, v{d}[v{d}]", .{ v.dest_or_src, v.array, v.index }),
        .aget_char => |v| try writer.print("aget-char v{d}, v{d}[v{d}]", .{ v.dest_or_src, v.array, v.index }),
        .aget_short => |v| try writer.print("aget-short v{d}, v{d}[v{d}]", .{ v.dest_or_src, v.array, v.index }),
        .aput => |v| try writer.print("aput v{d}, v{d}[v{d}]", .{ v.dest_or_src, v.array, v.index }),
        .aput_wide => |v| try writer.print("aput-wide v{d}, v{d}[v{d}]", .{ v.dest_or_src, v.array, v.index }),
        .aput_object => |v| try writer.print("aput-object v{d}, v{d}[v{d}]", .{ v.dest_or_src, v.array, v.index }),
        .aput_boolean => |v| try writer.print("aput-boolean v{d}, v{d}[v{d}]", .{ v.dest_or_src, v.array, v.index }),
        .aput_byte => |v| try writer.print("aput-byte v{d}, v{d}[v{d}]", .{ v.dest_or_src, v.array, v.index }),
        .aput_char => |v| try writer.print("aput-char v{d}, v{d}[v{d}]", .{ v.dest_or_src, v.array, v.index }),
        .aput_short => |v| try writer.print("aput-short v{d}, v{d}[v{d}]", .{ v.dest_or_src, v.array, v.index }),

        .iget => |v| try writer.print("iget v{d}, v{d}, field@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iget_wide => |v| try writer.print("iget-wide v{d}, v{d}, field@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iget_object => |v| try writer.print("iget-object v{d}, v{d}, field@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iget_boolean => |v| try writer.print("iget-boolean v{d}, v{d}, field@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iget_byte => |v| try writer.print("iget-byte v{d}, v{d}, field@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iget_char => |v| try writer.print("iget-char v{d}, v{d}, field@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iget_short => |v| try writer.print("iget-short v{d}, v{d}, field@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iput => |v| try writer.print("iput v{d}, v{d}, field@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iput_wide => |v| try writer.print("iput-wide v{d}, v{d}, field@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iput_object => |v| try writer.print("iput-object v{d}, v{d}, field@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iput_boolean => |v| try writer.print("iput-boolean v{d}, v{d}, field@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iput_byte => |v| try writer.print("iput-byte v{d}, v{d}, field@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iput_char => |v| try writer.print("iput-char v{d}, v{d}, field@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iput_short => |v| try writer.print("iput-short v{d}, v{d}, field@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),

        .sget => |v| try writer.print("sget v{d}, field@{d}", .{ v.dest_or_src, v.field_idx }),
        .sget_wide => |v| try writer.print("sget-wide v{d}, field@{d}", .{ v.dest_or_src, v.field_idx }),
        .sget_object => |v| try writer.print("sget-object v{d}, field@{d}", .{ v.dest_or_src, v.field_idx }),
        .sget_boolean => |v| try writer.print("sget-boolean v{d}, field@{d}", .{ v.dest_or_src, v.field_idx }),
        .sget_byte => |v| try writer.print("sget-byte v{d}, field@{d}", .{ v.dest_or_src, v.field_idx }),
        .sget_char => |v| try writer.print("sget-char v{d}, field@{d}", .{ v.dest_or_src, v.field_idx }),
        .sget_short => |v| try writer.print("sget-short v{d}, field@{d}", .{ v.dest_or_src, v.field_idx }),
        .sput => |v| try writer.print("sput v{d}, field@{d}", .{ v.dest_or_src, v.field_idx }),
        .sput_wide => |v| try writer.print("sput-wide v{d}, field@{d}", .{ v.dest_or_src, v.field_idx }),
        .sput_object => |v| try writer.print("sput-object v{d}, field@{d}", .{ v.dest_or_src, v.field_idx }),
        .sput_boolean => |v| try writer.print("sput-boolean v{d}, field@{d}", .{ v.dest_or_src, v.field_idx }),
        .sput_byte => |v| try writer.print("sput-byte v{d}, field@{d}", .{ v.dest_or_src, v.field_idx }),
        .sput_char => |v| try writer.print("sput-char v{d}, field@{d}", .{ v.dest_or_src, v.field_idx }),
        .sput_short => |v| try writer.print("sput-short v{d}, field@{d}", .{ v.dest_or_src, v.field_idx }),

        .iget_quick => |v| try writer.print("iget-quick v{d}, v{d}, field_offset@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iget_wide_quick => |v| try writer.print("iget-wide-quick v{d}, v{d}, field_offset@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iget_object_quick => |v| try writer.print("iget-object-quick v{d}, v{d}, field_offset@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iput_quick => |v| try writer.print("iput-quick v{d}, v{d}, field_offset@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iput_wide_quick => |v| try writer.print("iput-wide-quick v{d}, v{d}, field_offset@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),
        .iput_object_quick => |v| try writer.print("iput-object-quick v{d}, v{d}, field_offset@{d}", .{ v.dest_or_src, v.obj, v.field_idx }),

        .invoke_virtual_quick => |v| try writer.print("invoke-virtual-quick {{{d} args}} vtable@{d}", .{ v.args.len, v.target.method_idx }),
        .invoke_super_quick => |v| try writer.print("invoke-super-quick {{{d} args}} vtable@{d}", .{ v.args.len, v.target.method_idx }),

        .invoke => |v| {
            const kind_str = switch (v.kind) {
                .virtual => "invoke-virtual",
                .super => "invoke-super",
                .direct => "invoke-direct",
                .static => "invoke-static",
                .interface => "invoke-interface",
                .polymorphic => "invoke-polymorphic",
                .custom => "invoke-custom",
            };
            if (v.kind == .custom) {
                try writer.print("{s} {{{d} args}} call_site@{d}", .{
                    kind_str, v.args.len, v.target.call_site_idx,
                });
            } else {
                try writer.print("{s} {{{d} args}} {s}->{s} sig={s}", .{
                    kind_str, v.args.len, v.class_name, v.method_name, v.signature,
                });
            }
            if (v.dest) |d| try writer.print(" -> v{d}", .{d});
        },

        .neg_int => |v| try writer.print("neg-int v{d}, v{d}", .{ v.dest, v.src }),
        .not_int => |v| try writer.print("not-int v{d}, v{d}", .{ v.dest, v.src }),
        .neg_long => |v| try writer.print("neg-long v{d}, v{d}", .{ v.dest, v.src }),
        .not_long => |v| try writer.print("not-long v{d}, v{d}", .{ v.dest, v.src }),
        .neg_float => |v| try writer.print("neg-float v{d}, v{d}", .{ v.dest, v.src }),
        .neg_double => |v| try writer.print("neg-double v{d}, v{d}", .{ v.dest, v.src }),
        .int_to_long => |v| try writer.print("int-to-long v{d}, v{d}", .{ v.dest, v.src }),
        .int_to_float => |v| try writer.print("int-to-float v{d}, v{d}", .{ v.dest, v.src }),
        .int_to_double => |v| try writer.print("int-to-double v{d}, v{d}", .{ v.dest, v.src }),
        .long_to_int => |v| try writer.print("long-to-int v{d}, v{d}", .{ v.dest, v.src }),
        .long_to_float => |v| try writer.print("long-to-float v{d}, v{d}", .{ v.dest, v.src }),
        .long_to_double => |v| try writer.print("long-to-double v{d}, v{d}", .{ v.dest, v.src }),
        .float_to_int => |v| try writer.print("float-to-int v{d}, v{d}", .{ v.dest, v.src }),
        .float_to_long => |v| try writer.print("float-to-long v{d}, v{d}", .{ v.dest, v.src }),
        .float_to_double => |v| try writer.print("float-to-double v{d}, v{d}", .{ v.dest, v.src }),
        .double_to_int => |v| try writer.print("double-to-int v{d}, v{d}", .{ v.dest, v.src }),
        .double_to_long => |v| try writer.print("double-to-long v{d}, v{d}", .{ v.dest, v.src }),
        .double_to_float => |v| try writer.print("double-to-float v{d}, v{d}", .{ v.dest, v.src }),
        .int_to_byte => |v| try writer.print("int-to-byte v{d}, v{d}", .{ v.dest, v.src }),
        .int_to_char => |v| try writer.print("int-to-char v{d}, v{d}", .{ v.dest, v.src }),
        .int_to_short => |v| try writer.print("int-to-short v{d}, v{d}", .{ v.dest, v.src }),

        .add_int => |v| try writer.print("add-int v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .sub_int => |v| try writer.print("sub-int v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .mul_int => |v| try writer.print("mul-int v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .div_int => |v| try writer.print("div-int v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .rem_int => |v| try writer.print("rem-int v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .and_int => |v| try writer.print("and-int v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .or_int => |v| try writer.print("or-int v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .xor_int => |v| try writer.print("xor-int v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .shl_int => |v| try writer.print("shl-int v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .shr_int => |v| try writer.print("shr-int v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .ushr_int => |v| try writer.print("ushr-int v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .add_long => |v| try writer.print("add-long v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .sub_long => |v| try writer.print("sub-long v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .mul_long => |v| try writer.print("mul-long v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .div_long => |v| try writer.print("div-long v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .rem_long => |v| try writer.print("rem-long v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .and_long => |v| try writer.print("and-long v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .or_long => |v| try writer.print("or-long v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .xor_long => |v| try writer.print("xor-long v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .shl_long => |v| try writer.print("shl-long v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .shr_long => |v| try writer.print("shr-long v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .ushr_long => |v| try writer.print("ushr-long v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .add_float => |v| try writer.print("add-float v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .sub_float => |v| try writer.print("sub-float v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .mul_float => |v| try writer.print("mul-float v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .div_float => |v| try writer.print("div-float v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .rem_float => |v| try writer.print("rem-float v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .add_double => |v| try writer.print("add-double v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .sub_double => |v| try writer.print("sub-double v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .mul_double => |v| try writer.print("mul-double v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .div_double => |v| try writer.print("div-double v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),
        .rem_double => |v| try writer.print("rem-double v{d}, v{d}, v{d}", .{ v.dest, v.src1, v.src2 }),

        .add_int_lit16 => |v| try writer.print("add-int/lit16 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .rsub_int_lit16 => |v| try writer.print("rsub-int/lit16 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .mul_int_lit16 => |v| try writer.print("mul-int/lit16 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .div_int_lit16 => |v| try writer.print("div-int/lit16 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .rem_int_lit16 => |v| try writer.print("rem-int/lit16 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .and_int_lit16 => |v| try writer.print("and-int/lit16 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .or_int_lit16 => |v| try writer.print("or-int/lit16 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .xor_int_lit16 => |v| try writer.print("xor-int/lit16 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),

        .add_int_lit8 => |v| try writer.print("add-int/lit8 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .rsub_int_lit8 => |v| try writer.print("rsub-int/lit8 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .mul_int_lit8 => |v| try writer.print("mul-int/lit8 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .div_int_lit8 => |v| try writer.print("div-int/lit8 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .rem_int_lit8 => |v| try writer.print("rem-int/lit8 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .and_int_lit8 => |v| try writer.print("and-int/lit8 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .or_int_lit8 => |v| try writer.print("or-int/lit8 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .xor_int_lit8 => |v| try writer.print("xor-int/lit8 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .shl_int_lit8 => |v| try writer.print("shl-int/lit8 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .shr_int_lit8 => |v| try writer.print("shr-int/lit8 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
        .ushr_int_lit8 => |v| try writer.print("ushr-int/lit8 v{d}, v{d}, #{d}", .{ v.dest, v.src, v.lit }),
    }
}
