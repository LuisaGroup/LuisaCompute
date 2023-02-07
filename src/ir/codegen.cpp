//
// Created by Mike Smith on 2023/1/13.
//

#include <core/logging.h>
#include <ir/codegen.h>

namespace luisa::compute {

luisa::string CppSourceBuilder::_generate_type(const ir::Type *type) noexcept {
    auto iter = _type_names.find(type);
    if (iter == _type_names.end()) {
        auto primitive_name = [](const auto &prim) noexcept {
            using namespace std::string_view_literals;
            switch (prim) {
                case ir::Primitive::Bool: return "lc_bool"sv;
                case ir::Primitive::Int32: return "lc_int"sv;
                case ir::Primitive::Uint32: return "lc_uint"sv;
                case ir::Primitive::Int64: return "lc_long"sv;
                case ir::Primitive::Uint64: return "lc_ulong"sv;
                case ir::Primitive::Float32: return "lc_float"sv;
                case ir::Primitive::Float64: return "lc_double"sv;
            }
            LUISA_ERROR_WITH_LOCATION("Unsupported primitive type.");
        };
        switch (type->tag) {
            case ir::Type::Tag::Void: {
                iter = _type_names.emplace(type, "void").first;
                break;
            }
            case ir::Type::Tag::Primitive: {
                auto name = primitive_name(type->primitive._0);
                iter = _type_names.emplace(type, name).first;
                break;
            }
            case ir::Type::Tag::Vector: {
                auto &&elem = type->vector._0.element;
                switch (elem.tag) {
                    case ir::VectorElementType::Tag::Scalar: {
                        auto name = luisa::format("{}{}", primitive_name(elem.scalar._0),
                                                  type->vector._0.length);
                        iter = _type_names.emplace(type, name).first;
                        break;
                    }
                    case ir::VectorElementType::Tag::Vector: {
                        // TODO: support vector of vector
                        LUISA_ERROR_WITH_LOCATION("Vector of vector is not supported.");
                    }
                }
                break;
            }
            case ir::Type::Tag::Matrix: {
                auto &&elem = type->matrix._0.element;
                switch (elem.tag) {
                    case ir::VectorElementType::Tag::Scalar: {
                        auto dim = type->matrix._0.dimension;
                        auto name = luisa::format("{}{}x{}", primitive_name(elem.scalar._0),
                                                  dim, dim);
                        iter = _type_names.emplace(type, name).first;
                        break;
                    }
                    case ir::VectorElementType::Tag::Vector: {
                        // TODO: support matrix of vector
                        LUISA_ERROR_WITH_LOCATION("Matrix of vector is not supported.");
                    }
                }
                break;
            }
            case ir::Type::Tag::Struct: {
                auto &&s = type->struct_._0;
                auto &&fields = type->struct_._0.fields;
                luisa::string body;
                luisa::string zero_init;
                luisa::string one_init;
                luisa::string accum_grad;
                for (auto i = 0u; i < fields.len; i++) {
                    auto field = _generate_type(fields.ptr[i].get());
                    body.append(luisa::format("  {} m{};\n", field, i));
                    zero_init.append(luisa::format("  s.m{} = lc_zero<{}>();\n", i, field));
                    one_init.append(luisa::format("  s.m{} = lc_one<{}>();\n", i, field));
                    accum_grad.append(luisa::format("  lc_accumulate_grad(&dst->m{}, grad.m{});\n", i, i));
                }
                auto hash = hash64(body, hash64(s.alignment));
                auto name = luisa::format("lc_struct_{:016x}", hash);
                auto decl = luisa::format("struct alignas({}) {} {{\n"
                                          "{}"
                                          "}};\n\n",
                                          s.alignment, name, body);
                auto zero = luisa::format("template<>\n"
                                          "[[nodiscard]] __device__ inline auto lc_zero<{}>() noexcept {{\n"
                                          "  {} s;\n"
                                          "{}"
                                          "  return s;\n"
                                          "}}\n\n",
                                          name, name, zero_init);
                auto one = luisa::format("template<>\n"
                                         "[[nodiscard]] __device__ inline auto lc_one<{}>() noexcept {{\n"
                                         "  {} s;\n"
                                         "{}"
                                         "  return s;\n"
                                         "}}\n\n",
                                         name, name, one_init);
                auto grad = luisa::format("__device__ inline void lc_accumulate_grad({} *dst, {} grad) noexcept {{\n"
                                          "{}"
                                          "}}\n\n",
                                          name, name, accum_grad);
                _types.append(decl)
                    .append(zero)
                    .append(one)
                    .append(grad);
                iter = _type_names.emplace(type, name).first;
                break;
            }
            case ir::Type::Tag::Array: {
                auto elem = _generate_type(type->array._0.element.get());
                auto name = luisa::format("lc_array<{}, {}>", elem, type->array._0.length);
                iter = _type_names.emplace(type, name).first;
                break;
            }
        }
    }
    return iter->second;
}

luisa::string CppSourceBuilder::_generate_constant(const ir::Const &c) noexcept {
    switch (c.tag) {
        case ir::Const::Tag::Zero:
            return luisa::format("lc_zero<{}>()",
                                 _generate_type(c.zero._0.get()));
        case ir::Const::Tag::One:
            return luisa::format("lc_one<{}>()",
                                 _generate_type(c.one._0.get()));
        case ir::Const::Tag::Bool:
            return c.bool_._0 ? "true" : "false";
        case ir::Const::Tag::Int32:
            return luisa::format("lc_int({})", c.int32._0);
        case ir::Const::Tag::Uint32:
            return luisa::format("lc_uint({})", c.uint32._0);
        case ir::Const::Tag::Int64:
            return luisa::format("lc_long({})", c.int64._0);
        case ir::Const::Tag::Uint64:
            return luisa::format("lc_ulong({})", c.uint64._0);
        case ir::Const::Tag::Float32:
            return luisa::format("lc_float({})", c.float32._0);
        case ir::Const::Tag::Float64:
            return luisa::format("lc_double({})", c.float64._0);
        case ir::Const::Tag::Generic: {
            // decode the const into literals
            auto decode_primitive = [](ir::Primitive t, const uint8_t *data) noexcept {
                auto decode = [data]<typename T>() noexcept {
                    T x{};
                    std::memcpy(&x, data, sizeof(T));
                    return std::make_pair(x, data + sizeof(T));
                };
                switch (t) {
                    case ir::Primitive::Bool: {
                        auto [x, next] = decode.operator()<bool>();
                        return std::make_pair(luisa::string{x ? "true" : "false"}, next);
                    }
                    case ir::Primitive::Int32: {
                        auto [x, next] = decode.operator()<int32_t>();
                        return std::make_pair(luisa::format("lc_int({})", x), next);
                    }
                    case ir::Primitive::Uint32: {
                        auto [x, next] = decode.operator()<uint32_t>();
                        return std::make_pair(luisa::format("lc_uint({})", x), next);
                    }
                    case ir::Primitive::Int64: {
                        auto [x, next] = decode.operator()<int64_t>();
                        return std::make_pair(luisa::format("lc_long({})", x), next);
                    }
                    case ir::Primitive::Uint64: {
                        auto [x, next] = decode.operator()<uint64_t>();
                        return std::make_pair(luisa::format("lc_ulong({})", x), next);
                    }
                    case ir::Primitive::Float32: {
                        auto [x, next] = decode.operator()<float>();
                        return std::make_pair(luisa::format("lc_float({})", x), next);
                    }
                    case ir::Primitive::Float64: {
                        auto [x, next] = decode.operator()<double>();
                        return std::make_pair(luisa::format("lc_double({})", x), next);
                    }
                }
                LUISA_ERROR_WITH_LOCATION("Invalid primitive type.");
            };
            switch (auto t = c.generic._1.get(); t->tag) {
                case ir::Type::Tag::Void:
                    LUISA_ERROR_WITH_LOCATION("Void type cannot be encoded into a constant.");
                case ir::Type::Tag::Primitive:
                    return decode_primitive(t->primitive._0, c.generic._0.ptr).first;
                case ir::Type::Tag::Vector: [[fallthrough]];
                case ir::Type::Tag::Matrix: {
                    auto &&elem = t->tag == ir::Type::Tag::Vector ?
                                      t->vector._0.element :
                                      t->matrix._0.element;
                    switch (elem.tag) {
                        case ir::VectorElementType::Tag::Scalar: {
                            auto v = luisa::format("{}{{", _generate_type(t));
                            auto &&prim_type = elem.scalar._0;
                            auto &&data = c.generic._0;
                            auto [first, remaining] = decode_primitive(prim_type, data.ptr);
                            v.append(first);
                            while (remaining < data.ptr + data.len) {
                                auto [next, next_remaining] = decode_primitive(prim_type, remaining);
                                v.append(", ").append(next);
                                remaining = next_remaining;
                            }
                            v.append("}");
                            return v;
                        }
                        case ir::VectorElementType::Tag::Vector: {
                            LUISA_ERROR_WITH_LOCATION("Vector of vector is not supported.");
                        }
                    }
                }
                default: {
                    LUISA_ASSERT(t->tag == ir::Type::Tag::Array ||
                                     t->tag == ir::Type::Tag::Struct,
                                 "Invalid type.");
                    auto &&g = c.generic._0;
                    if (auto iter = _constant_names.find(g); iter != _constant_names.end()) {
                        return iter->second;
                    }
                    auto hash = BoxedSliceHash<uint8_t>{}(g);
                    auto name = luisa::format("lc_const_{:016x}", hash);
                    auto decl = luisa::format("__constant__ alignas(16) const unsigned char {}[] = {{", name);
                    for (auto i = 0u; i < g.len; i++) {
                        if (i % 16u == 0u) { decl.append("\n  "); }
                        decl.append(luisa::format("0x{:02x}, ", static_cast<uint>(g.ptr[i])));
                    }
                    decl.append("\n};\n\n");
                    _symbols.append(decl);
                    _constant_names.emplace(g, name);
                    return luisa::format("(*reinterpret_cast<const {} *>(&{}[0]))",
                                         _generate_type(t), name);
                }
            }
        }
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported constant type.");
}

void CppSourceBuilder::_collect_phis(const ir::BasicBlock *bb) noexcept {
    _iterate(bb, [this](const ir::Node *node) noexcept {
        auto instr = node->instruction.get();
        switch (instr->tag) {
            case ir::Instruction::Tag::Phi: {
                auto &&incomings = instr->phi._0;
                for (auto i = 0u; i < incomings.len; i++) {
                    auto &&incoming = incomings.ptr[i];
                    auto src_block = incoming.block.get();
                    auto src_value = ir::luisa_compute_ir_node_get(incoming.value);
                    _ctx->block_to_phis[src_block].emplace_back(PhiAssignment{node, src_value});
                }
                break;
            }
            case ir::Instruction::Tag::Loop: {
                _collect_phis(instr->loop.body.get());
                break;
            }
            case ir::Instruction::Tag::GenericLoop: {
                // TODO: this should have been lowered before codegen
                _collect_phis(instr->generic_loop.prepare.get());
                _collect_phis(instr->generic_loop.body.get());
                _collect_phis(instr->generic_loop.update.get());
                break;
            }
            case ir::Instruction::Tag::If: {
                _collect_phis(instr->if_.true_branch.get());
                _collect_phis(instr->if_.false_branch.get());
                break;
            }
            case ir::Instruction::Tag::Switch: {
                auto &&cases = instr->switch_.cases;
                for (auto i = 0u; i < cases.len; i++) {
                    _collect_phis(cases.ptr[i].block.get());
                }
                _collect_phis(instr->switch_.default_.get());
                break;
            }
            case ir::Instruction::Tag::AdScope: {
                // TODO: this should have been lowered before codegen
                _collect_phis(instr->ad_scope.forward.get());
                _collect_phis(instr->ad_scope.backward.get());
                _collect_phis(instr->ad_scope.epilogue.get());
                break;
            }
            case ir::Instruction::Tag::AdDetach: {
                // TODO: this should have been lowered before codegen
                _collect_phis(instr->ad_detach._0.get());
                break;
            }
            default: break;
        }
    });
}

luisa::string CppSourceBuilder::_generate_callable(const ir::CallableModule &callable) noexcept {
    if (auto iter = _callable_names.find(callable.module.entry.get());
        iter != _callable_names.end()) { return iter->second; }
    Context ctx{.module = callable.module};
    auto old_ctx = _ctx;
    _ctx = &ctx;
    auto entry = callable.module.entry.get();
    _collect_phis(entry);
    auto name = luisa::format("lc_callable_{}", _callable_names.size());
    _callable_names.emplace(entry, name);
    ctx.signature = luisa::format("__device__ auto {}(", name);
    auto args = callable.args;
    for (auto i = 0u; i < args.len; i++) {
        auto node = ir::luisa_compute_ir_node_get(args.ptr[i]);
        _generate_argument(node, i == args.len - 1u);
    }
    ctx.signature.append(") noexcept");
    _generate_block(entry);
    _symbols.append(ctx.signature)
        .append(" {\n")
        .append("  /* local variables */\n")
        .append(ctx.locals)
        .append("  /* body */\n")
        .append(ctx.body)
        .append("}\n\n");
    _ctx = old_ctx;
    return name;
}

void CppSourceBuilder::_generate_kernel(const ir::KernelModule &kernel) noexcept {
    _symbols.append(luisa::format("[[nodiscard]] __device__ constexpr lc_uint3 lc_block_size() noexcept {{\n  return lc_make_uint3({}, {}, {});\n}}\n\n",
                                  kernel.block_size[0], kernel.block_size[1], kernel.block_size[2]));
    Context ctx{.module = kernel.module};
    _ctx = &ctx;
    _collect_phis(kernel.module.entry.get());
    auto entry = kernel.module.entry.get();
    _collect_phis(entry);
    ctx.signature = "extern \"C\" __global__ void kernel_main(";
    auto args = kernel.args;
    auto captures = kernel.captures;
    for (auto i = 0u; i < captures.len; i++) {
        auto node = ir::luisa_compute_ir_node_get(captures.ptr[i].node);
        _generate_argument(node, false);
        ctx.signature.append(" /* captured */");
    }
    for (auto i = 0u; i < args.len; i++) {
        auto node = ir::luisa_compute_ir_node_get(args.ptr[i]);
        _generate_argument(node, false);
    }
    ctx.signature.append("\n    lc_uint3 dispatch_size)");
    auto shared = kernel.shared;
    for (auto i = 0u; i < shared.len; i++) {
        auto node = ir::luisa_compute_ir_node_get(shared.ptr[i]);
        ctx.locals.append(luisa::format(
            "  __shared__ {} {};\n",
            _generate_type(node->type_.get()),
            _generate_node(node)));
    }
    auto return_if_out_of_bounds = "if(lc_any(lc_dispatch_id() >= lc_dispatch_size())) return;";
    _generate_block(entry);
    _symbols.append(ctx.signature)
        .append(" {\n")
        .append("  /* local variables */\n")
        .append(ctx.locals)
        .append(return_if_out_of_bounds)
        .append("  /* body */\n")
        .append(ctx.body)
        .append("}\n");
}

luisa::string CppSourceBuilder::build(const ir::KernelModule &m) noexcept {
    CppSourceBuilder builder;
    builder._generate_kernel(m);
    luisa::string source;
    source.swap(builder._types);
    source.append(builder._symbols);
    return source;
}

luisa::string CppSourceBuilder::_generate_node(const ir::Node *node) noexcept {
    if (auto iter = _ctx->node_to_var.find(node);
        iter != _ctx->node_to_var.end()) { return iter->second; }
    auto name = [this, node, index = _ctx->node_to_var.size()] {
        switch (node->instruction->tag) {
            case ir::Instruction::Tag::Buffer:
                return luisa::format("b{}", index);
            case ir::Instruction::Tag::Bindless:
                return luisa::format("bl{}", index);
            case ir::Instruction::Tag::Texture2D:
                return luisa::format("t2d{}", index);
            case ir::Instruction::Tag::Texture3D:
                return luisa::format("t3d{}", index);
            case ir::Instruction::Tag::Accel:
                return luisa::format("a{}", index);
            case ir::Instruction::Tag::Shared:
                return luisa::format("s{}", index);
            case ir::Instruction::Tag::Uniform:
                return luisa::format("u{}", index);
            case ir::Instruction::Tag::Local:
                return luisa::format("v{}", index);
            case ir::Instruction::Tag::Argument:
                return luisa::format("arg{}", index);
            case ir::Instruction::Tag::UserData:
                return luisa::format("ud{}", index);
            case ir::Instruction::Tag::Const:
                return _generate_constant(node->instruction->const_._0);
            case ir::Instruction::Tag::Call:
                if (node->type_->tag == ir::Type::Tag::Void) { break; }
                return luisa::format("f{}", index);
            case ir::Instruction::Tag::Phi:
                return luisa::format("phi{}", index);
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid node type.");
    }();
    _ctx->node_to_var.emplace(node, name);
    return name;
}

luisa::string CppSourceBuilder::_generate_node(ir::NodeRef node) noexcept {
    return _generate_node(ir::luisa_compute_ir_node_get(node));
}

void CppSourceBuilder::_generate_argument(const ir::Node *node, bool is_last) noexcept {
    _ctx->signature.append("\n    ");
    auto arg_name = _generate_node(node);
    switch (node->instruction->tag) {
        case ir::Instruction::Tag::Buffer:
            _ctx->signature.append(luisa::format(
                "LCBuffer<{}> {}", _generate_type(node->type_.get()), arg_name));
            break;
        case ir::Instruction::Tag::Bindless:
            _ctx->signature.append(luisa::format(
                "LCBindlessArray {}", arg_name));
            break;
        case ir::Instruction::Tag::Texture2D:
            _ctx->signature.append(luisa::format(
                "LCTexture2D<{}> {}", _generate_type(node->type_.get()), arg_name));
            break;
        case ir::Instruction::Tag::Texture3D:
            _ctx->signature.append(luisa::format(
                "LCTexture3D<{}> {}", _generate_type(node->type_.get()), arg_name));
            break;
        case ir::Instruction::Tag::Accel:
            _ctx->signature.append(luisa::format(
                "LCAccel {}", arg_name));
            break;
        case ir::Instruction::Tag::Uniform:
            LUISA_ASSERT(_ctx->module.kind == ir::ModuleKind::Kernel,
                         "Uniforms are only allowed in kernels.");
            _ctx->signature.append(luisa::format(
                "{} {}", _generate_type(node->type_.get()), arg_name));
            break;
        case ir::Instruction::Tag::Argument:
            LUISA_ASSERT(_ctx->module.kind == ir::ModuleKind::Function,
                         "Arguments are only allowed in callables.");
            _ctx->signature.append(luisa::format(
                "{} {}{}", _generate_type(node->type_.get()),
                node->instruction->argument.by_value ? "" : "&",
                arg_name));
            break;
        default: LUISA_ERROR_WITH_LOCATION("Invalid argument type.");
    }
    if (!is_last) { _ctx->signature.append(","); }
}

void CppSourceBuilder::_generate_indent(uint depth) noexcept {
    for (auto i = 0u; i <= depth; i++) { _ctx->body.append("  "); }
}

void CppSourceBuilder::_generate_block(const ir::BasicBlock *bb, uint indent) noexcept {
    _iterate(bb, [this, indent](const ir::Node *node) noexcept {
        switch (node->instruction->tag) {
            case ir::Instruction::Tag::Local: _generate_instr_local(node, indent); break;
            case ir::Instruction::Tag::UserData: _generate_instr_user_data(node, indent); break;
            case ir::Instruction::Tag::Invalid: _generate_instr_invalid(node, indent); break;
            case ir::Instruction::Tag::Const: _generate_instr_const(node, indent); break;
            case ir::Instruction::Tag::Update: _generate_instr_update(node, indent); break;
            case ir::Instruction::Tag::Call: _generate_instr_call(node, indent); break;
            case ir::Instruction::Tag::Phi: _generate_instr_phi(node, indent); break;
            case ir::Instruction::Tag::Return: _generate_instr_return(node, indent); break;
            case ir::Instruction::Tag::Loop: _generate_instr_loop(node, indent); break;
            case ir::Instruction::Tag::GenericLoop: _generate_instr_generic_loop(node, indent); break;
            case ir::Instruction::Tag::Break: _generate_instr_break(node, indent); break;
            case ir::Instruction::Tag::Continue: _generate_instr_continue(node, indent); break;
            case ir::Instruction::Tag::If: _generate_instr_if(node, indent); break;
            case ir::Instruction::Tag::Switch: _generate_instr_switch(node, indent); break;
            case ir::Instruction::Tag::AdScope: _generate_instr_ad_scope(node, indent); break;
            case ir::Instruction::Tag::AdDetach: _generate_instr_ad_detach(node, indent); break;
            case ir::Instruction::Tag::Comment: _generate_instr_comment(node, indent); break;
            case ir::Instruction::Tag::Debug: _generate_instr_debug(node, indent); break;
            default: LUISA_ERROR_WITH_LOCATION("Invalid instruction in body.");
        }
    });
    // process phi nodes
    if (auto iter = _ctx->block_to_phis.find(bb);
        iter != _ctx->block_to_phis.end()) {
        for (auto phi : iter->second) {
            _generate_indent(indent);
            _ctx->body.append(luisa::format(
                "{} = {};\n",
                _generate_node(phi.dst),
                _generate_node(phi.src)));
        }
    }
}

void CppSourceBuilder::_generate_instr_local(const ir::Node *node, uint indent) noexcept {
    _generate_indent(indent);
    _ctx->body.append(luisa::format(
        "{} {} = {};\n",
        _generate_type(node->type_.get()),
        _generate_node(node),
        _generate_node(node->instruction->local.init)));
}

void CppSourceBuilder::_generate_instr_user_data(const ir::Node *node, uint indent) noexcept {
    LUISA_ERROR_WITH_LOCATION("Instruction 'UserData' is not implemented.");
}

void CppSourceBuilder::_generate_instr_invalid(const ir::Node *node, uint indent) noexcept {
    _generate_indent(indent);
    _ctx->body.append("/* basic block sentinel */\n");
}

void CppSourceBuilder::_generate_instr_const(const ir::Node *node, uint indent) noexcept {
    /* do nothing; will be inlined */
}

void CppSourceBuilder::_generate_instr_update(const ir::Node *node, uint indent) noexcept {
    _generate_indent(indent);
    _generate_node(node->instruction->update.var);
    _ctx->body.append(luisa::format(
        "{} = {};\n",
        _generate_node(node->instruction->update.var),
        _generate_node(node->instruction->update.value)));
}

void CppSourceBuilder::_generate_instr_call(const ir::Node *node, uint indent) noexcept {
    _generate_indent(indent);
    auto call = [this, node, indent]() noexcept -> luisa::string {
        auto &&[func, arg_slice] = node->instruction->call;
        auto args = luisa::span{arg_slice.ptr, arg_slice.len};
        if (node->type_->tag != ir::Type::Tag::Void) {
            if (func.tag == ir::Func::Tag::GetElementPtr) {
                auto qualifier = ir::luisa_compute_ir_node_get(args[0])->instruction.get()->tag == ir::Instruction::Tag::Local ? "" : "const ";
                _ctx->body.append(luisa::format(
                    "{}{} &{} = ",
                    qualifier,
                    _generate_type(node->type_.get()),
                    _generate_node(node)));
            } else if (func.tag != ir::Func::Tag::GradientMarker &&
                       func.tag != ir::Func::Tag::AccGrad &&
                       func.tag != ir::Func::Tag::Assume) {
                if (func.tag != ir::Func::Tag::InsertElement) {
                    _ctx->body.append("const ");
                }
                _ctx->body.append(luisa::format(
                    "{} {} = ",
                    _generate_type(node->type_.get()),
                    _generate_node(node)));
            }
        }
       
        auto constant_index = [](ir::NodeRef index) noexcept -> uint64_t {
            auto index_node = ir::luisa_compute_ir_node_get(index);
            LUISA_ASSERT(index_node->instruction->tag == ir::Instruction::Tag::Const,
                         "GetElementPtr's index must be a constant uint32.");
            auto &&c = index_node->instruction->const_._0;
            switch (c.tag) {
                case ir::Const::Tag::Zero: return 0;
                case ir::Const::Tag::One: return 1;
                case ir::Const::Tag::Int32: return c.int32._0;
                case ir::Const::Tag::Uint32: return c.uint32._0;
                case ir::Const::Tag::Int64: return c.int64._0;
                case ir::Const::Tag::Uint64: return c.uint64._0;
                case ir::Const::Tag::Generic: {
                    auto t = index_node->type_.get();
                    LUISA_ASSERT(t->tag == ir::Type::Tag::Primitive,
                                 "Invalid GetElementPtr index type.");
                    auto do_cast = [&c]<typename T>() noexcept {
                        T x{};
                        std::memcpy(&x, c.generic._0.ptr, sizeof(T));
                        return static_cast<uint64_t>(x);
                    };
                    switch (t->primitive._0) {
                        case ir::Primitive::Int32: return do_cast.operator()<int32_t>();
                        case ir::Primitive::Uint32: return do_cast.operator()<uint32_t>();
                        case ir::Primitive::Int64: return do_cast.operator()<int64_t>();
                        case ir::Primitive::Uint64: return do_cast.operator()<uint64_t>();
                        default: break;
                    }
                }
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Invalid GetElementPtr index.");
        };
        switch (func.tag) {
            case ir::Func::Tag::ZeroInitializer:
                LUISA_ASSERT(args.empty(), "ZeroInitializer takes no arguments.");
                return luisa::format("lc_zero<{}>();", _generate_type(node->type_.get()));
            case ir::Func::Tag::Assume:
                LUISA_ASSERT(args.size() == 1u, "Assume takes 1 argument.");
                return luisa::format("lc_assume({});", _generate_node(args[0]));
            case ir::Func::Tag::Unreachable:
                LUISA_ASSERT(args.empty(), "Unreachable takes no arguments.");
                return luisa::format("lc_unreachable<{}>();",
                                     _generate_type(node->type_.get()));
            case ir::Func::Tag::Assert:
                LUISA_ASSERT(args.size() == 1u, "Assert takes 1 argument.");
                return luisa::format("lc_assert({});", _generate_node(args[0]));
            case ir::Func::Tag::ThreadId:
                LUISA_ASSERT(args.empty(), "ThreadId takes no arguments.");
                return "lc_thread_id();";
            case ir::Func::Tag::BlockId:
                LUISA_ASSERT(args.empty(), "BlockId takes no arguments.");
                return "lc_block_id();";
            case ir::Func::Tag::DispatchId:
                LUISA_ASSERT(args.empty(), "DispatchId takes no arguments.");
                return "lc_dispatch_id();";
            case ir::Func::Tag::DispatchSize:
                LUISA_ASSERT(args.empty(), "DispatchSize takes no arguments.");
                return "lc_dispatch_size();";
            case ir::Func::Tag::RequiresGradient: {
                LUISA_ASSERT(args.size() == 1u, "RequiresGradient takes 1 argument.");
                auto var = ir::luisa_compute_ir_node_get(args[0]);
                auto name = _generate_node(args[0]);
                return luisa::format("/* requires_grad({}) */", name);
            }
            case ir::Func::Tag::Detach: {
                return luisa::format("{};", _generate_node(args[0]));
            }
            case ir::Func::Tag::Gradient:
                LUISA_ASSERT(args.size() == 1u, "Gradient takes 1 argument.");
                return luisa::format("{}_grad;", _generate_node(args[0]));
            case ir::Func::Tag::GradientMarker: {
                LUISA_ASSERT(args.size() == 2u, "GradientMarker takes 2 arguments.");
                auto var = ir::luisa_compute_ir_node_get(args[0]);
                auto var_name = _generate_node(var);
                if (_ctx->grads.emplace(var).second) {
                    auto type = _generate_type(var->type_.get());
                    _ctx->locals.append(luisa::format(
                        "  {} {}_grad = lc_zero<{}>();\n",
                        type, var_name, type));
                }
                return luisa::format("{}_grad = {}; // gradient_marker({});",
                                     var_name, _generate_node(args[1]), var_name);
            }
            case ir::Func::Tag::AccGrad:
                // TODO: generate lc_accumulate_grad for structures
                LUISA_ASSERT(args.size() == 2u, "AccGrad takes 2 arguments.");
                return luisa::format("lc_accumulate_grad(&({}), {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::InstanceToWorldMatrix:
                LUISA_ASSERT(args.size() == 1u, "InstanceToWorldMatrix takes 1 argument.");
                return luisa::format("lc_accel_instance_transform({});", _generate_node(args[0]));
            case ir::Func::Tag::TraceClosest:
                LUISA_ASSERT(args.size() == 2u, "TraceClosest takes 2 arguments.");
                return luisa::format("lc_accel_trace_closest({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::TraceAny:
                LUISA_ASSERT(args.size() == 2u, "TraceAny takes 2 arguments.");
                return luisa::format("lc_accel_trace_any({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::SetInstanceTransform:
                LUISA_ASSERT(args.size() == 3u, "SetInstanceTransform takes 3 arguments.");
                return luisa::format("lc_accel_set_instance_transform({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::SetInstanceVisibility:
                LUISA_ASSERT(args.size() == 3u, "SetInstanceVisibility takes 3 arguments.");
                return luisa::format("lc_accel_set_instance_visibility({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::Load:
                LUISA_ASSERT(args.size() == 1u, "Load takes 1 argument.");
                return luisa::format("{};", _generate_node(args[0]));
            case ir::Func::Tag::Cast: {
                LUISA_ASSERT(args.size() == 1u, "Cast takes 1 argument.");
                auto is_vector = node->type_.get()->tag == ir::Type::Tag::Vector;
                if (is_vector) {
                    auto vt = _generate_type(node->type_.get());
                    return luisa::format("lc_make_{}({});",
                                         vt.substr(3),
                                         _generate_node(args[0]));
                } else {
                    return luisa::format("static_cast<{}>({});",
                                         _generate_type(node->type_.get()),
                                         _generate_node(args[0]));
                }
            }
            case ir::Func::Tag::Bitcast:
                LUISA_ASSERT(args.size() == 1u, "Bitcast takes 1 argument.");
                return luisa::format("lc_bit_cast<{}>({});",
                                     _generate_type(node->type_.get()),
                                     _generate_node(args[0]));
            case ir::Func::Tag::Add:
                LUISA_ASSERT(args.size() == 2u, "Add takes 2 arguments.");
                return luisa::format("{} + {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Sub:
                LUISA_ASSERT(args.size() == 2u, "Sub takes 2 arguments.");
                return luisa::format("{} - {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Mul:
                LUISA_ASSERT(args.size() == 2u, "Mul takes 2 arguments.");
                return luisa::format("{} * {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Div:
                LUISA_ASSERT(args.size() == 2u, "Div takes 2 arguments.");
                return luisa::format("{} / {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Rem:
                LUISA_ASSERT(args.size() == 2u, "Rem takes 2 arguments.");
                return luisa::format("{} % {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::BitAnd:
                LUISA_ASSERT(args.size() == 2u, "BitAnd takes 2 arguments.");
                return luisa::format("{} & {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::BitOr:
                LUISA_ASSERT(args.size() == 2u, "BitOr takes 2 arguments.");
                return luisa::format("{} | {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::BitXor:
                LUISA_ASSERT(args.size() == 2u, "BitXor takes 2 arguments.");
                return luisa::format("{} ^ {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Shl:
                LUISA_ASSERT(args.size() == 2u, "Shl takes 2 arguments.");
                return luisa::format("{} << {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Shr:
                LUISA_ASSERT(args.size() == 2u, "Shr takes 2 arguments.");
                return luisa::format("{} >> {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::RotRight:
                LUISA_ASSERT(args.size() == 2u, "RotRight takes 2 arguments.");
                return luisa::format("lc_rotate_right({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::RotLeft:
                LUISA_ASSERT(args.size() == 2u, "RotLeft takes 2 arguments.");
                return luisa::format("lc_rotate_left({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Eq:
                LUISA_ASSERT(args.size() == 2u, "Eq takes 2 arguments.");
                return luisa::format("{} == {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Ne:
                LUISA_ASSERT(args.size() == 2u, "Ne takes 2 arguments.");
                return luisa::format("{} != {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Lt:
                LUISA_ASSERT(args.size() == 2u, "Lt takes 2 arguments.");
                return luisa::format("{} < {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Le:
                LUISA_ASSERT(args.size() == 2u, "Le takes 2 arguments.");
                return luisa::format("{} <= {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Gt:
                LUISA_ASSERT(args.size() == 2u, "Gt takes 2 arguments.");
                return luisa::format("{} > {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Ge:
                LUISA_ASSERT(args.size() == 2u, "Ge takes 2 arguments.");
                return luisa::format("{} >= {};",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::MatCompMul:
                LUISA_ASSERT(args.size() == 2u, "MatCompMul takes 2 arguments.");
                return luisa::format("lc_mat_comp_mul({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Neg:
                LUISA_ASSERT(args.size() == 1u, "Neg takes 1 argument.");
                return luisa::format("-{};", _generate_node(args[0]));
            case ir::Func::Tag::Not:
                LUISA_ASSERT(args.size() == 1u, "Not takes 1 argument.");
                return luisa::format("!{};", _generate_node(args[0]));
            case ir::Func::Tag::BitNot: {
                LUISA_ASSERT(args.size() == 1u, "BitNot takes 1 argument.");
                auto arg = ir::luisa_compute_ir_node_get(args[0]);
                auto is_bool = arg->type_->tag == ir::Type::Tag::Primitive &&
                               arg->type_->primitive._0 == ir::Primitive::Bool;
                return luisa::format("{}{};", is_bool ? "!" : "~", _generate_node(arg));
            }
            case ir::Func::Tag::All:
                LUISA_ASSERT(args.size() == 1u, "All takes 1 argument.");
                return luisa::format("lc_all({});", _generate_node(args[0]));
            case ir::Func::Tag::Any:
                LUISA_ASSERT(args.size() == 1u, "Any takes 1 argument.");
                return luisa::format("lc_any({});", _generate_node(args[0]));
            case ir::Func::Tag::Select:
                LUISA_ASSERT(args.size() == 3u, "Select takes 3 arguments.");
                return luisa::format("lc_select({}, {}, {});",
                                     _generate_node(args[2]),
                                     _generate_node(args[1]),
                                     _generate_node(args[0]));
            case ir::Func::Tag::Clamp:
                LUISA_ASSERT(args.size() == 3u, "Clamp takes 3 arguments.");
                return luisa::format("lc_clamp({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::Lerp:
                LUISA_ASSERT(args.size() == 3u, "Lerp takes 3 arguments.");
                return luisa::format("lc_lerp({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::Step:
                LUISA_ASSERT(args.size() == 2u, "Step takes 2 arguments.");
                return luisa::format("lc_step({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Abs:
                LUISA_ASSERT(args.size() == 1u, "Abs takes 1 argument.");
                return luisa::format("lc_abs({});", _generate_node(args[0]));
            case ir::Func::Tag::Min:
                LUISA_ASSERT(args.size() == 2u, "Min takes 2 arguments.");
                return luisa::format("lc_min({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Max:
                LUISA_ASSERT(args.size() == 2u, "Max takes 2 arguments.");
                return luisa::format("lc_max({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::ReduceSum:
                LUISA_ASSERT(args.size() == 1u, "ReduceSum takes 1 argument.");
                return luisa::format("lc_reduce_sum({});", _generate_node(args[0]));
            case ir::Func::Tag::ReduceProd:
                LUISA_ASSERT(args.size() == 1u, "ReduceProd takes 1 argument.");
                return luisa::format("lc_reduce_prod({});", _generate_node(args[0]));
            case ir::Func::Tag::ReduceMin:
                LUISA_ASSERT(args.size() == 1u, "ReduceMin takes 1 argument.");
                return luisa::format("lc_reduce_min({});", _generate_node(args[0]));
            case ir::Func::Tag::ReduceMax:
                LUISA_ASSERT(args.size() == 1u, "ReduceMax takes 1 argument.");
                return luisa::format("lc_reduce_max({});", _generate_node(args[0]));
            case ir::Func::Tag::Clz:
                LUISA_ASSERT(args.size() == 1u, "Clz takes 1 argument.");
                return luisa::format("lc_clz({});", _generate_node(args[0]));
            case ir::Func::Tag::Ctz:
                LUISA_ASSERT(args.size() == 1u, "Ctz takes 1 argument.");
                return luisa::format("lc_ctz({});", _generate_node(args[0]));
            case ir::Func::Tag::PopCount:
                LUISA_ASSERT(args.size() == 1u, "PopCount takes 1 argument.");
                return luisa::format("lc_popcount({});", _generate_node(args[0]));
            case ir::Func::Tag::Reverse:
                LUISA_ASSERT(args.size() == 1u, "Reverse takes 1 argument.");
                return luisa::format("lc_reverse({});", _generate_node(args[0]));
            case ir::Func::Tag::IsInf:
                LUISA_ASSERT(args.size() == 1u, "IsInf takes 1 argument.");
                return luisa::format("lc_isinf({});", _generate_node(args[0]));
            case ir::Func::Tag::IsNan:
                LUISA_ASSERT(args.size() == 1u, "IsNan takes 1 argument.");
                return luisa::format("lc_isnan({});", _generate_node(args[0]));
            case ir::Func::Tag::Acos:
                LUISA_ASSERT(args.size() == 1u, "Acos takes 1 argument.");
                return luisa::format("lc_acos({});", _generate_node(args[0]));
            case ir::Func::Tag::Acosh:
                LUISA_ASSERT(args.size() == 1u, "Acosh takes 1 argument.");
                return luisa::format("lc_acosh({});", _generate_node(args[0]));
            case ir::Func::Tag::Asin:
                LUISA_ASSERT(args.size() == 1u, "Asin takes 1 argument.");
                return luisa::format("lc_asin({});", _generate_node(args[0]));
            case ir::Func::Tag::Asinh:
                LUISA_ASSERT(args.size() == 1u, "Asinh takes 1 argument.");
                return luisa::format("lc_asinh({});", _generate_node(args[0]));
            case ir::Func::Tag::Atan:
                LUISA_ASSERT(args.size() == 1u, "Atan takes 1 argument.");
                return luisa::format("lc_atan({});", _generate_node(args[0]));
            case ir::Func::Tag::Atan2:
                LUISA_ASSERT(args.size() == 2u, "Atan2 takes 2 arguments.");
                return luisa::format("lc_atan2({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Atanh:
                LUISA_ASSERT(args.size() == 1u, "Atanh takes 1 argument.");
                return luisa::format("lc_atanh({});", _generate_node(args[0]));
            case ir::Func::Tag::Cos:
                LUISA_ASSERT(args.size() == 1u, "Cos takes 1 argument.");
                return luisa::format("lc_cos({});", _generate_node(args[0]));
            case ir::Func::Tag::Cosh:
                LUISA_ASSERT(args.size() == 1u, "Cosh takes 1 argument.");
                return luisa::format("lc_cosh({});", _generate_node(args[0]));
            case ir::Func::Tag::Sin:
                LUISA_ASSERT(args.size() == 1u, "Sin takes 1 argument.");
                return luisa::format("lc_sin({});", _generate_node(args[0]));
            case ir::Func::Tag::Sinh:
                LUISA_ASSERT(args.size() == 1u, "Sinh takes 1 argument.");
                return luisa::format("lc_sinh({});", _generate_node(args[0]));
            case ir::Func::Tag::Tan:
                LUISA_ASSERT(args.size() == 1u, "Tan takes 1 argument.");
                return luisa::format("lc_tan({});", _generate_node(args[0]));
            case ir::Func::Tag::Tanh:
                LUISA_ASSERT(args.size() == 1u, "Tanh takes 1 argument.");
                return luisa::format("lc_tanh({});", _generate_node(args[0]));
            case ir::Func::Tag::Exp:
                LUISA_ASSERT(args.size() == 1u, "Exp takes 1 argument.");
                return luisa::format("lc_exp({});", _generate_node(args[0]));
            case ir::Func::Tag::Exp2:
                LUISA_ASSERT(args.size() == 1u, "Exp2 takes 1 argument.");
                return luisa::format("lc_exp2({});", _generate_node(args[0]));
            case ir::Func::Tag::Exp10:
                LUISA_ASSERT(args.size() == 1u, "Exp10 takes 1 argument.");
                return luisa::format("lc_exp10({});", _generate_node(args[0]));
            case ir::Func::Tag::Log:
                LUISA_ASSERT(args.size() == 1u, "Log takes 1 argument.");
                return luisa::format("lc_log({});", _generate_node(args[0]));
            case ir::Func::Tag::Log2:
                LUISA_ASSERT(args.size() == 1u, "Log2 takes 1 argument.");
                return luisa::format("lc_log2({});", _generate_node(args[0]));
            case ir::Func::Tag::Log10:
                LUISA_ASSERT(args.size() == 1u, "Log10 takes 1 argument.");
                return luisa::format("lc_log10({});", _generate_node(args[0]));
            case ir::Func::Tag::Powi:
                LUISA_ASSERT(args.size() == 2u, "Powi takes 2 arguments.");
                return luisa::format("lc_powi({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Powf:
                LUISA_ASSERT(args.size() == 2u, "Powf takes 2 arguments.");
                return luisa::format("lc_powf({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Sqrt:
                LUISA_ASSERT(args.size() == 1u, "Sqrt takes 1 argument.");
                return luisa::format("lc_sqrt({});", _generate_node(args[0]));
            case ir::Func::Tag::Rsqrt:
                LUISA_ASSERT(args.size() == 1u, "Rsqrt takes 1 argument.");
                return luisa::format("lc_rsqrt({});", _generate_node(args[0]));
            case ir::Func::Tag::Ceil:
                LUISA_ASSERT(args.size() == 1u, "Ceil takes 1 argument.");
                return luisa::format("lc_ceil({});", _generate_node(args[0]));
            case ir::Func::Tag::Floor:
                LUISA_ASSERT(args.size() == 1u, "Floor takes 1 argument.");
                return luisa::format("lc_floor({});", _generate_node(args[0]));
            case ir::Func::Tag::Fract:
                LUISA_ASSERT(args.size() == 1u, "Fract takes 1 argument.");
                return luisa::format("lc_fract({});", _generate_node(args[0]));
            case ir::Func::Tag::Trunc:
                LUISA_ASSERT(args.size() == 1u, "Trunc takes 1 argument.");
                return luisa::format("lc_trunc({});", _generate_node(args[0]));
            case ir::Func::Tag::Round:
                LUISA_ASSERT(args.size() == 1u, "Round takes 1 argument.");
                return luisa::format("lc_round({});", _generate_node(args[0]));
            case ir::Func::Tag::Fma:
                LUISA_ASSERT(args.size() == 3u, "Fma takes 3 arguments.");
                return luisa::format("lc_fma({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::Copysign:
                LUISA_ASSERT(args.size() == 2u, "Copysign takes 2 arguments.");
                return luisa::format("lc_copysign({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Cross:
                LUISA_ASSERT(args.size() == 2u, "Cross takes 2 arguments.");
                return luisa::format("lc_cross({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Dot:
                LUISA_ASSERT(args.size() == 2u, "Dot takes 2 arguments.");
                return luisa::format("lc_dot({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::OuterProduct:
                LUISA_ASSERT(args.size() == 2u, "OuterProduct takes 2 arguments.");
                return luisa::format("lc_outer_product({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::Length:
                LUISA_ASSERT(args.size() == 1u, "Length takes 1 argument.");
                return luisa::format("lc_length({});", _generate_node(args[0]));
            case ir::Func::Tag::LengthSquared:
                LUISA_ASSERT(args.size() == 1u, "LengthSquared takes 1 argument.");
                return luisa::format("lc_length_squared({});", _generate_node(args[0]));
            case ir::Func::Tag::Normalize:
                LUISA_ASSERT(args.size() == 1u, "Normalize takes 1 argument.");
                return luisa::format("lc_normalize({});", _generate_node(args[0]));
            case ir::Func::Tag::Faceforward:
                LUISA_ASSERT(args.size() == 3u, "Faceforward takes 3 arguments.");
                return luisa::format("lc_faceforward({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::Determinant:
                LUISA_ASSERT(args.size() == 1u, "Determinant takes 1 argument.");
                return luisa::format("lc_determinant({});", _generate_node(args[0]));
            case ir::Func::Tag::Transpose:
                LUISA_ASSERT(args.size() == 1u, "Transpose takes 1 argument.");
                return luisa::format("lc_transpose({});", _generate_node(args[0]));
            case ir::Func::Tag::Inverse:
                LUISA_ASSERT(args.size() == 1u, "Inverse takes 1 argument.");
                return luisa::format("lc_inverse({});", _generate_node(args[0]));
            case ir::Func::Tag::SynchronizeBlock:
                LUISA_ASSERT(args.empty(), "SynchronizeBlock takes no arguments.");
                return "lc_synchronize_block();";
            case ir::Func::Tag::AtomicExchange:
                LUISA_ASSERT(args.size() == 3u, "AtomicExchange takes 3 arguments.");
                return luisa::format("lc_atomic_exchange({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::AtomicCompareExchange:
                LUISA_ASSERT(args.size() == 4u, "AtomicCompareExchange takes 4 arguments.");
                return luisa::format("lc_atomic_compare_exchange({}, {}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]),
                                     _generate_node(args[3]));
            case ir::Func::Tag::AtomicFetchAdd:
                LUISA_ASSERT(args.size() == 3u, "AtomicFetchAdd takes 3 arguments.");
                return luisa::format("lc_atomic_fetch_add({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::AtomicFetchSub:
                LUISA_ASSERT(args.size() == 3u, "AtomicFetchSub takes 3 arguments.");
                return luisa::format("lc_atomic_fetch_sub({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::AtomicFetchAnd:
                LUISA_ASSERT(args.size() == 3u, "AtomicFetchAnd takes 3 arguments.");
                return luisa::format("lc_atomic_fetch_and({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::AtomicFetchOr:
                LUISA_ASSERT(args.size() == 3u, "AtomicFetchOr takes 3 arguments.");
                return luisa::format("lc_atomic_fetch_or({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::AtomicFetchXor:
                LUISA_ASSERT(args.size() == 3u, "AtomicFetchXor takes 3 arguments.");
                return luisa::format("lc_atomic_fetch_xor({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::AtomicFetchMin:
                LUISA_ASSERT(args.size() == 3u, "AtomicFetchMin takes 3 arguments.");
                return luisa::format("lc_atomic_fetch_min({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::AtomicFetchMax:
                LUISA_ASSERT(args.size() == 3u, "AtomicFetchMax takes 3 arguments.");
                return luisa::format("lc_atomic_fetch_max({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::BufferRead:
                LUISA_ASSERT(args.size() == 2u, "BufferRead takes 2 arguments.");
                return luisa::format("lc_buffer_read({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::BufferWrite:
                LUISA_ASSERT(args.size() == 3u, "BufferWrite takes 3 arguments.");
                return luisa::format("lc_buffer_write({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::BufferSize:
                LUISA_ASSERT(args.size() == 1u, "BufferSize takes 1 argument.");
                return luisa::format("lc_buffer_size({});", _generate_node(args[0]));
            case ir::Func::Tag::TextureRead: {
                LUISA_ASSERT(args.size() == 2u, "TextureRead takes 2 arguments.");
                auto tex = ir::luisa_compute_ir_node_get(args[0]);
                return luisa::format("lc_texture_read({}, {});",
                                     _generate_type(tex->type_.get()),
                                     _generate_node(tex),
                                     _generate_node(args[1]));
            }
            case ir::Func::Tag::TextureWrite: {
                LUISA_ASSERT(args.size() == 3u, "TextureWrite takes 3 arguments.");
                auto tex = ir::luisa_compute_ir_node_get(args[0]);
                return luisa::format("lc_texture_write({}, {}, {});",
                                     _generate_type(tex->type_.get()),
                                     _generate_node(tex),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            }
            case ir::Func::Tag::BindlessTexture2dSample:
                LUISA_ASSERT(args.size() == 3u, "BindlessTexture2dSample takes 3 arguments.");
                return luisa::format("lc_bindless_texture_sample2d({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::BindlessTexture2dSampleLevel:
                LUISA_ASSERT(args.size() == 4u, "BindlessTexture2dSampleLevel takes 4 arguments.");
                return luisa::format("lc_bindless_texture_sample2d_level({}, {}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]),
                                     _generate_node(args[3]));
            case ir::Func::Tag::BindlessTexture2dSampleGrad:
                LUISA_ASSERT(args.size() == 5u, "BindlessTexture2dSampleGrad takes 5 arguments.");
                return luisa::format("lc_bindless_texture_sample2d_grad({}, {}, {}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]),
                                     _generate_node(args[3]),
                                     _generate_node(args[4]));
            case ir::Func::Tag::BindlessTexture3dSample:
                LUISA_ASSERT(args.size() == 3u, "BindlessTexture3dSample takes 3 arguments.");
                return luisa::format("lc_bindless_texture_sample3d({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::BindlessTexture3dSampleLevel:
                LUISA_ASSERT(args.size() == 4u, "BindlessTexture3dSampleLevel takes 4 arguments.");
                return luisa::format("lc_bindless_texture_sample3d_level({}, {}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]),
                                     _generate_node(args[3]));
            case ir::Func::Tag::BindlessTexture3dSampleGrad:
                LUISA_ASSERT(args.size() == 5u, "BindlessTexture3dSampleGrad takes 5 arguments.");
                return luisa::format("lc_bindless_texture_sample3d_grad({}, {}, {}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]),
                                     _generate_node(args[3]),
                                     _generate_node(args[4]));
            case ir::Func::Tag::BindlessTexture2dRead:
                LUISA_ASSERT(args.size() == 2u, "BindlessTexture2dRead takes 2 arguments.");
                return luisa::format("lc_bindless_texture_read2d({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::BindlessTexture3dRead:
                LUISA_ASSERT(args.size() == 2u, "BindlessTexture3dRead takes 2 arguments.");
                return luisa::format("lc_bindless_texture_read3d({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::BindlessTexture2dReadLevel:
                LUISA_ASSERT(args.size() == 3u, "BindlessTexture2dReadLevel takes 3 arguments.");
                return luisa::format("lc_bindless_texture_read2d_level({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::BindlessTexture3dReadLevel:
                LUISA_ASSERT(args.size() == 3u, "BindlessTexture3dReadLevel takes 3 arguments.");
                return luisa::format("lc_bindless_texture_read3d_level({}, {}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            case ir::Func::Tag::BindlessTexture2dSize:
                LUISA_ASSERT(args.size() == 1u, "BindlessTexture2dSize takes 1 argument.");
                return luisa::format("lc_bindless_texture_size2d({});", _generate_node(args[0]));
            case ir::Func::Tag::BindlessTexture3dSize:
                LUISA_ASSERT(args.size() == 1u, "BindlessTexture3dSize takes 1 argument.");
                return luisa::format("lc_bindless_texture_size3d({});", _generate_node(args[0]));
            case ir::Func::Tag::BindlessTexture2dSizeLevel:
                LUISA_ASSERT(args.size() == 2u, "BindlessTexture2dSizeLevel takes 2 arguments.");
                return luisa::format("lc_bindless_texture_size2d_level({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::BindlessTexture3dSizeLevel:
                LUISA_ASSERT(args.size() == 2u, "BindlessTexture3dSizeLevel takes 2 arguments.");
                return luisa::format("lc_bindless_texture_size3d_level({}, {});",
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::BindlessBufferRead:
                LUISA_ASSERT(args.size() == 2u, "BindlessBufferRead takes 2 arguments.");
                return luisa::format("lc_bindless_buffer_read<{}>({}, {});",
                                     _generate_type(node->type_.get()),
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            case ir::Func::Tag::BindlessBufferSize:
                LUISA_ASSERT(args.size() == 1u, "BindlessBufferSize takes 1 argument.");
                return luisa::format("lc_bindless_buffer_size({});", _generate_node(args[0]));
            case ir::Func::Tag::BindlessBufferType:
                LUISA_ASSERT(args.size() == 1u, "BindlessBufferType takes 1 argument.");
                return luisa::format("lc_bindless_buffer_type({});", _generate_node(args[0]));
            case ir::Func::Tag::Vec: {
                LUISA_ASSERT(args.size() == 1u, "Vec takes 1 argument.");
                auto type = _generate_type(node->type_.get());
                LUISA_ASSERT(type.starts_with("lc_"), "Invalid vector type '{}'.", type);
                return luisa::format("lc_make_{}({});",
                                     luisa::string_view{type}.substr(3u),
                                     _generate_node(args[0]));
            }
            case ir::Func::Tag::Vec2: {
                LUISA_ASSERT(args.size() == 2u, "Vec2 takes 2 arguments.");
                auto type = _generate_type(node->type_.get());
                LUISA_ASSERT(type.starts_with("lc_"), "Invalid vector type '{}'.", type);
                return luisa::format("lc_make_{}({}, {});",
                                     luisa::string_view{type}.substr(3u),
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            }
            case ir::Func::Tag::Vec3: {
                LUISA_ASSERT(args.size() == 3u, "Vec3 takes 3 arguments.");
                auto type = _generate_type(node->type_.get());
                LUISA_ASSERT(type.starts_with("lc_"), "Invalid vector type '{}'.", type);
                return luisa::format("lc_make_{}({}, {}, {});",
                                     luisa::string_view{type}.substr(3u),
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            }
            case ir::Func::Tag::Vec4: {
                LUISA_ASSERT(args.size() == 4u, "Vec4 takes 4 arguments.");
                auto type = _generate_type(node->type_.get());
                LUISA_ASSERT(type.starts_with("lc_"), "Invalid vector type '{}'.", type);
                return luisa::format("lc_make_{}({}, {}, {}, {});",
                                     luisa::string_view{type}.substr(3u),
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]),
                                     _generate_node(args[3]));
            }
            case ir::Func::Tag::Permute: {
                auto src = _generate_node(args[0]);
                auto type = _generate_type(node->type_.get());
                LUISA_ASSERT(type.starts_with("lc_"), "Invalid vector type '{}'.", type);
                auto invoke = luisa::format("lc_make_{}(", luisa::string_view{type}.substr(3u));
                for (auto i = 1u; i < args.size(); i++) {
                    invoke.append(luisa::format("{}[{}]", src, _generate_node(args[i])));
                    if (i != args.size() - 1u) { invoke.append(", "); }
                }
                invoke.append(");");
                return invoke;
            }
            case ir::Func::Tag::InsertElement: {
                LUISA_ASSERT(args.size() == 3u, "InsertElement takes 3 arguments.");
                auto self = ir::luisa_compute_ir_node_get(args[0]);
                _ctx->body.append(luisa::format("{};\n", _generate_node(args[0])));// make a copy
                _generate_indent(indent);
                auto target = _generate_node(node);
                if (self->type_->tag == ir::Type::Tag::Struct) {
                    auto index = constant_index(args[2]);
                    LUISA_ASSERT(index < self->type_->struct_._0.fields.len,
                                 "InsertElement index out of range.");
                    return luisa::format("{}.m{} = {};",
                                         target, index,
                                         _generate_node(args[1]));
                }
                return luisa::format("{}[{}] = {};",
                                     target,
                                     _generate_node(args[2]),
                                     _generate_node(args[1]));
            }
            case ir::Func::Tag::ExtractElement: [[fallthrough]];
            case ir::Func::Tag::GetElementPtr: {
                auto op = func.tag == ir::Func::Tag::ExtractElement ?
                              "ExtractElement" :
                              "GetElementPtr";
                LUISA_ASSERT(args.size() == 2u, "{} takes 2 arguments.", op);
                auto self = ir::luisa_compute_ir_node_get(args[0]);
                if (self->type_->tag == ir::Type::Tag::Struct) {
                    auto index = constant_index(args[1]);
                    LUISA_ASSERT(index < self->type_->struct_._0.fields.len,
                                 "{} index out of range.", op);
                    return luisa::format("{}.m{};", _generate_node(self), index);
                }
                return luisa::format("{}[{}];",
                                     _generate_node(self),
                                     _generate_node(args[1]));
            }
            case ir::Func::Tag::Struct: {
                auto invoke = luisa::format("{}{{", _generate_type(node->type_.get()));
                for (auto i = 0u; i < args.size(); i++) {
                    invoke += _generate_node(args[i]);
                    if (i != args.size() - 1u) { invoke += ", "; }
                }
                invoke.append("};");
                return invoke;
            }
            case ir::Func::Tag::Mat: {
                LUISA_ASSERT(args.size() == 1u, "Mat takes 1 argument.");
                auto type = _generate_type(node->type_.get());
                LUISA_ASSERT(type.starts_with("lc_"), "Invalid matrix type '{}'.", type);
                return luisa::format("{}::full({});",
                                     type,
                                     _generate_node(args[0]));
            }
            case ir::Func::Tag::Mat2: {
                LUISA_ASSERT(args.size() == 2u, "Mat2 takes 2 arguments.");
                auto type = _generate_type(node->type_.get());
                LUISA_ASSERT(type.starts_with("lc_"), "Invalid matrix type '{}'.", type);
                return luisa::format("lc_make_{}({}, {});",
                                     luisa::string_view{type}.substr(3u),
                                     _generate_node(args[0]),
                                     _generate_node(args[1]));
            }
            case ir::Func::Tag::Mat3: {
                LUISA_ASSERT(args.size() == 3u, "Mat3 takes 3 arguments.");
                auto type = _generate_type(node->type_.get());
                LUISA_ASSERT(type.starts_with("lc_"), "Invalid matrix type '{}'.", type);
                return luisa::format("lc_make_{}({}, {}, {});",
                                     luisa::string_view{type}.substr(3u),
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]));
            }
            case ir::Func::Tag::Mat4: {
                LUISA_ASSERT(args.size() == 4u, "Mat4 takes 4 arguments.");
                auto type = _generate_type(node->type_.get());
                LUISA_ASSERT(type.starts_with("lc_"), "Invalid matrix type '{}'.", type);
                return luisa::format("lc_make_{}({}, {}, {}, {});",
                                     luisa::string_view{type}.substr(3u),
                                     _generate_node(args[0]),
                                     _generate_node(args[1]),
                                     _generate_node(args[2]),
                                     _generate_node(args[3]));
            }
            case ir::Func::Tag::Callable: {
                auto p_callable = ir::luisa_compute_ir_get_symbol(func.callable._0);
                LUISA_ASSERT(p_callable != nullptr, "Invalid callable.");
                auto callable = _generate_callable(p_callable->data);
                auto invoke = luisa::format("{}(", callable);
                for (auto i = 0u; i < args.size(); i++) {
                    invoke.append(_generate_node(args[i]));
                    if (i != args.size() - 1u) { invoke.append(", "); }
                }
                invoke.append(");");
                return invoke;
            }
            case ir::Func::Tag::CpuCustomOp:
                LUISA_ERROR_WITH_LOCATION("CpuCustomOp is not implemented.");
        }
    }();
    _ctx->body.append(call).append("\n");
}

void CppSourceBuilder::_generate_instr_phi(const ir::Node *node, uint indent) noexcept {
    auto dst = _generate_node(node);
    auto type = _generate_type(node->type_.get());
    _ctx->locals.append(luisa::format(
        "  {} {} = lc_zero<{}>();\n",
        type, dst, type));
    _generate_indent(indent);
    _ctx->body.append(luisa::format(
        "/* phi-node for {} */\n", dst));
}

void CppSourceBuilder::_generate_instr_return(const ir::Node *node, uint indent) noexcept {
    _generate_indent(indent);
    if (auto ret = node->instruction->return_._0; ret != ir::INVALID_REF) {
        _ctx->body.append(luisa::format(
            "return {};\n", _generate_node(ret)));
    } else {
        _ctx->body.append("return;\n");
    }
}

void CppSourceBuilder::_generate_instr_loop(const ir::Node *node, uint indent) noexcept {
    // loop {
    //     body();
    //     if (!cond) {
    //         break;
    //     }
    // }
    _generate_indent(indent);
    _ctx->body.append("for (;;) {\n");
    auto old_flag = _ctx->in_generic_loop;
    _ctx->in_generic_loop = false;
    _generate_block(node->instruction->loop.body.get(), indent + 1u);
    _ctx->in_generic_loop = old_flag;
    _generate_indent(indent + 1u);
    _ctx->body.append(luisa::format(
        "if (!({})) {{ break; }}\n",
        _generate_node(node->instruction->loop.cond)));
    _generate_indent(indent);
    _ctx->body.append("}\n");
}

void CppSourceBuilder::_generate_instr_generic_loop(const ir::Node *node, uint indent) noexcept {
    // for (;;) {
    //     prepare;
    //     if (!cond) { break; }
    //     body;
    //     update;// continue goes here
    // }
    // =>
    // for (;;) {
    //     bool loop_break = false;
    //     prepare();
    //     if (!cond()) break;
    //     do {// body
    //         // break => { loop_break = true; break; }
    //         // continue => { break; }
    //     } while(false);
    //     if (loop_break) break;
    //     update();
    // }
    _generate_indent(indent);
    _ctx->body.append("for (;;) {\n");
    _generate_indent(indent + 1u);
    _ctx->body.append("bool loop_break = false;\n");
    auto old_flag = _ctx->in_generic_loop;
    // prepare
    _generate_block(node->instruction->generic_loop.prepare.get(), indent + 1u);
    // cond
    _generate_indent(indent + 1u);
    _ctx->body.append(luisa::format(
        "if (!({})) {{ break; }}\n",
        _generate_node(node->instruction->generic_loop.cond)));
    // body
    _ctx->in_generic_loop = true;
    _generate_indent(indent + 1u);
    _ctx->body.append("do {\n");
    _generate_block(node->instruction->generic_loop.body.get(), indent + 2u);
    _generate_indent(indent + 1u);
    _ctx->body.append("} while (false);\n");
    _ctx->in_generic_loop = old_flag;
    // break
    _generate_indent(indent + 1u);
    _ctx->body.append("if (loop_break) break;\n");
    // update
    _generate_block(node->instruction->generic_loop.update.get(), indent + 1u);
    // end
    _generate_indent(indent);
    _ctx->body.append("}\n");
}

void CppSourceBuilder::_generate_instr_break(const ir::Node *node, uint indent) noexcept {
    if (_ctx->in_generic_loop) {
        _generate_indent(indent);
        _ctx->body.append("loop_break = true;\n");
    }
    _generate_indent(indent);
    _ctx->body.append("break;\n");
}

void CppSourceBuilder::_generate_instr_continue(const ir::Node *node, uint indent) noexcept {
    _generate_indent(indent);
    if (_ctx->in_generic_loop) {
        _ctx->body.append("break;\n");
    } else {
        _ctx->body.append("continue;\n");
    }
}

void CppSourceBuilder::_generate_instr_if(const ir::Node *node, uint indent) noexcept {
    _generate_indent(indent);
    _ctx->body.append(luisa::format(
        "if ({}) {{\n",
        _generate_node(node->instruction->if_.cond)));
    _generate_block(node->instruction->if_.true_branch.get(), indent + 1u);
    _generate_indent(indent);
    _ctx->body.append("} else {\n");
    _generate_block(node->instruction->if_.false_branch.get(), indent + 1u);
    _generate_indent(indent);
    _ctx->body.append("}\n");
}

void CppSourceBuilder::_generate_instr_switch(const ir::Node *node, uint indent) noexcept {
    _generate_indent(indent);
    auto &&instr = node->instruction->switch_;
    _ctx->body.append(luisa::format(
        "switch ({}) {{\n", _generate_node(instr.value)));
    for (auto &&c : luisa::span{instr.cases.ptr, instr.cases.len}) {
        _generate_indent(indent + 1u);
        _ctx->body.append(luisa::format("case {}: {{\n", c.value));
        _generate_block(c.block.get(), indent + 2);
        _generate_indent(indent + 2u);
        _ctx->body.append("break;\n");
        _generate_indent(indent + 1u);
        _ctx->body.append("}\n");
    }
    if (auto d = instr.default_.get()) {
        _generate_indent(indent + 1u);
        _ctx->body.append("default: {\n");
        _generate_block(d, indent + 2u);
        _generate_indent(indent + 2u);
        _ctx->body.append("break;\n");
        _generate_indent(indent + 1u);
        _ctx->body.append("}\n");
    }
    _generate_indent(indent);
    _ctx->body.append("}\n");
}

void CppSourceBuilder::_generate_instr_ad_scope(const ir::Node *node, uint indent) noexcept {
    // forward
    _generate_indent(indent);
    _ctx->body.append("/* ADScope Forward Begin */\n");
    _generate_block(node->instruction->ad_scope.forward.get(), indent + 1u);
    _generate_indent(indent);
    _ctx->body.append("/* ADScope Forward End */\n");
    // backward
    _generate_indent(indent);
    _ctx->body.append("/* ADScope Backward Begin */\n");
    _generate_block(node->instruction->ad_scope.backward.get(), indent + 1u);
    _generate_indent(indent);
    _ctx->body.append("/* ADScope Backward End */\n");
    // epilogue
    _generate_indent(indent);
    _ctx->body.append("/* ADScope Epilogue Begin */\n");
    _generate_block(node->instruction->ad_scope.epilogue.get(), indent + 1u);
    _generate_indent(indent);
    _ctx->body.append("/* ADScope Epilogue End */\n");
}
void CppSourceBuilder::_generate_instr_ad_detach(const ir::Node *node, uint indent) noexcept {
    _generate_indent(indent);
    _ctx->body.append("/* AD Detach Begin */\n");
    _generate_block(node->instruction->ad_detach._0.get(), indent + 1u);
    _generate_indent(indent);
    _ctx->body.append("/* AD Detach End */\n");
}
void CppSourceBuilder::_generate_instr_comment(const ir::Node *node, uint indent) noexcept {
    auto s = node->instruction->comment._0;
    _generate_indent(indent);
    _ctx->body.append(luisa::format(
        "/* {} */\n",
        luisa::string_view{reinterpret_cast<const char *>(s.ptr), s.len}));
}

void CppSourceBuilder::_generate_instr_debug(const ir::Node *node, uint indent) noexcept {
    LUISA_WARNING_WITH_LOCATION("Instruction 'Debug' is not implemented.");
    _generate_indent(indent);
    auto s = node->instruction->debug._0;
    _ctx->body.append(luisa::format(
        "/* Debug: {} */\n",
        luisa::string_view{reinterpret_cast<const char *>(s.ptr), s.len}));
    //    auto s = node->instruction->debug._0;
    //    _generate_indent(indent);
    //    _ctx->body.append(luisa::format(
    //        "lc_debug(\"{}\");\n",
    //        luisa::string_view{reinterpret_cast<const char *>(s.ptr), s.len}));
}

}// namespace luisa::compute
