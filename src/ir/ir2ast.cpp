#pragma clang diagnostic push
#pragma ide diagnostic ignored "*-no-recursion"

#include <luisa/ast/op.h>
#include <luisa/ast/variable.h>
#include <luisa/core/logging.h>
#include <luisa/core/magic_enum.h>
#include <luisa/runtime/rtx/ray.h>
#include <luisa/runtime/rtx/hit.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/rust/ir.hpp>
#include <luisa/ir/ir2ast.h>
#include <luisa/ir/ir.h>

namespace luisa::compute {

void IR2AST::_convert_block(const ir::BasicBlock *block) noexcept {
    auto node_ref = block->first;
    while (node_ref != ir::INVALID_REF) {
        auto node = ir::luisa_compute_ir_node_get(node_ref);
        switch (node->instruction->tag) {
            case ir::Instruction::Tag::Local: _convert_instr_local(node); break;
            case ir::Instruction::Tag::UserData: _convert_instr_user_data(node); break;
            case ir::Instruction::Tag::Invalid: _convert_instr_invalid(node); break;
            case ir::Instruction::Tag::Const: _convert_instr_const(node); break;
            case ir::Instruction::Tag::Update: _convert_instr_update(node); break;
            case ir::Instruction::Tag::Call: {
                if (node->type_->tag == ir::Type::Tag::Void) {
                    _convert_instr_call(node);
                } else {
                    auto expr = _convert_node(node);
                }
                break;
            }
            case ir::Instruction::Tag::Phi: _convert_instr_phi(node); break;
            case ir::Instruction::Tag::Return: _convert_instr_return(node); break;
            case ir::Instruction::Tag::Loop: _convert_instr_loop(node); break;
            case ir::Instruction::Tag::GenericLoop: _convert_instr_generic_loop(node); break;
            case ir::Instruction::Tag::Break: _convert_instr_break(node); break;
            case ir::Instruction::Tag::Continue: _convert_instr_continue(node); break;
            case ir::Instruction::Tag::If: _convert_instr_if(node); break;
            case ir::Instruction::Tag::Switch: _convert_instr_switch(node); break;
            case ir::Instruction::Tag::AdScope: _convert_instr_ad_scope(node); break;
            case ir::Instruction::Tag::AdDetach: _convert_instr_ad_detach(node); break;
            case ir::Instruction::Tag::RayQuery: _convert_instr_ray_query(node); break;
            case ir::Instruction::Tag::Comment: _convert_instr_comment(node); break;
            case ir::Instruction::Tag::Print: LUISA_NOT_IMPLEMENTED();// TODO: print
            case ir::Instruction::Tag::Buffer: LUISA_ERROR_WITH_LOCATION("Unexpected instruction 'Buffer' in function body."); break;
            case ir::Instruction::Tag::Bindless: LUISA_ERROR_WITH_LOCATION("Unexpected instruction 'Bindless' in function body."); break;
            case ir::Instruction::Tag::Texture2D: LUISA_ERROR_WITH_LOCATION("Unexpected instruction 'Texture2D' in function body."); break;
            case ir::Instruction::Tag::Texture3D: LUISA_ERROR_WITH_LOCATION("Unexpected instruction 'Texture3D' in function body."); break;
            case ir::Instruction::Tag::Accel: LUISA_ERROR_WITH_LOCATION("Unexpected instruction 'Accel' in function body."); break;
            case ir::Instruction::Tag::Shared: LUISA_ERROR_WITH_LOCATION("Unexpected instruction 'Shared' in function body."); break;
            case ir::Instruction::Tag::Uniform: LUISA_ERROR_WITH_LOCATION("Unexpected instruction 'Uniform' in function body."); break;
            case ir::Instruction::Tag::Argument: LUISA_ERROR_WITH_LOCATION("Unexpected instruction 'Argument' in function body."); break;
        }
        node_ref = node->next;
    }
    if (auto iter = _ctx->block_to_phis.find(block);
        iter != _ctx->block_to_phis.end()) {
        for (auto phi : iter->second) {
            _ctx->function_builder->comment_("phi node assignment");
            _ctx->function_builder->assign(_convert_node(phi.dst), _convert_node(phi.src));
        }
    }
}

const Expression *IR2AST::_convert_node(ir::NodeRef node_ref) noexcept {
    return _convert_node(ir::luisa_compute_ir_node_get(node_ref));
}

const Expression *IR2AST::_convert_node(const ir::Node *node) noexcept {
    if (auto iter = _ctx->node_to_exprs.find(node); iter != _ctx->node_to_exprs.end()) {
        return iter->second;
    }
    auto type = _convert_type(node->type_.get());

    auto expr = [&, index = _ctx->node_to_exprs.size()]() -> const Expression * {
        switch (node->instruction->tag) {
            case ir::Instruction::Tag::Buffer: return _ctx->function_builder->buffer(Type::buffer(type));
            case ir::Instruction::Tag::Bindless: return _ctx->function_builder->bindless_array();
            case ir::Instruction::Tag::Texture2D: [[fallthrough]];
            case ir::Instruction::Tag::Texture3D: {
                // for Texture{2|3}D, type is vector<primitive,4>
                // where primitive could be int, float or uint
                auto dimension = node->instruction->tag == ir::Instruction::Tag::Texture2D ? 2u : 3u;
                auto texture_type = Type::texture(type, dimension);
                return _ctx->function_builder->texture(texture_type);
            }
            case ir::Instruction::Tag::Accel: return _ctx->function_builder->accel();
            case ir::Instruction::Tag::Shared: return _ctx->function_builder->shared(type);
            case ir::Instruction::Tag::UserData: return _ctx->function_builder->literal(Type::of<float>(), 0.0f);
            case ir::Instruction::Tag::Const: return _convert_constant(node->instruction->const_._0);
            case ir::Instruction::Tag::Call: {
                auto ret = _convert_instr_call(node);
                if (node->instruction->call._0.tag == ir::Func::Tag::GetElementPtr) {
                    // we should not make a local copy for GEP, as
                    // it might appear in the LHS in assignment.
                    return ret;
                }
                if (ret != nullptr) {
                    LUISA_ASSERT(ret->type() == type,
                                 "Type mismatch: expected {}, got {} (op = {}).",
                                 type->description(), ret->type()->description(),
                                 to_underlying(node->instruction->call._0.tag));
                    auto local = _ctx->function_builder->local(type);
                    _ctx->function_builder->assign(local, ret);
                    _ctx->node_to_exprs.emplace(node, local);
                    ret = local;
                }
                return ret;
            }
            case ir::Instruction::Tag::Phi: {
                auto local = _ctx->function_builder->local(type);
                _ctx->node_to_exprs.emplace(node, local);
                return local;
            }
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid node type: {}.", to_string(node->instruction->tag));
    }();
    return expr;
}

void IR2AST::_convert_instr_local(const ir::Node *node) noexcept {
    auto init = _convert_node(node->instruction->local.init);
    auto iter = _ctx->node_to_exprs.find(node);
    LUISA_ASSERT(iter != _ctx->node_to_exprs.end(),
                 "Local variable not found in node_to_exprs.");
    auto expr = iter->second;

    // assign the init value to the variable
    _ctx->function_builder->assign(expr, init);
}

void IR2AST::_convert_instr_user_data(const ir::Node *_user_data) noexcept {
}

void IR2AST::_convert_instr_invalid(const ir::Node *invalid) noexcept {
    // do nothing
}

void IR2AST::_convert_instr_const(const ir::Node *const_) noexcept {
    // do nothing
}

void IR2AST::_convert_instr_update(const ir::Node *node) noexcept {
    auto lhs = _convert_node(node->instruction->update.var);
    auto rhs = _convert_node(node->instruction->update.value);
    _ctx->function_builder->assign(lhs, rhs);
}

namespace detail {

[[nodiscard]] inline const Expression *
ir2ast_convert_ray_ast_type_2_ir_type(FunctionBuilder *b, const Expression *expr) noexcept {
    auto ft = Type::of<float>();
    auto vt = Type::structure(4u, ft, ft, ft);
    auto rt = Type::structure(16, vt, ft, vt, ft);
    // if the types are the same (i.e. Ray), no need to convert
    if (expr->type() == rt) { return expr; }
    LUISA_ASSERT(expr->type() == Type::of<Ray>(),
                 "Invalid ray type: {}.",
                 expr->type()->description());
    // if the ray is not a local variable, make a local copy first
    if (expr->tag() != Expression::Tag::REF) {
        auto ref = b->local(expr->type());
        b->assign(ref, expr);
        expr = ref;
    }
    auto ut = Type::of<uint>();
    auto at = Type::array(ft, 3u);

    auto u0 = b->literal(ut, 0u);
    auto u1 = b->literal(ut, 1u);
    auto u2 = b->literal(ut, 2u);
    auto u3 = b->literal(ut, 3u);
    // @Mike-Leo-Smith: check pls
    // decompose ray
    auto o = b->member(at, expr, 0u);
    auto ox = b->access(ft, o, u0);
    auto oy = b->access(ft, o, u1);
    auto oz = b->access(ft, o, u2);
    auto tmin = b->member(ft, expr, 1u);
    auto d = b->member(at, expr, 2u);
    auto dx = b->access(ft, d, u0);
    auto dy = b->access(ft, d, u1);
    auto dz = b->access(ft, d, u2);
    auto tmax = b->member(ft, expr, 3u);
    auto ray = b->local(rt);
    auto o_ = b->member(vt, ray, 0u);
    auto d_ = b->member(vt, ray, 2u);

    b->assign(b->member(ft, o_, 0), ox);
    b->assign(b->member(ft, o_, 1), oy);
    b->assign(b->member(ft, o_, 2), oz);
    b->assign(b->member(ft, ray, 1), tmin);
    b->assign(b->member(ft, d_, 0), dx);
    b->assign(b->member(ft, d_, 1), dy);
    b->assign(b->member(ft, d_, 2), dz);
    b->assign(b->member(ft, ray, 3), tmax);
    return ray;
}
[[nodiscard]] inline const Expression *
ir2ast_convert_ray_ir_type_2_ast_type(FunctionBuilder *b, const Expression *expr) noexcept {
    // if the types are the same (i.e. Ray), no need to convert
    if (expr->type() == Type::of<Ray>()) { return expr; }
    auto ft = Type::of<float>();
    auto vt = Type::structure(4u, ft, ft, ft);
    auto rt = Type::structure(16, vt, ft, vt, ft);
    LUISA_ASSERT(expr->type() == rt,
                 "Invalid ray type: {}.",
                 expr->type()->description());
    // if the ray is not a local variable, make a local copy first
    if (expr->tag() != Expression::Tag::REF) {
        auto ref = b->local(expr->type());
        b->assign(ref, expr);
        expr = ref;
    }
    // decompose the ray
    auto o = b->member(vt, expr, 0u);
    auto ox = b->member(ft, o, 0u);
    auto oy = b->member(ft, o, 1u);
    auto oz = b->member(ft, o, 2u);
    auto tmin = b->member(ft, expr, 1u);
    auto d = b->member(vt, expr, 2u);
    auto dx = b->member(ft, d, 0u);
    auto dy = b->member(ft, d, 1u);
    auto dz = b->member(ft, d, 2u);
    auto tmax = b->member(ft, expr, 3u);
    auto at = Type::array(ft, 3u);
    auto ray = b->local(Type::of<Ray>());
    auto o_ = b->member(at, ray, 0u);
    auto d_ = b->member(at, ray, 2u);
    auto ut = Type::of<uint>();
    auto u0 = b->literal(ut, 0u);
    auto u1 = b->literal(ut, 1u);
    auto u2 = b->literal(ut, 2u);
    b->assign(b->access(ft, o_, u0), ox);
    b->assign(b->access(ft, o_, u1), oy);
    b->assign(b->access(ft, o_, u2), oz);
    b->assign(b->member(ft, ray, 1u), tmin);
    b->assign(b->access(ft, d_, u0), dx);
    b->assign(b->access(ft, d_, u1), dy);
    b->assign(b->access(ft, d_, u2), dz);
    b->assign(b->member(ft, ray, 3u), tmax);
    return ray;
}

[[nodiscard]] inline const Expression *
ir2ast_convert_triangle_hit(FunctionBuilder *b, const Type *dst_ht, const Expression *expr) noexcept {
    LUISA_ASSERT(expr->type() == Type::of<TriangleHit>(),
                 "Invalid triangle hit type: {}.",
                 expr->type()->description());
    if (dst_ht == Type::of<TriangleHit>()) { return expr; }
    auto ft = Type::of<float>();
    auto ut = Type::of<uint>();
    auto vt = Type::of<float2>();
    auto ht = Type::structure(8u, ut, ut, ft, ft, ft);
    LUISA_ASSERT(dst_ht == ht,
                 "Invalid triangle hit type: {}.",
                 expr->type()->description());
    if (expr->tag() != Expression::Tag::REF) {
        auto ref = b->local(expr->type());
        b->assign(ref, expr);
        expr = ref;
    }
    auto bary = b->member(vt, expr, 2u);
    auto bary_x = b->access(ft, bary, b->literal(ut, 0u));
    auto bary_y = b->access(ft, bary, b->literal(ut, 1u));
    auto hit = b->local(dst_ht);
    b->assign(b->member(ut, hit, 0u), b->member(ut, expr, 0u));// inst
    b->assign(b->member(ut, hit, 1u), b->member(ut, expr, 1u));// prim
    b->assign(b->member(ft, hit, 2u), bary_x);                 // bary_x
    b->assign(b->member(ft, hit, 3u), bary_y);                 // bary_y
    b->assign(b->member(ft, hit, 4u), b->member(ft, expr, 3u));// ray_t
    return hit;
}

[[nodiscard]] inline const Expression *
ir2ast_convert_committed_hit(FunctionBuilder *b, const Type *dst_ht, const Expression *expr) noexcept {
    LUISA_ASSERT(expr->type() == Type::of<CommittedHit>(),
                 "Invalid committed hit type: {}.",
                 expr->type()->description());
    if (dst_ht == Type::of<CommittedHit>()) { return expr; }
    auto ft = Type::of<float>();
    auto ut = Type::of<uint>();
    auto vt = Type::of<float2>();
    auto ht = Type::structure(8u, ut, ut, ft, ft, ut, ft);
    LUISA_ASSERT(dst_ht == ht,
                 "Invalid committed hit type: {}.",
                 expr->type()->description());
    if (expr->tag() != Expression::Tag::REF) {
        auto ref = b->local(expr->type());
        b->assign(ref, expr);
        expr = ref;
    }
    auto bary = b->member(ft, expr, 2u);
    auto bary_x = b->access(ft, bary, b->literal(ut, 0u));
    auto bary_y = b->access(ft, bary, b->literal(ut, 1u));
    auto hit = b->local(dst_ht);
    b->assign(b->member(ut, hit, 0u), b->member(ut, expr, 0u));// inst
    b->assign(b->member(ut, hit, 1u), b->member(ut, expr, 1u));// prim
    b->assign(b->member(ft, hit, 2u), bary_x);                 // bary_x
    b->assign(b->member(ft, hit, 3u), bary_y);                 // bary_y
    b->assign(b->member(ut, hit, 4u), b->member(ut, expr, 3u));// hit_type
    b->assign(b->member(ft, hit, 5u), b->member(ft, expr, 4u));// ray_t
    return hit;
}

}// namespace detail

const Expression *IR2AST::_convert_instr_call(const ir::Node *node) noexcept {
    auto type = _convert_type(node->type_.get());
    auto &&[func, arg_slice] = node->instruction->call;
    auto args = luisa::span{arg_slice.ptr, arg_slice.len};
    auto function_name = to_string(func.tag);
    auto builtin_func = [&](size_t arg_num, CallOp call_op) -> const Expression * {
        auto argument_information = arg_num == 0 ?
                                        "no arguments" :
                                    arg_num == 1 ?
                                        "1 argument" :
                                        luisa::format("{} arguments", arg_num);
        LUISA_ASSERT(args.size() == arg_num, "`{}` takes {}, got {}.",
                     function_name, argument_information, args.size());
        auto converted_args = luisa::vector<const Expression *>{};
        for (const auto &arg : args) {
            converted_args.push_back(_convert_node(arg));
        }
        if (call_op == CallOp::RAY_TRACING_TRACE_CLOSEST ||
            call_op == CallOp::RAY_TRACING_TRACE_ANY ||
            call_op == CallOp::RAY_TRACING_QUERY_ALL ||
            call_op == CallOp::RAY_TRACING_QUERY_ANY) {
            // converted_args[1] = detail::ir2ast_convert_ray_ir_type_2_ast_type(
            //     _ctx->function_builder.get(),
            //     converted_args[1]);
        }
        if (call_op == CallOp::RAY_TRACING_QUERY_ANY) {
            auto type = Type::of<RayQueryAny>();
            auto local = _ctx->function_builder->local(type);
            auto call = _ctx->function_builder->call(type, call_op, converted_args);
            _ctx->function_builder->assign(local, call);
            return local;
        }
        if (call_op == CallOp::RAY_TRACING_QUERY_ALL) {
            auto type = Type::of<RayQueryAll>();
            auto local = _ctx->function_builder->local(type);
            auto call = _ctx->function_builder->call(type, call_op, converted_args);
            _ctx->function_builder->assign(local, call);
            return local;
        }
        if (call_op == CallOp::RAY_QUERY_WORLD_SPACE_RAY) {
            auto ray = _ctx->function_builder->call(Type::of<Ray>(), call_op, converted_args);
            return ray;//detail::ir2ast_convert_ray_ast_type_2_ir_type(_ctx->function_builder.get(), ray);
        }
        if (type == nullptr) {
            _ctx->function_builder->call(
                call_op, luisa::span{converted_args});
            return nullptr;
        } else {
            if (call_op == CallOp::RAY_TRACING_TRACE_CLOSEST) {
                auto ret = _ctx->function_builder->call(
                    Type::of<TriangleHit>(), call_op, converted_args);
                return detail::ir2ast_convert_triangle_hit(
                    _ctx->function_builder.get(), type, ret);
            }
            if (call_op == CallOp::RAY_QUERY_COMMITTED_HIT) {
                auto ret = _ctx->function_builder->call(
                    Type::of<CommittedHit>(), call_op, converted_args);
                return detail::ir2ast_convert_committed_hit(
                    _ctx->function_builder.get(), type, ret);
            }
            auto ret = _ctx->function_builder->call(
                type, call_op, luisa::span{converted_args});
            return ret;
        }
    };
    auto unary_op = [&](UnaryOp un_op) -> const Expression * {
        LUISA_ASSERT(args.size() == 1u, "`{}` takes 1 argument, got {}.", function_name, args.size());
        return _ctx->function_builder->unary(type, un_op, _convert_node(args[0]));
    };
    auto binary_op = [&](BinaryOp bin_op) -> const Expression * {
        LUISA_ASSERT(args.size() == 2u, "`{}` takes 2 arguments, got {}.", function_name, args.size());
        return _ctx->function_builder->binary(type, bin_op, _convert_node(args[0]), _convert_node(args[1]));
    };
    auto make_vector = [&](size_t length) -> const Expression * {
        LUISA_ASSERT(args.size() == length, "`MakeVec` takes {} argument(s), got {}.", length, args.size());
        auto inner_type = ir::luisa_compute_ir_node_get(args[0])->type_.get();
        LUISA_ASSERT(inner_type->tag == ir::Type::Tag::Primitive, "`MakeVec` supports primitive type only, got {}.", to_string(inner_type->tag));
        LUISA_ASSERT(type->is_vector(), "`MakeVec` must return a vector, got {}.", type->description());

        auto converted_args = luisa::vector<const Expression *>{};
        for (const auto &arg : args) {
            converted_args.push_back(_convert_node(arg));
        }
        auto vector_op = _decide_make_vector_op(_convert_primitive_type(inner_type->primitive._0), type->dimension());
        return _ctx->function_builder->call(type, vector_op, luisa::span{converted_args});
    };
    auto rotate = [&](BinaryOp this_op) -> const Expression * {
        LUISA_ASSERT(this_op == BinaryOp::SHL || this_op == BinaryOp::SHR, "rotate is only valid with SHL and SHR.");
        LUISA_ASSERT(args.size() == 2u, "{} takes 2 arguments, got {}.", function_name, args.size());
        auto lhs = _convert_node(args[0]);
        auto rhs = _convert_node(args[1]);
        auto lhs_bit_length = (uint)lhs->type()->size() * 8;
        // this_op == SHL -> complement_op == SHR, vice versa
        auto complement_op = this_op == BinaryOp::SHL ? BinaryOp::SHR : BinaryOp::SHL;

        // rot_left(a, b) = (a << b) | (a >> (bit_length(a) - b))
        // rot_right(a, b) = (a >> b) | (a << (bit_length(a) - b))
        auto part1 = _ctx->function_builder->binary(
            lhs->type(),
            this_op,
            lhs,
            rhs);
        // (a << b) for rot_left, (a >> b) for rot_right
        auto shl_length = _ctx->function_builder->binary(
            rhs->type(),
            BinaryOp::SUB,
            _ctx->function_builder->literal(rhs->type(), lhs_bit_length),
            rhs);// bit_length(a) - b
        auto part2 = _ctx->function_builder->binary(
            lhs->type(),
            complement_op,
            lhs,
            shl_length);// (a >> (bit_length(a) - b)) for rot_left, (a << (bit_length(a) - b)) for rot_right
        return _ctx->function_builder->binary(lhs->type(), BinaryOp::BIT_OR, part1, part2);
    };
    auto make_matrix = [&](size_t dimension) -> const Expression * {
        LUISA_ASSERT(args.size() == dimension, "`Mat` takes {} argument(s), got {}.", dimension, args.size());
        LUISA_ASSERT(type->is_matrix(), "`Mat{}` must return a matrix, got {}.", dimension, type->description());
        auto matrix_dimension = type->dimension();
        auto converted_args = luisa::vector<const Expression *>{};
        for (const auto &arg : args) {
            converted_args.push_back(_convert_node(arg));
        }
        auto matrix_op = _decide_make_matrix_op(matrix_dimension);
        if (dimension == 1u) {// Mat is a broadcasting operation in IR
            auto col_type = Type::vector(type->element(), type->dimension());
            auto vector_op = _decide_make_vector_op(type->element(), matrix_dimension);
            auto col = _ctx->function_builder->call(col_type, vector_op, converted_args);
            converted_args.clear();
            converted_args.reserve(matrix_dimension);
            for (auto i = 0u; i < matrix_dimension; i++) {
                converted_args.emplace_back(col);
            }
        }
        return _ctx->function_builder->call(type, matrix_op, luisa::span{converted_args});
    };

    auto constant_index = [](ir::NodeRef index) noexcept -> uint64_t {
        auto node = ir::luisa_compute_ir_node_get(index);
        LUISA_ASSERT(node->instruction->tag == ir::Instruction::Tag::Const,
                     "Index must be constant uint32.");
        auto &&c = node->instruction->const_._0;
        switch (c.tag) {
            case ir::Const::Tag::Zero: return 0;
            case ir::Const::Tag::One: return 1;
            case ir::Const::Tag::Int16: return c.int16._0;
            case ir::Const::Tag::Uint16: return c.uint16._0;
            case ir::Const::Tag::Int32: return c.int32._0;
            case ir::Const::Tag::Uint32: return c.uint32._0;
            case ir::Const::Tag::Int64: return c.int64._0;
            case ir::Const::Tag::Uint64: return c.uint64._0;
            case ir::Const::Tag::Generic: {
                auto t = node->type_.get();
                LUISA_ASSERT(t->tag == ir::Type::Tag::Primitive, "Invalid index type: {}.", to_string(t->tag));
                auto do_cast = [&c]<typename T>() noexcept {
                    T x{};
                    std::memcpy(&x, c.generic._0.ptr, sizeof(T));
                    return static_cast<uint64_t>(x);
                };
                switch (t->primitive._0) {
                    case ir::Primitive::Int16: return do_cast.operator()<int16_t>();
                    case ir::Primitive::Uint16: return do_cast.operator()<uint16_t>();
                    case ir::Primitive::Int32: return do_cast.operator()<int32_t>();
                    case ir::Primitive::Uint32: return do_cast.operator()<uint32_t>();
                    case ir::Primitive::Int64: return do_cast.operator()<int64_t>();
                    case ir::Primitive::Uint64: return do_cast.operator()<uint64_t>();
                    default: break;
                }
            }
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid index.");
    };
    switch (func.tag) {
        case ir::Func::Tag::Pack: return builtin_func(3, CallOp::PACK);
        case ir::Func::Tag::Unpack: return builtin_func(2, CallOp::UNPACK);
        case ir::Func::Tag::ZeroInitializer: return builtin_func(0, CallOp::ZERO);
        case ir::Func::Tag::Assume: return builtin_func(1, CallOp::ASSUME);
        case ir::Func::Tag::Unreachable: return builtin_func(0, CallOp::UNREACHABLE);
        case ir::Func::Tag::Assert: return builtin_func(1, CallOp::ASSERT);
        case ir::Func::Tag::ThreadId: {
            LUISA_ASSERT(args.empty(), "`ThreadId` takes no arguments.");
            return _ctx->function_builder->thread_id();
        }
        case ir::Func::Tag::BlockId: {
            LUISA_ASSERT(args.empty(), "`BlockId` takes no arguments.");
            return _ctx->function_builder->block_id();
        }
        case ir::Func::Tag::DispatchId: {
            LUISA_ASSERT(args.empty(), "`DispatchId` takes no arguments.");
            return _ctx->function_builder->dispatch_id();
        }
        case ir::Func::Tag::DispatchSize: {
            LUISA_ASSERT(args.empty(), "`DispatchSize` takes no arguments.");
            return _ctx->function_builder->dispatch_size();
        }
        case ir::Func::Tag::WarpSize: {
            LUISA_ASSERT(args.empty(), "`WarpSize` takes no arguments.");
            return _ctx->function_builder->warp_lane_count();
        }
        case ir::Func::Tag::WarpLaneId: {
            LUISA_ASSERT(args.empty(), "`WarpLaneId` takes no arguments.");
            return _ctx->function_builder->warp_lane_id();
        }
        case ir::Func::Tag::RequiresGradient: return builtin_func(1, CallOp::REQUIRES_GRADIENT);
        case ir::Func::Tag::Gradient: return builtin_func(1, CallOp::GRADIENT);
        case ir::Func::Tag::GradientMarker: return builtin_func(2, CallOp::GRADIENT_MARKER);
        case ir::Func::Tag::AccGrad: return builtin_func(2, CallOp::ACCUMULATE_GRADIENT);
        case ir::Func::Tag::Detach: return builtin_func(1, CallOp::DETACH);
        case ir::Func::Tag::Backward: return builtin_func(1, CallOp::BACKWARD);
        case ir::Func::Tag::PropagateGrad: LUISA_NOT_IMPLEMENTED();// TODO
        case ir::Func::Tag::OutputGrad: LUISA_NOT_IMPLEMENTED();   // TODO
        case ir::Func::Tag::RayTracingInstanceTransform: return builtin_func(2, CallOp::RAY_TRACING_INSTANCE_TRANSFORM);
        case ir::Func::Tag::RayTracingInstanceUserId: return builtin_func(2, CallOp::RAY_TRACING_INSTANCE_USER_ID);
        case ir::Func::Tag::RayTracingSetInstanceTransform: return builtin_func(3, CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM);
        case ir::Func::Tag::RayTracingSetInstanceVisibility: return builtin_func(3, CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY);
        case ir::Func::Tag::RayTracingSetInstanceOpacity: return builtin_func(3, CallOp::RAY_TRACING_SET_INSTANCE_OPACITY);
        case ir::Func::Tag::RayTracingSetInstanceUserId: return builtin_func(3, CallOp::RAY_TRACING_SET_INSTANCE_USER_ID);
        case ir::Func::Tag::RayTracingTraceClosest: return builtin_func(3, CallOp::RAY_TRACING_TRACE_CLOSEST);
        case ir::Func::Tag::RayTracingTraceAny: return builtin_func(3, CallOp::RAY_TRACING_TRACE_ANY);
        case ir::Func::Tag::RayTracingQueryAll: return builtin_func(3, CallOp::RAY_TRACING_QUERY_ALL);
        case ir::Func::Tag::RayTracingQueryAny: return builtin_func(3, CallOp::RAY_TRACING_QUERY_ANY);
        case ir::Func::Tag::RayQueryProceduralCandidateHit: return builtin_func(1, CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT);
        case ir::Func::Tag::RayQueryTriangleCandidateHit: return builtin_func(1, CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT);
        case ir::Func::Tag::RayQueryCommittedHit: return builtin_func(1, CallOp::RAY_QUERY_COMMITTED_HIT);
        case ir::Func::Tag::RayQueryCommitTriangle: return builtin_func(1, CallOp::RAY_QUERY_COMMIT_TRIANGLE);
        case ir::Func::Tag::RayQueryCommitProcedural: return builtin_func(2, CallOp::RAY_QUERY_COMMIT_PROCEDURAL);
        case ir::Func::Tag::RayQueryTerminate: return builtin_func(1, CallOp::RAY_QUERY_TERMINATE);
        case ir::Func::Tag::RayQueryWorldSpaceRay: return builtin_func(1, CallOp::RAY_QUERY_WORLD_SPACE_RAY);
        case ir::Func::Tag::RasterDiscard: return builtin_func(0, CallOp::RASTER_DISCARD);
        // case ir::Func::Tag::IndirectClearDispatchBuffer: return builtin_func(1, CallOp::INDIRECT_CLEAR_DISPATCH_BUFFER);
        // case ir::Func::Tag::IndirectEmplaceDispatchKernel: return builtin_func(4, CallOp::INDIRECT_EMPLACE_DISPATCH_KERNEL);
        case ir::Func::Tag::Load: {
            LUISA_ASSERT(args.size() == 1u, "`Load` takes 1 argument.");
            return _convert_node(args[0]);
        }
        case ir::Func::Tag::Cast: {
            LUISA_ASSERT(args.size() == 1u, "Cast takes 1 argument.");
            switch (type->tag()) {
                case Type::Tag::VECTOR: {
                    auto primitive = type->element();
                    auto length = type->dimension();
                    auto vector_op = _decide_make_vector_op(primitive, length);
                    return _ctx->function_builder->call(type, vector_op, {_convert_node(args[0])});
                }
                default: return _ctx->function_builder->cast(type, CastOp::STATIC, _convert_node(args[0]));
            }
        }
        case ir::Func::Tag::Bitcast: {
            LUISA_ASSERT(args.size() == 1u, "BitCast takes 1 argument.");
            return _ctx->function_builder->cast(type, CastOp::BITWISE, _convert_node(args[0]));
        }
        case ir::Func::Tag::Add: return binary_op(BinaryOp::ADD);
        case ir::Func::Tag::Sub: return binary_op(BinaryOp::SUB);
        case ir::Func::Tag::Mul: return binary_op(BinaryOp::MUL);
        case ir::Func::Tag::Div: return binary_op(BinaryOp::DIV);
        case ir::Func::Tag::Rem: {
            if (type->is_float32() || type->is_float32_vector()) {
                // implemented as x - y * trunc(x / y)
                auto x = _convert_node(args[0]);
                auto y = _convert_node(args[1]);
                auto div = _ctx->function_builder->binary(type, BinaryOp::DIV, x, y);
                auto trunc = _ctx->function_builder->call(type, CallOp::TRUNC, {div});
                auto mul = _ctx->function_builder->binary(type, BinaryOp::MUL, y, trunc);
                return _ctx->function_builder->binary(type, BinaryOp::SUB, x, mul);
            }
            return binary_op(BinaryOp::MOD);
        }
        case ir::Func::Tag::BitAnd: return binary_op(BinaryOp::BIT_AND);
        case ir::Func::Tag::BitOr: return binary_op(BinaryOp::BIT_OR);
        case ir::Func::Tag::BitXor: return binary_op(BinaryOp::BIT_XOR);
        case ir::Func::Tag::Shl: return binary_op(BinaryOp::SHL);
        case ir::Func::Tag::Shr: return binary_op(BinaryOp::SHR);
        case ir::Func::Tag::RotRight: return rotate(BinaryOp::SHR);
        case ir::Func::Tag::RotLeft: return rotate(BinaryOp::SHL);
        case ir::Func::Tag::Eq: return binary_op(BinaryOp::EQUAL);
        case ir::Func::Tag::Ne: return binary_op(BinaryOp::NOT_EQUAL);
        case ir::Func::Tag::Lt: return binary_op(BinaryOp::LESS);
        case ir::Func::Tag::Le: return binary_op(BinaryOp::LESS_EQUAL);
        case ir::Func::Tag::Gt: return binary_op(BinaryOp::GREATER);
        case ir::Func::Tag::Ge: return binary_op(BinaryOp::GREATER_EQUAL);
        case ir::Func::Tag::MatCompMul: return builtin_func(2, CallOp::MATRIX_COMPONENT_WISE_MULTIPLICATION);
        case ir::Func::Tag::Neg: return unary_op(UnaryOp::MINUS);
        case ir::Func::Tag::Not: return unary_op(UnaryOp::NOT);
        case ir::Func::Tag::BitNot: {
            if (type->is_bool() || (type->is_vector() && type->element()->is_bool())) {
                return unary_op(UnaryOp::NOT);
            } else {
                return unary_op(UnaryOp::BIT_NOT);
            }
        };
        case ir::Func::Tag::All: return builtin_func(1, CallOp::ALL);
        case ir::Func::Tag::Any: return builtin_func(1, CallOp::ANY);
        case ir::Func::Tag::Select: {
            LUISA_ASSERT(args.size() == 3u, "Select takes 3 arguments.");
            // In IR the argument order is (condition, value_true, value_false)
            // However in AST it is (value_false, value_true, condition)
            return _ctx->function_builder->call(
                type, CallOp::SELECT,
                {
                    _convert_node(args[2]),
                    _convert_node(args[1]),
                    _convert_node(args[0]),
                });
        }
        case ir::Func::Tag::Clamp: return builtin_func(3, CallOp::CLAMP);
        case ir::Func::Tag::Saturate: return builtin_func(1, CallOp::SATURATE);
        case ir::Func::Tag::Lerp: return builtin_func(3, CallOp::LERP);
        case ir::Func::Tag::SmoothStep: return builtin_func(3, CallOp::SMOOTHSTEP);
        case ir::Func::Tag::Step: return builtin_func(2, CallOp::STEP);
        case ir::Func::Tag::Abs: return builtin_func(1, CallOp::ABS);
        case ir::Func::Tag::Min: return builtin_func(2, CallOp::MIN);
        case ir::Func::Tag::Max: return builtin_func(2, CallOp::MAX);
        case ir::Func::Tag::ReduceSum: return builtin_func(1, CallOp::REDUCE_SUM);
        case ir::Func::Tag::ReduceProd: return builtin_func(1, CallOp::REDUCE_PRODUCT);
        case ir::Func::Tag::ReduceMin: return builtin_func(1, CallOp::REDUCE_MIN);
        case ir::Func::Tag::ReduceMax: return builtin_func(1, CallOp::REDUCE_MAX);
        case ir::Func::Tag::Clz: return builtin_func(1, CallOp::CLZ);
        case ir::Func::Tag::Ctz: return builtin_func(1, CallOp::CTZ);
        case ir::Func::Tag::PopCount: return builtin_func(1, CallOp::POPCOUNT);
        case ir::Func::Tag::Reverse: return builtin_func(1, CallOp::REVERSE);
        case ir::Func::Tag::IsInf: return builtin_func(1, CallOp::ISINF);
        case ir::Func::Tag::IsNan: return builtin_func(1, CallOp::ISNAN);
        case ir::Func::Tag::Acos: return builtin_func(1, CallOp::ACOS);
        case ir::Func::Tag::Acosh: return builtin_func(1, CallOp::ACOSH);
        case ir::Func::Tag::Asin: return builtin_func(1, CallOp::ASIN);
        case ir::Func::Tag::Asinh: return builtin_func(1, CallOp::ASINH);
        case ir::Func::Tag::Atan: return builtin_func(1, CallOp::ATAN);
        case ir::Func::Tag::Atan2: return builtin_func(2, CallOp::ATAN2);
        case ir::Func::Tag::Atanh: return builtin_func(1, CallOp::ATANH);
        case ir::Func::Tag::Cos: return builtin_func(1, CallOp::COS);
        case ir::Func::Tag::Cosh: return builtin_func(1, CallOp::COSH);
        case ir::Func::Tag::Sin: return builtin_func(1, CallOp::SIN);
        case ir::Func::Tag::Sinh: return builtin_func(1, CallOp::SINH);
        case ir::Func::Tag::Tan: return builtin_func(1, CallOp::TAN);
        case ir::Func::Tag::Tanh: return builtin_func(1, CallOp::TANH);
        case ir::Func::Tag::Exp: return builtin_func(1, CallOp::EXP);
        case ir::Func::Tag::Exp2: return builtin_func(1, CallOp::EXP2);
        case ir::Func::Tag::Exp10: return builtin_func(1, CallOp::EXP10);
        case ir::Func::Tag::Log: return builtin_func(1, CallOp::LOG);
        case ir::Func::Tag::Log2: return builtin_func(1, CallOp::LOG2);
        case ir::Func::Tag::Log10: return builtin_func(1, CallOp::LOG10);
        case ir::Func::Tag::Powi: return builtin_func(2, CallOp::POW);
        case ir::Func::Tag::Powf: return builtin_func(2, CallOp::POW);
        case ir::Func::Tag::Sqrt: return builtin_func(1, CallOp::SQRT);
        case ir::Func::Tag::Rsqrt: return builtin_func(1, CallOp::RSQRT);
        case ir::Func::Tag::Ceil: return builtin_func(1, CallOp::CEIL);
        case ir::Func::Tag::Floor: return builtin_func(1, CallOp::FLOOR);
        case ir::Func::Tag::Fract: return builtin_func(1, CallOp::FRACT);
        case ir::Func::Tag::Trunc: return builtin_func(1, CallOp::TRUNC);
        case ir::Func::Tag::Round: return builtin_func(1, CallOp::ROUND);
        case ir::Func::Tag::Fma: return builtin_func(3, CallOp::FMA);
        case ir::Func::Tag::Copysign: return builtin_func(2, CallOp::COPYSIGN);
        case ir::Func::Tag::Cross: return builtin_func(2, CallOp::CROSS);
        case ir::Func::Tag::Dot: return builtin_func(2, CallOp::DOT);
        case ir::Func::Tag::OuterProduct: return builtin_func(2, CallOp::OUTER_PRODUCT);
        case ir::Func::Tag::Length: return builtin_func(1, CallOp::LENGTH);
        case ir::Func::Tag::LengthSquared: return builtin_func(1, CallOp::LENGTH_SQUARED);
        case ir::Func::Tag::Normalize: return builtin_func(1, CallOp::NORMALIZE);
        case ir::Func::Tag::Faceforward: return builtin_func(3, CallOp::FACEFORWARD);
        case ir::Func::Tag::Reflect: return builtin_func(2, CallOp::REFLECT);
        case ir::Func::Tag::Determinant: return builtin_func(1, CallOp::DETERMINANT);
        case ir::Func::Tag::Transpose: return builtin_func(1, CallOp::TRANSPOSE);
        case ir::Func::Tag::Inverse: return builtin_func(1, CallOp::INVERSE);
        case ir::Func::Tag::SynchronizeBlock: return builtin_func(0, CallOp::SYNCHRONIZE_BLOCK);
        case ir::Func::Tag::AtomicRef: LUISA_ERROR_WITH_LOCATION("AtomicRef should have been lowered.");
        case ir::Func::Tag::AtomicExchange: return builtin_func(args.size(), CallOp::ATOMIC_EXCHANGE);
        case ir::Func::Tag::AtomicCompareExchange: return builtin_func(args.size(), CallOp::ATOMIC_COMPARE_EXCHANGE);
        case ir::Func::Tag::AtomicFetchAdd: return builtin_func(args.size(), CallOp::ATOMIC_FETCH_ADD);
        case ir::Func::Tag::AtomicFetchSub: return builtin_func(args.size(), CallOp::ATOMIC_FETCH_SUB);
        case ir::Func::Tag::AtomicFetchAnd: return builtin_func(args.size(), CallOp::ATOMIC_FETCH_AND);
        case ir::Func::Tag::AtomicFetchOr: return builtin_func(args.size(), CallOp::ATOMIC_FETCH_OR);
        case ir::Func::Tag::AtomicFetchXor: return builtin_func(args.size(), CallOp::ATOMIC_FETCH_XOR);
        case ir::Func::Tag::AtomicFetchMin: return builtin_func(args.size(), CallOp::ATOMIC_FETCH_MIN);
        case ir::Func::Tag::AtomicFetchMax: return builtin_func(args.size(), CallOp::ATOMIC_FETCH_MAX);
        case ir::Func::Tag::BufferRead: return builtin_func(2, CallOp::BUFFER_READ);
        case ir::Func::Tag::BufferWrite: return builtin_func(3, CallOp::BUFFER_WRITE);
        case ir::Func::Tag::BufferSize: return builtin_func(1, CallOp::BUFFER_SIZE);
        case ir::Func::Tag::ByteBufferRead: return builtin_func(2, CallOp::BYTE_BUFFER_READ);
        case ir::Func::Tag::ByteBufferWrite: return builtin_func(3, CallOp::BYTE_BUFFER_WRITE);
        case ir::Func::Tag::ByteBufferSize: return builtin_func(1, CallOp::BYTE_BUFFER_SIZE);
        case ir::Func::Tag::Texture2dRead: return builtin_func(2, CallOp::TEXTURE_READ);
        case ir::Func::Tag::Texture3dRead: return builtin_func(2, CallOp::TEXTURE_READ);
        case ir::Func::Tag::Texture2dWrite: return builtin_func(3, CallOp::TEXTURE_WRITE);
        case ir::Func::Tag::Texture3dWrite: return builtin_func(3, CallOp::TEXTURE_WRITE);
        case ir::Func::Tag::Texture2dSize: return builtin_func(1, CallOp::TEXTURE_SIZE);
        case ir::Func::Tag::Texture3dSize: return builtin_func(1, CallOp::TEXTURE_SIZE);
        case ir::Func::Tag::BindlessTexture2dSample: return builtin_func(3, CallOp::BINDLESS_TEXTURE2D_SAMPLE);
        case ir::Func::Tag::BindlessTexture2dSampleLevel: return builtin_func(4, CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL);
        case ir::Func::Tag::BindlessTexture2dSampleGrad: return builtin_func(5, CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD);
        case ir::Func::Tag::BindlessTexture2dSampleGradLevel: return builtin_func(6, CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL);
        case ir::Func::Tag::BindlessTexture3dSample: return builtin_func(3, CallOp::BINDLESS_TEXTURE3D_SAMPLE);
        case ir::Func::Tag::BindlessTexture3dSampleLevel: return builtin_func(4, CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL);
        case ir::Func::Tag::BindlessTexture3dSampleGrad: return builtin_func(5, CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD);
        case ir::Func::Tag::BindlessTexture3dSampleGradLevel: return builtin_func(6, CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL);
        case ir::Func::Tag::BindlessTexture2dRead: return builtin_func(3, CallOp::BINDLESS_TEXTURE2D_READ);
        case ir::Func::Tag::BindlessTexture3dRead: return builtin_func(3, CallOp::BINDLESS_TEXTURE3D_READ);
        case ir::Func::Tag::BindlessTexture2dReadLevel: return builtin_func(4, CallOp::BINDLESS_TEXTURE2D_READ_LEVEL);
        case ir::Func::Tag::BindlessTexture3dReadLevel: return builtin_func(4, CallOp::BINDLESS_TEXTURE3D_READ_LEVEL);
        case ir::Func::Tag::BindlessTexture2dSize: return builtin_func(2, CallOp::BINDLESS_TEXTURE2D_SIZE);
        case ir::Func::Tag::BindlessTexture3dSize: return builtin_func(2, CallOp::BINDLESS_TEXTURE3D_SIZE);
        case ir::Func::Tag::BindlessTexture2dSizeLevel: return builtin_func(3, CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL);
        case ir::Func::Tag::BindlessTexture3dSizeLevel: return builtin_func(3, CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL);
        case ir::Func::Tag::BindlessByteBufferRead: return builtin_func(3, CallOp::BINDLESS_BYTE_BUFFER_READ);
        case ir::Func::Tag::BindlessBufferRead: return builtin_func(3, CallOp::BINDLESS_BUFFER_READ);
        case ir::Func::Tag::BindlessBufferSize: return builtin_func(3, CallOp::BINDLESS_BUFFER_SIZE);
        case ir::Func::Tag::BindlessBufferType: return builtin_func(2, CallOp::BINDLESS_BUFFER_TYPE);
        case ir::Func::Tag::Vec: return make_vector(1);
        case ir::Func::Tag::Vec2: return make_vector(2);
        case ir::Func::Tag::Vec3: return make_vector(3);
        case ir::Func::Tag::Vec4: return make_vector(4);
        case ir::Func::Tag::Permute: {
            auto src = _convert_node(args[0]);
            LUISA_ASSERT(type->is_vector() || type->is_matrix(), "Invalid vector type");
            LUISA_ASSERT(args.size() == type->dimension() + 1, "Vector type inconsistent with arguments.");
            auto inner_type = type->element();

            auto reconstruct_args = luisa::vector<const Expression *>{};
            for (auto i = 1u; i < args.size(); i++) {
                auto reconstruct_arg = _ctx->function_builder->access(inner_type, src, _convert_node(args[i]));
                reconstruct_args.push_back(reconstruct_arg);
            }
            auto op = type->is_vector() ?
                          _decide_make_vector_op(inner_type, type->dimension()) :
                          _decide_make_matrix_op(type->dimension());
            return _ctx->function_builder->call(type, op, luisa::span{reconstruct_args});
        }
        case ir::Func::Tag::InsertElement: {
            // InsertElement(self, new_value, indices...)
            // First we make a copy of self, then we assign the new value to the specified index.
            LUISA_ASSERT(args.size() >= 3u, "`InsertElement` takes 3 arguments.");
            auto old = _convert_node(args[0]);
            auto copy = _ctx->function_builder->local(old->type());
            _ctx->function_builder->assign(copy, old);
            auto self = static_cast<const Expression *>(copy);
            auto indices = args.subspan(2u);
            for (auto i : indices) {
                if (auto self_type = self->type(); self_type->is_structure()) {
                    auto member_index = constant_index(i);
                    auto member_type = self_type->members()[member_index];
                    self = _ctx->function_builder->member(member_type, self, member_index);
                } else if (self_type->is_array() || self_type->is_vector()) {
                    auto index = _convert_node(i);
                    auto inner_type = self_type->element();
                    self = _ctx->function_builder->access(inner_type, self, index);
                } else {
                    LUISA_ASSERT(self_type->is_matrix(), "Invalid type.");
                    auto index = _convert_node(i);
                    auto inner_type = Type::vector(self_type->element(), self_type->dimension());
                    self = _ctx->function_builder->access(inner_type, self, index);
                }
            }
            auto elem = _convert_node(args[1]);
            LUISA_ASSERT(self->type() == elem->type(), "Type mismatch.");
            _ctx->function_builder->assign(self, elem);
            return copy;
        }
        case ir::Func::Tag::ExtractElement: [[fallthrough]];
        case ir::Func::Tag::GetElementPtr: {
            LUISA_ASSERT(args.size() >= 2u, "{} takes at least 2 arguments.", to_string(func.tag));
            auto self = _convert_node(args[0]);
            auto indices = args.subspan(1u);
            for (auto i : indices) {
                if (auto self_type = self->type(); self_type->is_structure()) {
                    auto member_index = constant_index(i);
                    auto member_type = self_type->members()[member_index];
                    self = _ctx->function_builder->member(member_type, self, member_index);
                } else if (self_type->is_vector() || self_type->is_array()) {
                    auto index = _convert_node(i);
                    auto inner_type = self_type->element();
                    self = _ctx->function_builder->access(inner_type, self, index);
                } else {
                    LUISA_ASSERT(self_type->is_matrix(), "Invalid type.");
                    auto index = _convert_node(i);
                    auto inner_type = Type::vector(self_type->element(), self_type->dimension());
                    self = _ctx->function_builder->access(inner_type, self, index);
                }
            }
            LUISA_ASSERT(self->type() == type, "Type mismatch.");
            return self;
        }
        case ir::Func::Tag::Struct: {
            auto alignment = node->type_->struct_._0.alignment;
            auto fields = luisa::vector<const Type *>{};
            auto converted_args = luisa::vector<const Expression *>{};
            for (const auto &arg_ref : args) {
                auto arg = ir::luisa_compute_ir_node_get(arg_ref);
                fields.push_back(_convert_type(arg->type_.get()));
            }
            auto struct_type = Type::structure(alignment, fields);

            auto struct_instance = _ctx->function_builder->local(struct_type);
            for (auto member_index = 0u; member_index < args.size(); member_index++) {
                auto member_type = struct_type->members()[member_index];
                auto access = _ctx->function_builder->member(member_type, struct_instance, member_index);
                _ctx->function_builder->assign(access, _convert_node(args[member_index]));
            }
            return struct_instance;
        }
        case ir::Func::Tag::Array: {
            LUISA_ASSERT(type->is_array(), "Invalid array type.");
            auto element_type = type->element();
            auto array_instance = _ctx->function_builder->local(type);
            LUISA_ASSERT(args.size() == type->dimension(), "Array type inconsistent with arguments. Expected {}, found {}", type->count(), args.size());
            for (auto i = 0u; i < args.size(); i++) {
                auto index = _ctx->function_builder->literal(Type::of<uint>(), i);
                auto access = _ctx->function_builder->access(element_type, array_instance, index);
                auto elem = _convert_node(args[i]);
                _ctx->function_builder->assign(access, elem);
            }
            return array_instance;
        }
        case ir::Func::Tag::Mat: return make_matrix(1);
        case ir::Func::Tag::Mat2: return make_matrix(2);
        case ir::Func::Tag::Mat3: return make_matrix(3);
        case ir::Func::Tag::Mat4: return make_matrix(4);
        case ir::Func::Tag::Callable: {
            auto p_callable = func.callable._0._0;
            LUISA_ASSERT(!p_callable.is_null(), "Invalid callable.");
            auto converted_args = luisa::vector<const Expression *>{};
            for (const auto &arg : args) {
                converted_args.push_back(_convert_node(arg));
            }
            auto callable_fb = convert_callable(p_callable.get());
            auto callable = callable_fb->function();
            return _ctx->function_builder->call(type, callable, luisa::span{converted_args});
        }
        case ir::Func::Tag::CpuCustomOp:
            LUISA_ERROR_WITH_LOCATION("CpuCustomOp is not implemented.");
        case ir::Func::Tag::Unknown0: [[fallthrough]];
        case ir::Func::Tag::Unknown1: LUISA_NOT_IMPLEMENTED();
        case ir::Func::Tag::ShaderExecutionReorder: return builtin_func(2, CallOp::SHADER_EXECUTION_REORDER);

        case ir::Func::Tag::WarpIsFirstActiveLane: return builtin_func(0, CallOp::WARP_IS_FIRST_ACTIVE_LANE);
        case ir::Func::Tag::WarpFirstActiveLane: return builtin_func(0, CallOp::WARP_FIRST_ACTIVE_LANE);
        case ir::Func::Tag::WarpActiveAllEqual: return builtin_func(1, CallOp::WARP_ACTIVE_ALL_EQUAL);
        case ir::Func::Tag::WarpActiveBitAnd: return builtin_func(1, CallOp::WARP_ACTIVE_BIT_AND);
        case ir::Func::Tag::WarpActiveBitOr: return builtin_func(1, CallOp::WARP_ACTIVE_BIT_OR);
        case ir::Func::Tag::WarpActiveBitXor: return builtin_func(1, CallOp::WARP_ACTIVE_BIT_XOR);
        case ir::Func::Tag::WarpActiveCountBits: return builtin_func(1, CallOp::WARP_ACTIVE_COUNT_BITS);
        case ir::Func::Tag::WarpActiveMax: return builtin_func(1, CallOp::WARP_ACTIVE_MAX);
        case ir::Func::Tag::WarpActiveMin: return builtin_func(1, CallOp::WARP_ACTIVE_MIN);
        case ir::Func::Tag::WarpActiveProduct: return builtin_func(1, CallOp::WARP_ACTIVE_PRODUCT);
        case ir::Func::Tag::WarpActiveSum: return builtin_func(1, CallOp::WARP_ACTIVE_SUM);
        case ir::Func::Tag::WarpActiveAll: return builtin_func(1, CallOp::WARP_ACTIVE_ALL);
        case ir::Func::Tag::WarpActiveAny: return builtin_func(1, CallOp::WARP_ACTIVE_ANY);
        case ir::Func::Tag::WarpActiveBitMask: return builtin_func(1, CallOp::WARP_ACTIVE_BIT_MASK);
        case ir::Func::Tag::WarpPrefixCountBits: return builtin_func(1, CallOp::WARP_PREFIX_COUNT_BITS);
        case ir::Func::Tag::WarpPrefixSum: return builtin_func(1, CallOp::WARP_PREFIX_SUM);
        case ir::Func::Tag::WarpPrefixProduct: return builtin_func(1, CallOp::WARP_PREFIX_PRODUCT);
        case ir::Func::Tag::WarpReadLaneAt: return builtin_func(2, CallOp::WARP_READ_LANE);
        case ir::Func::Tag::WarpReadFirstLane: return builtin_func(1, CallOp::WARP_READ_FIRST_ACTIVE_LANE);

        case ir::Func::Tag::IndirectDispatchSetCount: return builtin_func(2, CallOp::INDIRECT_SET_DISPATCH_COUNT);
        case ir::Func::Tag::IndirectDispatchSetKernel: return builtin_func(5, CallOp::INDIRECT_SET_DISPATCH_KERNEL);
    }
    return nullptr;
}

void IR2AST::_convert_instr_phi(const ir::Node *phi) noexcept {
    // do nothing
}

void IR2AST::_convert_instr_return(const ir::Node *node) noexcept {
    if (auto ret = node->instruction->return_._0; ret != ir::INVALID_REF) {
        _ctx->function_builder->return_(_convert_node(ret));
    } else {
        _ctx->function_builder->return_(nullptr);
    }
}

void IR2AST::_convert_instr_loop(const ir::Node *node) noexcept {
    // loop {
    //     body();
    //     if (!cond) {
    //         break;
    //     }
    // }
    auto loop_scope = _ctx->function_builder->loop_();
    _ctx->function_builder->with(loop_scope->body(), [&] {
        // body
        auto old_generic_loop_break = std::exchange(_ctx->generic_loop_break, nullptr);
        _convert_block(node->instruction->loop.body.get());
        _ctx->generic_loop_break = old_generic_loop_break;
        // if (!cond) break;
        auto cond = _convert_node(node->instruction->loop.cond);
        auto not_cond = _ctx->function_builder->unary(Type::of<bool>(), UnaryOp::NOT, cond);
        auto if_scope = _ctx->function_builder->if_(not_cond);
        _ctx->function_builder->with(if_scope->true_branch(), [&] {
            _ctx->function_builder->break_();
        });
    });
}

void IR2AST::_convert_instr_generic_loop(const ir::Node *node) noexcept {
    // template:
    // loop {
    //     loop_break = false;
    //     prepare();
    //     if (!cond()) break;
    //     loop {
    //         body {
    //             // break => { loop_break = true; break; }
    //             // continue => { break; }
    //         }
    //         break;
    //     }
    //     if (loop_break) break;
    //     update();
    // }
    auto generic_loop_break = _ctx->function_builder->local(Type::of<bool>());
    auto loop = _ctx->function_builder->loop_();
    _ctx->function_builder->with(loop->body(), [&] {
        // bool loop_break = false;
        auto false_ = _ctx->function_builder->literal(Type::of<bool>(), false);
        _ctx->function_builder->assign(generic_loop_break, false_);
        // prepare();
        _convert_block(node->instruction->generic_loop.prepare.get());
        // if (!cond()) break;
        auto cond = _convert_node(node->instruction->generic_loop.cond);
        auto not_cond = _ctx->function_builder->unary(Type::of<bool>(), UnaryOp::NOT, cond);
        auto if_ = _ctx->function_builder->if_(not_cond);
        _ctx->function_builder->with(if_->true_branch(), [&] {
            _ctx->function_builder->break_();
        });
        // loop
        auto loop_once = _ctx->function_builder->loop_();
        _ctx->function_builder->with(loop_once->body(), [&] {
            // body
            auto old_generic_loop_break = std::exchange(_ctx->generic_loop_break, generic_loop_break);
            _convert_block(node->instruction->generic_loop.body.get());
            _ctx->generic_loop_break = old_generic_loop_break;
            // break;
            _ctx->function_builder->break_();
        });
        // if (loop_break) break;
        auto if_break = _ctx->function_builder->if_(generic_loop_break);
        _ctx->function_builder->with(if_break->true_branch(), [&] {
            _ctx->function_builder->break_();
        });
        // update();
        _convert_block(node->instruction->generic_loop.update.get());
    });
}

void IR2AST::_convert_instr_break(const ir::Node *node) noexcept {
    if (_ctx->generic_loop_break) {// inside generic loop
        // break => { loop_break = true; break; }
        auto true_ = _ctx->function_builder->literal(Type::of<bool>(), true);
        _ctx->function_builder->assign(_ctx->generic_loop_break, true_);
    }
    _ctx->function_builder->break_();
}

void IR2AST::_convert_instr_continue(const ir::Node *node) noexcept {
    if (_ctx->generic_loop_break) {// inside generic loop
        // continue => { break; }
        _ctx->function_builder->break_();
    } else {
        _ctx->function_builder->continue_();
    }
}

void IR2AST::_convert_instr_if(const ir::Node *node) noexcept {
    auto cond = _convert_node(node->instruction->if_.cond);
    auto if_scope = _ctx->function_builder->if_(cond);
    _ctx->function_builder->with(if_scope->true_branch(), [&] {
        _convert_block(node->instruction->if_.true_branch.get());
    });
    _ctx->function_builder->with(if_scope->false_branch(), [&] {
        _convert_block(node->instruction->if_.false_branch.get());
    });
}

void IR2AST::_convert_instr_switch(const ir::Node *node) noexcept {
    auto value = _convert_node(node->instruction->switch_.value);
    auto switch_scope = _ctx->function_builder->switch_(value);
    _ctx->function_builder->with(switch_scope->body(), [&] {
        auto data = node->instruction->switch_.cases.ptr;
        auto len = node->instruction->switch_.cases.len;
        for (auto i = 0; i < len; i++) {
            auto value = _ctx->function_builder->literal(Type::of<int>(), data[i].value);
            auto case_scope = _ctx->function_builder->case_(value);
            _ctx->function_builder->with(case_scope->body(), [&] {
                _convert_block(data[i].block.get());
            });
        }
        if (node->instruction->switch_.default_.get() != nullptr) {
            auto default_scope = _ctx->function_builder->default_();
            _ctx->function_builder->with(default_scope->body(), [&] {
                _convert_block(node->instruction->switch_.default_.get());
            });
        }
    });
}

void IR2AST::_convert_instr_ad_scope(const ir::Node *node) noexcept {
    _ctx->function_builder->comment_("ADScope Begin");
    _convert_block(node->instruction->ad_scope.body.get());
    _ctx->function_builder->comment_("ADScope End");
}

void IR2AST::_convert_instr_ad_detach(const ir::Node *node) noexcept {
    _ctx->function_builder->comment_("AD Detach Begin");
    _convert_block(node->instruction->ad_detach._0.get());
    _ctx->function_builder->comment_("AD Detach End");
}
void IR2AST::_convert_instr_ray_query(const ir::Node *node) noexcept {
    _ctx->function_builder->comment_("Ray Query Begin");
    auto rq = static_cast<const RefExpr *>(_convert_node(node->instruction->ray_query.ray_query));
    auto rq_scope = _ctx->function_builder->ray_query_(rq);
    _ctx->function_builder->with(rq_scope->on_triangle_candidate(), [&] {
        _convert_block(node->instruction->ray_query.on_triangle_hit.get());
    });
    _ctx->function_builder->with(rq_scope->on_procedural_candidate(), [&] {
        _convert_block(node->instruction->ray_query.on_procedural_hit.get());
    });

    _ctx->function_builder->comment_("Ray Query End");
}
void IR2AST::_convert_instr_comment(const ir::Node *node) noexcept {
    auto comment_body = node->instruction->comment._0;
    auto len = comment_body.ptr[comment_body.len - 1] == 0 ?
                   comment_body.len - 1 :
                   comment_body.len;
    auto comment_content = luisa::string{reinterpret_cast<const char *>(comment_body.ptr), len};
    _ctx->function_builder->comment_(comment_content);
}

const Expression *IR2AST::_convert_constant(const ir::Const &const_) noexcept {

    auto b = _ctx->function_builder.get();
    switch (const_.tag) {
        case ir::Const::Tag::Zero: return b->call(_convert_type(const_.zero._0.get()), CallOp::ZERO, {});
        case ir::Const::Tag::One: return b->call(_convert_type(const_.one._0.get()), CallOp::ONE, {});
        case ir::Const::Tag::Bool: return b->literal(Type::of<bool>(), const_.bool_._0);
        case ir::Const::Tag::Int32: return b->literal(Type::of<int>(), const_.int32._0);
        case ir::Const::Tag::Uint32: return b->literal(Type::of<uint>(), const_.uint32._0);
        case ir::Const::Tag::Int64: return b->literal(Type::of<slong>(), const_.int64._0);
        case ir::Const::Tag::Uint64: return b->literal(Type::of<ulong>(), const_.uint64._0);
        case ir::Const::Tag::Float16: return b->literal(Type::of<half>(), luisa::bit_cast<half>(const_.float16._0));
        case ir::Const::Tag::Float32: return b->literal(Type::of<float>(), const_.float32._0);
        case ir::Const::Tag::Float64: return b->literal(Type::of<double>(), const_.float64._0);
        case ir::Const::Tag::Generic: {
            auto type = _convert_type(const_.generic._1.get());
            auto [data, size, _] = const_.generic._0;
            if (type->is_scalar() || type->is_vector() || type->is_matrix()) {

#define LUISA_IR2AST_DECODE_CONST(T)                                 \
    if (type == Type::of<T>()) {                                     \
        return b->literal(type, *reinterpret_cast<const T *>(data)); \
    }

#define LUISA_IR2AST_DECODE_CONST_VEC(T) \
    LUISA_IR2AST_DECODE_CONST(T)         \
    LUISA_IR2AST_DECODE_CONST(T##2)      \
    LUISA_IR2AST_DECODE_CONST(T##3)      \
    LUISA_IR2AST_DECODE_CONST(T##4)

                LUISA_IR2AST_DECODE_CONST_VEC(bool)
                LUISA_IR2AST_DECODE_CONST_VEC(int)
                LUISA_IR2AST_DECODE_CONST_VEC(uint)
                LUISA_IR2AST_DECODE_CONST_VEC(short)
                LUISA_IR2AST_DECODE_CONST_VEC(ushort)
                LUISA_IR2AST_DECODE_CONST_VEC(slong)
                LUISA_IR2AST_DECODE_CONST_VEC(ulong)
                LUISA_IR2AST_DECODE_CONST_VEC(half)
                LUISA_IR2AST_DECODE_CONST_VEC(float)
                LUISA_IR2AST_DECODE_CONST_VEC(double)

                LUISA_IR2AST_DECODE_CONST(float2x2)
                LUISA_IR2AST_DECODE_CONST(float3x3)
                LUISA_IR2AST_DECODE_CONST(float4x4)

#undef LUISA_IR2AST_DECODE_CONST_VEC
#undef LUISA_IR2AST_DECODE_CONST

                LUISA_NOT_IMPLEMENTED();
            }
            auto c = ConstantData::create(type, data, size);
            return b->constant(c);
        }
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unreachable");
}

const Type *IR2AST::_convert_primitive_type(const ir::Primitive &type) noexcept {
    switch (type) {
        case ir::Primitive::Bool: return Type::of<bool>();
        case ir::Primitive::Float16: return Type::of<half>();
        case ir::Primitive::Float32: return Type::of<float>();
        case ir::Primitive::Int8: return Type::of<int8_t>();
        case ir::Primitive::Uint8: return Type::of<uint8_t>();
        case ir::Primitive::Int16: return Type::of<short>();
        case ir::Primitive::Uint16: return Type::of<ushort>();
        case ir::Primitive::Int32: return Type::of<int>();
        case ir::Primitive::Uint32: return Type::of<uint>();
        case ir::Primitive::Float64: return Type::of<double>();
        case ir::Primitive::Int64: return Type::of<slong>();
        case ir::Primitive::Uint64: return Type::of<ulong>();
        default: LUISA_ERROR_WITH_LOCATION("Invalid primitive type.");
    }
}

const Type *IR2AST::_convert_type(const ir::Type *type) noexcept {
    switch (type->tag) {
        case ir::Type::Tag::Void: return nullptr;
        case ir::Type::Tag::Primitive: return _convert_primitive_type(type->primitive._0);
        case ir::Type::Tag::Vector: {
            switch (type->vector._0.element.tag) {
                case ir::VectorElementType::Tag::Scalar: {
                    auto element_type = _convert_primitive_type(type->vector._0.element.scalar._0);
                    return Type::vector(element_type, type->vector._0.length);
                }
                case ir::VectorElementType::Tag::Vector: LUISA_ERROR_WITH_LOCATION("Vector of vectors is not supported.");
            }
        }
        case ir::Type::Tag::Matrix: return Type::matrix(type->matrix._0.dimension);
        case ir::Type::Tag::Array: {
            auto element_type = _convert_type(type->array._0.element.get());
            return Type::array(element_type, type->array._0.length);
        }
        case ir::Type::Tag::Struct: {
            auto struct_type = type->struct_._0;
            luisa::vector<const Type *> fields;
            fields.reserve(struct_type.fields.len);
            for (auto i = 0; i < struct_type.fields.len; i++) {
                fields.push_back(_convert_type(struct_type.fields.ptr[i].get()));
            }
            return Type::structure(struct_type.alignment, fields);
        }
        case ir::Type::Tag::Opaque: {
            auto opaque_type = luisa::string_view((const char *)type->opaque._0.ptr);
            return Type::custom(opaque_type);
        }
        case ir::Type::Tag::UserData: {
            return Type::of<float>();
        }
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid type. {}", to_string(type->tag));
}

void IR2AST::_collect_phis(const ir::BasicBlock *bb) noexcept {
    _iterate(bb, [this](const ir::Node *node) noexcept {
        auto instr = node->instruction.get();
        switch (instr->tag) {
            case ir::Instruction::Tag::Phi: {
                auto &&incomings = instr->phi._0;
                for (auto i = 0u; i < incomings.len; i++) {
                    const auto &incoming = incomings.ptr[i];
                    auto src_block = incoming.block.get();
                    auto src_value = ir::luisa_compute_ir_node_get(incoming.value);
                    _ctx->block_to_phis[src_block].push_back(PhiAssignment{.dst = node, .src = src_value});
                }
                break;
            }
            case ir::Instruction::Tag::Loop: {
                _collect_phis(instr->loop.body.get());
                break;
            }
            case ir::Instruction::Tag::GenericLoop: {
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
                const auto &cases = instr->switch_.cases;
                for (auto i = 0u; i < cases.len; i++) {
                    _collect_phis(cases.ptr[i].block.get());
                }
                _collect_phis(instr->switch_.default_.get());
                break;
            }
            case ir::Instruction::Tag::AdScope: {
                _collect_phis(instr->ad_scope.body.get());
                break;
            }
            case ir::Instruction::Tag::AdDetach: {
                _collect_phis(instr->ad_detach._0.get());
                break;
            }
            case ir::Instruction::Tag::RayQuery: {
                _collect_phis(instr->ray_query.on_triangle_hit.get());
                _collect_phis(instr->ray_query.on_procedural_hit.get());
                break;
            }
            default: break;
        }
    });
}

void IR2AST::_process_local_declarations(const ir::BasicBlock *bb) noexcept {
    if (bb == nullptr) { return; }
    _iterate(bb, [this](const ir::Node *node) noexcept {
        switch (auto instr = node->instruction.get(); instr->tag) {
            case ir::Instruction::Tag::Buffer: break;
            case ir::Instruction::Tag::Bindless: break;
            case ir::Instruction::Tag::Texture2D: break;
            case ir::Instruction::Tag::Texture3D: break;
            case ir::Instruction::Tag::Accel: break;
            case ir::Instruction::Tag::Shared: break;
            case ir::Instruction::Tag::Uniform: break;
            case ir::Instruction::Tag::Local: {
                auto type = _convert_type(node->type_.get());
                auto variable = _ctx->function_builder->local(type);
                _ctx->node_to_exprs.emplace(node, variable);
                break;
            }
            case ir::Instruction::Tag::Argument: break;
            case ir::Instruction::Tag::UserData: break;
            case ir::Instruction::Tag::Invalid: break;
            case ir::Instruction::Tag::Const: break;
            case ir::Instruction::Tag::Update: break;
            case ir::Instruction::Tag::Call: break;
            case ir::Instruction::Tag::Phi: break;
            case ir::Instruction::Tag::Return: break;
            case ir::Instruction::Tag::Loop: {
                _process_local_declarations(instr->loop.body.get());
                break;
            }
            case ir::Instruction::Tag::GenericLoop: {
                _process_local_declarations(instr->generic_loop.prepare.get());
                _process_local_declarations(instr->generic_loop.body.get());
                _process_local_declarations(instr->generic_loop.update.get());
                break;
            }
            case ir::Instruction::Tag::Break: break;
            case ir::Instruction::Tag::Continue: break;
            case ir::Instruction::Tag::If: {
                _process_local_declarations(instr->if_.true_branch.get());
                _process_local_declarations(instr->if_.false_branch.get());
                break;
            }
            case ir::Instruction::Tag::Switch: {
                auto &&s = instr->switch_.cases;
                for (auto i = 0u; i < s.len; i++) {
                    _process_local_declarations(s.ptr[i].block.get());
                }
                _process_local_declarations(instr->switch_.default_.get());
                break;
            }
            case ir::Instruction::Tag::AdScope: {
                _process_local_declarations(instr->ad_scope.body.get());
                break;
            }
            case ir::Instruction::Tag::AdDetach: {
                _process_local_declarations(instr->ad_detach._0.get());
                break;
            }
            case ir::Instruction::Tag::RayQuery: {
                _process_local_declarations(instr->ray_query.on_triangle_hit.get());
                _process_local_declarations(instr->ray_query.on_procedural_hit.get());
                break;
            }
            case ir::Instruction::Tag::Comment: break;
            case ir::Instruction::Tag::Print: break;
        }
    });
}

[[nodiscard]] CallOp IR2AST::_decide_make_vector_op(const Type *primitive, size_t length) noexcept {
    LUISA_ASSERT(primitive->is_scalar(), "Only scalar types are allowed here, got {}.", primitive->description());
    switch (primitive->tag()) {
        case Type::Tag::BOOL:
            switch (length) {
                case 2: return CallOp::MAKE_BOOL2;
                case 3: return CallOp::MAKE_BOOL3;
                case 4: return CallOp::MAKE_BOOL4;
                default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
            }
        case Type::Tag::FLOAT16:
            switch (length) {
                case 2: return CallOp::MAKE_HALF2;
                case 3: return CallOp::MAKE_HALF3;
                case 4: return CallOp::MAKE_HALF4;
                default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
            }
        case Type::Tag::FLOAT32:
            switch (length) {
                case 2: return CallOp::MAKE_FLOAT2;
                case 3: return CallOp::MAKE_FLOAT3;
                case 4: return CallOp::MAKE_FLOAT4;
                default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
            }
        case Type::Tag::FLOAT64:
            switch (length) {
                case 2: return CallOp::MAKE_DOUBLE2;
                case 3: return CallOp::MAKE_DOUBLE3;
                case 4: return CallOp::MAKE_DOUBLE4;
                default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
            }
        case Type::Tag::INT16:
            switch (length) {
                case 2: return CallOp::MAKE_SHORT2;
                case 3: return CallOp::MAKE_SHORT3;
                case 4: return CallOp::MAKE_SHORT4;
                default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
            }
        case Type::Tag::UINT16:
            switch (length) {
                case 2: return CallOp::MAKE_USHORT2;
                case 3: return CallOp::MAKE_USHORT3;
                case 4: return CallOp::MAKE_USHORT4;
                default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
            }
        case Type::Tag::INT32:
            switch (length) {
                case 2: return CallOp::MAKE_INT2;
                case 3: return CallOp::MAKE_INT3;
                case 4: return CallOp::MAKE_INT4;
                default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
            }
        case Type::Tag::UINT32:
            switch (length) {
                case 2: return CallOp::MAKE_UINT2;
                case 3: return CallOp::MAKE_UINT3;
                case 4: return CallOp::MAKE_UINT4;
                default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
            }
        case Type::Tag::INT64:
            switch (length) {
                case 2: return CallOp::MAKE_LONG2;
                case 3: return CallOp::MAKE_LONG3;
                case 4: return CallOp::MAKE_LONG4;
                default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
            }
        case Type::Tag::UINT64:
            switch (length) {
                case 2: return CallOp::MAKE_ULONG2;
                case 3: return CallOp::MAKE_ULONG3;
                case 4: return CallOp::MAKE_ULONG4;
                default: LUISA_ERROR_WITH_LOCATION("Vectors with length other than 2, 3 and 4 are not supported.");
            }
        default: LUISA_ERROR_WITH_LOCATION("Unsupported vector element type: {}.",
                                           primitive->description());
    }
}

[[nodiscard]] CallOp IR2AST::_decide_make_matrix_op(size_t dimension) noexcept {
    switch (dimension) {
        case 2: return CallOp::MAKE_FLOAT2X2;
        case 3: return CallOp::MAKE_FLOAT3X3;
        case 4: return CallOp::MAKE_FLOAT4X4;
        default: LUISA_ERROR_WITH_LOCATION("Matrices with dimension other than 2, 3 and 4 are not supported.");
    }
}

[[nodiscard]] const RefExpr *IR2AST::_convert_argument(const ir::Node *node) noexcept {
    auto type = _convert_type(node->type_.get());
    switch (node->instruction->tag) {
        case ir::Instruction::Tag::Uniform: return _ctx->function_builder->argument(type); ;
        case ir::Instruction::Tag::Argument: {
            if (node->instruction->argument.by_value) {
                return _ctx->function_builder->argument(type);
            } else {
                return _ctx->function_builder->reference(type);
            }
        }
        // Uniform is the argument of a kernel
        // while Argument is the argument of a callable
        case ir::Instruction::Tag::Texture2D: [[fallthrough]];
        case ir::Instruction::Tag::Texture3D: {
            auto dimension = node->instruction->tag == ir::Instruction::Tag::Texture2D ? 2u : 3u;
            auto texture_type = Type::texture(type, dimension);
            return _ctx->function_builder->texture(texture_type);
        }
        case ir::Instruction::Tag::Buffer: {
            const luisa::compute::Type *buffer_type = nullptr;
            if (type == Type::of<void>() || type == Type::of<ubyte>()) {
                buffer_type = Type::of<ByteBuffer>();
            } else {
                buffer_type = Type::buffer(type);
            }
            return _ctx->function_builder->buffer(buffer_type);
        }
        case ir::Instruction::Tag::Bindless: return _ctx->function_builder->bindless_array();
        case ir::Instruction::Tag::Accel: return _ctx->function_builder->accel();
        default: LUISA_ERROR_WITH_LOCATION("Invalid argument type: {}.", to_string(node->instruction->tag));
    }
}

[[nodiscard]] const RefExpr *IR2AST::_convert_captured(const ir::Capture &captured) noexcept {
    auto node = ir::luisa_compute_ir_node_get(captured.node);
    auto type = _convert_type(node->type_.get());
    switch (captured.binding.tag) {
        case ir::Binding::Tag::Buffer: {
            const luisa::compute::Type *buffer_type = nullptr;
            if (type == Type::of<void>() || type == Type::of<ubyte>()) {
                buffer_type = Type::of<ByteBuffer>();
            } else {
                buffer_type = Type::buffer(type);
            }
            auto &&[handle, offset_bytes, size_bytes] = captured.binding.buffer._0;
            return _ctx->function_builder->buffer_binding(buffer_type, handle, offset_bytes, size_bytes);
        }
        case ir::Binding::Tag::Accel: {
            auto handle = captured.binding.accel._0.handle;
            return _ctx->function_builder->accel_binding(handle);
        }
        case ir::Binding::Tag::BindlessArray: {
            auto handle = captured.binding.bindless_array._0.handle;
            return _ctx->function_builder->bindless_array_binding(handle);
        }
        case ir::Binding::Tag::Texture: {
            auto dimension = [&]() {
                switch (node->instruction->tag) {
                    case ir::Instruction::Tag::Texture2D: return 2u;
                    case ir::Instruction::Tag::Texture3D: return 3u;
                    default: LUISA_ERROR_WITH_LOCATION("Binding tag {} inconsistent with instruction tag {}.",
                                                       to_string(captured.binding.tag),
                                                       to_string(node->instruction->tag));
                }
            }();
            auto texture_type = Type::texture(type, dimension);
            auto &&[handle, level] = captured.binding.texture._0;
            return _ctx->function_builder->texture_binding(texture_type, handle, level);
        }
        default: LUISA_ERROR_WITH_LOCATION("Invalid binding tag {}.", to_string(captured.binding.tag));
    }
}

[[nodiscard]] luisa::shared_ptr<detail::FunctionBuilder> IR2AST::convert_kernel(const ir::KernelModule *kernel) noexcept {

    LUISA_VERBOSE("IR2AST: converting kernel (ptr = {}).",
                  (void *)(kernel));

    IR2ASTContext ctx{
        .module = kernel->module,
        .generic_loop_break = nullptr,
        .function_builder = luisa::make_shared<detail::FunctionBuilder>(Function::Tag::KERNEL)};

    // do the conversion
    {
        auto old_ctx = _ctx;
        _ctx = &ctx;
        detail::FunctionBuilder::FunctionStackGuard guard{_ctx->function_builder.get()};
        _ctx->function_builder->with(_ctx->function_builder->body(), [&]() {
            auto entry = kernel->module.entry.get();
            _collect_phis(entry);

            _ctx->function_builder->set_block_size(uint3{kernel->block_size[0], kernel->block_size[1], kernel->block_size[2]});

            auto captures = kernel->captures;
            auto args = kernel->args;
            for (auto i = 0; i < captures.len; i++) {
                auto captured = captures.ptr[i];
                auto node = ir::luisa_compute_ir_node_get(captured.node);
                auto binding = _convert_captured(captured);
                _ctx->node_to_exprs.emplace(node, binding);
            }
            for (auto i = 0; i < args.len; i++) {
                auto arg = ir::luisa_compute_ir_node_get(args.ptr[i]);
                auto argument = _convert_argument(arg);
                _ctx->node_to_exprs.emplace(arg, argument);
            }

            auto shared = kernel->shared;
            for (auto i = 0; i < shared.len; i++) {
                auto shared_var = ir::luisa_compute_ir_node_get(shared.ptr[i]);
                auto type = _convert_type(shared_var->type_.get());
                auto shared_var_expr = _ctx->function_builder->shared(type);
                _ctx->node_to_exprs.emplace(shared_var, shared_var_expr);
            }
            _process_local_declarations(entry);
            _convert_block(entry);
        });
        _ctx = old_ctx;
    }
    LUISA_VERBOSE("IR2AST: converted kernel (ptr = {}, hash = {:016x}).",
                  (void *)(kernel), ctx.function_builder->hash());
    return ctx.function_builder;
}

[[nodiscard]] luisa::shared_ptr<detail::FunctionBuilder> IR2AST::convert_callable(const ir::CallableModule *callable) noexcept {
    if (auto iter = _converted_callables.find(callable);
        iter != _converted_callables.end()) {
        return iter->second;
    }
    LUISA_VERBOSE("IR2AST: converting callable (ptr = {}).",
                  (void *)(callable), callable->captures.len);
    IR2ASTContext ctx{
        .module = callable->module,
        .generic_loop_break = nullptr,
        .function_builder = luisa::make_shared<detail::FunctionBuilder>(Function::Tag::CALLABLE)};
    _converted_callables.emplace(callable, ctx.function_builder);

    // do the conversion
    {
        auto old_ctx = _ctx;
        _ctx = &ctx;
        detail::FunctionBuilder::FunctionStackGuard guard{_ctx->function_builder.get()};
        _ctx->function_builder->with(_ctx->function_builder->body(), [&]() {
            for (auto i = 0; i < callable->captures.len; i++) {
                auto captured = callable->captures.ptr[i];
                auto node = ir::luisa_compute_ir_node_get(captured.node);
                auto binding = _convert_captured(captured);
                _ctx->node_to_exprs.emplace(node, binding);
            }
            for (auto i = 0; i < callable->args.len; i++) {
                auto arg = ir::luisa_compute_ir_node_get(callable->args.ptr[i]);
                _ctx->node_to_exprs.emplace(arg, _convert_argument(arg));
            }
            auto entry = callable->module.entry.get();
            _collect_phis(entry);
            _process_local_declarations(entry);
            _convert_block(entry);
        });
        _ctx = old_ctx;
    }
    auto callable_hash = ctx.function_builder->hash();
    LUISA_VERBOSE("IR2AST: converted callable (ptr = {}, hash = {:016x}).",
                  (void *)(callable), callable_hash,
                  ctx.function_builder->arguments().size());
    return std::move(ctx.function_builder);
}

const Type *IR2AST::get_type(const ir::Type *type) noexcept {
    return _convert_type(type);
}

[[nodiscard]] const Type *IR2AST::get_type(const ir::NodeRef node_ref) noexcept {
    auto node = ir::luisa_compute_ir_node_get(node_ref);
    //    LUISA_VERBOSE("node type is {}", luisa::to_underlying(node->instruction->tag));
    switch (node->instruction->tag) {
        case ir::Instruction::Tag::Buffer: return Type::buffer(_convert_type(node->type_.get()));
        case ir::Instruction::Tag::Texture2D: return Type::texture(_convert_type(node->type_.get()), 2u);
        case ir::Instruction::Tag::Texture3D: return Type::texture(_convert_type(node->type_.get()), 3u);
        case ir::Instruction::Tag::Bindless: return Type::from("bindless_array");
        case ir::Instruction::Tag::Accel: return Type::from("accel");
        default: return _convert_type(node->type_.get());
    }
}

[[nodiscard]] luisa::shared_ptr<detail::FunctionBuilder> IR2AST::build(const ir::KernelModule *kernel) noexcept {
    IR2AST builder{};
    return builder.convert_kernel(kernel);
}

[[nodiscard]] luisa::shared_ptr<detail::FunctionBuilder> IR2AST::build(const ir::CallableModule *callable) noexcept {
    IR2AST builder{};
    return builder.convert_callable(callable);
}

}// namespace luisa::compute

#pragma clang diagnostic pop
