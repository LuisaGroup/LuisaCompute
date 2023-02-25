#include <core/logging.h>
#include <ir/ir.hpp>

namespace luisa::compute {


using ir::Func;
using ir::Instruction;
struct Converter {
    detail::FunctionBuilder *builder;
    luisa::unordered_map<const ir::Type *, const Type *> type_map;
    luisa::unordered_map<ir::NodeRef, const Expression *, NodeRefHash> node_map;
    // luisa::unordered_map<const
    const Type *convert(const ir::Type *ty) noexcept {
        if (auto it = type_map.find(ty); it != type_map.end()) {
            return it->second;
        }
        const Type *converted = nullptr;
        switch (ty->tag) {
            case ir::Type::Tag::Primitive: {
                auto prim = ty->primitive._0;
                switch (prim) {
                    case ir::Primitive::Bool:
                        converted = Type::of<bool>();
                        break;
                    case ir::Primitive::Int32:
                        converted = Type::of<int32_t>();
                        break;
                    case ir::Primitive::Uint32:
                        converted = Type::of<uint32_t>();
                        break;
                    // case ir::Primitive::Int64:
                    //     converted = Type::of<int64_t>();
                    //     break;
                    // case ir::Primitive::Uint64:
                    //     converted = Type::of<uint64_t>();
                    //     break;
                    case ir::Primitive::Float32:
                        converted = Type::of<float>();
                        break;
                    // case ir::Primitive::Float64:
                    //     converted = Type::of<double>();
                    //     break;
                    default:
                        LUISA_ERROR_WITH_LOCATION("Unsupported primitive type.");
                }
                break;
            }
            default: abort();
        }
        type_map[ty] = converted;
        return converted;
    }
    const Expression *convert(const ir::NodeRef &node_ref) noexcept {
        if (auto it = node_map.find(node_ref); it != node_map.end()) {
            return it->second;
        }
        auto e = _convert(node_ref);
        node_map[node_ref] = e;
        return e;
    }
    const Expression *_convert(const ir::NodeRef &node_ref) noexcept {
        auto node = ir::luisa_compute_ir_node_get(node_ref);
        auto inst = node->instruction;
        auto ty = convert(node->type_.get());
        switch (inst->tag) {
            case Instruction::Tag::Local: {
                return builder->local(ty);
            }
            case Instruction::Tag::Call: {
                auto call = inst->call;
                auto func = call._0.tag;
                auto &args_v = call._1;
                auto args = args_v.ptr;
                if (func == Func::Tag::Gradient) {
                    return nullptr;
                }
                if (func == Func::Tag::RequiresGradient) {
                    return nullptr;
                }
                if (func == Func::Tag::GradientMarker) {
                    auto v = convert(args[0]);
                    auto grad = convert(args[1]);
                }
                auto v = [&] {
                    switch (func) {
                        case Func::Tag::Add:
                            return builder->binary(ty, BinaryOp::ADD, convert(args[0]), convert(args[1]));
                        case Func::Tag::Sub:
                            return builder->binary(ty, BinaryOp::SUB, convert(args[0]), convert(args[1]));
                        case Func::Tag::Mul:
                            return builder->binary(ty, BinaryOp::MUL, convert(args[0]), convert(args[1]));
                        case Func::Tag::Div:
                            return builder->binary(ty, BinaryOp::DIV, convert(args[0]), convert(args[1]));
                        case Func::Tag::Rem:
                            // TODO: this is actually different
                            return builder->binary(ty, BinaryOp::MOD, convert(args[0]), convert(args[1]));
                        case Func::Tag::BitAnd:
                            return builder->binary(ty, BinaryOp::BIT_AND, convert(args[0]), convert(args[1]));
                        case Func::Tag::BitOr:
                            return builder->binary(ty, BinaryOp::BIT_OR, convert(args[0]), convert(args[1]));
                        case Func::Tag::BitXor:
                            return builder->binary(ty, BinaryOp::BIT_XOR, convert(args[0]), convert(args[1]));
                        case Func::Tag::Shl:
                            return builder->binary(ty, BinaryOp::SHL, convert(args[0]), convert(args[1]));
                        case Func::Tag::Shr:
                            return builder->binary(ty, BinaryOp::SHR, convert(args[0]), convert(args[1]));
                        case Func::Tag::RotLeft:
                        case Func::Tag::RotRight:
                            LUISA_ERROR_WITH_LOCATION("Ask the author to implement this.");
                            // return builder->binary(ty, BinaryOp::ROT_LEFT, convert(args[0]), convert(args[1]));
                        default:
                            break;
                    }
                    LUISA_ERROR_WITH_LOCATION("Ask the author to implement this.");
                }();
                auto a = builder->local(ty);
                builder->assign(a, v);
                return a;
            } break;

            default:
                LUISA_ERROR_WITH_LOCATION("unreachable");
        }
    }
};
void convert_to_ast(const ir::Module *module, detail::FunctionBuilder *builder) noexcept {
}

struct ToIR {
    const ScopeStmt *stmt;
    luisa::unordered_map<const Expression *, ir::NodeRef> expr_map;
    luisa::unordered_map<const Type *, ir::CArc<ir::Type>> type_map;
    luisa::unordered_map<uint32_t, ir::NodeRef> var_map;
    ir::IrBuilder *var_def_builder = nullptr;
    ir::CArc<ir::Type> _build_type(const Type *ty) noexcept {
        return {};
    }
    ir::CArc<ir::Type> build_type(const Type *ty) noexcept {
        auto it = type_map.find(ty);
        if (it != type_map.end()) {
            return it->second;
        }
        auto ir_ty = _build_type(ty);
        type_map[ty] = ir_ty;
        return ir_ty;
    }
    ir::NodeRef build_expr(const Expression *expr, ir::IrBuilder *builder) {
        auto it = expr_map.find(expr);
        if (it != expr_map.end()) {
            return it->second;
        }
        auto node = _build_expr(expr, builder);
        expr_map[expr] = node;
        return node;
    }
    ir::NodeRef _build_expr(const Expression *expr, ir::IrBuilder *builder) {
        auto type = build_type(expr->type());
        if (auto binop = dynamic_cast<const BinaryExpr *>(expr)) {
            auto op = binop->op();
            ir::Func func;
            switch (op) {
                case BinaryOp::ADD:
                    func.tag = ir::Func::Tag::Add;
                    break;
                case BinaryOp::SUB:
                    func.tag = ir::Func::Tag::Sub;
                    break;
                case BinaryOp::MUL:
                    func.tag = ir::Func::Tag::Mul;
                    break;
                default:
                    abort();
            }
            return build_call(builder, func, {build_expr(binop->lhs(), builder), build_expr(binop->rhs(), builder)}, type);
        } else if (auto unary = dynamic_cast<const UnaryExpr *>(expr)) {
            auto op = unary->op();
            ir::Func func;
            switch (op) {
                case UnaryOp::MINUS:
                    func.tag = ir::Func::Tag::Neg;
                    break;
                case UnaryOp::NOT:
                case UnaryOp::BIT_NOT:
                    func.tag = ir::Func::Tag::BitNot;
                    break;
                case UnaryOp::PLUS:
                    return build_expr(unary->operand(), builder);
                default:
                    abort();
            }
            return build_call(builder, func, {build_expr(unary->operand(), builder)}, type);
        } else if (auto var = dynamic_cast<const RefExpr *>(expr)) {
            auto v = var->variable();
            auto vid = v.uid();
            if (auto it = var_map.find(vid); it != var_map.end()) {
                return it->second;
            }
            auto node = ir::luisa_compute_ir_build_local_zero_init(builder, type);
            var_map[vid] = node;
            return node;
        } else if (auto literal = dynamic_cast<const LiteralExpr *>(expr)) {
            auto value = literal->value();
            ir::NodeRef node;
            ir::Const cst;
            luisa::visit(
                [&](auto &&value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, float>) {
                        cst.tag = ir::Const::Tag::Float32;
                        cst.float32._0 = value;
                        node = ir::luisa_compute_ir_build_const(builder, cst);
                    } else if constexpr (std::is_same_v<T, int32_t>) {
                        cst.tag = ir::Const::Tag::Int32;
                        cst.int32._0 = value;
                        node = ir::luisa_compute_ir_build_const(builder, cst);
                    } else if constexpr (std::is_same_v<T, uint32_t>) {
                        cst.tag = ir::Const::Tag::Uint32;
                        cst.uint32._0 = value;
                        node = ir::luisa_compute_ir_build_const(builder, cst);
                    } else if constexpr (std::is_same_v<T, bool>) {
                        cst.tag = ir::Const::Tag::Bool;
                        cst.bool_._0 = value;
                        node = ir::luisa_compute_ir_build_const(builder, cst);
                    } else {
                        LUISA_ERROR_WITH_LOCATION("unreachable");
                    }
                },
                value);
        }
    }
    ir::Pooled<ir::BasicBLock> build_block(const ScopeStmt *stmt) {
        auto builder = ir::luisa_compute_ir_new_builder();
        for (auto s : stmt->statements()) {
            if (auto expr = dynamic_cast<const ExprStmt *>(s)) {
                build_expr(expr->expression(), &builder);
            } else if (auto expr = dynamic_cast<const AssignStmt *>(s)) {
                auto lhs = expr->lhs();
                auto rhs = expr->rhs();
                auto lhs_node = build_expr(lhs, &builder);
                auto rhs_node = build_expr(rhs, &builder);
                ir::luisa_compute_ir_build_update(&builder, lhs_node, rhs_node);
            }
        }
        return ir::luisa_compute_ir_build_finish(builder);
    }
};

LC_IR_API ir::Module convert_to_ir(const ScopeStmt *stmt) noexcept {
    ir::Module m;

    return m;
}

struct AdContext {
    ScopeStmt *stmt;
};

thread_local AdContext *ad_context = nullptr;

LC_IR_API void begin_autodiff() noexcept {
    LUISA_ASSERT(!ad_context, "autodiff context already exists");
    ad_context = new AdContext();
    auto func_builder = detail::FunctionBuilder::current();
    auto true_ = func_builder->literal(Type::of<bool>(), true);
    auto if_ = func_builder->if_(true_);
    ad_context->stmt = if_->true_branch();
}

LC_IR_API AutodiffResult end_autodiff() noexcept {
    LUISA_ASSERT(ad_context, "no autodiff context exists");
    auto func_builder = detail::FunctionBuilder::current();
    func_builder->pop_scope(ad_context->stmt);
    (void)func_builder->pop_stmt();
    auto m = convert_to_ir(ad_context->stmt);

    auto pipeline = ir::luisa_compute_ir_transform_pipeline_new();
    ir::luisa_compute_ir_transform_pipeline_add_transform(pipeline, "ssa");
    ir::luisa_compute_ir_transform_pipeline_add_transform(pipeline, "autodiff");
    m = ir::luisa_compute_ir_transform_pipeline_transform(pipeline, m);
    ir::luisa_compute_ir_transform_pipeline_destroy(pipeline);
    convert_to_ast(&m, func_builder);
    delete ad_context;
    ad_context = nullptr;
    return {};
}

}// namespace luisa::compute
