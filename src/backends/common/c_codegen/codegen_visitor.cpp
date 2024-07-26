#include "codegen_visitor.h"
#include <luisa/core/logging.h>
#include <luisa/core/mathematics.h>
#include "../shader_print_formatter.h"
namespace luisa::compute {
CodegenVisitor::~CodegenVisitor() {}
void CodegenVisitor::visit(const UnaryExpr *expr) {
    if (expr->op() == UnaryOp::PLUS) {
        expr->operand()->accept(*this);
        return;
    }
    if (expr->type()->is_vector()) {
        auto func = utils.gen_vec_unary(expr->op(), expr->type());
        sb << func << '(';
        expr->operand()->accept(*this);
        sb << ')';
    } else {
        sb << '(';
        switch (expr->op()) {
            case UnaryOp::MINUS:
                sb << '-';
                break;
            case UnaryOp::NOT:
                sb << '!';
                break;
            case UnaryOp::BIT_NOT:
                sb << '~';
                break;
        }
        expr->operand()->accept(*this);
        sb << ')';
    }
}
void CodegenVisitor::visit(const FuncRefExpr *expr) {
    Function func(expr->func());
    sb << "((uint64_t)(&custom_" << luisa::format("{}", utils.func_index(func)) << "))";
}
void CodegenVisitor::visit(const BinaryExpr *expr) {
    auto lhs = expr->lhs();
    auto rhs = expr->rhs();
    if (lhs->type()->is_scalar() && rhs->type()->is_scalar()) {
        sb << '(';
        lhs->accept(*this);
        switch (expr->op()) {
            case BinaryOp::ADD: sb << '+'; break;
            case BinaryOp::SUB: sb << '-'; break;
            case BinaryOp::MUL: sb << '*'; break;
            case BinaryOp::DIV: sb << '/'; break;
            case BinaryOp::MOD: sb << '%'; break;
            case BinaryOp::BIT_AND: sb << '&'; break;
            case BinaryOp::BIT_OR: sb << '|'; break;
            case BinaryOp::BIT_XOR: sb << '^'; break;
            case BinaryOp::SHL: sb << "<<"; break;
            case BinaryOp::SHR: sb << ">>"; break;
            case BinaryOp::AND: sb << "&&"; break;
            case BinaryOp::OR: sb << "||"; break;

            case BinaryOp::LESS: sb << '<'; break;
            case BinaryOp::GREATER: sb << '>'; break;
            case BinaryOp::LESS_EQUAL: sb << "<="; break;
            case BinaryOp::GREATER_EQUAL: sb << ">="; break;
            case BinaryOp::EQUAL: sb << "=="; break;
            case BinaryOp::NOT_EQUAL: sb << "!="; break;
        }
        rhs->accept(*this);
        sb << ')';
    } else if (lhs->type()->is_vector()) {
        auto func = utils.gen_vec_binary(expr->op(), lhs->type(), rhs->type());
        sb << func << '(';
        lhs->accept(*this);
        sb << ',';
        rhs->accept(*this);
        sb << ')';
    } else if (lhs->type()->is_matrix()) {
        LUISA_ASSERT(expr->op() == BinaryOp::MUL, "Matrix only support mul.");
        sb << "mul_";
        utils.get_type_name(sb, lhs->type());
        sb << '_';
        utils.get_type_name(sb, rhs->type());
        sb << '(';
        lhs->accept(*this);
        sb << ',';
        rhs->accept(*this);
        sb << ')';
    }
}
void CodegenVisitor::visit(const MemberExpr *expr) {
    if (expr->is_swizzle()) {
        if (expr->swizzle_size() == 1) {
            sb << "GET(";
            utils.get_type_name(sb, expr->type());
            sb << ", ";
            expr->self()->accept(*this);
            sb << ", " << luisa::format("{}", expr->swizzle_index(0)) << ')';
        } else {
            luisa::fixed_vector<uint, 4> swizzles;
            for (auto &i : vstd::range(expr->swizzle_size())) {
                swizzles.emplace_back(expr->swizzle_index(i));
            }
            auto swizzle_name = utils.gen_vec_swizzle(swizzles, expr->swizzle_code(), expr->self()->type());
            sb << swizzle_name << '(';
            expr->self()->accept(*this);
            sb << ')';
        }
    } else {
        expr->self()->accept(*this);
        sb << ".v";
        vstd::to_string(expr->member_index(), sb);
    }
}
void CodegenVisitor::visit(const AccessExpr *expr) {
    bool is_deref = false;
    if (expr->index()->tag() == Expression::Tag::LITERAL) {
        auto lit = static_cast<LiteralExpr const *>(expr->index());
        is_deref = luisa::visit(
            [&]<typename T>(T const &t) {
                if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
                    return t == 0;
                }
                return false;
            },
            lit->value());
    }
    auto self = expr->range();
    auto self_type = self->type();
    auto is_valid_struct = [&](Type const *t) {
        if (!t->is_structure()) return false;
        auto mem = t->members();
        if (mem.size() < 2) return false;
        return mem[0]->is_uint64() && mem[1]->is_uint64();
    };
    auto is_lvalue = [&](auto &is_lvalue, Expression const *expr) -> bool {
        switch (expr->tag()) {
            case Expression::Tag::UNARY: return is_lvalue(is_lvalue, static_cast<UnaryExpr const *>(expr)->operand());
            case Expression::Tag::BINARY: return false;
            case Expression::Tag::MEMBER: return is_lvalue(is_lvalue, static_cast<MemberExpr const *>(expr)->self());
            case Expression::Tag::ACCESS: return is_lvalue(is_lvalue, static_cast<AccessExpr const *>(expr)->range());
            case Expression::Tag::LITERAL: return true;
            case Expression::Tag::REF: return true;
            case Expression::Tag::CONSTANT: return true;
            case Expression::Tag::CALL: return false;
            case Expression::Tag::CAST: return is_lvalue(is_lvalue, static_cast<CastExpr const *>(expr)->expression());
            case Expression::Tag::TYPE_ID:
            case Expression::Tag::STRING_ID:
            case Expression::Tag::FUNC_REF:
                return true;
            default:
                LUISA_ERROR("Not accessable.");
                return false;
        }
    };
    if (self_type->is_vector() || self_type->is_array() || is_valid_struct(self_type)) {
        auto arg_types = {self_type, expr->index()->type()};
        auto is_rval = !is_lvalue(is_lvalue, expr->range());
        sb << "(*" << utils.gen_access(expr->type(), arg_types, is_rval) << '(';
        if(!is_rval){
            sb << '&';
        }
        self->accept(*this);
        sb << ", ";
        expr->index()->accept(*this);
        sb << "))";
    } else {
        if (is_deref) {
            sb << "DEREF(";
            utils.get_type_name(sb, expr->type());
            sb << ", ";
            self->accept(*this);
            sb << ')';
        } else {
            sb << "ACCESS(";
            utils.get_type_name(sb, expr->type());
            sb << ", ";
            self->accept(*this);
            sb << ", ";
            expr->index()->accept(*this);
            sb << ')';
        }
    }
}
void CodegenVisitor::visit(const LiteralExpr *expr) {
    sb << '(';
    luisa::visit(
        [&]<typename T>(T const &value) -> void {
            PrintValue<T> prt;
            prt(value, sb);
        },
        expr->value());
    sb << ')';
}
void CodegenVisitor::visit(const RefExpr *expr) {
    auto &&var = expr->variable();
    if (var.is_reference()) {
        sb << "(*";
        utils.gen_var_name(sb, expr->variable());
        sb << ')';
    } else {
        utils.gen_var_name(sb, expr->variable());
    }
}
void CodegenVisitor::visit(const ConstantExpr *expr) {
    sb << luisa::format("c{}", expr->data().hash());
}
void CodegenVisitor::visit(const CallExpr *expr) {
    auto args = expr->arguments();
    auto check_atomic = [](Type const *type) {
        LUISA_ASSERT(type->is_scalar() && (type->is_int32() || type->is_uint32() || type->is_int64() || type->is_uint64()), "Bad atomic type");
    };
    switch (expr->op()) {
        case CallOp::EXTERNAL: {
            utils.call_external_func(sb, this, expr);
        }
            return;
        case CallOp::CUSTOM: {
            auto func = expr->custom();
            auto &&func_args = func.arguments();
            sb << "custom_" << luisa::format("{}", utils.func_index(func))
               << '(';
            bool comma = false;
            auto func_arg_iter = func_args.begin();
            for (auto &i : args) {
                if (comma) {
                    sb << ", ";
                }
                comma = true;
                if (func_arg_iter->is_reference()) {
                    sb << "&(";
                    i->accept(*this);
                    sb << ')';
                } else {
                    i->accept(*this);
                }
                func_arg_iter++;
            }
            sb << ')';
        }
            return;
        case CallOp::ATOMIC_EXCHANGE:
            check_atomic(args[0]->type());
            sb << "skr_atomic_exchange_explicit((volatile ";
            utils.get_type_name(sb, args[0]->type());
            sb << "*)&";
            args[0]->accept(*this);
            sb << ", ";
            args[1]->accept(*this);
            sb << ", skr_memory_order_seq_cst)";
            return;
        case CallOp::ATOMIC_COMPARE_EXCHANGE:
            check_atomic(args[0]->type());
            sb << "skr_atomic_compare_exchange_weak((volatile ";
            utils.get_type_name(sb, args[0]->type());
            sb << "*)&";
            args[0]->accept(*this);
            sb << ", ";
            args[1]->accept(*this);
            sb << ", ";
            args[2]->accept(*this);
            sb << ')';
            return;
        case CallOp::ATOMIC_FETCH_ADD:
            sb << "skr_atomic_fetch_add_explicit((volatile ";
            check_atomic(args[0]->type());
            utils.get_type_name(sb, args[0]->type());
            sb << "*)&";
            args[0]->accept(*this);
            sb << ", ";
            args[1]->accept(*this);
            sb << ", skr_memory_order_seq_cst)";
            return;
        case CallOp::ATOMIC_FETCH_SUB:
            sb << "skr_atomic_fetch_sub_explicit((volatile ";
            check_atomic(args[0]->type());
            utils.get_type_name(sb, args[0]->type());
            sb << "*)&";
            args[0]->accept(*this);
            sb << ", ";
            args[1]->accept(*this);
            sb << ", skr_memory_order_seq_cst)";
            return;
        case CallOp::ATOMIC_FETCH_AND:
            sb << "skr_atomic_fetch_and_explicit((volatile ";
            check_atomic(args[0]->type());
            utils.get_type_name(sb, args[0]->type());
            sb << "*)&";
            args[0]->accept(*this);
            sb << ", ";
            args[1]->accept(*this);
            sb << ", skr_memory_order_seq_cst)";
            return;
        case CallOp::ATOMIC_FETCH_OR:
            sb << "skr_atomic_fetch_or_explicit((volatile ";
            check_atomic(args[0]->type());
            utils.get_type_name(sb, args[0]->type());
            sb << "*)&";
            args[0]->accept(*this);
            sb << ", ";
            args[1]->accept(*this);
            sb << ", skr_memory_order_seq_cst)";
            return;
        case CallOp::ATOMIC_FETCH_XOR:
            sb << "skr_atomic_fetch_xor_explicit((volatile ";
            check_atomic(args[0]->type());
            utils.get_type_name(sb, args[0]->type());
            sb << "*)&";
            args[0]->accept(*this);
            sb << ", ";
            args[1]->accept(*this);
            sb << ", skr_memory_order_seq_cst)";
            return;
        case CallOp::ADDRESS_OF:
            sb << "ADDR_OF(";
            args[0]->accept(*this);
            sb << ')';
            return;
        case CallOp::BUFFER_ADDRESS:
            sb << '(';
            args[0]->accept(*this);
            sb << ".ptr)";
            return;
        case CallOp::BYTE_BUFFER_SIZE:
            sb << '(';
            args[0]->accept(*this);
            sb << ".len)";
            return;
        case CallOp::BUFFER_SIZE:
            sb << '(';
            args[0]->accept(*this);
            sb << ".len / " << luisa::format("{}", args[0]->type()->element()->size()) << ')';
            return;
        case CallOp::ASSUME:
            sb << "LUISA_ASSUME(";
            args[0]->accept(*this);
            sb << ')';
            return;
        case CallOp::UNREACHABLE:
            sb << "LUISA_UNREACHABLE()";
            return;
        case CallOp::MAKE_FLOAT2X2: {
            if (args.size() == 2) {
                sb << "make_float2x2_0";
            } else {
                sb << "make_float2x2_1";
            }
        } break;
        case CallOp::MAKE_FLOAT3X3: {
            if (args.size() == 3) {
                sb << "make_float3x3_0";
            } else {
                sb << "make_float3x3_1";
            }
        } break;
        case CallOp::MAKE_FLOAT4X4: {
            if (args.size() == 4) {
                sb << "make_float4x4_0";
            } else {
                sb << "make_float4x4_1";
            }
        } break;
        case CallOp::TRANSPOSE: {
            sb << "transpose_";
            utils.get_type_name(sb, expr->type());
        } break;
        case CallOp::INVERSE: {
            sb << "inverse_";
            utils.get_type_name(sb, expr->type());
        } break;
        case CallOp::ZERO: {
            sb << "memzero(&(";
            args[0]->accept(*this);
            sb << "), "
               << luisa::format("{}", args[0]->type()->size())
               << ')';
        } break;
        case CallOp::ONE: {
            sb << "memone(&(";
            args[0]->accept(*this);
            sb << "), "
               << luisa::format("{}", args[0]->type()->size())
               << ')';
        } break;
        default: {
            luisa::fixed_vector<Type const *, 4> types;
            vstd::push_back_func(types, args.size(), [&](size_t i) { return args[i]->type(); });
            sb << utils.gen_callop(expr->op(), expr->type(), types);
        } break;
    }
    sb << '(';
    bool comma = false;
    for (auto &i : args) {
        if (comma) {
            sb << ", ";
        }
        comma = true;
        i->accept(*this);
    }
    sb << ')';
}
void CodegenVisitor::visit(const CastExpr *expr) {
    sb << "((";
    utils.get_type_name(sb, expr->type());
    sb << ')';
    expr->expression()->accept(*this);
    sb << ')';
}
void CodegenVisitor::visit(const TypeIDExpr *expr) {
    sb << luisa::format("{}", expr->data_type()->hash());
}
void CodegenVisitor::visit(const StringIDExpr *expr) {
    sb << luisa::format("{}", luisa::hash<luisa::string_view>{}(expr->data()));
}

void CodegenVisitor::visit(const BreakStmt *stmt) {
    sb << "break;";
}
void CodegenVisitor::visit(const ContinueStmt *stmt) {
    sb << "continue;";
}
void CodegenVisitor::visit(const ReturnStmt *stmt) {
    sb << "return";
    if (stmt->expression()) {
        sb << ' ';
        stmt->expression()->accept(*this);
    }
    sb << ';';
}
void CodegenVisitor::visit(const ScopeStmt *stmt) {
    sb << "{\n";
    for (auto &i : stmt->statements()) {
        i->accept(*this);
        sb << '\n';
        if (i->tag() == Statement::Tag::RETURN ||
            i->tag() == Statement::Tag::CONTINUE ||
            i->tag() == Statement::Tag::BREAK) {
            break;
        }
    }
    sb << "}\n";
}
void CodegenVisitor::visit(const IfStmt *stmt) {
    sb << "if(";
    stmt->condition()->accept(*this);
    sb << ")";
    stmt->true_branch()->accept(*this);
    sb << "else";
    stmt->false_branch()->accept(*this);
}
void CodegenVisitor::visit(const LoopStmt *stmt) {
    sb << "while(true)";
    stmt->body()->accept(*this);
}
void CodegenVisitor::visit(const ExprStmt *stmt) {
    stmt->expression()->accept(*this);
    sb << ';';
}
void CodegenVisitor::visit(const SwitchStmt *stmt) {
    sb << "switch(";
    stmt->expression()->accept(*this);
    sb << ")";
    stmt->body()->accept(*this);
}
void CodegenVisitor::visit(const SwitchCaseStmt *stmt) {
    sb << "case ";
    stmt->expression()->accept(*this);
    sb << ":\n";
    stmt->body()->accept(*this);
}
void CodegenVisitor::visit(const SwitchDefaultStmt *stmt) {
    sb << "default:\n";
    stmt->body()->accept(*this);
}
void CodegenVisitor::visit(const AssignStmt *stmt) {
    stmt->lhs()->accept(*this);
    sb << "=(";
    stmt->rhs()->accept(*this);
    sb << ");";
}
void CodegenVisitor::visit(const ForStmt *stmt) {
    sb << "for(";
    //    state->variable()->accept(*this);
    sb << ';';
    stmt->condition()->accept(*this);
    sb << ';';
    stmt->variable()->accept(*this);
    sb << "+=";
    stmt->step()->accept(*this);
    sb << ")";
    stmt->body()->accept(*this);
}
void CodegenVisitor::visit(const CommentStmt *stmt) {
    sb << "/*" << stmt->comment() << "*/";
}
void CodegenVisitor::visit(const RayQueryStmt *stmt) {
    LUISA_ERROR("Ray query not implemented.");
}
void CodegenVisitor::visit(const PrintStmt *stmt) {
    sb << "{\nchar print_str[] = {";
    bool comma = false;
    for (auto &i : stmt->format()) {
        comma = true;
        sb << luisa::format("{}", (uint)i)
           << ',';
    };
    sb
        << "0};\n"
           "push_str(print_str, "
        << luisa::format("{}ull", stmt->format().size())
        << ");\n";
    for (auto &i : stmt->arguments()) {
        sb << "push_";
        utils.get_type_name(sb, i->type());
        sb << '(';
        i->accept(*this);
        sb << ");\n";
    }
    sb << "invoke_print();\n}";
}
CodegenVisitor::CodegenVisitor(
    vstd::StringBuilder &sb,
    luisa::string_view entry_name,
    Clanguage_CodegenUtils &utils,
    Function func)
    : utils(utils), sb(sb) {
    if (func.tag() == Function::Tag::CALLABLE) {
        utils.print_function_declare(sb, func);
    } else {
        utils.print_kernel_declare(sb, func);
    }
    sb << "{\n";
    for (auto &i : func.local_variables()) {
        utils.get_type_name(sb, i.type());
        sb << ' ';
        if (i.tag() == Variable::Tag::REFERENCE) {
            sb << '*';
        }
        utils.gen_var_name(sb, i);
        sb << ";\n";
    }
    func.body()->accept(*this);
    sb << "}\n";
}

}// namespace luisa::compute