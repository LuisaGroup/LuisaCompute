

#include <Codegen/DxCodegen.h>
#include <Codegen/StructGenerator.h>
namespace toolhub::directx {

void StringStateVisitor::visit(const UnaryExpr *expr) {
    str << "(";
    switch (expr->op()) {
        case UnaryOp::PLUS://+x
            str << '+';
            break;
        case UnaryOp::MINUS://-x
            str << '-';
            break;
        case UnaryOp::NOT://!x
            str << '!';
            break;
        case UnaryOp::BIT_NOT://~x
            str << '~';
            break;
    }
    expr->operand()->accept(*this);
    str << ")";
}
void StringStateVisitor::visit(const BinaryExpr *expr) {
    str << "(";
    auto op = expr->op();
    auto IsMulFuncCall = [&]() -> bool {
        if (op == BinaryOp::MUL) {
            if ((expr->lhs()->type()->is_matrix() &&
                 (!expr->rhs()->type()->is_scalar())) ||
                (expr->rhs()->type()->is_matrix() &&
                 (!expr->lhs()->type()->is_scalar()))) {
                return true;
            }
        }
        return false;
    };
    if (IsMulFuncCall()) {
        str << "Mul("sv;
        expr->lhs()->accept(*this);
        str << ',';
        expr->rhs()->accept(*this);//Reverse matrix
        str << ')';

    } else if (op == BinaryOp::AND) {
        str << "and(";
        expr->lhs()->accept(*this);
        str << ",";
        expr->rhs()->accept(*this);
        str << ")";
    } else if (op == BinaryOp::OR) {
        str << "or(";
        expr->lhs()->accept(*this);
        str << ",";
        expr->rhs()->accept(*this);
        str << ")";
    } else {

        expr->lhs()->accept(*this);
        switch (op) {
            case BinaryOp::ADD:
                str << '+';
                break;
            case BinaryOp::SUB:
                str << '-';
                break;
            case BinaryOp::MUL:
                str << '*';
                break;
            case BinaryOp::DIV:
                str << '/';
                break;
            case BinaryOp::MOD:
                str << '%';
                break;
            case BinaryOp::BIT_AND:
                str << '&';
                break;
            case BinaryOp::BIT_OR:
                str << '|';
                break;
            case BinaryOp::BIT_XOR:
                str << '^';
                break;
            case BinaryOp::SHL:
                str << "<<"sv;
                break;
            case BinaryOp::SHR:
                str << ">>"sv;
                break;
            case BinaryOp::LESS:
                str << '<';
                break;
            case BinaryOp::GREATER:
                str << '>';
                break;
            case BinaryOp::LESS_EQUAL:
                str << "<="sv;
                break;
            case BinaryOp::GREATER_EQUAL:
                str << ">="sv;
                break;
            case BinaryOp::EQUAL:
                str << "=="sv;
                break;
            case BinaryOp::NOT_EQUAL:
                str << "!="sv;
                break;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "Not implemented.");
        }
        expr->rhs()->accept(*this);
    }
    str << ")";
}
void StringStateVisitor::visit(const MemberExpr *expr) {
    str << "(";
    char const *xyzw = "xyzw";
    if (expr->is_swizzle()) {
        expr->self()->accept(*this);
        str << '.';
        for (auto i : vstd::range(static_cast<uint32_t>(expr->swizzle_size()))) {
            str << xyzw[expr->swizzle_index(i)];
        }

    } else {
        vstd::string curStr;
        StringStateVisitor vis(f, curStr);
        expr->self()->accept(vis);
        auto &&selfStruct = CodegenUtility::GetStruct(expr->self()->type());
        auto &&structVar = selfStruct->GetStructVar(expr->member_index());
        str << curStr << '.' << structVar;
    }
    str << ")";
}
void StringStateVisitor::visit(const AccessExpr *expr) {
    str << "(";
    auto t = expr->range()->type();
    if (expr->range()->tag() == Expression::Tag::REF) {
        auto variable = static_cast<RefExpr const *>(expr->range())->variable();
        if (variable.tag() == Variable::Tag::SHARED) {
            expr->range()->accept(*this);
            str << '[';
            expr->index()->accept(*this);
            str << ']';
            return;
        }
    }
    switch (t->tag()) {
        case Type::Tag::BUFFER:
        case Type::Tag::VECTOR:
            expr->range()->accept(*this);
            str << '[';
            expr->index()->accept(*this);
            str << ']';
            break;
        case Type::Tag::MATRIX:
            expr->range()->accept(*this);
            str << '[';
            expr->index()->accept(*this);
            str << ']';
            if (t->dimension() == 3u) {
                str << ".xyz";
            }
            break;
        default: {
            expr->range()->accept(*this);
            str << ".v[";
            expr->index()->accept(*this);
            str << ']';
        } break;
    }
    str << ")";
}
void StringStateVisitor::visit(const RefExpr *expr) {
    Variable v = expr->variable();
    vstd::string tempStr;
    CodegenUtility::GetVariableName(v, tempStr);
    CodegenUtility::RegistStructType(v.type());
    str << tempStr;
}

void StringStateVisitor::visit(const LiteralExpr *expr) {
    LiteralExpr::Value const &value = expr->value();
    eastl::visit(
        [&](auto &&value) -> void {
            using T = std::remove_cvref_t<decltype(value)>;
            PrintValue<T> prt;
            prt(value, (str));
        },
        expr->value());
}
void StringStateVisitor::visit(const CallExpr *expr) {
    CodegenUtility::GetFunctionName(expr, str, *this);
}
void StringStateVisitor::visit(const CastExpr *expr) {
    str << "(";
    switch (expr->op()) {
        case CastOp::STATIC:
            str << '(';
            CodegenUtility::GetTypeName(*expr->type(), str, Usage::READ);
            str << ')';
            expr->expression()->accept(*this);
            break;
        case CastOp::BITWISE: {
            auto type = expr->type();
            while (type->is_vector()) {
                type = type->element();
            }
            switch (type->tag()) {
                case Type::Tag::FLOAT:
                    str << "asfloat"sv;
                    break;
                case Type::Tag::INT:
                    str << "asint"sv;
                    break;
                case Type::Tag::UINT:
                    str << "asuint"sv;
                    break;
                default:
                    LUISA_ERROR_WITH_LOCATION(
                        "Bitwise cast not implemented for type '{}'.",
                        expr->type()->description());
            }
            str << '(';
            expr->expression()->accept(*this);
            str << ')';
        } break;
    }
    str << ")";
}

void StringStateVisitor::visit(const ConstantExpr *expr) {
    CodegenUtility::GetConstName(expr->data(), str);
}

void StringStateVisitor::visit(const BreakStmt *state) {
    str << "break;\n";
}
void StringStateVisitor::visit(const ContinueStmt *state) {

    str << "continue;\n";
}
void StringStateVisitor::visit(const ReturnStmt *state) {
    if (state->expression()) {
        str << "return ";
        state->expression()->accept(*this);
        str << ";\n";
    } else {
        str << "return;\n";
    }
}
void StringStateVisitor::visit(const ScopeStmt *state) {
    str << "{\n";
    for (auto &&i : state->statements()) {
        i->accept(*this);
    }
    str << "}\n";
}
void StringStateVisitor::visit(const CommentStmt *state) {
    str << "/* " << state->comment() << " */\n";
}
void StringStateVisitor::visit(const IfStmt *state) {
    str << "if(";
    state->condition()->accept(*this);
    str << ")";
    state->true_branch()->accept(*this);
    if (!state->false_branch()->statements().empty()) {
        str << "else";
        state->false_branch()->accept(*this);
    }
}
void StringStateVisitor::visit(const LoopStmt *state) {
    str << "while(true)";
    state->body()->accept(*this);
}
void StringStateVisitor::visit(const ExprStmt *state) {
    state->expression()->accept(*this);
    str << ";\n";
}
void StringStateVisitor::visit(const SwitchStmt *state) {
    str << "switch(";
    state->expression()->accept(*this);
    str << ")";
    state->body()->accept(*this);
}
void StringStateVisitor::visit(const SwitchCaseStmt *state) {
    str << "case ";
    state->expression()->accept(*this);
    str << ":";
    state->body()->accept(*this);
}
void StringStateVisitor::visit(const SwitchDefaultStmt *state) {
    str << "default:";
    state->body()->accept(*this);
}

void StringStateVisitor::visit(const AssignStmt *state) {
    state->lhs()->accept(*this);
    str << '=';

    state->rhs()->accept(*this);
    str << ";\n";
}
void StringStateVisitor::visit(const ForStmt *state) {
    str << "for(";

    //    state->variable()->accept(*this);
    str << ';';
    state->condition()->accept(*this);
    str << ';';
    state->variable()->accept(*this);
    str << "+=";
    state->step()->accept(*this);
    str << ")";
    state->body()->accept(*this);
}

StringStateVisitor::StringStateVisitor(
    Function f,
    vstd::string &str)
    : str(str), f(f) {}

void StringStateVisitor::visit(const Function &f) {
    auto func = [&]<bool collectShared>() {
        auto process_variable = [&](auto v) noexcept {
            if (v.tag() == Variable::Tag::LOCAL && v.type()->is_structure()) {
                vstd::string typeName;
                CodegenUtility::GetTypeName(*v.type(), typeName, f.variable_usage(v.uid()));
                str << typeName << ' ';
                CodegenUtility::GetVariableName(v, str);
                str << "=("sv << typeName << ")0;\n";
            } else {
                CodegenUtility::GetTypeName(*v.type(), str, f.variable_usage(v.uid()));
                str << ' ';
                CodegenUtility::GetVariableName(v, str);
                str << ";\n";
            }
            if constexpr (collectShared) {
                if (v.tag() == Variable::Tag::SHARED) {
                    sharedVariables->emplace(v);
                }
            }
        };
        for (auto v : f.local_variables()) { process_variable(v); }
        for (auto v : f.shared_variables()) { process_variable(v); }
    };
    if (sharedVariables) {
        func.operator()<true>();
    } else {
        func.operator()<false>();
    }
    f.body()->accept(*this);
}

StringStateVisitor::~StringStateVisitor() = default;
}// namespace toolhub::directx
