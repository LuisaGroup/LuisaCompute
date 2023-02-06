#include "dx_codegen.h"
#include "struct_generator.h"
#include "codegen_stack_data.h"
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
    auto op = expr->op();
    auto IsMulFuncCall = [&]() -> bool {
        if (op == BinaryOp::MUL) [[unlikely]] {
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
        str << "(";
        expr->lhs()->accept(*this);
        switch (op) {
            case BinaryOp::ADD:
                str << '+';
                break;
            case BinaryOp::SUB:
                str << '-';
                break;
            case BinaryOp::MUL: {
                str << '*';
            } break;
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
        str << ")";
    }
}
void StringStateVisitor::visit(const MemberExpr *expr) {
    char const *xyzw = "xyzw";
    if (expr->is_swizzle()) {
        expr->self()->accept(*this);
        str << '.';
        for (auto i : vstd::range(expr->swizzle_size())) {
            str << xyzw[expr->swizzle_index(i)];
        }

    } else {
        vstd::StringBuilder curStr;
        StringStateVisitor vis(f, curStr);
        expr->self()->accept(vis);
        auto &&selfStruct = CodegenUtility::GetStruct(expr->self()->type());
        auto &&structVar = selfStruct->GetStructVar(expr->member_index());
        str << curStr << '.' << structVar;
        auto t = expr->type();
        if (t->is_vector() && t->dimension() == 3) {
            str << ".v"sv;
        }
    }
}
void StringStateVisitor::visit(const AccessExpr *expr) {
    auto t = expr->range()->type();
    auto PrintOrigin = [&] {
        expr->range()->accept(*this);
        str << '[';
        expr->index()->accept(*this);
        str << ']';
        if (t->is_matrix() && t->dimension() == 3u) {
            str << ".xyz"sv;
        }
        else if(t->is_array() && t->element()->is_vector() && t->element()->dimension() == 3){
            str << ".v"sv;
        }
    };
    if (expr->range()->tag() == Expression::Tag::REF) {
        auto variable = static_cast<RefExpr const *>(expr->range())->variable();
        if (variable.tag() == Variable::Tag::SHARED) {
            PrintOrigin();
            return;
        }
    } else if (expr->range()->tag() == Expression::Tag::CONSTANT) {
        PrintOrigin();
        return;
    }
    switch (t->tag()) {
        case Type::Tag::BUFFER:
        case Type::Tag::VECTOR: {
            expr->range()->accept(*this);
            str << '[';
            expr->index()->accept(*this);
            str << ']';
        } break;
        case Type::Tag::MATRIX: {
            expr->range()->accept(*this);
            str << '[';
            expr->index()->accept(*this);
            str << ']';
            if (t->dimension() == 3u) {
                str << ".xyz"sv;
            }
        } break;
        default: {
            expr->range()->accept(*this);
            str << ".v[";
            expr->index()->accept(*this);
            str << ']';
            if (t->element()->is_vector() && t->element()->dimension() == 3) {
                str << ".v"sv;
            }
        } break;
    }
}
void StringStateVisitor::visit(const RefExpr *expr) {
    Variable v = expr->variable();
    vstd::StringBuilder tempStr;
    CodegenUtility::GetVariableName(v, tempStr);
    CodegenUtility::RegistStructType(v.type());
    str << tempStr;
    auto t = expr->type();
}

void StringStateVisitor::visit(const LiteralExpr *expr) {
    luisa::visit(
        [&]<typename T>(T const &value) -> void {
            PrintValue<T> prt;
            prt(value, str);
        },
        expr->value());
}
void StringStateVisitor::visit(const CallExpr *expr) {
    CodegenUtility::GetFunctionName(expr, str, *this);
}
void StringStateVisitor::visit(const CastExpr *expr) {
    if (expr->type() == expr->expression()->type()) [[unlikely]] {
        expr->expression()->accept(*this);
        return;
    }
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
    CodegenUtility::GetConstName(expr->data().hash(), expr->data(), str);
}

void StringStateVisitor::visit(const BreakStmt *state) {
#ifdef USE_SPIRV
    auto stackData = CodegenUtility::StackData();
    if (!CodegenStackData::ThreadLocalSpirv() || !stackData->tempSwitchExpr)
#endif
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
    for (auto &&i : state->statements()) {
        i->accept(*this);
        switch (state->tag()) {
            case Statement::Tag::BREAK:
            case Statement::Tag::CONTINUE:
            case Statement::Tag::RETURN:
                return;
            default: break;
        }
    }
}
void StringStateVisitor::visit(const CommentStmt *state) {
    // str << "/* " << state->comment() << " */\n";
}
void StringStateVisitor::visit(const IfStmt *state) {
    str << "if(";
    state->condition()->accept(*this);
    str << ")";
    {
        Scope scope{this};
        state->true_branch()->accept(*this);
    }
    if (!state->false_branch()->statements().empty()) {
        str << "else";
        {
            Scope scope{this};
            state->false_branch()->accept(*this);
        }
    }
}
void StringStateVisitor::visit(const LoopStmt *state) {
    str << "while(true)";
    {
        Scope scope{this};
        state->body()->accept(*this);
    }
}
void StringStateVisitor::visit(const ExprStmt *state) {
    state->expression()->accept(*this);
    str << ";\n";
}
void StringStateVisitor::visit(const SwitchStmt *state) {
#ifdef USE_SPIRV
    if (CodegenStackData::ThreadLocalSpirv()) {
        auto stackData = CodegenUtility::StackData();
        stackData->tempSwitchExpr = state->expression();
        stackData->tempSwitchCounter = 0;
        state->body()->accept(*this);
        stackData->tempSwitchExpr = nullptr;
    } else
#endif
    {

        str << "switch(";
        state->expression()->accept(*this);
        str << ")";
        {
            Scope scope{this};
            state->body()->accept(*this);
        }
    }
}
void StringStateVisitor::visit(const SwitchCaseStmt *state) {
#ifdef USE_SPIRV
    if (CodegenStackData::ThreadLocalSpirv()) {
        auto stackData = CodegenUtility::StackData();
        if (stackData->tempSwitchCounter == 0) {
            str << "if("sv;
        } else {
            str << "else if("sv;
        }
        ++stackData->tempSwitchCounter;
        CodegenUtility::StackData()->tempSwitchExpr->accept(*this);
        str << "=="sv;
        state->expression()->accept(*this);
        str << ')';
        {
            Scope scope{this};
            state->body()->accept(*this);
        }
    } else
#endif
    {
        str << "case ";
        state->expression()->accept(*this);
        str << ":";
        {
            Scope scope{this};
            state->body()->accept(*this);
        }
    }
}
void StringStateVisitor::visit(const SwitchDefaultStmt *state) {
#ifdef USE_SPIRV
    if (CodegenStackData::ThreadLocalSpirv()) {
        auto stackData = CodegenUtility::StackData();
        if (stackData->tempSwitchCounter == 0) {
            {
                Scope scope{this};
                state->body()->accept(*this);
            }
        } else {
            str << "else";
            {
                Scope scope{this};
                state->body()->accept(*this);
            }
        }
        ++stackData->tempSwitchCounter;

    } else
#endif
    {
        str << "default:";
        {
            Scope scope{this};
            state->body()->accept(*this);
        }
    }
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
    {
        Scope scope{this};
        state->body()->accept(*this);
    }
}
StringStateVisitor::StringStateVisitor(
    Function f,
    vstd::StringBuilder &str)
    : f(f), str(str) {
}
void StringStateVisitor::VisitFunction(Function func) {
    for (auto &&v : func.local_variables()) {
        Usage usage = func.variable_usage(v.uid());
        if (usage == Usage::NONE) [[unlikely]] {
            continue;
        }
        if ((static_cast<uint32_t>(usage) & static_cast<uint32_t>(Usage::WRITE)) == 0) {
            str << "const "sv;
        }
#if false// clear struct
        if (v.type()->is_structure()) {
            vstd::StringBuilder typeName;
            CodegenUtility::GetTypeName(*v.type(), typeName, f.variable_usage(v.uid()));
            str << typeName << ' ';
            CodegenUtility::GetVariableName(v, str);
            str << "=("sv << typeName << ")0;\n";
        } else
#endif
        {
            CodegenUtility::GetTypeName(*v.type(), str, f.variable_usage(v.uid()));
            str << ' ';
            CodegenUtility::GetVariableName(v, str);
            str << ";\n";
        }
    }
    if (sharedVariables) {
        for (auto &&v : func.shared_variables()) {
            sharedVariables->emplace(v.hash(), v);
        }
    }
    func.body()->accept(*this);
}
StringStateVisitor::~StringStateVisitor() = default;
StringStateVisitor::Scope::Scope(StringStateVisitor *self)
    : self(self) {
    self->str << "{\n"sv;
}
StringStateVisitor::Scope::~Scope() {
    self->str << "}\n"sv;
}
}// namespace toolhub::directx
