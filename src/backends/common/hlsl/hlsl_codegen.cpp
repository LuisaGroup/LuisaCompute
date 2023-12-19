#include "hlsl_codegen.h"
#include "struct_generator.h"
#include "codegen_stack_data.h"
namespace lc::hlsl {
void StringStateVisitor::visit(const UnaryExpr *expr) {
    literalBrace = true;
    auto fallBackLiteral = vstd::scope_exit([this]() { literalBrace = false; });
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
    literalBrace = true;
    auto fallBackLiteral = vstd::scope_exit([this]() { literalBrace = false; });
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
        str << "_Mul("sv;
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
        accessCount++;
        expr->self()->accept(*this);
        accessCount--;
        str << '.';
        for (auto i : vstd::range(expr->swizzle_size())) {
            str << xyzw[expr->swizzle_index(i)];
        }

    } else {
        vstd::StringBuilder curStr;
        StringStateVisitor vis(f, curStr, util);
        expr->self()->accept(vis);
        str << curStr << ".v"sv << vstd::to_string(expr->member_index());
        auto t = expr->type();
        if (t->is_vector() && t->dimension() == 3) {
            str << ".v"sv;
        }
    }
}
void StringStateVisitor::visit(const AccessExpr *expr) {
    auto t = expr->range()->type();
    auto basicAccess = [&]() {
        accessCount++;
        expr->range()->accept(*this);
        accessCount--;
        str << '[';
        expr->index()->accept(*this);
        str << ']';
    };
    if (expr->range()->tag() == Expression::Tag::REF) {
        auto variable = static_cast<RefExpr const *>(expr->range())->variable();
        if (variable.tag() == Variable::Tag::SHARED) {
            accessCount++;
            expr->range()->accept(*this);
            accessCount--;
            str << '[';
            expr->index()->accept(*this);
            str << ']';
            if (accessCount == 0 && t->is_matrix() && t->dimension() == 3u) {
                str << ".xyz"sv;
            } else if (t->is_array() && t->element()->is_vector() && t->element()->dimension() == 3) {
                str << ".v"sv;
            }
            return;
        }
    }
    //    else if (expr->range()->tag() == Expression::Tag::CONSTANT) {
    //        basicAccess();
    //        return;
    //    }
    switch (t->tag()) {
        case Type::Tag::BUFFER:
        case Type::Tag::VECTOR: {
            basicAccess();
        } break;
        case Type::Tag::MATRIX: {
            accessCount++;
            expr->range()->accept(*this);
            accessCount--;
            str << '[';
            expr->index()->accept(*this);
            str << ']';
            if (accessCount == 0 && t->dimension() == 3u) {
                str << ".xyz"sv;
            }
        } break;
        default: {
            accessCount++;
            expr->range()->accept(*this);
            accessCount--;
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
    util->GetVariableName(v, tempStr);
    util->RegistStructType(v.type());
    str << tempStr;
}

void StringStateVisitor::visit(const LiteralExpr *expr) {
    if (literalBrace) {
        str << '(';
    }
    luisa::visit(
        [&]<typename T>(T const &value) -> void {
            PrintValue<T> prt;
            prt(value, str);
        },
        expr->value());
    if (literalBrace) {
        str << ')';
    }
}
void StringStateVisitor::visit(const CallExpr *expr) {
    util->GetFunctionName(expr, str, *this);
}

void StringStateVisitor::visit(const StringIDExpr *expr) {
    str << "((";
    util->GetTypeName(*expr->type(), str, Usage::READ);
    str << ")" << luisa::hash_value(expr->data()) << "ull)";
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
            util->GetTypeName(*expr->type(), str, Usage::READ);
            str << ')';
            expr->expression()->accept(*this);
            break;
        case CastOp::BITWISE: {
            auto type = expr->type();
            while (type->is_vector()) {
                type = type->element();
            }
            switch (type->tag()) {
                case Type::Tag::FLOAT16:
                case Type::Tag::FLOAT32:
                    str << "asfloat"sv;
                    break;
                case Type::Tag::INT16:
                case Type::Tag::INT32:
                case Type::Tag::INT64:
                    str << "asint"sv;
                    break;
                case Type::Tag::UINT16:
                case Type::Tag::UINT32:
                case Type::Tag::UINT64:
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
    util->GetConstName(expr->data().hash(), expr->data(), str);
}

void StringStateVisitor::visit(const BreakStmt *state) {
#ifdef USE_SPIRV
    auto stackData = util->StackData();
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
        switch (i->tag()) {
            case Statement::Tag::BREAK:
            case Statement::Tag::CONTINUE:
            case Statement::Tag::RETURN:
                return;
            default: break;
        }
    }
}
void StringStateVisitor::visit(const AutoDiffStmt *stmt) {
    visit(stmt->body());
}
void StringStateVisitor::visit(const PrintStmt *stmt) {
    vstd::fixed_vector<Type const *, 16> types;
    vstd::push_back_func(
        types,
        stmt->arguments().size(),
        [&](size_t i) {
            return stmt->arguments()[i]->type();
        });
    auto structType = Type::structure(types);
    vstd::string counterName = "_p";
    vstd::to_string(printCount, counterName);
    str << "uint "sv << counterName << ";\nInterlockedAdd(_printCounter[0],"sv;
    size_t align = std::max<size_t>(structType->alignment(), 4);
    size_t ele_size = structType->size() + align;
    ele_size = ((ele_size + 15ull) & (~15ull));
    vstd::to_string(ele_size, str);
    str << ',' << counterName << ");\n"sv
        << "if("sv << counterName << "<1048576){\n"sv;
    {
        vstd::StringBuilder typeName;
        util->GetTypeName(*structType, typeName, Usage::READ_WRITE);
        vstd::string dataName = "_pv";
        vstd::to_string(printCount, dataName);
        str << typeName << ' ' << dataName << "=("sv << typeName << ")0;\n"sv;
        for (auto i : vstd::range(types.size())) {
            str << dataName << ".v"sv;
            vstd::to_string(i, str);
            auto t = types[i];
            if (t->is_vector() && t->dimension() == 3) {
                str << ".v"sv;
            }
            str << '=';
            stmt->arguments()[i]->accept(*this);
            str << ";\n"sv;
        }
        str << "_printBuffer.Store("sv
            << counterName << ',';
        auto printerIdx = util->AddPrinter(stmt->format(), structType);
        vstd::to_string(printerIdx, str);
        str << ");\n"sv
            << "_printBuffer.Store("sv
            << counterName << '+' + vstd::to_string(align) + ',' << dataName << ");\n"sv;
        printCount++;
    }
    str << "}\n"sv;
}
void StringStateVisitor::visit(const CommentStmt *state) {
#ifndef NDEBUG
    str << "/* " << state->comment() << " */\n";
#endif
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
        auto stackData = util->StackData();
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
        auto stackData = util->StackData();
        if (stackData->tempSwitchCounter == 0) {
            str << "if("sv;
        } else {
            str << "else if("sv;
        }
        ++stackData->tempSwitchCounter;
        util->StackData()->tempSwitchExpr->accept(*this);
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
        if (std::none_of(state->body()->statements().cbegin(),
                         state->body()->statements().cend(),
                         [](const auto &stmt) {
                             return stmt->tag() == Statement::Tag::BREAK;
                         })) {
            str << "break;\n";
        }
    }
}
void StringStateVisitor::visit(const SwitchDefaultStmt *state) {
#ifdef USE_SPIRV
    if (CodegenStackData::ThreadLocalSpirv()) {
        auto stackData = util->StackData();
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
        if (std::none_of(state->body()->statements().cbegin(),
                         state->body()->statements().cend(),
                         [](const auto &stmt) {
                             return stmt->tag() == Statement::Tag::BREAK;
                         })) {
            str << "break;\n";
        }
    }
}

void StringStateVisitor::visit(const AssignStmt *state) {
    auto is_shared = [&](const Expression *x) noexcept {
        if (x->tag() == Expression::Tag::REF) {
            auto v = static_cast<RefExpr const *>(x)->variable();
            if (v.tag() == Variable::Tag::SHARED) {
                return true;
            }
        }
        return false;
    };
    auto is_custom = [&](const Expression *x, Variable &v) noexcept {
        if (x->tag() == Expression::Tag::REF) {
            v = static_cast<RefExpr const *>(x)->variable();
            if (v.type()->is_custom()) {
                return true;
            }
        }
        return false;
    };
    // shared variables are not wrapped in array
    // structs, so some hack is necessary
    bool isLazyDecl = false;
    Variable rqVar;
    bool lhs_is_shared{false};
    if (is_custom(state->lhs(), rqVar)) {
        auto iter = lazyDeclVars.find(rqVar);
        if (iter != lazyDeclVars.end()) {
            util->GetTypeName(*rqVar.type(), str, Usage::READ);
            str << ' ';
            util->GetVariableName(rqVar, str);
            lazyDeclVars.erase(iter);
            isLazyDecl = true;
        }
    }
    auto rhs_is_shared = is_shared(state->rhs());
    if (!isLazyDecl) {
        lhs_is_shared = is_shared(state->lhs());
        if (!lhs_is_shared && rhs_is_shared) {
            str << "(";
            state->lhs()->accept(*this);
            str << ").v";
        } else {
            state->lhs()->accept(*this);
        }
    }
    str << '=';
    if (lhs_is_shared && !rhs_is_shared) {
        str << "(";
        state->rhs()->accept(*this);
        str << ").v";
    } else {
        state->rhs()->accept(*this);
    }
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
void StringStateVisitor::visit(const RayQueryStmt *stmt) {
    str << "{\n"sv;
    str << "while("sv;
    stmt->query()->accept(*this);
    str << ".Proceed()){\n"sv
        << "if("sv;
    stmt->query()->accept(*this);
    str << ".CandidateType()==CANDIDATE_NON_OPAQUE_TRIANGLE){\n"sv;
    stmt->on_triangle_candidate()->accept(*this);
    str << "}else{\n"sv;
    stmt->on_procedural_candidate()->accept(*this);
    str << "}}}\n"sv;
}
StringStateVisitor::StringStateVisitor(
    Function f,
    vstd::StringBuilder &str,
    CodegenUtility *util)
    : f(f), util(util), str(str) {
}
void StringStateVisitor::VisitFunction(
#ifdef LUISA_ENABLE_IR
    vstd::unordered_set<Variable> const &grad_vars,
#endif
    Function func) {
    lazyDeclVars.clear();
    for (auto &&v : func.local_variables()) {
        Usage usage = func.variable_usage(v.uid());
        if (usage == Usage::NONE) [[unlikely]] {
            continue;
        }
        if (v.type()->tag() == Type::Tag::CUSTOM) {
            auto desc = v.type()->description();
            // rayquery need specialization to workaround DXC's bug
            if (desc == "LC_RayQueryAll"sv || desc == "LC_RayQueryAny"sv) {
                lazyDeclVars.emplace(v);
                continue;
            }
        }
        if ((static_cast<uint32_t>(usage) & static_cast<uint32_t>(Usage::WRITE)) == 0) {
            str << "const "sv;
        }
        vstd::StringBuilder typeName;
        util->GetTypeName(*v.type(), typeName, f.variable_usage(v.uid()));
        vstd::StringBuilder varName;
        util->GetVariableName(v, varName);

        str << typeName << ' ' << varName;
        if (!(v.type()->is_resource() || v.type()->is_custom())) [[likely]] {
            str << "=("sv << typeName << ")0"sv;
        }
        str << ";\n";
    }
#ifdef LUISA_ENABLE_IR
    for (auto v : grad_vars) {
        vstd::StringBuilder typeName;
        util->GetTypeName(*v.type(), typeName, f.variable_usage(v.uid()));
        vstd::StringBuilder varName;
        util->GetVariableName(v, varName);
        str << typeName << ' ' << varName << "_grad=("sv << typeName << ")0;\n"sv;
    }
#endif
    if (sharedVariables) {
        size_t shared_size{};
        for (auto &&v : func.shared_variables()) {
            // FIXME: redundant creation of string
            vstd::StringBuilder typeName;
            util->GetTypeName(*v.type(), typeName, f.variable_usage(v.uid()));
            sharedVariables->emplace(v.hash(), v);
            shared_size += v.type()->size();
        }
        LUISA_ASSERT(shared_size <= 32768, "Shared memory size must be less than 64kb.");
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
}// namespace lc::hlsl
