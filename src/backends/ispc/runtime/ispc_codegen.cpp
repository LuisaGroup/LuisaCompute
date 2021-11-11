#pragma vengine_package ispc_vsproject

#include "ispc_codegen.h"

namespace lc::ispc {
void StringExprVisitor::visit(const UnaryExpr *expr) {

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
    StringExprVisitor vis(str);
    expr->operand()->accept(vis);
}
void StringExprVisitor::visit(const BinaryExpr *expr) {
    auto IsMulFuncCall = [&]() -> bool {
        if (expr->op() == BinaryOp::MUL) {
            if ((expr->lhs()->type()->is_matrix() && (!expr->rhs()->type()->is_scalar())) || (expr->rhs()->type()->is_matrix() && (!expr->lhs()->type()->is_scalar()))) {
                return true;
            }
        }
        return false;
    };
    if (IsMulFuncCall()) {
        str << "mul("sv;
        expr->rhs()->accept(*this);//Reverse matrix
        str << ',';
        expr->lhs()->accept(*this);
        str << ')';

    } else {

        expr->lhs()->accept(*this);
        switch (expr->op()) {
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
            case BinaryOp::AND:
                str << "&&"sv;
                break;
            case BinaryOp::OR:
                str << "||"sv;
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
        }
        expr->rhs()->accept(*this);
    }
}
void StringExprVisitor::visit(const MemberExpr *expr) {
    expr->self()->accept(*this);
    if (expr->is_swizzle()) {
        char const *xyzw = "xyzw";
        str << '.';
        for (auto i : vstd::range(static_cast<uint32_t>(expr->swizzle_size()))) {
            str << xyzw[expr->swizzle_index(i)];
        }
    } else {
        str << ".v"sv;
        vstd::to_string(static_cast<uint64_t>(expr->member_index()), (str));
    }
}
void StringExprVisitor::visit(const AccessExpr *expr) {
    expr->range()->accept(*this);
    auto t = expr->range()->type();
    if (t && (t->is_buffer() || t->is_vector()))
        str << '[';
    else
        str << ".v[";
    expr->index()->accept(*this);
    str << ']';
}
void StringExprVisitor::visit(const RefExpr *expr) {
    Variable v = expr->variable();
    CodegenUtility::RegistStructType(v.type());
    CodegenUtility::GetVariableName(v, str);
}
template<typename T>
struct PrintValue;
template<>
struct PrintValue<float> {
    void operator()(float const &v, std::string &str) {
        vstd::to_string(v, str);
    }
};
template<>
struct PrintValue<int> {
    void operator()(int const &v, std::string &str) {
        vstd::to_string(v, str);
    }
};
template<>
struct PrintValue<uint> {
    void operator()(uint const &v, std::string &str) {
        vstd::to_string(v, str);
    }
};

template<>
struct PrintValue<bool> {
    void operator()(bool const &v, std::string &str) {
        if (v)
            str << "true";
        else
            str << "false";
    }
};
template<typename EleType, uint64 N>
struct PrintValue<Vector<EleType, N>> {
    using T = Vector<EleType, N>;
    void PureRun(T const &v, std::string &varName) {
        for (uint64 i = 0; i < N; ++i) {
            vstd::to_string(v[i], varName);
            varName += ',';
        }
        auto &&last = varName.end() - 1;
        if (*last == ',')
            varName.erase(last);
    }
    void operator()(T const &v, std::string &varName) {
        if constexpr (N > 1) {
            if constexpr (std::is_same_v<EleType, float>) {
                varName << "_float";
            } else if constexpr (std::is_same_v<EleType, uint>) {
                varName << "_uint";
            } else if constexpr (std::is_same_v<EleType, int>) {
                varName << "_int";
            } else if constexpr (std::is_same_v<EleType, bool>) {
                varName << "_bool";
            }
            vstd::to_string(N, varName);
            varName << '(';
            PureRun(v, varName);
            varName << ')';
        } else {
            PureRun(v, varName);     
        }
    }

};

template<uint64 N>
struct PrintValue<Matrix<N>> {
    using T = Matrix<N>;
    using EleType = float;
    void operator()(T const &v, std::string &varName) {
        varName << "_float";
        auto ss = vstd::to_string(N);
        varName << ss << 'x' << ss << '(';
        PrintValue<Vector<EleType, N>> vecPrinter;
        for (uint64 i = 0; i < N; ++i) {
            vecPrinter.PureRun(v[i], varName);
            varName += ',';
        }
        auto &&last = varName.end() - 1;
        if (*last == ',')
            varName.erase(last);
        varName << ')';
    }
};

template<>
struct PrintValue<LiteralExpr::MetaValue> {
    void operator()(const LiteralExpr::MetaValue &s, std::string &varName) const noexcept {
        // TODO...
    }
};

void StringExprVisitor::visit(const LiteralExpr *expr) {
    LiteralExpr::Value const &value = expr->value();
    std::visit([&](auto &&value) -> void {
        using T = std::remove_cvref_t<decltype(value)>;
        PrintValue<T> prt;
        prt(value, (str));
    },
               expr->value());
}
void StringExprVisitor::visit(const CallExpr *expr) {
    CodegenUtility::GetFunctionName(expr, str)(*this);
}
void StringExprVisitor::visit(const CastExpr *expr) {
    str << '(';
    CodegenUtility::GetTypeName(*expr->type(), str);
    str << ')';
    StringExprVisitor vis(str);
    expr->expression()->accept(vis);
}

void StringExprVisitor::visit(const ConstantExpr *expr) {
    CodegenUtility::GetConstName(expr->data(), str);
}
StringExprVisitor::StringExprVisitor(std::string &str)
    : str(str) {
}
StringExprVisitor::~StringExprVisitor() {}

void StringStateVisitor::visit(const BreakStmt *state) {
    str << "break;\n";
}
void StringStateVisitor::visit(const ContinueStmt *state) {
    str << "continue;\n";
}
void StringStateVisitor::visit(const ReturnStmt *state) {
    if (state->expression()) {
        StringExprVisitor vis(str);
        str << "return ";
        state->expression()->accept(vis);
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
}
void StringStateVisitor::visit(const IfStmt *state) {
    stmtCount = std::numeric_limits<uint64>::max();
    str << "if(";
    StringExprVisitor vis(str);
    state->condition()->accept(vis);
    str << "){\n";
    state->true_branch()->accept(*this);
    str << "}else{\n";
    state->false_branch()->accept(*this);
    str << "}\n";
}
void StringStateVisitor::visit(const LoopStmt *state) {
    stmtCount = std::numeric_limits<uint64>::max();
    str << "while(1){\n";
    str << "}\n";
}
void StringStateVisitor::visit(const ExprStmt *state) {
    stmtCount++;
    StringExprVisitor vis(str);
    state->expression()->accept(vis);
}
void StringStateVisitor::visit(const SwitchStmt *state) {
   stmtCount++;
    str << "switch(";
    StringExprVisitor vis(str);
    state->expression()->accept(vis);
    str << "){\n";
    state->body()->accept(*this);
    str << "}\n";
}
void StringStateVisitor::visit(const SwitchCaseStmt *state) {
    stmtCount++;
    str << "case ";
    StringExprVisitor vis(str);
    state->expression()->accept(vis);
    str << ":{\n";
    state->body()->accept(*this);
    str << "}\n";
}
void StringStateVisitor::visit(const SwitchDefaultStmt *state) {
    stmtCount++;
    str << "default:{\n";
    state->body()->accept(*this);
    str << "}\n";
}
void StringStateVisitor::visit(const AssignStmt *state) {
    stmtCount++;
    StringExprVisitor vis(str);
    state->lhs()->accept(vis);
    switch (state->op()) {
        case AssignOp::ASSIGN:
            str << '=';
            break;
        case AssignOp::ADD_ASSIGN:
            str << "+=";
            break;
        case AssignOp::SUB_ASSIGN:
            str << "-=";
            break;
        case AssignOp::MUL_ASSIGN:
            str << "*=";
            break;
        case AssignOp::DIV_ASSIGN:
            str << "/=";
            break;
        case AssignOp::MOD_ASSIGN:
            str << "%=";
            break;
        case AssignOp::BIT_AND_ASSIGN:
            str << "&=";
            break;
        case AssignOp::BIT_OR_ASSIGN:
            str << "|=";
            break;
        case AssignOp::BIT_XOR_ASSIGN:
            str << "^=";
            break;
        case AssignOp::SHL_ASSIGN:
            str << "<<=";
            break;
        case AssignOp::SHR_ASSIGN:
            str << ">>=";
            break;
    }
    state->rhs()->accept(vis);
    str << ";\n";
}
void StringStateVisitor::visit(const ForStmt *state) {
    stmtCount = std::numeric_limits<uint64>::max();
    str << "for(";
    StringExprVisitor vis(str);
    state->variable()->accept(vis);
    str << ';';
    state->condition()->accept(vis);
    str << ';';
    state->step()->accept(vis);
    str << "){\n";
    str << "}\n";
}
StringStateVisitor::StringStateVisitor(std::string &str)
    : str(str) {
}
void StringStateVisitor::visit(const MetaStmt *stmt) {
    str << "{\n";
    for (auto &&v : stmt->variables()) {
        CodegenUtility::GetTypeName(*v.type(), str);
        str << ' ';
        CodegenUtility::GetVariableName(v, str);
        str << ";\n";
    }
    stmt->scope()->accept(*this);
    str << "}\n";
}
StringStateVisitor::~StringStateVisitor() = default;
}// namespace lc::ispc
