#pragma vengine_package ispc_vsproject

#include "ispc_codegen.h"
namespace lc::ispc {
void StringExprVisitor::visit(const UnaryExpr *expr) {

    switch (expr->op()) {
        case UnaryOp::PLUS://+x
            str += '+';
            break;
        case UnaryOp::MINUS://-x
            str += '-';
            break;
        case UnaryOp::NOT://!x
            str += '!';
            break;
        case UnaryOp::BIT_NOT://~x
            str += '~';
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
    StringExprVisitor vis(str);
    if (IsMulFuncCall()) {
        str += "mul("sv;
        expr->rhs()->accept(vis);//Reverse matrix
        str += ',';
        expr->lhs()->accept(vis);
        str += ')';

    } else {

        expr->lhs()->accept(vis);
        switch (expr->op()) {
            case BinaryOp::ADD:
                str += '+';
                break;
            case BinaryOp::SUB:
                str += '-';
                break;
            case BinaryOp::MUL:
                str += '*';
                break;
            case BinaryOp::DIV:
                str += '/';
                break;
            case BinaryOp::MOD:
                str += '%';
                break;
            case BinaryOp::BIT_AND:
                str += '&';
                break;
            case BinaryOp::BIT_OR:
                str += '|';
                break;
            case BinaryOp::BIT_XOR:
                str += '^';
                break;
            case BinaryOp::SHL:
                str += "<<"sv;
                break;
            case BinaryOp::SHR:
                str += ">>"sv;
                break;
            case BinaryOp::AND:
                str += "&&"sv;
                break;
            case BinaryOp::OR:
                str += "||"sv;
                break;
            case BinaryOp::LESS:
                str += '<';
                break;
            case BinaryOp::GREATER:
                str += '>';
                break;
            case BinaryOp::LESS_EQUAL:
                str += "<="sv;
                break;
            case BinaryOp::GREATER_EQUAL:
                str += ">="sv;
                break;
            case BinaryOp::EQUAL:
                str += "=="sv;
                break;
            case BinaryOp::NOT_EQUAL:
                str += "!="sv;
                break;
        }
        expr->rhs()->accept(vis);
    }
}
void StringExprVisitor::visit(const MemberExpr *expr) {
    expr->self()->accept(*this);
    if (expr->is_swizzle()) {
        char const *xyzw = "xyzw";
        str << '.';
        for (auto i : vstd::range(expr->swizzle_size())) {
            str << xyzw[expr->swizzle_index(i)];
        }
    } else {
        str += ".v"sv;
        vstd::to_string(expr->member_index(), (str));
    }
}
void StringExprVisitor::visit(const AccessExpr *expr) {
    expr->range()->accept(*this);
    str += '[';
    StringExprVisitor vis(str);
    expr->index()->accept(vis);
    str += ']';
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
template<typename EleType, size_t N>
struct PrintValue<Vector<EleType, N>> {
    using T = typename Vector<EleType, N>;
    void operator()(T const &v, std::string &varName) {
        for (size_t i = 0; i < N; ++i) {
            vstd::to_string(v[i], varName);
            varName += ',';
        }
        auto &&last = varName.end() - 1;
        if (*last == ',')
            varName.erase(last);
    }
};

template<size_t N>
struct PrintValue<Matrix<N>> {
    using T = Matrix<N>;
    using EleType = float;
    void operator()(T const &v, std::string &varName) {
        PrintValue<Vector<EleType, N>> vecPrinter;
        for (size_t i = 0; i < N; ++i) {
            vecPrinter(v[i], varName);
            varName += ',';
        }
        auto &&last = varName.end() - 1;
        if (*last == ',')
            varName.erase(last);
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
    CodegenUtility::GetFunctionName(expr, str);
    str += '(';
    auto &&args = expr->arguments();
    StringExprVisitor vis(str);
    for (auto &&i : args) {
        i->accept(vis);
        str += ',';
    }
    (str)[str.size() - 1] = ')';
}
void StringExprVisitor::visit(const CastExpr *expr) {
    str += '(';
    CodegenUtility::GetTypeName(*expr->type(), str);
    str += ')';
    StringExprVisitor vis(str);
    expr->expression()->accept(vis);
}
void StringExprVisitor::visit(const ConstantExpr *expr) {
    str << "uniform const ";
    auto data = expr->data();
    auto &&view = data.view();
    auto typeName = CodegenUtility::GetBasicTypeName(view.index());
    str << typeName << ' ' << 'c';
    vstd::to_string(constCount, str);
    constCount++;
    str << "[]={";
    std::visit(
        [&](auto &&arr) {
            for (auto const &ele : arr) {
                PrintValue<std::remove_cvref_t<std::remove_cvref_t<decltype(arr)>::element_type>> prt;
                prt(ele, str);
            }
        },
        view);
    auto last = str.end() - 1;
    if (*last == ',')
        *last = '}';
    else
        str << '}';
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
    StringExprVisitor vis(str);
    str << "return ";
    state->expression()->accept(vis);
    str << ";\n";
}
void StringStateVisitor::visit(const ScopeStmt *state) {
    str << "{\n";
    for (auto&& i : state->statements()) {
        i->accept(*this);
    }
    str << "}\n";
}
void StringStateVisitor::visit(const CommentStmt *state) {

}
void StringStateVisitor::visit(const IfStmt *state) {
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
    str << "while(1){\n";
    str << "}\n";
}
void StringStateVisitor::visit(const ExprStmt *state) {
    StringExprVisitor vis(str);
    state->expression()->accept(vis);
}
void StringStateVisitor::visit(const SwitchStmt *state) {
    str << "switch(";
    StringExprVisitor vis(str);
    state->expression()->accept(vis);
    str << "){\n";
    state->body()->accept(*this);
    str << "}\n";
}
void StringStateVisitor::visit(const SwitchCaseStmt *state) {
    str << "case ";
    StringExprVisitor vis(str);
    state->expression()->accept(vis);
    str << ":{\n";
    state->body()->accept(*this);
    str << "}\n";
}
void StringStateVisitor::visit(const SwitchDefaultStmt *state) {
    str << "default:{\n";
    state->body()->accept(*this);
    str << "}\n";
}
void StringStateVisitor::visit(const AssignStmt *state) {
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
void StringStateVisitor::visit(const ForStmt * state) {
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
    CodegenUtility::ClearStructType();
}
StringStateVisitor::~StringStateVisitor() {}
}// namespace lc::ispc