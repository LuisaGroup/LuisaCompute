#pragma vengine_package ispc_vsproject

#include <backends/ispc/runtime/ispc_codegen.h>

namespace lc::ispc {
struct VariableInfo {
    size_t startUsingStmt = 0;
    size_t endUsingStmt = 0;
    size_t scope = 0;
    //vstd::vector<vstd::vector<std
};
class CodegenGlobalData {
    /* vstd::HashMap<uint64, VariableInfo> varInfos;
    size_t stmtIndex = 0;*/
};
template<typename T>
struct VisitStruct {
    T t;
    VisitStruct(T t)
        : t(t) {
        t->BeforeVisit();
    }
    ~VisitStruct() {
        t->AfterVisit();
    }
};
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
    expr->operand()->accept(*this);
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
    if (t && (t->is_buffer() || t->is_vector() || t->is_array()))
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
void StringExprVisitor::visit(const LiteralExpr *expr) {

    LiteralExpr::Value const &value = expr->value();
    luisa::visit([&](auto &&value) -> void {
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
    expr->expression()->accept(*this);
}

void StringExprVisitor::visit(const ConstantExpr *expr) {

    CodegenUtility::GetConstName(expr->data(), str);
}
StringExprVisitor::StringExprVisitor(luisa::string &str,
                                     CodegenGlobalData *ptr)
    : str(str), VisitorBase(ptr) {
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
        StringExprVisitor vis(str, ptr);
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
    StringExprVisitor vis(str, ptr);
    state->condition()->accept(vis);
    str << ")";
    state->true_branch()->accept(*this);
    if (!state->false_branch()->statements().empty()) {
        str << "else";
        state->false_branch()->accept(*this);
    }
}
void StringStateVisitor::visit(const LoopStmt *state) {

    stmtCount = std::numeric_limits<uint64>::max();
    str << "while(1)";
    state->body()->accept(*this);
}
void StringStateVisitor::visit(const ExprStmt *state) {

    stmtCount++;
    StringExprVisitor vis(str, ptr);
    state->expression()->accept(vis);
    str << ";\n";
}
void StringStateVisitor::visit(const SwitchStmt *state) {

    stmtCount++;
    str << "switch(";
    StringExprVisitor vis(str, ptr);
    state->expression()->accept(vis);
    str << ")";
    state->body()->accept(*this);
}
void StringStateVisitor::visit(const SwitchCaseStmt *state) {

    stmtCount++;
    str << "case ";
    StringExprVisitor vis(str, ptr);
    state->expression()->accept(vis);
    str << ":";
    state->body()->accept(*this);
}
void StringStateVisitor::visit(const SwitchDefaultStmt *state) {

    stmtCount++;
    str << "default:";
    state->body()->accept(*this);
}
void StringStateVisitor::visit(const AssignStmt *state) {

    stmtCount++;
    StringExprVisitor vis(str, ptr);
    state->lhs()->accept(vis);
    str << '=';
    state->rhs()->accept(vis);
    str << ";\n";
}
void StringStateVisitor::visit(const ForStmt *state) {

    stmtCount = std::numeric_limits<uint64>::max();
    str << "for(";
    StringExprVisitor vis(str, ptr);
    //    state->variable()->accept(vis);
    str << ';';
    state->condition()->accept(vis);
    str << ';';
    state->variable()->accept(vis);
    str << "+=";
    state->step()->accept(vis);
    str << ")";
    state->body()->accept(*this);
}
StringStateVisitor::StringStateVisitor(luisa::string &str, CodegenGlobalData *ptr)
    : str(str), VisitorBase(ptr) {
}
void StringStateVisitor::visit(const MetaStmt *stmt) {

    str << "{ // begin: " << stmt->info() << "\n";
    for (auto &&v : stmt->variables()) {
        CodegenUtility::GetTypeName(*v.type(), str);
        str << ' ';
        CodegenUtility::GetVariableName(v, str);
        str << ";\n";
    }
    stmt->scope()->accept(*this);
    str << "} // end: " << stmt->info() << "\n";
}
StringStateVisitor::~StringStateVisitor() = default;
}// namespace lc::ispc
