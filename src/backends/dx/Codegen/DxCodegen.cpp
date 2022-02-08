#pragma vengine_package vengine_directx

#include <Codegen/DxCodegen.h>
#include <Codegen/StructGenerator.h>
#include <Codegen/StructVariableTracker.h>
namespace toolhub::directx {
static bool IsFloat3x3(Type const &t) {
    return t.is_matrix() && t.dimension() == 3;
}
void StringStateVisitor::InsertString() {
    if (preprocStr.empty()) return;
    size_t afterSize = str.size() - lastIdx;
    str.resize(str.size() + preprocStr.size());
    auto beforePtr = str.data() + lastIdx;
    if (afterSize > 0) {
        memmove(
            beforePtr + preprocStr.size(),
            beforePtr,
            afterSize);
    }
    memcpy(beforePtr, preprocStr.data(), preprocStr.size());
    preprocStr.clear();
    lastIdx += preprocStr.size();
}
void StringStateVisitor::SetStub() {
    lastIdx = str.size();
}

StringStateVisitor::Tracker::Tracker(StringStateVisitor *self)
    : self(self) {
    self->SetStub();
}
StringStateVisitor::Tracker::~Tracker() {
    self->InsertString();
}
void StringStateVisitor::visit(const UnaryExpr *expr) {

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
void StringStateVisitor::visit(const BinaryExpr *expr) {

    auto IsMulFuncCall = [&]() -> bool {
        if (expr->op() == BinaryOp::MUL) {
            if ((expr->lhs()->type()->is_matrix() && (!expr->rhs()->type()->is_scalar())) || (expr->rhs()->type()->is_matrix() && (!expr->lhs()->type()->is_scalar()))) {
                return true;
            }
        }
        return false;
    };
    if (IsMulFuncCall()) {
        if (IsFloat3x3(*expr->lhs()->type()) || IsFloat3x3(*expr->rhs()->type())) {
            str << "mul("sv;
        } else {
            str << "FMul("sv;
        }
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
void StringStateVisitor::visit(const MemberExpr *expr) {

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
        auto &&structVar = selfStruct->GetVariable(expr->member_index());
        if (structVar.boolOffset != StructVariable::OFFSET_NPOS) {
            auto tmpName = CodegenUtility::GetNewTempVarName();
            auto realTmpName = CodegenUtility::GetTracker()->CreateTempVar(
                CodegenUtility::GetScope(),
                preprocStr,
                curStr,
                structVar.name,
                tmpName,
                AssignSetter::IsAssigning());
            str << realTmpName << '.' << vstd::string_view(xyzw + structVar.boolOffset, structVar.boolVecDim);

        } else {
            str << curStr << '.' << structVar.name;
        }
    }
}
void StringStateVisitor::visit(const AccessExpr *expr) {

    expr->range()->accept(*this);
    auto t = expr->range()->type();
    if (t && (t->is_buffer() || t->is_vector()))
        str << '[';
    else
        str << ".v[";
    expr->index()->accept(*this);
    str << ']';
}
void StringStateVisitor::visit(const RefExpr *expr) {

    Variable v = expr->variable();
    vstd::string tempStr;
    CodegenUtility::GetVariableName(v, tempStr);
    if (v.type()->tag() == Type::Tag::STRUCTURE) {
        CodegenUtility::GetTracker()->ClearTempVar(
            CodegenUtility::GetScope(),
            preprocStr,
            tempStr);
    }
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
    //TODO: bool & bool vector
    str << '(';
    CodegenUtility::GetTypeName(*expr->type(), str, Usage::READ);
    str << ')';
    expr->expression()->accept(*this);
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
    Tracker t(this);
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
    CodegenUtility::AddScope(1);
    InsertString();
    for (auto &&i : state->statements()) {
        i->accept(*this);
    }
    CodegenUtility::GetTracker()->RemoveStack(str);
    InsertString();
    CodegenUtility::AddScope(-1);
    str << "}\n";
}
void StringStateVisitor::visit(const CommentStmt *state) {
}
void StringStateVisitor::visit(const IfStmt *state) {
    SetStub();
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
    SetStub();
    str << "while(true)";
    state->body()->accept(*this);
}
void StringStateVisitor::visit(const ExprStmt *state) {
    Tracker t(this);
    state->expression()->accept(*this);
    str << ";\n";
}
void StringStateVisitor::visit(const SwitchStmt *state) {
    SetStub();
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
    Tracker t(this);
    {
        AssignSetter setter;
        state->lhs()->accept(*this);
        str << '=';
    }
    state->rhs()->accept(*this);
    str << ";\n";
}
void StringStateVisitor::visit(const ForStmt *state) {
    SetStub();
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
    : str(str), f(f) {
}
void StringStateVisitor::visit(const MetaStmt *stmt) {
    for (auto &&v : stmt->variables()) {
        CodegenUtility::GetTypeName(*v.type(), str, f.variable_usage(v.uid()));
        str << ' ';
        CodegenUtility::GetVariableName(v, str);
        str << ";\n";
    }
    SetStub();
    stmt->scope()->accept(*this);
}
StringStateVisitor::~StringStateVisitor() = default;
}// namespace toolhub::directx
