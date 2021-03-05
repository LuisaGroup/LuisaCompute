//
// Created by Mike Smith on 2021/3/5.
//

#include <ast/codegen.h>

namespace luisa::compute {

Codegen::Scratch::Scratch() noexcept { _buffer.reserve(4096u); }
std::string_view Codegen::Scratch::view() const noexcept { return _buffer; }

namespace detail {

template<typename T>
[[nodiscard]] inline auto to_string(T x) noexcept {
    static thread_local std::array<char, 128u> s;
    auto [_, size] = fmt::format_to_n(s.begin(), s.size(), FMT_STRING("{}"), x);
    return std::string_view{s.data(), size};
}

}// namespace detail

Codegen::Scratch &Codegen::Scratch::operator<<(bool x) noexcept {
    return *this << detail::to_string(x);
}

Codegen::Scratch &Codegen::Scratch::operator<<(float x) noexcept {
    return *this << detail::to_string(x);
}

Codegen::Scratch &Codegen::Scratch::operator<<(int x) noexcept {
    return *this << detail::to_string(x);
}

Codegen::Scratch &Codegen::Scratch::operator<<(uint x) noexcept {
    return *this << detail::to_string(x);
}

Codegen::Scratch &Codegen::Scratch::operator<<(std::string_view s) noexcept {
    _buffer.append(s);
    return *this;
}

Codegen::Scratch &Codegen::Scratch::operator<<(int64_t x) noexcept {
    return *this << detail::to_string(x);
}

Codegen::Scratch &Codegen::Scratch::operator<<(uint64_t x) noexcept {
    return *this << detail::to_string(x);
}

Codegen::Scratch &Codegen::Scratch::operator<<(size_t x) noexcept {
    return *this << detail::to_string(x);
}

Codegen::Scratch &Codegen::Scratch::operator<<(const char *s) noexcept {
    return *this << std::string_view{s};
}

Codegen::Scratch &Codegen::Scratch::operator<<(const std::string &s) noexcept {
    return *this << std::string_view{s};
}

void Codegen::Scratch::clear() noexcept { _buffer.clear(); }

void CppCodegen::visit(const UnaryExpr *expr) {
    switch (expr->op()) {
        case UnaryOp::PLUS:
            break;
        case UnaryOp::MINUS:
            break;
        case UnaryOp::NOT:
            break;
        case UnaryOp::BIT_NOT:
            break;
        default: break;
    }
}

void CppCodegen::visit(const BinaryExpr *expr) {
}

void CppCodegen::visit(const MemberExpr *expr) {
}

void CppCodegen::visit(const AccessExpr *expr) {
}

void CppCodegen::visit(const LiteralExpr *expr) {
}

void CppCodegen::visit(const RefExpr *expr) {
}

void CppCodegen::visit(const CallExpr *expr) {
}

void CppCodegen::visit(const CastExpr *expr) {
}

void CppCodegen::visit(const BreakStmt *stmt) {
}

void CppCodegen::visit(const ContinueStmt *stmt) {
}

void CppCodegen::visit(const ReturnStmt *stmt) {
}

void CppCodegen::visit(const ScopeStmt *stmt) {
}

void CppCodegen::visit(const DeclareStmt *stmt) {
}

void CppCodegen::visit(const IfStmt *stmt) {
}

void CppCodegen::visit(const WhileStmt *stmt) {
}

void CppCodegen::visit(const ExprStmt *stmt) {
}

void CppCodegen::visit(const SwitchStmt *stmt) {
}

void CppCodegen::visit(const SwitchCaseStmt *stmt) {
}

void CppCodegen::visit(const SwitchDefaultStmt *stmt) {
}

void CppCodegen::visit(const AssignStmt *stmt) {
}

void CppCodegen::emit(Function f) { _emit_function(f); }

void CppCodegen::_emit_function(Function f) noexcept {
    _emit_type_declarations();
}

void CppCodegen::_emit_variable_name(Variable v) noexcept {
    switch (v.tag()) {
        case Variable::Tag::LOCAL: _scratch << "v" << v.uid(); break;
        case Variable::Tag::SHARED: _scratch << "s" << v.uid(); break;
        case Variable::Tag::CONSTANT: _scratch << "c" << v.uid(); break;
        case Variable::Tag::UNIFORM: _scratch << "u" << v.uid(); break;
        case Variable::Tag::BUFFER: _scratch << "b" << v.uid(); break;
        case Variable::Tag::TEXTURE: _scratch << "t" << v.uid(); break;
        case Variable::Tag::THREAD_ID: _scratch << "tid"; break;
        case Variable::Tag::BLOCK_ID: _scratch << "bid"; break;
        case Variable::Tag::DISPATCH_ID: _scratch << "did"; break;
        default: break;
    }
}

void CppCodegen::_emit_type_declarations() noexcept {
    Type::traverse(*this);
}

void CppCodegen::visit(const Type *type) const noexcept {
    if (type->is_structure()) {
        _scratch << "struct alignas(" << type->alignment() << ") S" << type->hash() << " {\n";
        for (auto i = 0u; i < type->members().size(); i++) {
            if (auto m = type->members()[i]; m->is_structure()) {
                _scratch << "  S" << m->hash();
            } else {
                _scratch << "  " << m->description();
            }
            _scratch << " m" << i << ";\n";
        }
        _scratch << "};\n";
    }
}

}// namespace luisa::compute
