//
// Created by Mike Smith on 2021/3/6.
//

#include <string_view>

#include <core/hash.h>
#include <ast/type_registry.h>
#include <ast/constant_data.h>
#include <compile/cpp_codegen.h>

namespace luisa::compute::compile {

void CppCodegen::visit(const UnaryExpr *expr) {
    switch (expr->op()) {
        case UnaryOp::PLUS: _scratch << "+"; break;
        case UnaryOp::MINUS: _scratch << "-"; break;
        case UnaryOp::NOT: _scratch << "!"; break;
        case UnaryOp::BIT_NOT: _scratch << "~"; break;
        default: break;
    }
    expr->operand()->accept(*this);
}

void CppCodegen::visit(const BinaryExpr *expr) {
    _scratch << "(";
    expr->lhs()->accept(*this);
    switch (expr->op()) {
        case BinaryOp::ADD: _scratch << " + "; break;
        case BinaryOp::SUB: _scratch << " - "; break;
        case BinaryOp::MUL: _scratch << " * "; break;
        case BinaryOp::DIV: _scratch << " / "; break;
        case BinaryOp::MOD: _scratch << " % "; break;
        case BinaryOp::BIT_AND: _scratch << " & "; break;
        case BinaryOp::BIT_OR: _scratch << " | "; break;
        case BinaryOp::BIT_XOR: _scratch << " ^ "; break;
        case BinaryOp::SHL: _scratch << " << "; break;
        case BinaryOp::SHR: _scratch << " >> "; break;
        case BinaryOp::AND: _scratch << " && "; break;
        case BinaryOp::OR: _scratch << " || "; break;
        case BinaryOp::LESS: _scratch << " < "; break;
        case BinaryOp::GREATER: _scratch << " > "; break;
        case BinaryOp::LESS_EQUAL: _scratch << " <= "; break;
        case BinaryOp::GREATER_EQUAL: _scratch << " >= "; break;
        case BinaryOp::EQUAL: _scratch << " == "; break;
        case BinaryOp::NOT_EQUAL: _scratch << " != "; break;
    }
    expr->rhs()->accept(*this);
    _scratch << ")";
}

void CppCodegen::visit(const MemberExpr *expr) {
    expr->self()->accept(*this);
    if (expr->self()->type()->is_vector()) {
        static constexpr std::string_view xyzw[]{".x", ".y", ".z", ".w"};
        _scratch << xyzw[expr->member_index()];
    } else {
        _scratch << ".m" << expr->member_index();
    }
}

void CppCodegen::visit(const AccessExpr *expr) {
    expr->range()->accept(*this);
    _scratch << "[";
    expr->index()->accept(*this);
    _scratch << "]";
}

namespace detail {

class LiteralPrinter {

private:
    Codegen::Scratch &_s;

public:
    explicit LiteralPrinter(Codegen::Scratch &s) noexcept : _s{s} {}
    void operator()(bool v) const noexcept { _s << v; }
    void operator()(float v) const noexcept {
        if (std::isnan(v)) { LUISA_ERROR_WITH_LOCATION("Encountered with NaN."); }
        if (std::isinf(v)) {
            _s << (v < 0.0f ? "(1.0f/-0.0f)" : "1.0f/+0.0f");
        } else {
            _s << v << "f";
        }
    }
    void operator()(int v) const noexcept { _s << v; }
    void operator()(uint v) const noexcept { _s << v << "u"; }

    template<typename T, size_t N>
    void operator()(Vector<T, N> v) const noexcept {
        auto t = Type::of<T>();
        _s << t->description() << N << "(";
        for (auto i = 0u; i < N; i++) {
            (*this)(v[i]);
            _s << ", ";
        }
        _s.pop_back();
        _s.pop_back();
        _s << ")";
    }

    void operator()(float3x3 m) const noexcept {
        _s << "float3x3(";
        for (auto col = 0u; col < 3u; col++) {
            for (auto row = 0u; row < 3u; row++) {
                (*this)(m[col][row]);
                _s << ", ";
            }
        }
        _s.pop_back();
        _s.pop_back();
        _s << ")";
    }

    void operator()(float4x4 m) const noexcept {
        _s << "float4x4(";
        for (auto col = 0u; col < 4u; col++) {
            for (auto row = 0u; row < 4u; row++) {
                (*this)(m[col][row]);
            }
        }
        _s << ")";
    }
};

}// namespace detail

void CppCodegen::visit(const LiteralExpr *expr) {
    std::visit(detail::LiteralPrinter{_scratch}, expr->value());
}

void CppCodegen::visit(const RefExpr *expr) {
    _emit_variable_name(expr->variable());
}

void CppCodegen::visit(const CallExpr *expr) {
    _scratch << expr->name() << "(";
    if (!expr->arguments().empty()) {
        for (auto arg : expr->arguments()) {
            arg->accept(*this);
            _scratch << ", ";
        }
        _scratch.pop_back();
        _scratch.pop_back();
    }
    _scratch << ")";
}

void CppCodegen::visit(const CastExpr *expr) {
    switch (expr->op()) {
        case CastOp::STATIC:
            _scratch << "static_cast<";
            _emit_type_name(expr->type());
            _scratch << ">(";
            break;
        case CastOp::BITWISE:
            _scratch << "as<";
            _emit_type_name(expr->type());
            _scratch << ">(";
            break;
        default: break;
    }
    expr->expression()->accept(*this);
    _scratch << ")";
}

void CppCodegen::visit(const BreakStmt *stmt) {
    _scratch << "break;";
}

void CppCodegen::visit(const ContinueStmt *stmt) {
    _scratch << "continue;";
}

void CppCodegen::visit(const ReturnStmt *stmt) {
    _scratch << "return";
    if (auto expr = stmt->expression(); expr != nullptr) {
        _scratch << " ";
        expr->accept(*this);
    }
    _scratch << ";";
}

void CppCodegen::visit(const ScopeStmt *stmt) {
    _scratch << "{";
    _emit_statements(stmt->statements());
    _scratch << "}";
}

void CppCodegen::visit(const DeclareStmt *stmt) {
    _emit_variable_decl(stmt->variable());
    _scratch << "{";
    if (!stmt->initializer().empty()) {
        for (auto init : stmt->initializer()) {
            init->accept(*this);
            _scratch << ", ";
        }
        _scratch.pop_back();
        _scratch.pop_back();
    }
    _scratch << "};";
}

void CppCodegen::visit(const IfStmt *stmt) {
    _scratch << "if (";
    stmt->condition()->accept(*this);
    _scratch << ") ";
    stmt->true_branch()->accept(*this);
    if (auto fb = stmt->false_branch(); fb != nullptr && !fb->statements().empty()) {
        _scratch << " else ";
        if (auto elif = dynamic_cast<const IfStmt *>(fb->statements().front());
            fb->statements().size() == 1u && elif != nullptr) {
            elif->accept(*this);
        } else {
            fb->accept(*this);
        }
    }
}

void CppCodegen::visit(const WhileStmt *stmt) {
    _scratch << "while (";
    stmt->condition()->accept(*this);
    _scratch << ") ";
    stmt->body()->accept(*this);
}

void CppCodegen::visit(const ExprStmt *stmt) {
    stmt->expression()->accept(*this);
    _scratch << ";";
}

void CppCodegen::visit(const SwitchStmt *stmt) {
    _scratch << "switch (";
    stmt->expression()->accept(*this);
    _scratch << ") ";
    stmt->body()->accept(*this);
}

void CppCodegen::visit(const SwitchCaseStmt *stmt) {
    _scratch << "case ";
    stmt->expression()->accept(*this);
    _scratch << ": ";
    stmt->body()->accept(*this);
}

void CppCodegen::visit(const SwitchDefaultStmt *stmt) {
    _scratch << "default: ";
    stmt->body()->accept(*this);
}

void CppCodegen::visit(const AssignStmt *stmt) {
    stmt->lhs()->accept(*this);
    switch (stmt->op()) {
        case AssignOp::ASSIGN: _scratch << " = "; break;
        case AssignOp::ADD_ASSIGN: _scratch << " += "; break;
        case AssignOp::SUB_ASSIGN: _scratch << " -= "; break;
        case AssignOp::MUL_ASSIGN: _scratch << " *= "; break;
        case AssignOp::DIV_ASSIGN: _scratch << " /= "; break;
        case AssignOp::MOD_ASSIGN: _scratch << " %= "; break;
        case AssignOp::BIT_AND_ASSIGN: _scratch << " &= "; break;
        case AssignOp::BIT_OR_ASSIGN: _scratch << " |= "; break;
        case AssignOp::BIT_XOR_ASSIGN: _scratch << " ^= "; break;
        case AssignOp::SHL_ASSIGN: _scratch << " <<= "; break;
        case AssignOp::SHR_ASSIGN: _scratch << " >>= "; break;
        default: break;
    }
    stmt->rhs()->accept(*this);
    _scratch << ";";
}

void CppCodegen::emit(Function f) {
    _emit_type_decl();
    _emit_function(f);
}

void CppCodegen::_emit_function(Function f) noexcept {

    if (auto iter = std::find(
            _generated_functions.cbegin(), _generated_functions.cend(), f.uid());
        iter != _generated_functions.cend()) { return; }
    _generated_functions.emplace_back(f.uid());

    for (auto callable : f.custom_callables()) {
        _emit_function(Function::callable(callable));
    }

    _function = f;
    _indent = 0u;

    // constants
    if (!f.constants().empty()) {
        for (auto c : f.constants()) { _emit_constant(c); }
        _scratch << "\n";
    }

    // signature
    if (f.tag() == Function::Tag::KERNEL) {
        _scratch << "__kernel__ void kernel_" << f.uid();
    } else if (f.tag() == Function::Tag::CALLABLE) {
        _scratch << "__device__ ";
        if (f.return_type() != nullptr) {
            _emit_type_name(f.return_type());
        } else {
            _scratch << "void";
        }
        _scratch << " custom_" << f.uid();
    } else {
        LUISA_ERROR_WITH_LOCATION("Invalid function type.");
    }
    // argument list
    _scratch << "(";
    for (auto arg : f.arguments()) {
        _scratch << "\n    ";
        if (f.tag() == Function::Tag::KERNEL && arg.tag() == Variable::Tag::UNIFORM) {
            _scratch << "__uniform__ ";
        }
        _emit_variable_decl(arg);
        if (arg.tag() == Variable::Tag::BUFFER) {
            _scratch << " ";
            _emit_access_attribute(arg);
        }
        _scratch << ",";
    }
    for (auto image : f.captured_images()) {
        _scratch << "\n    ";
        _emit_variable_decl(image.variable);
        _scratch << " ";
        _emit_access_attribute(image.variable);
        _scratch << ",";
    }
    for (auto buffer : f.captured_buffers()) {
        _scratch << "\n    ";
        _emit_variable_decl(buffer.variable);
        _scratch << " ";
        _emit_access_attribute(buffer.variable);
        _scratch << ",";
    }
    for (auto builtin : f.builtin_variables()) {
        _scratch << "\n    ";
        _emit_variable_decl(builtin);
        switch (builtin.tag()) {
            case Variable::Tag::THREAD_ID: _scratch << " [[thread_id]]"; break;
            case Variable::Tag::BLOCK_ID: _scratch << " [[block_id]]"; break;
            case Variable::Tag::DISPATCH_ID: _scratch << " [[dispatch_id]]"; break;
            case Variable::Tag::LAUNCH_SIZE: _scratch << " [[launch_size]]"; break;
            default: break;
        }
        _scratch << ",";
    }
    if (!f.arguments().empty()
        || !f.captured_images().empty()
        || !f.captured_buffers().empty()
        || !f.builtin_variables().empty()) {
        _scratch.pop_back();
    }
    _scratch << ") {";
    if (!f.shared_variables().empty()) {
        _scratch << "\n";
        for (auto s : f.shared_variables()) {
            _scratch << "\n  ";
            _emit_variable_decl(s);
            _scratch << ";";
        }
        _scratch << "\n";
    }
    _emit_statements(f.body()->statements());
    _scratch << "}\n\n";
}

void CppCodegen::_emit_variable_name(Variable v) noexcept {
    switch (v.tag()) {
        case Variable::Tag::LOCAL: _scratch << "v" << v.uid(); break;
        case Variable::Tag::SHARED: _scratch << "s" << v.uid(); break;
        case Variable::Tag::UNIFORM: _scratch << "u" << v.uid(); break;
        case Variable::Tag::BUFFER: _scratch << "b" << v.uid(); break;
        case Variable::Tag::IMAGE: _scratch << "i" << v.uid(); break;
        case Variable::Tag::THREAD_ID: _scratch << "tid"; break;
        case Variable::Tag::BLOCK_ID: _scratch << "bid"; break;
        case Variable::Tag::DISPATCH_ID: _scratch << "did"; break;
        case Variable::Tag::LAUNCH_SIZE: _scratch << "ls"; break;
        default: break;
    }
}

void CppCodegen::_emit_type_decl() noexcept {
    Type::traverse(*this);
}

void CppCodegen::visit(const Type *type) noexcept {
    if (type->is_structure()) {
        _scratch << "struct alignas(" << type->alignment() << ") ";
        _emit_type_name(type);
        _scratch << " {\n";
        for (auto i = 0u; i < type->members().size(); i++) {
            _scratch << "  ";
            _emit_type_name(type->members()[i]);
            _scratch << " m" << i << ";\n";
        }
        _scratch << "};\n\n";
    }
}

void CppCodegen::_emit_type_name(const Type *type) noexcept {

    switch (type->tag()) {
        case Type::Tag::BOOL: _scratch << "bool"; break;
        case Type::Tag::FLOAT: _scratch << "float"; break;
        case Type::Tag::INT: _scratch << "int"; break;
        case Type::Tag::UINT: _scratch << "uint"; break;
        case Type::Tag::VECTOR:
            _emit_type_name(type->element());
            _scratch << type->dimension();
            break;
        case Type::Tag::MATRIX:
            _scratch << "float"
                     << type->dimension()
                     << "x"
                     << type->dimension();
            break;
        case Type::Tag::ARRAY:
            _scratch << "array<";
            _emit_type_name(type->element());
            _scratch << ", ";
            _scratch << type->dimension() << ">";
            break;
        case Type::Tag::ATOMIC:
            _scratch << "atomic<";
            _emit_type_name(type->element());
            _scratch << ">";
            break;
        case Type::Tag::STRUCTURE:
            _scratch << "S" << hash_to_string(type->hash());
            break;
    }
}

void CppCodegen::_emit_variable_decl(Variable v) noexcept {
    switch (v.tag()) {
        case Variable::Tag::BUFFER:
            _scratch << "__device__ ";
            _emit_type_name(v.type());
            _scratch << " *";
            break;
        case Variable::Tag::IMAGE:
            _scratch << "image<float, ";
            if (auto usage = _function.variable_usage(v.uid());
                usage == Variable::Usage::READ_WRITE) {
                _scratch << "access::read_write> ";
            } else if (usage == Variable::Usage::WRITE) {
                _scratch << "access::write> ";
            } else {
                _scratch << "access::read> ";
            }
            _emit_variable_name(v);
            break;
        case Variable::Tag::UNIFORM:
        case Variable::Tag::THREAD_ID:
        case Variable::Tag::BLOCK_ID:
        case Variable::Tag::DISPATCH_ID:
        case Variable::Tag::LAUNCH_SIZE:
        case Variable::Tag::LOCAL:
            _emit_type_name(v.type());
            _scratch << " ";
            break;
        case Variable::Tag::SHARED:
            _scratch << "__shared__ ";
            _emit_type_name(v.type());
            _scratch << " ";
            break;
    }
    _emit_variable_name(v);
}

void CppCodegen::_emit_indent() noexcept {
    for (auto i = 0u; i < _indent; i++) { _scratch << "  "; }
}

void CppCodegen::_emit_statements(std::span<const Statement *const> stmts) noexcept {
    _indent++;
    for (auto s : stmts) {
        _scratch << "\n";
        _emit_indent();
        s->accept(*this);
    }
    _indent--;
    if (!stmts.empty()) {
        _scratch << "\n";
        _emit_indent();
    }
}

void CppCodegen::_emit_constant(Function::ConstantBinding c) noexcept {

    if (std::find(_generated_constants.cbegin(),
                  _generated_constants.cend(), c.hash)
        != _generated_constants.cend()) { return; }
    _generated_constants.emplace_back(c.hash);

    _scratch << "__constant__ ";
    _emit_type_name(c.type);
    _scratch << " c" << hash_to_string(c.hash) << "{";
    auto count = c.type->dimension();
    static constexpr auto wrap = 16u;
    using namespace std::string_view_literals;
    std::visit(
        [count, this](auto ptr) {
            detail::LiteralPrinter print{_scratch};
            for (auto i = 0u; i < count; i++) {
                if (count > wrap && i % wrap == 0u) { _scratch << "\n    "; }
                print(ptr[i]);
                _scratch << ", ";
            }
        },
        ConstantData::view(c.hash));
    if (count > 0u) {
        _scratch.pop_back();
        _scratch.pop_back();
    }
    _scratch << "};\n";
}

void CppCodegen::visit(const ConstantExpr *expr) {
    _scratch << "c" << hash_to_string(expr->hash());
}

void CppCodegen::visit(const ForStmt *stmt) {

    _scratch << "for (";

    if (auto init = stmt->initialization(); init != nullptr) {
        init->accept(*this);
    } else {
        _scratch << ";";
    }

    if (auto cond = stmt->condition(); cond != nullptr) {
        _scratch << " ";
        cond->accept(*this);
    }
    _scratch << ";";

    if (auto update = stmt->update(); update != nullptr) {
        _scratch << " ";
        update->accept(*this);
        if (_scratch.back() == ';') { _scratch.pop_back(); }
    }

    _scratch << ") ";
    stmt->body()->accept(*this);
}

void CppCodegen::_emit_access_attribute(Variable v) noexcept {
    switch (_function.variable_usage(v.uid())) {
        case Variable::Usage::NONE: _scratch << "[[access::none]]"; break;
        case Variable::Usage::READ: _scratch << "[[access::read]]"; break;
        case Variable::Usage::WRITE: _scratch << "[[access::write]]"; break;
        case Variable::Usage::READ_WRITE: _scratch << "[[access::read_write]]"; break;
        default: _scratch << "[[access::unknown]]"; break;
    }
}

}// namespace luisa::compute::compile
