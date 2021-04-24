//
// Created by Mike Smith on 2021/3/25.
//

#import <span>

#import <core/hash.h>
#import <ast/type_registry.h>
#import <ast/variable.h>
#import <backends/metal/metal_codegen.h>

namespace luisa::compute::metal {

void MetalCodegen::visit(const UnaryExpr *expr) {
    switch (expr->op()) {
        case UnaryOp::PLUS: _scratch << "+"; break;
        case UnaryOp::MINUS: _scratch << "-"; break;
        case UnaryOp::NOT: _scratch << "!"; break;
        case UnaryOp::BIT_NOT: _scratch << "~"; break;
    }
    expr->operand()->accept(*this);
}

void MetalCodegen::visit(const BinaryExpr *expr) {
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

void MetalCodegen::visit(const MemberExpr *expr) {
    expr->self()->accept(*this);
    if (expr->self()->type()->is_vector()) {
        static constexpr std::string_view xyzw[]{".x", ".y", ".z", ".w"};
        _scratch << xyzw[expr->member_index()];
    } else {
        _scratch << ".m" << expr->member_index();
    }
}

void MetalCodegen::visit(const AccessExpr *expr) {
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
            _s << (v < 0.0f ? "(-INFINITY)" : "(+INFINITY)");
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

void MetalCodegen::visit(const LiteralExpr *expr) {
    std::visit(detail::LiteralPrinter{_scratch}, expr->value());
}

void MetalCodegen::visit(const RefExpr *expr) {
    auto v = expr->variable();
    if (_function.tag() == Function::Tag::KERNEL
        && (v.tag() == Variable::Tag::UNIFORM
            || v.tag() == Variable::Tag::BUFFER
            || v.tag() == Variable::Tag::TEXTURE
            || v.tag() == Variable::Tag::LAUNCH_SIZE)) {
        _scratch << "arg.";
    }
    _emit_variable_name(expr->variable());
}

void MetalCodegen::visit(const CallExpr *expr) {
    switch (expr->op()) {
        case CallOp::CUSTOM: _scratch << "custom_" << expr->uid(); break;
        case CallOp::ALL: _scratch << "all"; break;
        case CallOp::ANY: _scratch << "any"; break;
        case CallOp::NONE: _scratch << "none"; break;
        case CallOp::TEXTURE_READ: _scratch << "texture_read"; break;
        case CallOp::TEXTURE_WRITE:
            _scratch << "texture_write";
            break;
            // TODO...
#define LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL(type, tag)               \
    case CallOp::MAKE_##tag##2: _scratch << "make_" #type "2"; break; \
    case CallOp::MAKE_##tag##3: _scratch << "make_" #type "3"; break; \
    case CallOp::MAKE_##tag##4: _scratch << "make_" #type "4"; break;
            LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL(bool, BOOL)
            LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL(int, INT)
            LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL(uint, UINT)
            LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL(float, FLOAT)
#undef LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL
    }
    _scratch << "(";
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

void MetalCodegen::visit(const CastExpr *expr) {
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
    }
    expr->expression()->accept(*this);
    _scratch << ")";
}

void MetalCodegen::visit(const BreakStmt *stmt) {
    _scratch << "break;";
}

void MetalCodegen::visit(const ContinueStmt *stmt) {
    _scratch << "continue;";
}

void MetalCodegen::visit(const ReturnStmt *stmt) {
    _scratch << "return";
    if (auto expr = stmt->expression(); expr != nullptr) {
        _scratch << " ";
        expr->accept(*this);
    }
    _scratch << ";";
}

void MetalCodegen::visit(const ScopeStmt *stmt) {
    _scratch << "{";
    _emit_statements(stmt->statements());
    _scratch << "}";
}

void MetalCodegen::visit(const DeclareStmt *stmt) {
    auto v = stmt->variable();
    _scratch << "auto ";
    _emit_variable_name(v);
    _scratch << " = ";
    _emit_type_name(v.type());
    _scratch << (v.type()->is_structure() ? "{" : "(");
    if (!stmt->initializer().empty()) {
        for (auto init : stmt->initializer()) {
            init->accept(*this);
            _scratch << ", ";
        }
        _scratch.pop_back();
        _scratch.pop_back();
    }
    _scratch << (v.type()->is_structure() ? "};" : ");");
}

void MetalCodegen::visit(const IfStmt *stmt) {
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

void MetalCodegen::visit(const WhileStmt *stmt) {
    _scratch << "while (";
    stmt->condition()->accept(*this);
    _scratch << ") ";
    stmt->body()->accept(*this);
}

void MetalCodegen::visit(const ExprStmt *stmt) {
    stmt->expression()->accept(*this);
    _scratch << ";";
}

void MetalCodegen::visit(const SwitchStmt *stmt) {
    _scratch << "switch (";
    stmt->expression()->accept(*this);
    _scratch << ") ";
    stmt->body()->accept(*this);
}

void MetalCodegen::visit(const SwitchCaseStmt *stmt) {
    _scratch << "case ";
    stmt->expression()->accept(*this);
    _scratch << ": ";
    stmt->body()->accept(*this);
}

void MetalCodegen::visit(const SwitchDefaultStmt *stmt) {
    _scratch << "default: ";
    stmt->body()->accept(*this);
}

void MetalCodegen::visit(const AssignStmt *stmt) {
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
    }
    stmt->rhs()->accept(*this);
    _scratch << ";";
}

void MetalCodegen::emit(Function f) {
    _scratch << R"(#include <metal_stdlib>

using namespace metal;

template<typename T, access a>
struct Image {
  texture2d<T, a> handle;
  uint2 offset;
};

template<typename T, access a>
struct Volume {
  texture3d<T, a> handle;
  uint3 offset;
};

[[nodiscard]] auto srgb_to_linear(float4 c) {
  auto srgb = saturate(c.rgb);
  auto rgb = select(
    srgb * (1.0f / 12.92f),
    pow((srgb + 0.055f) / 1.055f, 2.4f),
    srgb >= 0.0404482362771082f);
  return float4(rgb, c.a);
}

[[nodiscard]] auto linear_to_srgb(float4 c) {
  auto rgb = saturate(c.rgb);
  auto srgb = select(
    rgb * 12.92f,
    1.055f * pow(rgb, 1.0f / 2.4f) - 0.055f,
    rgb >= 0.00313066844250063f);
  return float4(srgb, c.a);
}

)";
    _emit_type_decl();
    _emit_function(f);
}

void MetalCodegen::_emit_function(Function f) noexcept {

    if (std::find(_generated_functions.cbegin(),
                  _generated_functions.cend(),
                  f.uid())
        != _generated_functions.cend()) { return; }
    _generated_functions.emplace_back(f.uid());

    for (auto intrinsic : f.builtin_callables()) {
        _emit_intrinsic(intrinsic);
    }

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

    if (f.tag() == Function::Tag::KERNEL) {

        // argument buffer
        _scratch << "struct Argument {";
        for (auto buffer : f.captured_buffers()) {
            _scratch << "\n  ";
            _emit_variable_decl(buffer.variable);
            _scratch << ";";
        }
        for (auto image : f.captured_images()) {
            _scratch << "\n  ";
            _emit_variable_decl(image.variable);
            _scratch << ";";
        }
        for (auto arg : f.arguments()) {
            _scratch << "\n  ";
            _emit_variable_decl(arg);
            _scratch << ";";
        }
        _scratch << "\n  const uint3 ls;\n};\n\n";

        // function signature
        _scratch << "[[kernel]] // block_size = ("
                 << f.block_size().x << ", "
                 << f.block_size().y << ", "
                 << f.block_size().z << ")\n"
                 << "void kernel_" << f.uid()
                 << "(\n    device const Argument &arg,";
        for (auto builtin : f.builtin_variables()) {
            if (builtin.tag() != Variable::Tag::LAUNCH_SIZE) {
                _scratch << "\n    ";
                _emit_variable_decl(builtin);
                _scratch << ",";
            }
        }
        _scratch.pop_back();
    } else if (f.tag() == Function::Tag::CALLABLE) {
        if (f.return_type() != nullptr) {
            _emit_type_name(f.return_type());
        } else {
            _scratch << "void";
        }
        _scratch << " custom_" << f.uid() << "(";
        for (auto buffer : f.captured_buffers()) {
            _scratch << "\n    ";
            _emit_variable_decl(buffer.variable);
            _scratch << ",";
        }
        for (auto tex : f.captured_images()) {
            _scratch << "\n    ";
            _emit_variable_decl(tex.variable);
            _scratch << ",";
        }
        for (auto arg : f.arguments()) {
            _scratch << "\n    ";
            _emit_variable_decl(arg);
            _scratch << ",";
        }
        if (!f.arguments().empty()
            || !f.captured_images().empty()
            || !f.captured_buffers().empty()) {
            _scratch.pop_back();
        }
    } else {
        LUISA_ERROR_WITH_LOCATION("Invalid function type.");
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

void MetalCodegen::_emit_variable_name(Variable v) noexcept {
    switch (v.tag()) {
        case Variable::Tag::LOCAL: _scratch << "v" << v.uid(); break;
        case Variable::Tag::SHARED: _scratch << "s" << v.uid(); break;
        case Variable::Tag::UNIFORM: _scratch << "u" << v.uid(); break;
        case Variable::Tag::BUFFER: _scratch << "b" << v.uid(); break;
        case Variable::Tag::TEXTURE: _scratch << "i" << v.uid(); break;
        case Variable::Tag::THREAD_ID: _scratch << "tid"; break;
        case Variable::Tag::BLOCK_ID: _scratch << "bid"; break;
        case Variable::Tag::DISPATCH_ID: _scratch << "did"; break;
        case Variable::Tag::LAUNCH_SIZE: _scratch << "ls"; break;
    }
}

void MetalCodegen::_emit_type_decl() noexcept {
    Type::traverse(*this);
}

void MetalCodegen::visit(const Type *type) noexcept {
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

void MetalCodegen::_emit_type_name(const Type *type) noexcept {

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
            _scratch << "atomic_";
            _emit_type_name(type->element());
            break;
        case Type::Tag::STRUCTURE:
            _scratch << "S" << hash_to_string(type->hash());
            break;
        default:
            break;
    }
}

void MetalCodegen::_emit_variable_decl(Variable v) noexcept {
    switch (v.tag()) {
        case Variable::Tag::BUFFER:
            _scratch << "device ";
            if (_function.variable_usage(v.uid()) == Variable::Usage::READ) {
                _scratch << "const ";
            }
            _emit_type_name(v.type()->element());
            _scratch << " *";
            _emit_variable_name(v);
            break;
        case Variable::Tag::TEXTURE:
            if (auto d = v.type()->dimension(); d == 2u) {
                _scratch << "Image<";
            } else {// d == 3u
                _scratch << "Volume<";
            }
            _emit_type_name(v.type()->element());
            _scratch << ", ";
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
            _scratch << "const ";
            _emit_type_name(v.type());
            _scratch << " ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::THREAD_ID:
            _scratch << "const ";
            _emit_type_name(v.type());
            _scratch << " ";
            _emit_variable_name(v);
            _scratch << " [[thread_position_in_threadgroup]]";
            break;
        case Variable::Tag::BLOCK_ID:
            _scratch << "const ";
            _emit_type_name(v.type());
            _scratch << " ";
            _emit_variable_name(v);
            _scratch << " [[threadgroup_position_in_grid]]";
            break;
        case Variable::Tag::DISPATCH_ID:
            _scratch << "const ";
            _emit_type_name(v.type());
            _scratch << " ";
            _emit_variable_name(v);
            _scratch << " [[thread_position_in_grid]]";
            break;
        case Variable::Tag::LAUNCH_SIZE:
            _scratch << "const ";
            _emit_type_name(v.type());
            _scratch << " ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::LOCAL:
            if (_function.variable_usage(v.uid()) == Variable::Usage::READ) {
                _scratch << "const ";
            }
            _emit_type_name(v.type());
            _scratch << " ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::SHARED:
            _scratch << "threadgroup ";
            _emit_type_name(v.type());
            _scratch << " ";
            _emit_variable_name(v);
            break;
    }
}

void MetalCodegen::_emit_indent() noexcept {
    for (auto i = 0u; i < _indent; i++) { _scratch << "  "; }
}

void MetalCodegen::_emit_statements(std::span<const Statement *const> stmts) noexcept {
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

void MetalCodegen::_emit_constant(Function::ConstantBinding c) noexcept {

    if (std::find(_generated_constants.cbegin(),
                  _generated_constants.cend(), c.hash)
        != _generated_constants.cend()) { return; }
    _generated_constants.emplace_back(c.hash);

    _scratch << "constant ";
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

void MetalCodegen::visit(const ConstantExpr *expr) {
    _scratch << "c" << hash_to_string(expr->hash());
}

void MetalCodegen::visit(const ForStmt *stmt) {

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

void MetalCodegen::_emit_intrinsic(CallOp intrinsic) noexcept {

    if (std::find(_generated_intrinsics.cbegin(),
                  _generated_intrinsics.cend(),
                  intrinsic)
        != _generated_intrinsics.cend()) { return; }
    _generated_intrinsics.emplace_back(intrinsic);

    switch (intrinsic) {
        case CallOp::CUSTOM:
            LUISA_ERROR_WITH_LOCATION("Invalid built-in function with custom tag.");
        case CallOp::ALL:
        case CallOp::ANY: break;
        case CallOp::NONE:
            _scratch << R"(template<typename T>
[[nodiscard]] auto none(T v) { return !any(v); }

)";
            break;
        case CallOp::TEXTURE_READ:
            _scratch << R"(template<typename T, access a>
[[nodiscard]] auto texture_read(Image<T, a> t, uint2 uv) {
  return t.handle.read(uv + t.offset);
}

template<typename T, access a>
[[nodiscard]] auto texture_read(Volume<T, a> t, uint3 uvw) {
  return t.handle.read(uvw + t.offset);
}

)";
            break;
        case CallOp::TEXTURE_WRITE:
            _scratch << R"(template<typename T, access a, typename Value>
void texture_write(Image<T, a> t, uint2 uv, Value value) {
  t.handle.write(value, uv + t.offset);
}

template<typename T, access a, typename Value>
void texture_write(Volume<T, a> t, uint3 uvw, Value value) {
  t.handle.write(value, uvw + t.offset);
}

)";
            break;
            // TODO...

#define LUISA_METAL_CODEGEN_MAKE_VECTOR_IMPL(type, tag)                                                          \
    case CallOp::MAKE_##tag##2:                                                                                  \
        _scratch << "[[nodiscard]] auto make_" #type "2(" #type " x, " #type " y) {\n"                           \
                    "  return " #type "2{x, y};\n"                                                               \
                    "}\n\n";                                                                                     \
        break;                                                                                                   \
    case CallOp::MAKE_##tag##3:                                                                                  \
        _scratch << "[[nodiscard]] auto make_" #type "3(" #type " x, " #type " y, " #type " z) {\n"              \
                    "  return " #type "3{x, y, z};\n"                                                            \
                    "}\n\n";                                                                                     \
        break;                                                                                                   \
    case CallOp::MAKE_##tag##4:                                                                                  \
        _scratch << "[[nodiscard]] auto make_" #type "4(" #type " x, " #type " y, " #type " z, " #type " w) {\n" \
                    "  return " #type "4{x, y, z, w};\n"                                                         \
                    "}\n\n";                                                                                     \
        break;
            LUISA_METAL_CODEGEN_MAKE_VECTOR_IMPL(bool, BOOL)
            LUISA_METAL_CODEGEN_MAKE_VECTOR_IMPL(int, INT)
            LUISA_METAL_CODEGEN_MAKE_VECTOR_IMPL(uint, UINT)
            LUISA_METAL_CODEGEN_MAKE_VECTOR_IMPL(float, FLOAT)
#undef LUISA_METAL_CODEGEN_MAKE_VECTOR_IMPL
    }
}

}
