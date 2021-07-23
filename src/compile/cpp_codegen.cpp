//
// Created by Mike Smith on 2021/3/6.
//

#include <string_view>

#include <core/hash.h>
#include <ast/type_registry.h>
#include <ast/constant_data.h>
#include <compile/cpp_codegen.h>

namespace luisa::compute {

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
    if (expr->is_swizzle()) {
        static constexpr std::string_view xyzw[]{"x", "y", "z", "w"};
        _scratch << ".";
        for (auto i = 0u; i < expr->swizzle_size(); i++) {
            _scratch << xyzw[expr->swizzle_index(i)];
        }
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
        if (std::isnan(v)) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Encountered with NaN."); }
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

    void operator()(float2x2 m) const noexcept {
        _s << "float3x3(";
        for (auto col = 0u; col < 2u; col++) {
            for (auto row = 0u; row < 2u; row++) {
                (*this)(m[col][row]);
                _s << ", ";
            }
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

    switch (expr->op()) {
        case CallOp::CUSTOM: _scratch << "custom_" << hash_to_string(expr->custom().hash()); break;
        case CallOp::ALL: _scratch << "all"; break;
        case CallOp::ANY: _scratch << "any"; break;
        case CallOp::NONE: _scratch << "none"; break;
        case CallOp::SELECT: _scratch << "select"; break;
        case CallOp::CLAMP: _scratch << "clamp"; break;
        case CallOp::LERP: _scratch << "mix"; break;
        case CallOp::SATURATE: _scratch << "saturate"; break;
        case CallOp::SIGN: _scratch << "sign"; break;
        case CallOp::STEP: _scratch << "step"; break;
        case CallOp::SMOOTHSTEP: _scratch << "smoothstep"; break;
        case CallOp::ABS: _scratch << "abs"; break;
        case CallOp::MIN: _scratch << "min"; break;
        case CallOp::MAX: _scratch << "max"; break;
        case CallOp::CLZ: _scratch << "clz"; break;
        case CallOp::CTZ: _scratch << "ctz"; break;
        case CallOp::POPCOUNT: _scratch << "popcount"; break;
        case CallOp::REVERSE: _scratch << "reverse_bits"; break;
        case CallOp::ISINF: _scratch << "precise::isinf"; break;
        case CallOp::ISNAN: _scratch << "precise::isnan"; break;
        case CallOp::ACOS: _scratch << "acos"; break;
        case CallOp::ACOSH: _scratch << "acosh"; break;
        case CallOp::ASIN: _scratch << "asin"; break;
        case CallOp::ASINH: _scratch << "asinh"; break;
        case CallOp::ATAN: _scratch << "atan"; break;
        case CallOp::ATAN2: _scratch << "atan2"; break;
        case CallOp::ATANH: _scratch << "atanh"; break;
        case CallOp::COS: _scratch << "cos"; break;
        case CallOp::COSH: _scratch << "cosh"; break;
        case CallOp::SIN: _scratch << "sin"; break;
        case CallOp::SINH: _scratch << "sinh"; break;
        case CallOp::TAN: _scratch << "tan"; break;
        case CallOp::TANH: _scratch << "tanh"; break;
        case CallOp::EXP: _scratch << "exp"; break;
        case CallOp::EXP2: _scratch << "exp2"; break;
        case CallOp::EXP10: _scratch << "exp10"; break;
        case CallOp::LOG: _scratch << "log"; break;
        case CallOp::LOG2: _scratch << "log2"; break;
        case CallOp::LOG10: _scratch << "log10"; break;
        case CallOp::POW: _scratch << "pow"; break;
        case CallOp::SQRT: _scratch << "sqrt"; break;
        case CallOp::RSQRT: _scratch << "rsqrt"; break;
        case CallOp::CEIL: _scratch << "ceil"; break;
        case CallOp::FLOOR: _scratch << "floor"; break;
        case CallOp::FRACT: _scratch << "fract"; break;
        case CallOp::TRUNC: _scratch << "trunc"; break;
        case CallOp::ROUND: _scratch << "round"; break;
        case CallOp::MOD: _scratch << "mod"; break;
        case CallOp::FMOD: _scratch << "fmod"; break;
        case CallOp::DEGREES: _scratch << "degrees"; break;
        case CallOp::RADIANS: _scratch << "radians"; break;
        case CallOp::FMA: _scratch << "fma"; break;
        case CallOp::COPYSIGN: _scratch << "copysign"; break;
        case CallOp::CROSS: _scratch << "cross"; break;
        case CallOp::DOT: _scratch << "dot"; break;
        case CallOp::DISTANCE: _scratch << "distance"; break;
        case CallOp::DISTANCE_SQUARED: _scratch << "distance_squared"; break;
        case CallOp::LENGTH: _scratch << "length"; break;
        case CallOp::LENGTH_SQUARED: _scratch << "length_squared"; break;
        case CallOp::NORMALIZE: _scratch << "normalize"; break;
        case CallOp::FACEFORWARD: _scratch << "faceforward"; break;
        case CallOp::DETERMINANT: _scratch << "determinant"; break;
        case CallOp::TRANSPOSE: _scratch << "transpose"; break;
        case CallOp::INVERSE: _scratch << "inverse"; break;
        case CallOp::GROUP_MEMORY_BARRIER: _scratch << "group_memory_barrier"; break;
        case CallOp::DEVICE_MEMORY_BARRIER: _scratch << "device_memory_barrier"; break;
        case CallOp::ALL_MEMORY_BARRIER: _scratch << "all_memory_barrier"; break;
        case CallOp::ATOMIC_LOAD: _scratch << "atomic_load"; break;
        case CallOp::ATOMIC_STORE: _scratch << "atomic_store"; break;
        case CallOp::ATOMIC_EXCHANGE: _scratch << "atomic_exchange"; break;
        case CallOp::ATOMIC_COMPARE_EXCHANGE: _scratch << "atomic_compare_exchange"; break;
        case CallOp::ATOMIC_FETCH_ADD: _scratch << "atomic_fetch_add"; break;
        case CallOp::ATOMIC_FETCH_SUB: _scratch << "atomic_fetch_sub"; break;
        case CallOp::ATOMIC_FETCH_AND: _scratch << "atomic_fetch_and"; break;
        case CallOp::ATOMIC_FETCH_OR: _scratch << "atomic_fetch_or"; break;
        case CallOp::ATOMIC_FETCH_XOR: _scratch << "atomic_fetch_xor"; break;
        case CallOp::ATOMIC_FETCH_MIN: _scratch << "atomic_fetch_min"; break;
        case CallOp::ATOMIC_FETCH_MAX: _scratch << "atomic_fetch_max"; break;
        case CallOp::TEXTURE_READ: _scratch << "texture_read"; break;
        case CallOp::TEXTURE_WRITE: _scratch << "texture_write"; break;
        case CallOp::TEXTURE_HEAP_SAMPLE2D: _scratch << "texture_heap_sample2d"; break;
        case CallOp::TEXTURE_HEAP_SAMPLE2D_LEVEL: _scratch << "texture_heap_sample2d_level"; break;
        case CallOp::TEXTURE_HEAP_SAMPLE2D_GRAD: _scratch << "texture_heap_sample2d_grad"; break;
        case CallOp::TEXTURE_HEAP_SAMPLE3D: _scratch << "texture_heap_sample3d"; break;
        case CallOp::TEXTURE_HEAP_SAMPLE3D_LEVEL: _scratch << "texture_heap_sample3d_level"; break;
        case CallOp::TEXTURE_HEAP_SAMPLE3D_GRAD: _scratch << "texture_heap_sample3d_grad"; break;
        case CallOp::TEXTURE_HEAP_READ2D: _scratch << "texture_heap_read2d"; break;
        case CallOp::TEXTURE_HEAP_READ3D: _scratch << "texture_heap_read3d"; break;
        case CallOp::TEXTURE_HEAP_READ2D_LEVEL: _scratch << "texture_heap_read2d_level"; break;
        case CallOp::TEXTURE_HEAP_READ3D_LEVEL: _scratch << "texture_heap_read3d_level"; break;
        case CallOp::TEXTURE_HEAP_SIZE2D: _scratch << "texture_heap_size2d"; break;
        case CallOp::TEXTURE_HEAP_SIZE3D: _scratch << "texture_heap_size3d"; break;
        case CallOp::TEXTURE_HEAP_SIZE2D_LEVEL: _scratch << "texture_heap_size2d_level"; break;
        case CallOp::TEXTURE_HEAP_SIZE3D_LEVEL: _scratch << "texture_heap_size3d_level"; break;
#define LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL(type, tag)       \
    case CallOp::MAKE_##tag##2: _scratch << #type "2"; break; \
    case CallOp::MAKE_##tag##3: _scratch << #type "3"; break; \
    case CallOp::MAKE_##tag##4: _scratch << #type "4"; break;
            LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL(bool, BOOL)
            LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL(int, INT)
            LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL(uint, UINT)
            LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL(float, FLOAT)
#undef LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL
        case CallOp::MAKE_FLOAT2X2: _scratch << "float2x2"; break;
        case CallOp::MAKE_FLOAT3X3: _scratch << "float3x3"; break;
        case CallOp::MAKE_FLOAT4X4: _scratch << "float4x4"; break;
        case CallOp::TRACE_CLOSEST: break;
        case CallOp::TRACE_ANY: break;
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

void CppCodegen::visit(const BreakStmt *) {
    _scratch << "break;";
}

void CppCodegen::visit(const ContinueStmt *) {
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

    if (auto iter = std::find(_generated_functions.cbegin(), _generated_functions.cend(), f);
        iter != _generated_functions.cend()) { return; }
    _generated_functions.emplace_back(f);

    for (auto callable : f.custom_callables()) { _emit_function(callable); }

    _function = f;
    _indent = 0u;

    // constants
    if (!f.constants().empty()) {
        for (auto c : f.constants()) { _emit_constant(c); }
        _scratch << "\n";
    }

    // signature
    if (f.tag() == Function::Tag::KERNEL) {
        _scratch << "__kernel__ void kernel_" << hash_to_string(f.hash());
    } else if (f.tag() == Function::Tag::CALLABLE) {
        _scratch << "__device__ ";
        if (f.return_type() != nullptr) {
            _emit_type_name(f.return_type());
        } else {
            _scratch << "void";
        }
        _scratch << " custom_" << hash_to_string(f.hash());
    } else [[unlikely]] {
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
    for (auto image : f.captured_textures()) {
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
            case Variable::Tag::DISPATCH_SIZE: _scratch << " [[launch_size]]"; break;
            default: break;
        }
        _scratch << ",";
    }
    if (!f.arguments().empty()
        || !f.captured_textures().empty()
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
        case Variable::Tag::TEXTURE: _scratch << "i" << v.uid(); break;
        case Variable::Tag::HEAP: _scratch << "h" << v.uid(); break;
        case Variable::Tag::ACCEL: _scratch << "a" << v.uid(); break;
        case Variable::Tag::THREAD_ID: _scratch << "tid"; break;
        case Variable::Tag::BLOCK_ID: _scratch << "bid"; break;
        case Variable::Tag::DISPATCH_ID: _scratch << "did"; break;
        case Variable::Tag::DISPATCH_SIZE: _scratch << "ls"; break;
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
        case Type::Tag::STRUCTURE:
            _scratch << "S" << hash_to_string(type->hash());
            break;
        default: break;
    }
}

void CppCodegen::_emit_variable_decl(Variable v) noexcept {
    switch (v.tag()) {
        case Variable::Tag::BUFFER:
            _scratch << "__device__ ";
            _emit_type_name(v.type()->element());
            _scratch << " *";
            break;
        case Variable::Tag::TEXTURE:
            _scratch << "image<";
            _emit_type_name(v.type()->element());
            _scratch << ", ";
            if (auto usage = _function.variable_usage(v.uid());
                usage == Usage::READ_WRITE) {
                _scratch << "access::read_write> ";
            } else if (usage == Usage::WRITE) {
                _scratch << "access::write> ";
            } else if (usage == Usage::READ) {
                _scratch << "access::read> ";
            }
            _emit_variable_name(v);
            break;
        case Variable::Tag::HEAP:
            _scratch << "heap ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::ACCEL:
            _scratch << "accel ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::UNIFORM:
        case Variable::Tag::THREAD_ID:
        case Variable::Tag::BLOCK_ID:
        case Variable::Tag::DISPATCH_ID:
        case Variable::Tag::DISPATCH_SIZE:
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
                  _generated_constants.cend(), c.data.hash())
        != _generated_constants.cend()) { return; }
    _generated_constants.emplace_back(c.data.hash());

    _scratch << "__constant__ ";
    _emit_type_name(c.type);
    _scratch << " c" << hash_to_string(c.data.hash()) << "{";
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
        c.data.view());
    if (count > 0u) {
        _scratch.pop_back();
        _scratch.pop_back();
    }
    _scratch << "};\n";
}

void CppCodegen::visit(const ConstantExpr *expr) {
    _scratch << "c" << hash_to_string(expr->data().hash());
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
        case Usage::NONE: _scratch << "[[access::none]]"; break;
        case Usage::READ: _scratch << "[[access::read]]"; break;
        case Usage::WRITE: _scratch << "[[access::write]]"; break;
        case Usage::READ_WRITE: _scratch << "[[access::read_write]]"; break;
        default: _scratch << "[[access::unknown]]"; break;
    }
}

}// namespace luisa::compute
