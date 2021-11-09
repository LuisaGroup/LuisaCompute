//
// Created by Mike on 2021/11/8.
//

#include <string_view>

#include <core/hash.h>
#include <ast/type_registry.h>
#include <ast/constant_data.h>
#include <ast/function_builder.h>
#include <backends/cuda/cuda_codegen.h>

namespace luisa::compute::cuda {

void CUDACodegen::visit(const UnaryExpr *expr) {
    switch (expr->op()) {
        case UnaryOp::PLUS: _scratch << "+"; break;
        case UnaryOp::MINUS: _scratch << "-"; break;
        case UnaryOp::NOT: _scratch << "!"; break;
        case UnaryOp::BIT_NOT: _scratch << "~"; break;
        default: break;
    }
    expr->operand()->accept(*this);
}

void CUDACodegen::visit(const BinaryExpr *expr) {
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

void CUDACodegen::visit(const MemberExpr *expr) {
    if (expr->is_swizzle()) {
        static constexpr std::string_view xyzw[]{"x", "y", "z", "w"};
        if (auto ss = expr->swizzle_size(); ss == 1u) {
            expr->self()->accept(*this);
            _scratch << ".";
            _scratch << xyzw[expr->swizzle_index(0)];
        } else {
            _scratch << "lc_make_";
            auto elem = expr->type()->element();
            switch (elem->tag()) {
                case Type::Tag::BOOL: _scratch << "bool"; break;
                case Type::Tag::INT: _scratch << "int"; break;
                case Type::Tag::UINT: _scratch << "uint"; break;
                case Type::Tag::FLOAT: _scratch << "float"; break;
                default: LUISA_ERROR_WITH_LOCATION(
                    "Invalid vector element type: {}.",
                    elem->description());
            }
            _scratch << ss << "(";
            for (auto i = 0u; i < ss; i++) {
                expr->self()->accept(*this);
                _scratch << "." << xyzw[expr->swizzle_index(i)] << ", ";
            }
            _scratch.pop_back();
            _scratch.pop_back();
            _scratch << ")";
        }
    } else {
        expr->self()->accept(*this);
        _scratch << ".m" << expr->member_index();
    }
}

void CUDACodegen::visit(const AccessExpr *expr) {
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
        _s << "lc_make_" << t->description() << N << "(";
        for (auto i = 0u; i < N; i++) {
            (*this)(v[i]);
            _s << ", ";
        }
        _s.pop_back();
        _s.pop_back();
        _s << ")";
    }

    void operator()(float2x2 m) const noexcept {
        _s << "lc_make_float2x2(";
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
        _s << "lc_make_float3x3(";
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
        _s << "lc_make_float4x4(";
        for (auto col = 0u; col < 4u; col++) {
            for (auto row = 0u; row < 4u; row++) {
                (*this)(m[col][row]);
            }
        }
        _s << ")";
    }

    void operator()(const LiteralExpr::MetaValue &s) const noexcept {
        // TODO...
    }
};

}// namespace detail

void CUDACodegen::visit(const LiteralExpr *expr) {
    std::visit(detail::LiteralPrinter{_scratch}, expr->value());
}

void CUDACodegen::visit(const RefExpr *expr) {
    _emit_variable_name(expr->variable());
}

void CUDACodegen::visit(const CallExpr *expr) {

    switch (expr->op()) {
        case CallOp::CUSTOM: _scratch << "custom_" << hash_to_string(expr->custom().hash()); break;
        case CallOp::ALL: _scratch << "lc_all"; break;
        case CallOp::ANY: _scratch << "lc_any"; break;
        case CallOp::SELECT: _scratch << "lc_select"; break;
        case CallOp::CLAMP: _scratch << "lc_clamp"; break;
        case CallOp::LERP: _scratch << "lc_lerp"; break;
        case CallOp::STEP: _scratch << "lc_step"; break;
        case CallOp::ABS: _scratch << "lc_abs"; break;
        case CallOp::MIN: _scratch << "lc_min"; break;
        case CallOp::MAX: _scratch << "lc_max"; break;
        case CallOp::CLZ: _scratch << "lc_clz"; break;
        case CallOp::CTZ: _scratch << "lc_ctz"; break;
        case CallOp::POPCOUNT: _scratch << "lc_popcount"; break;
        case CallOp::REVERSE: _scratch << "lc_reverse"; break;
        case CallOp::ISINF: _scratch << "lc_isinf"; break;
        case CallOp::ISNAN: _scratch << "lc_isnan"; break;
        case CallOp::ACOS: _scratch << "lc_acos"; break;
        case CallOp::ACOSH: _scratch << "lc_acosh"; break;
        case CallOp::ASIN: _scratch << "lc_asin"; break;
        case CallOp::ASINH: _scratch << "lc_asinh"; break;
        case CallOp::ATAN: _scratch << "lc_atan"; break;
        case CallOp::ATAN2: _scratch << "lc_atan2"; break;
        case CallOp::ATANH: _scratch << "lc_atanh"; break;
        case CallOp::COS: _scratch << "lc_cos"; break;
        case CallOp::COSH: _scratch << "lc_cosh"; break;
        case CallOp::SIN: _scratch << "lc_sin"; break;
        case CallOp::SINH: _scratch << "lc_sinh"; break;
        case CallOp::TAN: _scratch << "lc_tan"; break;
        case CallOp::TANH: _scratch << "lc_tanh"; break;
        case CallOp::EXP: _scratch << "lc_exp"; break;
        case CallOp::EXP2: _scratch << "lc_exp2"; break;
        case CallOp::EXP10: _scratch << "lc_exp10"; break;
        case CallOp::LOG: _scratch << "lc_log"; break;
        case CallOp::LOG2: _scratch << "lc_log2"; break;
        case CallOp::LOG10: _scratch << "lc_log10"; break;
        case CallOp::POW: _scratch << "lc_pow"; break;
        case CallOp::SQRT: _scratch << "lc_sqrt"; break;
        case CallOp::RSQRT: _scratch << "lc_rsqrt"; break;
        case CallOp::CEIL: _scratch << "lc_ceil"; break;
        case CallOp::FLOOR: _scratch << "lc_floor"; break;
        case CallOp::FRACT: _scratch << "lc_fract"; break;
        case CallOp::TRUNC: _scratch << "lc_trunc"; break;
        case CallOp::ROUND: _scratch << "lc_round"; break;
        case CallOp::FMA: _scratch << "lc_fma"; break;
        case CallOp::COPYSIGN: _scratch << "lc_copysign"; break;
        case CallOp::CROSS: _scratch << "lc_cross"; break;
        case CallOp::DOT: _scratch << "lc_dot"; break;
        case CallOp::LENGTH: _scratch << "lc_length"; break;
        case CallOp::LENGTH_SQUARED: _scratch << "lc_length_squared"; break;
        case CallOp::NORMALIZE: _scratch << "lc_normalize"; break;
        case CallOp::FACEFORWARD: _scratch << "lc_faceforward"; break;
        case CallOp::DETERMINANT: _scratch << "lc_determinant"; break;
        case CallOp::TRANSPOSE: _scratch << "lc_transpose"; break;
        case CallOp::INVERSE: _scratch << "lc_inverse"; break;
        case CallOp::SYNCHRONIZE_BLOCK:
            _scratch << "__syncthreads";
            break;
            // TODO: atomics
        case CallOp::ATOMIC_EXCHANGE: _scratch << "atomic_exchange"; break;
        case CallOp::ATOMIC_COMPARE_EXCHANGE: _scratch << "atomic_compare_exchange"; break;
        case CallOp::ATOMIC_FETCH_ADD: _scratch << "atomic_fetch_add"; break;
        case CallOp::ATOMIC_FETCH_SUB: _scratch << "atomic_fetch_sub"; break;
        case CallOp::ATOMIC_FETCH_AND: _scratch << "atomic_fetch_and"; break;
        case CallOp::ATOMIC_FETCH_OR: _scratch << "atomic_fetch_or"; break;
        case CallOp::ATOMIC_FETCH_XOR: _scratch << "atomic_fetch_xor"; break;
        case CallOp::ATOMIC_FETCH_MIN: _scratch << "atomic_fetch_min"; break;
        case CallOp::ATOMIC_FETCH_MAX: _scratch << "atomic_fetch_max"; break;
        case CallOp::TEXTURE_READ:
            _scratch << "lc_surf"
                     << expr->arguments().front()->type()->dimension() << "d_read<"
                     << "lc_" << expr->arguments().front()->type()->element()->description() << ">";
            break;
        case CallOp::TEXTURE_WRITE:
            _scratch << "lc_surf"
                     << expr->arguments().front()->type()->dimension() << "d_write<"
                     << "lc_" << expr->arguments().front()->type()->element()->description() << ">";
            break;
            // TODO: bindless resources
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE: _scratch << "texture_heap_sample2d"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: _scratch << "texture_heap_sample2d_level"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: _scratch << "texture_heap_sample2d_grad"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE: _scratch << "texture_heap_sample3d"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: _scratch << "texture_heap_sample3d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: _scratch << "texture_heap_sample3d_grad"; break;
        case CallOp::BINDLESS_TEXTURE2D_READ: _scratch << "texture_heap_read2d"; break;
        case CallOp::BINDLESS_TEXTURE3D_READ: _scratch << "texture_heap_read3d"; break;
        case CallOp::BINDLESS_TEXTURE2D_READ_LEVEL: _scratch << "texture_heap_read2d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_READ_LEVEL: _scratch << "texture_heap_read3d_level"; break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE: _scratch << "texture_heap_size2d"; break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE: _scratch << "texture_heap_size3d"; break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: _scratch << "texture_heap_size2d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: _scratch << "texture_heap_size3d_level"; break;
        case CallOp::BINDLESS_BUFFER_READ:
            _scratch << "buffer_heap_read<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
#define LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL(type, tag)                     \
    case CallOp::MAKE_##tag##2: _scratch << "lc_make_" << #type "2"; break; \
    case CallOp::MAKE_##tag##3: _scratch << "lc_make_" << #type "3"; break; \
    case CallOp::MAKE_##tag##4: _scratch << "lc_make_" << #type "4"; break;
            LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL(bool, BOOL)
            LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL(int, INT)
            LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL(uint, UINT)
            LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL(float, FLOAT)
#undef LUISA_METAL_CODEGEN_MAKE_VECTOR_CALL
        case CallOp::MAKE_FLOAT2X2: _scratch << "lc_make_float2x2"; break;
        case CallOp::MAKE_FLOAT3X3: _scratch << "lc_make_float3x3"; break;
        case CallOp::MAKE_FLOAT4X4:
            _scratch << "lc_make_float4x4";
            break;
            // TODO: RTX functions
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

void CUDACodegen::visit(const CastExpr *expr) {
    switch (expr->op()) {
        case CastOp::STATIC:
            _scratch << "static_cast<";
            _emit_type_name(expr->type());
            _scratch << ">(";
            break;
        case CastOp::BITWISE:
            _scratch << "reinterpret_cast<";
            _emit_type_name(expr->type());
            _scratch << " &>(";
            break;
        default: break;
    }
    expr->expression()->accept(*this);
    _scratch << ")";
}

void CUDACodegen::visit(const BreakStmt *) {
    _scratch << "break;";
}

void CUDACodegen::visit(const ContinueStmt *) {
    _scratch << "continue;";
}

void CUDACodegen::visit(const ReturnStmt *stmt) {
    _scratch << "return";
    if (auto expr = stmt->expression(); expr != nullptr) {
        _scratch << " ";
        expr->accept(*this);
    }
    _scratch << ";";
}

void CUDACodegen::visit(const ScopeStmt *stmt) {
    _scratch << "{";
    _emit_statements(stmt->statements());
    _scratch << "}";
}

void CUDACodegen::visit(const IfStmt *stmt) {
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

void CUDACodegen::visit(const LoopStmt *stmt) {
    _scratch << "for (;;) ";
    stmt->body()->accept(*this);
}

void CUDACodegen::visit(const ExprStmt *stmt) {
    stmt->expression()->accept(*this);
    _scratch << ";";
}

void CUDACodegen::visit(const SwitchStmt *stmt) {
    _scratch << "switch (";
    stmt->expression()->accept(*this);
    _scratch << ") ";
    stmt->body()->accept(*this);
}

void CUDACodegen::visit(const SwitchCaseStmt *stmt) {
    _scratch << "case ";
    stmt->expression()->accept(*this);
    _scratch << ": ";
    stmt->body()->accept(*this);
}

void CUDACodegen::visit(const SwitchDefaultStmt *stmt) {
    _scratch << "default: ";
    stmt->body()->accept(*this);
}

void CUDACodegen::visit(const AssignStmt *stmt) {
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

void CUDACodegen::emit(Function f) {
    _emit_type_decl();
    _emit_function(f);
}

void CUDACodegen::_emit_function(Function f) noexcept {

    if (auto iter = std::find(_generated_functions.cbegin(), _generated_functions.cend(), f);
        iter != _generated_functions.cend()) { return; }
    _generated_functions.emplace_back(f);

    for (auto &&callable : f.custom_callables()) { _emit_function(callable->function()); }

    _function = f;
    _indent = 0u;

    // constants
    if (!f.constants().empty()) {
        for (auto c : f.constants()) { _emit_constant(c); }
        _scratch << "\n";
    }

    // signature
    if (f.tag() == Function::Tag::KERNEL) {
        _scratch << "extern \"C\" __global__ void kernel_" << hash_to_string(f.hash());
    } else if (f.tag() == Function::Tag::CALLABLE) {
        _scratch << "__device__ inline ";
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
    for (auto buffer : f.captured_buffers()) {
        _scratch << "\n    ";
        _emit_variable_decl(buffer.variable);
        _scratch << ",";
    }
    for (auto image : f.captured_textures()) {
        _scratch << "\n    ";
        _emit_variable_decl(image.variable);
        _scratch << ",";
    }
    // TODO
    for (auto array : f.captured_bindless_arrays()) {
    }
    for (auto accel : f.captured_accels()) {
    }
    _scratch << "(";
    auto any_arg = false;
    for (auto arg : f.arguments()) {
        _scratch << "\n    ";
        _emit_variable_decl(arg);
        _scratch << ",";
        any_arg = true;
    }
    if (f.tag() == Function::Tag::KERNEL) {
        _scratch << "\n    lc_uint3 ls";// launch size
    } else {
        if (any_arg) { _scratch.pop_back(); }
    }
    _scratch << ") {";
    for (auto builtin : f.builtin_variables()) {
        switch (builtin.tag()) {
            case Variable::Tag::THREAD_ID:
                _scratch << "\n  auto tid = lc_make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);";
                break;
            case Variable::Tag::BLOCK_ID:
                _scratch << "\n  auto bid = lc_make_uint3(blockIdx.x, blockIdx.y, blockIdx.z);";
                break;
            case Variable::Tag::DISPATCH_ID:
                _scratch << "\n  auto did = lc_make_uint3("
                         << "\n    blockIdx.x * blockDim.x + threadIdx.x,"
                         << "\n    blockIdx.y * blockDim.y + threadIdx.y,"
                         << "\n    blockIdx.z * blockDim.z + threadIdx.z);";
                break;
            default: break;
        }
    }
    _emit_variable_declarations(f.body());
    _emit_statements(f.body()->scope()->statements());
    _scratch << "}\n\n";
}

void CUDACodegen::_emit_variable_name(Variable v) noexcept {
    switch (v.tag()) {
        case Variable::Tag::LOCAL: _scratch << "v" << v.uid(); break;
        case Variable::Tag::SHARED: _scratch << "s" << v.uid(); break;
        case Variable::Tag::REFERENCE: _scratch << "r" << v.uid(); break;
        case Variable::Tag::BUFFER: _scratch << "b" << v.uid(); break;
        case Variable::Tag::TEXTURE: _scratch << "i" << v.uid(); break;
        case Variable::Tag::BINDLESS_ARRAY: _scratch << "h" << v.uid(); break;
        case Variable::Tag::ACCEL: _scratch << "a" << v.uid(); break;
        case Variable::Tag::THREAD_ID: _scratch << "tid"; break;
        case Variable::Tag::BLOCK_ID: _scratch << "bid"; break;
        case Variable::Tag::DISPATCH_ID: _scratch << "did"; break;
        case Variable::Tag::DISPATCH_SIZE: _scratch << "ls"; break;
    }
}

void CUDACodegen::_emit_type_decl() noexcept {
    Type::traverse(*this);
}

void CUDACodegen::visit(const Type *type) noexcept {
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

void CUDACodegen::_emit_type_name(const Type *type) noexcept {

    switch (type->tag()) {
        case Type::Tag::BOOL: _scratch << "lc_bool"; break;
        case Type::Tag::FLOAT: _scratch << "lc_float"; break;
        case Type::Tag::INT: _scratch << "lc_int"; break;
        case Type::Tag::UINT: _scratch << "lc_uint"; break;
        case Type::Tag::VECTOR:
            _emit_type_name(type->element());
            _scratch << type->dimension();
            break;
        case Type::Tag::MATRIX:
            _scratch << "lc_float"
                     << type->dimension()
                     << "x"
                     << type->dimension();
            break;
        case Type::Tag::ARRAY:
            _scratch << "lc_array<";
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

void CUDACodegen::_emit_variable_decl(Variable v) noexcept {
    auto usage = _function.variable_usage(v.uid());
    auto readonly = usage == Usage::NONE || usage == Usage::READ;
    switch (v.tag()) {
        case Variable::Tag::SHARED:
            _scratch << "__shared__ ";
            _emit_type_name(v.type());
            _scratch << " ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::REFERENCE:
            if (readonly) { _scratch << "const "; }
            _emit_type_name(v.type());
            _scratch << " &";
            _emit_variable_name(v);
            break;
        case Variable::Tag::BUFFER:
            if (readonly) { _scratch << "const "; }
            _emit_type_name(v.type()->element());
            _scratch << " *__restrict__ ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::TEXTURE:
            _scratch << "const ";
            _scratch << "LCSurface ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::BINDLESS_ARRAY:
            if (readonly) { _scratch << "const "; }
            _scratch << "heap ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::ACCEL:
            if (readonly) { _scratch << "const "; }
            _scratch << "accel ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::LOCAL:
            if (readonly) { _scratch << "const "; }
            _emit_type_name(v.type());
            _scratch << " ";
            _emit_variable_name(v);
            break;
        default:
            break;
    }
}

void CUDACodegen::_emit_indent() noexcept {
    for (auto i = 0u; i < _indent; i++) { _scratch << "  "; }
}

void CUDACodegen::_emit_statements(std::span<const Statement *const> stmts) noexcept {
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

void CUDACodegen::_emit_constant(Function::ConstantBinding c) noexcept {

    if (std::find(_generated_constants.cbegin(),
                  _generated_constants.cend(), c.data.hash()) != _generated_constants.cend()) { return; }
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

void CUDACodegen::visit(const ConstantExpr *expr) {
    _scratch << "c" << hash_to_string(expr->data().hash());
}

void CUDACodegen::visit(const ForStmt *stmt) {
    _scratch << "for (; ";
    stmt->condition()->accept(*this);
    _scratch << "; ";
    stmt->variable()->accept(*this);
    _scratch << " += ";
    stmt->step()->accept(*this);
    _scratch << ") ";
    stmt->body()->accept(*this);
}

void CUDACodegen::visit(const CommentStmt *stmt) {
    _scratch << "/* " << stmt->comment() << " */";
}

void CUDACodegen::visit(const MetaStmt *stmt) {
    _scratch << "\n";
    _emit_indent();
    _scratch << "// meta region begin: " << stmt->info();
    for (auto s : stmt->scope()->statements()) {
        _scratch << "\n";
        _emit_indent();
        s->accept(*this);
    }
    _scratch << "\n";
    _emit_indent();
    _scratch << "// meta region end: " << stmt->info() << "\n";
}

void CUDACodegen::_emit_variable_declarations(const MetaStmt *meta) noexcept {
    for (auto v : meta->variables()) {
        _scratch << "\n  ";
        _emit_variable_decl(v);
        _scratch << ";";
    }
    for (auto m : meta->children()) {
        _emit_variable_declarations(m);
    }
}

}// namespace luisa::compute::cuda
