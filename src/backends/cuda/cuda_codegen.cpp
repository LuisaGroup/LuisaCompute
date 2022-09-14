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
            _s << (v < 0.0f ? " __int_as_float(0xff800000)" : " __int_as_float(0x7f800000)");
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
                _s << ", ";
            }
        }
        _s.pop_back();
        _s.pop_back();
        _s << ")";
    }
};

}// namespace detail

void CUDACodegen::visit(const LiteralExpr *expr) {
    luisa::visit(detail::LiteralPrinter{_scratch}, expr->value());
}

void CUDACodegen::visit(const RefExpr *expr) {
    _emit_variable_name(expr->variable());
}

void CUDACodegen::visit(const CallExpr *expr) {

    auto is_atomic = false;
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
        case CallOp::SYNCHRONIZE_BLOCK: _scratch << "__syncthreads"; break;
        case CallOp::ATOMIC_EXCHANGE:
            _scratch << "atomicExch";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_COMPARE_EXCHANGE:
            _scratch << "atomicCAS";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_FETCH_ADD:
            _scratch << "atomicAdd";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_FETCH_SUB:
            _scratch << "atomicSub";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_FETCH_AND:
            _scratch << "atomicAnd";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_FETCH_OR:
            _scratch << "atomicOr";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_FETCH_XOR:
            _scratch << "atomicXor";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_FETCH_MIN:
            _scratch << "atomicMin";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_FETCH_MAX:
            _scratch << "atomicMax";
            is_atomic = true;
            break;
        case CallOp::BUFFER_READ: _scratch << "lc_buffer_read"; break;
        case CallOp::BUFFER_WRITE: _scratch << "lc_buffer_write"; break;
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
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE: _scratch << "lc_bindless_texture_sample2d"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: _scratch << "lc_bindless_texture_sample2d_level"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: _scratch << "lc_bindless_texture_sample2d_grad"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE: _scratch << "lc_bindless_texture_sample3d"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: _scratch << "lc_bindless_texture_sample3d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: _scratch << "lc_bindless_texture_sample3d_grad"; break;
        case CallOp::BINDLESS_TEXTURE2D_READ: _scratch << "lc_bindless_texture_read2d"; break;
        case CallOp::BINDLESS_TEXTURE3D_READ: _scratch << "lc_bindless_texture_read3d"; break;
        case CallOp::BINDLESS_TEXTURE2D_READ_LEVEL: _scratch << "lc_bindless_texture_read2d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_READ_LEVEL: _scratch << "lc_bindless_texture_read3d_level"; break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE: _scratch << "lc_bindless_texture_size2d"; break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE: _scratch << "lc_bindless_texture_size3d"; break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: _scratch << "lc_bindless_texture_size2d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: _scratch << "lc_bindless_texture_size3d_level"; break;
        case CallOp::BINDLESS_BUFFER_READ:
            _scratch << "lc_bindless_buffer_read<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
#define LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(type, tag)                      \
    case CallOp::MAKE_##tag##2: _scratch << "lc_make_" << #type "2"; break; \
    case CallOp::MAKE_##tag##3: _scratch << "lc_make_" << #type "3"; break; \
    case CallOp::MAKE_##tag##4: _scratch << "lc_make_" << #type "4"; break;
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(bool, BOOL)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(int, INT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(uint, UINT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(float, FLOAT)
#undef LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL
        case CallOp::MAKE_FLOAT2X2: _scratch << "lc_make_float2x2"; break;
        case CallOp::MAKE_FLOAT3X3: _scratch << "lc_make_float3x3"; break;
        case CallOp::MAKE_FLOAT4X4: _scratch << "lc_make_float4x4"; break;
        case CallOp::ASSUME: _scratch << "__builtin_assume"; break;
        case CallOp::UNREACHABLE: _scratch << "__builtin_unreachable"; break;
        case CallOp::INSTANCE_TO_WORLD_MATRIX: _scratch << "lc_accel_instance_transform"; break;
        case CallOp::TRACE_CLOSEST: _scratch << "lc_trace_closest"; break;
        case CallOp::TRACE_ANY: _scratch << "lc_trace_any"; break;
        case CallOp::SET_INSTANCE_TRANSFORM: _scratch << "lc_accel_set_instance_transform"; break;
        case CallOp::SET_INSTANCE_VISIBILITY: _scratch << "lc_accel_set_instance_visibility"; break;
    }
    auto args = expr->arguments();
    if (is_atomic) {
        if (args.front()->type()->description() == "float") {
            _scratch << "_float";
        }
        _scratch << "(&(";
        args.front()->accept(*this);
        _scratch << ")";
        for (auto arg : args.subspan(1u)) {
            _scratch << ", ";
            arg->accept(*this);
        }
    } else {
        _scratch << "(";
        if (!args.empty()) {
            for (auto arg : args) {
                arg->accept(*this);
                _scratch << ", ";
            }
            _scratch.pop_back();
            _scratch.pop_back();
        }
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
            _scratch << "lc_bit_cast<";
            _emit_type_name(expr->type());
            _scratch << ">(";
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
    _scratch << " = ";
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

    _indent = 0u;
    _function = f;

    // constants
    if (!f.constants().empty()) {
        for (auto c : f.constants()) { _emit_constant(c); }
        _scratch << "\n";
    }

    // ray tracing kernels use __constant__ args
    if (f.tag() == Function::Tag::KERNEL && f.raytracing()) {
        _scratch << "struct alignas(16) Params {";
        for (auto arg : f.arguments()) {
            _scratch << "\n  alignas(16) ";
            _emit_variable_decl(arg, !arg.type()->is_buffer());
            _scratch << "{};";
        }
        _scratch << "\n};\n\nextern \"C\" "
                    "{ __constant__ Params params; }\n\n";
    }

    // signature
    if (f.tag() == Function::Tag::KERNEL) {
        _scratch << "extern \"C\" __global__ void "
                 << (f.raytracing() ? "__raygen__rg_" : "kernel_")
                 << hash_to_string(f.hash());
    } else if (f.tag() == Function::Tag::CALLABLE) {
        _scratch << "inline __device__ ";
        if (f.return_type() != nullptr) {
            _emit_type_name(f.return_type());
        } else {
            _scratch << "void";
        }
        _scratch << " custom_" << hash_to_string(f.hash());
    } else [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid function type.");
    }
    _scratch << "(";
    if (f.tag() == Function::Tag::KERNEL && f.raytracing()) {
        _scratch << ") {"
                 // block size
                 << "\n  constexpr auto bs = lc_make_uint3("
                 << f.block_size().x << ", "
                 << f.block_size().y << ", "
                 << f.block_size().z << ");"
                 // launch size
                 << "\n  const auto ls = lc_rtx_dispatch_size();"
                 // dispatch id
                 << "\n  const auto did = lc_rtx_dispatch_id();";
        for (auto builtin : f.builtin_variables()) {
            switch (builtin.tag()) {
                case Variable::Tag::THREAD_ID:
                    _scratch << "\n  const auto tid = lc_make_uint3("
                                "did.x % bs.x, "
                                "did.y % bs.y, "
                                "did.z % bs.z);";
                    break;
                case Variable::Tag::BLOCK_ID:
                    _scratch << "\n  const auto bid = lc_make_uint3("
                                "did.x / bs.x, "
                                "did.y / bs.y, "
                                "did.z / bs.z);";
                    break;
                default: break;
            }
        }
        for (auto arg : f.arguments()) {
            _scratch << "\n  ";
            if (auto usage = f.variable_usage(arg.uid());
                usage == Usage::WRITE || usage == Usage::READ_WRITE) {
                _scratch << "auto ";
            } else {
                _scratch << "const auto &";
            }
            _emit_variable_name(arg);
            _scratch << " = params.";
            _emit_variable_name(arg);
            _scratch << ";";
        }
    } else {
        auto any_arg = false;
        for (auto arg : f.arguments()) {
            _scratch << "\n    ";
            _emit_variable_decl(arg, false);
            _scratch << ",";
            any_arg = true;
        }
        if (f.tag() == Function::Tag::KERNEL) {
            _scratch << "\n"
                     << "    const lc_uint3 ls) {\n"
                     << "  const auto tid = lc_make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);\n"
                     << "  const auto bid = lc_make_uint3(blockIdx.x, blockIdx.y, blockIdx.z);\n"
                     << "  const auto did = lc_make_uint3(\n"
                     << "    blockIdx.x * blockDim.x + threadIdx.x,\n"
                     << "    blockIdx.y * blockDim.y + threadIdx.y,\n"
                     << "    blockIdx.z * blockDim.z + threadIdx.z);\n"
                     << "  if (lc_any(did >= ls)) { return; }";
        } else {
            if (any_arg) { _scratch.pop_back(); }
            _scratch << ") noexcept {";
        }
    }
    _indent = 1;
    _emit_variable_declarations(f);
    _indent = 0;
    _emit_statements(f.body()->statements());
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

static constexpr std::string_view ray_type_desc = "struct<16,array<float,3>,float,array<float,3>,float>";
static constexpr std::string_view hit_type_desc = "struct<16,uint,uint,vector<float,2>>";

void CUDACodegen::visit(const Type *type) noexcept {
    if (type->is_structure() &&
        type->description() != ray_type_desc &&
        type->description() != hit_type_desc) {
        _scratch << "struct alignas(" << type->alignment() << ") ";
        _emit_type_name(type);
        _scratch << " {\n";
        for (auto i = 0u; i < type->members().size(); i++) {
            _scratch << "  ";
            _emit_type_name(type->members()[i]);
            _scratch << " m" << i << "{};\n";
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
            if (auto desc = type->description(); desc == ray_type_desc) {
                _scratch << "LCRay";
            } else if (desc == hit_type_desc) {
                _scratch << "LCHit";
            } else {
                _scratch << "S" << hash_to_string(type->hash());
            }
            break;
        default: break;
    }
}

void CUDACodegen::_emit_variable_decl(Variable v, bool force_const) noexcept {
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
            if (readonly || force_const) {
                _scratch << "const ";
                _emit_type_name(v.type());
                _scratch << " ";
            } else {
                _emit_type_name(v.type());
                _scratch << " &";
            }
            _emit_variable_name(v);
            break;
        case Variable::Tag::BUFFER:
            if (readonly || force_const) { _scratch << "const "; }
            _emit_type_name(v.type()->element());
            _scratch << " *__restrict__ ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::TEXTURE:
            _scratch << "const LCSurface ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::BINDLESS_ARRAY:
            _scratch << "const LCBindlessArray ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::ACCEL:
            _scratch << "const LCAccel ";
            _emit_variable_name(v);
            break;
        default:
            _emit_type_name(v.type());
            _scratch << " ";
            _emit_variable_name(v);
            break;
    }
}

void CUDACodegen::_emit_indent() noexcept {
    for (auto i = 0u; i < _indent; i++) { _scratch << "  "; }
}

void CUDACodegen::_emit_statements(luisa::span<const Statement *const> stmts) noexcept {
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

void CUDACodegen::_emit_constant(Function::Constant c) noexcept {

    if (std::find(_generated_constants.cbegin(),
                  _generated_constants.cend(), c.data.hash()) != _generated_constants.cend()) { return; }
    _generated_constants.emplace_back(c.data.hash());

    _scratch << "__constant__ lc_constant ";
    _emit_type_name(c.type);
    _scratch << " c" << hash_to_string(c.data.hash()) << "{";
    auto count = c.type->dimension();
    static constexpr auto wrap = 16u;
    using namespace std::string_view_literals;
    luisa::visit(
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

void CUDACodegen::_emit_variable_declarations(Function f) noexcept {
    for (auto v : f.shared_variables()) {
        if (_function.variable_usage(v.uid()) != Usage::NONE) {
            _scratch << "\n";
            _emit_indent();
            _emit_variable_decl(v, false);
            _scratch << "{};";
        }
    }
    for (auto v : f.local_variables()) {
        if (_function.variable_usage(v.uid()) != Usage::NONE) {
            _scratch << "\n";
            _emit_indent();
            _emit_variable_decl(v, false);
            _scratch << "{};";
        }
    }
}

}// namespace luisa::compute::cuda
