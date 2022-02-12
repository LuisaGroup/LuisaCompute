//
// Created by Mike on 2021/11/8.
//

#include <string_view>

#include <core/hash.h>
#include <ast/type_registry.h>
#include <ast/constant_data.h>
#include <ast/function_builder.h>
#include <backends/ispc/ispc_codegen.h>

namespace luisa::compute::ispc {

void ISPCCodegen::visit(const UnaryExpr *expr) {
    switch (expr->op()) {
        case UnaryOp::PLUS: _scratch << "unary_plus("; break;
        case UnaryOp::MINUS: _scratch << "unary_minus("; break;
        case UnaryOp::NOT: _scratch << "unary_not("; break;
        case UnaryOp::BIT_NOT: _scratch << "unary_bit_not("; break;
        default: break;
    }
    expr->operand()->accept(*this);
    _scratch << ")";
}

void ISPCCodegen::visit(const BinaryExpr *expr) {
    if (expr->lhs()->type()->is_scalar() &&
        expr->rhs()->type()->is_scalar()) {
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
    } else {
        switch (expr->op()) {
            case BinaryOp::ADD: _scratch << "binary_add"; break;
            case BinaryOp::SUB: _scratch << "binary_sub"; break;
            case BinaryOp::MUL: _scratch << "binary_mul"; break;
            case BinaryOp::DIV: _scratch << "binary_div"; break;
            case BinaryOp::MOD: _scratch << "binary_mod"; break;
            case BinaryOp::BIT_AND: _scratch << "binary_bit_and"; break;
            case BinaryOp::BIT_OR: _scratch << "binary_bit_or"; break;
            case BinaryOp::BIT_XOR: _scratch << "binary_bit_xor"; break;
            case BinaryOp::SHL: _scratch << "binary_shl"; break;
            case BinaryOp::SHR: _scratch << "binary_shr"; break;
            case BinaryOp::AND: _scratch << "binary_and"; break;
            case BinaryOp::OR: _scratch << "binary_or"; break;
            case BinaryOp::LESS: _scratch << "binary_lt"; break;
            case BinaryOp::GREATER: _scratch << "binary_gt"; break;
            case BinaryOp::LESS_EQUAL: _scratch << "binary_le"; break;
            case BinaryOp::GREATER_EQUAL: _scratch << "binary_ge"; break;
            case BinaryOp::EQUAL: _scratch << "binary_eq"; break;
            case BinaryOp::NOT_EQUAL: _scratch << "binary_ne"; break;
        }
        _scratch << "(";
        expr->lhs()->accept(*this);
        _scratch << ", ";
        expr->rhs()->accept(*this);
        _scratch << ")";
    }
}

void ISPCCodegen::visit(const MemberExpr *expr) {
    if (expr->is_swizzle()) {
        static constexpr std::string_view xyzw[]{"x", "y", "z", "w"};
        if (auto ss = expr->swizzle_size(); ss == 1u) {
            expr->self()->accept(*this);
            _scratch << "._";
            _scratch << xyzw[expr->swizzle_index(0)];
        } else {
            _scratch << "make_";
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
                _scratch << "._" << xyzw[expr->swizzle_index(i)] << ", ";
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

void ISPCCodegen::visit(const AccessExpr *expr) {
    if (auto t = expr->range()->type(); t->is_array()) {
        _scratch << "array_access(";
    } else if (t->is_vector()) {
        _scratch << "vector_access(";
    } else if (t->is_matrix()) {
        _scratch << "matrix_access(";
    } else {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid range type for access expression: {}.",
            t->description());
    }
    expr->range()->accept(*this);
    _scratch << ", ";
    expr->index()->accept(*this);
    _scratch << ")";
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
            _s << (v < 0.0f ? "(1.0f/-0.0f)" : "(1.0f/+0.0f)");
        } else {
            _s << v << "f";
        }
    }
    void operator()(int v) const noexcept { _s << v; }
    void operator()(uint v) const noexcept { _s << v << "u"; }

    template<typename T, size_t N>
    void operator()(Vector<T, N> v) const noexcept {
        auto t = Type::of<T>();
        _s << "make_" << t->description() << N << "(";
        for (auto i = 0u; i < N; i++) {
            (*this)(v[i]);
            _s << ", ";
        }
        _s.pop_back();
        _s.pop_back();
        _s << ")";
    }

    void operator()(float2x2 m) const noexcept {
        _s << "make_float2x2(";
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
        _s << "make_float3x3(";
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
        _s << "make_float4x4(";
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

void ISPCCodegen::visit(const LiteralExpr *expr) {
    luisa::visit(detail::LiteralPrinter{_scratch}, expr->value());
}

void ISPCCodegen::visit(const RefExpr *expr) {
    _emit_variable_name(expr->variable());
}

void ISPCCodegen::visit(const CallExpr *expr) {

    auto is_atomic = false;
    switch (expr->op()) {
        case CallOp::CUSTOM:
            _scratch << "custom_"
                     << hash_to_string(expr->custom().hash());
            break;
        case CallOp::ALL: _scratch << "all"; break;
        case CallOp::ANY: _scratch << "any"; break;
        case CallOp::SELECT: {
            using namespace std::string_view_literals;
            auto pred_type = expr->arguments()[2]->type();
            auto is_scalar = pred_type->tag() == Type::Tag::BOOL;
            _scratch << (is_scalar ? "select_scalar"sv : "select"sv);
            break;
        }
        case CallOp::CLAMP: _scratch << "clamp"; break;
        case CallOp::LERP: _scratch << "lerp"; break;
        case CallOp::STEP: _scratch << "step"; break;
        case CallOp::ABS: _scratch << "abs"; break;
        case CallOp::MIN: _scratch << "min"; break;
        case CallOp::MAX: _scratch << "max"; break;
        case CallOp::CLZ: _scratch << "clz"; break;
        case CallOp::CTZ: _scratch << "ctz"; break;
        case CallOp::POPCOUNT: _scratch << "popcount"; break;
        case CallOp::REVERSE: _scratch << "reverse"; break;
        case CallOp::ISINF: _scratch << "is_inf"; break;
        case CallOp::ISNAN: _scratch << "is_nan"; break;
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
        case CallOp::FMA: _scratch << "fma"; break;
        case CallOp::COPYSIGN: _scratch << "copysign"; break;
        case CallOp::CROSS: _scratch << "cross"; break;
        case CallOp::DOT: _scratch << "dot"; break;
        case CallOp::LENGTH: _scratch << "length"; break;
        case CallOp::LENGTH_SQUARED: _scratch << "length_squared"; break;
        case CallOp::NORMALIZE: _scratch << "normalize"; break;
        case CallOp::FACEFORWARD: _scratch << "faceforward"; break;
        case CallOp::DETERMINANT: _scratch << "determinant"; break;
        case CallOp::TRANSPOSE: _scratch << "transpose"; break;
        case CallOp::INVERSE: _scratch << "inverse"; break;
        case CallOp::SYNCHRONIZE_BLOCK: _scratch << "barrier"; break;
        case CallOp::ATOMIC_EXCHANGE:
            _scratch << "atomic_swap";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_COMPARE_EXCHANGE:
            _scratch << "atomic_compare_exchange";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_FETCH_ADD:
            _scratch << "atomic_add";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_FETCH_SUB:
            _scratch << "atomic_subtract";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_FETCH_AND:
            _scratch << "atomic_and";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_FETCH_OR:
            _scratch << "atomic_or";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_FETCH_XOR:
            _scratch << "atomic_xor";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_FETCH_MIN:
            _scratch << "atomic_min";
            is_atomic = true;
            break;
        case CallOp::ATOMIC_FETCH_MAX:
            _scratch << "atomic_max";
            is_atomic = true;
            break;
        case CallOp::BUFFER_READ: _scratch << "buffer_read"; break;
        case CallOp::BUFFER_WRITE: _scratch << "buffer_write"; break;
        case CallOp::TEXTURE_READ:
            _scratch << "surf"
                     << expr->arguments().front()->type()->dimension()
                     << "d_read_"
                     << expr->arguments().front()->type()->element()->description();
            break;
        case CallOp::TEXTURE_WRITE:
            _scratch << "surf"
                     << expr->arguments().front()->type()->dimension()
                     << "d_write_"
                     << expr->arguments().front()->type()->element()->description();
            break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE: _scratch << "bindless_texture_sample2d"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: _scratch << "bindless_texture_sample2d_level"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: _scratch << "bindless_texture_sample2d_grad"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE: _scratch << "bindless_texture_sample3d"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: _scratch << "bindless_texture_sample3d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: _scratch << "bindless_texture_sample3d_grad"; break;
        case CallOp::BINDLESS_TEXTURE2D_READ: _scratch << "bindless_texture_read2d"; break;
        case CallOp::BINDLESS_TEXTURE3D_READ: _scratch << "bindless_texture_read3d"; break;
        case CallOp::BINDLESS_TEXTURE2D_READ_LEVEL: _scratch << "bindless_texture_read2d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_READ_LEVEL: _scratch << "bindless_texture_read3d_level"; break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE: _scratch << "bindless_texture_size2d"; break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE: _scratch << "bindless_texture_size3d"; break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: _scratch << "bindless_texture_size2d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: _scratch << "bindless_texture_size3d_level"; break;
        case CallOp::BINDLESS_BUFFER_READ: _scratch << "bindless_buffer_read"; break;
        case CallOp::MAKE_BOOL2: _scratch << "make_bool2"; break;
        case CallOp::MAKE_BOOL3: _scratch << "make_bool3"; break;
        case CallOp::MAKE_BOOL4: _scratch << "make_bool4"; break;
        case CallOp::MAKE_INT2: _scratch << "make_int2"; break;
        case CallOp::MAKE_INT3: _scratch << "make_int3"; break;
        case CallOp::MAKE_INT4: _scratch << "make_int4"; break;
        case CallOp::MAKE_UINT2: _scratch << "make_uint2"; break;
        case CallOp::MAKE_UINT3: _scratch << "make_uint3"; break;
        case CallOp::MAKE_UINT4: _scratch << "make_uint4"; break;
        case CallOp::MAKE_FLOAT2: _scratch << "make_float2"; break;
        case CallOp::MAKE_FLOAT3: _scratch << "make_float3"; break;
        case CallOp::MAKE_FLOAT4: _scratch << "make_float4"; break;
        case CallOp::MAKE_FLOAT2X2: _scratch << "make_float2x2"; break;
        case CallOp::MAKE_FLOAT3X3: _scratch << "make_float3x3"; break;
        case CallOp::MAKE_FLOAT4X4: _scratch << "make_float4x4"; break;
        case CallOp::ASSUME: _scratch << "assume"; break;
        case CallOp::UNREACHABLE: _scratch << "unreachable"; break;
        case CallOp::INSTANCE_TO_WORLD_MATRIX: _scratch << "accel_instance_transform"; break;
        case CallOp::TRACE_CLOSEST: _scratch << "trace_closest"; break;
        case CallOp::TRACE_ANY: _scratch << "trace_any"; break;
    }
    _scratch << "(";
    auto args = expr->arguments();
    if (is_atomic) {
        _scratch << "&(";
        args.front()->accept(*this);
        _scratch << ")";
        for (auto arg : args.subspan(1u)) {
            _scratch << ", ";
            arg->accept(*this);
        }
    } else if (!args.empty()) {
        if (expr->op() == CallOp::BINDLESS_BUFFER_READ) {
            _emit_type_name(expr->type());
            _scratch << ", ";
        }
        for (auto arg : args) {
            arg->accept(*this);
            _scratch << ", ";
        }
        _scratch.pop_back();
        _scratch.pop_back();
    }
    _scratch << ")";
}

void ISPCCodegen::visit(const CastExpr *expr) {
    switch (expr->op()) {
        case CastOp::STATIC:
            _scratch << "((";
            _emit_type_name(expr->type());
            _scratch << ")(";
            expr->expression()->accept(*this);
            _scratch << "))";
            break;
        case CastOp::BITWISE:
            _scratch << "(*((varying const ";
            _emit_type_name(expr->type());
            _scratch << " *)&(";
            expr->expression()->accept(*this);
            _scratch << ")))";
            break;
        default: break;
    }
}

void ISPCCodegen::visit(const BreakStmt *) {
    _scratch << "break;";
}

void ISPCCodegen::visit(const ContinueStmt *) {
    _scratch << "continue;";
}

void ISPCCodegen::visit(const ReturnStmt *stmt) {
    _scratch << "return";
    if (auto expr = stmt->expression(); expr != nullptr) {
        _scratch << " ";
        expr->accept(*this);
    }
    _scratch << ";";
}

void ISPCCodegen::visit(const ScopeStmt *stmt) {
    _scratch << "{";
    _emit_statements(stmt->statements());
    _scratch << "}";
}

void ISPCCodegen::visit(const IfStmt *stmt) {
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

void ISPCCodegen::visit(const LoopStmt *stmt) {
    _scratch << "for (;;) ";
    stmt->body()->accept(*this);
}

void ISPCCodegen::visit(const ExprStmt *stmt) {
    stmt->expression()->accept(*this);
    _scratch << ";";
}

void ISPCCodegen::visit(const SwitchStmt *stmt) {
    _scratch << "switch (";
    stmt->expression()->accept(*this);
    _scratch << ") ";
    stmt->body()->accept(*this);
}

void ISPCCodegen::visit(const SwitchCaseStmt *stmt) {
    _scratch << "case ";
    stmt->expression()->accept(*this);
    _scratch << ": ";
    stmt->body()->accept(*this);
}

void ISPCCodegen::visit(const SwitchDefaultStmt *stmt) {
    _scratch << "default: ";
    stmt->body()->accept(*this);
}

void ISPCCodegen::visit(const AssignStmt *stmt) {
    stmt->lhs()->accept(*this);
    _scratch << " = ";
    stmt->rhs()->accept(*this);
    _scratch << ";";
}

void ISPCCodegen::emit(Function f) {
    _scratch << "#include <ispc_device_library.isph>\n\n";
    _emit_type_decl();
    _emit_function(f);
}

void ISPCCodegen::_emit_function(Function f) noexcept {

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
        _scratch << "inline void "
                 << "kernel_"
                 << hash_to_string(f.hash());
    } else if (f.tag() == Function::Tag::CALLABLE) {
        _scratch << "inline ";
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
    auto any_arg = false;
    for (auto arg : f.arguments()) {
        _scratch << "\n    ";
        _emit_variable_decl(arg, false);
        _scratch << ",";
        any_arg = true;
    }
    if (f.tag() == Function::Tag::KERNEL) {
        _scratch << "\n"
                 << "    const uint3 tid,\n"
                 << "    const uint3 did,\n"
                 << "    const uniform uint3 bid,\n"
                 << "    const uniform uint3 ls) {"
                 << "  if (any(binary_ge(did, ls))) { return; }\n" ;
    } else {
        if (any_arg) { _scratch.pop_back(); }
        _scratch << ") {";
    }
    _indent = 1;
    _emit_variable_declarations(f.body());
    _indent = 0;
    _emit_statements(f.body()->scope()->statements());
    _scratch << "}\n\n";

    // entry point
    if (f.tag() == Function::Tag::KERNEL) {
        _scratch << "struct Params {\n";
        auto pad_count = 0u;
        auto size = 0u;
        for (auto arg : f.arguments()) {
            auto aligned_size = (size + 15u) / 16u * 16u;
            if (auto pad_size = aligned_size - size) {
                if (pad_size % 4u == 0u) {
                    _scratch << "    uint _pad_" << pad_count++
                             << "[" << pad_size / 4u << "];\n";
                } else {
                    _scratch << "    uint8 _pad_" << pad_count++
                             << "[" << pad_size << "];\n";
                }
            }
            _scratch << "    ";
            _emit_variable_decl(arg, !arg.type()->is_buffer());
            _scratch << ";\n";
            if (arg.type()->is_buffer()) {
                size = aligned_size + buffer_handle_size;
            } else if (arg.type()->is_texture()) {
                size = aligned_size + texture_handle_size;
            } else if (arg.type()->is_accel()) {
                size = aligned_size + accel_handle_size;
            } else if (arg.type()->is_bindless_array()) {
                size = aligned_size + bindless_array_handle_size;
            } else {
                size = aligned_size + arg.type()->size();
            }
        }
        _scratch << "};\n\n";
        _scratch << "export void kernel_main(\n"
                    "    const Params *uniform params,\n"
                    "    uniform const uint bx, uniform const uint by, uniform const uint bz,\n"
                    "    uniform const uint lx, uniform const uint ly, uniform const uint lz) {\n"
                    "    uniform const uint3 bid = make_uint3(bx, by, bz);\n"
                    "    uniform const uint3 ls = make_uint3(lx, ly, lz);\n";
        _scratch << "  foreach ("
                 << "k = 0..." << f.block_size().z << ", "
                 << "j = 0..." << f.block_size().y << ", "
                 << "i = 0..." << f.block_size().x << ") {\n"
                 << "    uint3 tid = make_uint3(i, j, k);\n"
                 << "    uint3 did = make_uint3("
                 << "bx * " << f.block_size().x << " + i, "
                 << "by * " << f.block_size().y << " + j, "
                 << "bz * " << f.block_size().z << " + k);\n";
        _scratch << "    kernel_" << hash_to_string(f.hash()) << "(";
        for (auto arg : f.arguments()) {
            _scratch << "params->";
            _emit_variable_name(arg);
            _scratch << ", ";
        }
        _scratch << "tid, did, bid, ls);\n";
        _scratch << "  }\n"
                 << "}";
    }
}

void ISPCCodegen::_emit_variable_name(Variable v) noexcept {
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

void ISPCCodegen::_emit_type_decl() noexcept {
    Type::traverse(*this);
}

static constexpr std::string_view float_array_3 = "array<float,3>";
static constexpr std::string_view ray_type_desc = "struct<16,array<float,3>,float,array<float,3>,float>";
static constexpr std::string_view hit_type_desc = "struct<16,uint,uint,vector<float,2>>";

void ISPCCodegen::visit(const Type *type) noexcept {
    if (type->is_array() &&
        type->description() != float_array_3) {
        _scratch << "make_array_type(";
        _emit_type_name(type);
        _scratch << ", ";
        _emit_type_name(type->element());
        _scratch << ", "
                 << type->dimension()
                 << ");\n\n";
    } else if (type->is_structure() &&
               type->description() != ray_type_desc &&
               type->description() != hit_type_desc) {
        _scratch << "struct ";
        _emit_type_name(type);
        _scratch << " {\n";
        auto pad_count = 0u;
        auto size = 0u;
        for (auto i = 0u; i < type->members().size(); i++) {
            auto member = type->members()[i];
            auto a = member->alignment();
            auto aligned_size = (size + a - 1u) / a * a;
            if (auto pad_size = aligned_size - size) {
                if (pad_size % 4u == 0u) {
                    _scratch << "    uint _pad_" << pad_count++
                             << "[" << pad_size / 4u << "];\n";
                } else {
                    _scratch << "    uint8 _pad_" << pad_count++
                             << "[" << pad_size << "];\n";
                }
            }
            _scratch << "    ";
            _emit_type_name(member);
            _scratch << " m" << i << ";\n";
            size = aligned_size + member->size();
        }
        if (auto pad_size = type->size() - size) {
            if (pad_size % 4u == 0u) {
                _scratch << "    uint _pad_" << pad_count
                         << "[" << pad_size / 4u << "];\n";
            } else {
                _scratch << "    uint8 _pad_" << pad_count
                         << "[" << pad_size << "];\n";
            }
        }
        _scratch << "};\n\n";
    }
}

void ISPCCodegen::_emit_type_name(const Type *type) noexcept {
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
            if (auto desc = type->description(); desc == float_array_3) {
                _scratch << "packed_float3";
            } else {
                _scratch << "A_" << hash_to_string(type->hash());
            }
            break;
        case Type::Tag::STRUCTURE:
            if (auto desc = type->description(); desc == ray_type_desc) {
                _scratch << "LCRay";
            } else if (desc == hit_type_desc) {
                _scratch << "LCHit";
            } else {
                _scratch << "S_" << hash_to_string(type->hash());
            }
            break;
        default: break;
    }
}

void ISPCCodegen::_emit_variable_decl(Variable v, bool force_const) noexcept {
    auto usage = _function.variable_usage(v.uid());
    auto readonly = usage == Usage::NONE || usage == Usage::READ;
    switch (v.tag()) {
        case Variable::Tag::SHARED:
            // TODO: support shared
            _scratch << "__shared__ ";
            _emit_type_name(v.type());
            _scratch << " ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::REFERENCE:
            if (readonly || force_const) { _scratch << "const "; }
            _emit_type_name(v.type());
            _scratch << " &";
            _emit_variable_name(v);
            break;
        case Variable::Tag::BUFFER:
            if (readonly || force_const) { _scratch << "uniform const "; }
            _emit_type_name(v.type()->element());
            _scratch << " *uniform ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::TEXTURE:
            _scratch << "uniform const LCSurface ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::BINDLESS_ARRAY:
            _scratch << "uniform const LCBindlessArray ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::ACCEL:
            _scratch << "uniform const LCAccel ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::LOCAL:
            _emit_type_name(v.type());
            _scratch << " ";
            _emit_variable_name(v);
            break;
        default:
            break;
    }
}

void ISPCCodegen::_emit_indent() noexcept {
    for (auto i = 0u; i < _indent; i++) { _scratch << "  "; }
}

void ISPCCodegen::_emit_statements(luisa::span<const Statement *const> stmts) noexcept {
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

void ISPCCodegen::_emit_constant(Function::Constant c) noexcept {

    if (std::find(_generated_constants.cbegin(),
                  _generated_constants.cend(), c.data.hash()) != _generated_constants.cend()) { return; }
    _generated_constants.emplace_back(c.data.hash());

    _scratch << "static const uniform ";
    _emit_type_name(c.type);
    _scratch << " c" << hash_to_string(c.data.hash()) << " = {{";
    auto count = c.type->dimension();
    static constexpr auto wrap = 16u;
    using namespace std::string_view_literals;
    luisa::visit(
        [count, this](auto ptr) {
            detail::LiteralPrinter print{_scratch};
            for (auto i = 0u; i < count; i++) {
                if (count > wrap && i % wrap == 0u) { _scratch << "\n    "; }
                auto value = ptr[i];
                using T = std::remove_cvref_t<decltype(value)>;
                if constexpr (is_scalar_v<T>) {
                    print(value);
                } else if constexpr (is_vector2_v<T>) {
                    _scratch << "{{ ";
                    print(value[0]);
                    _scratch << ", ";
                    print(value[1]);
                    _scratch << " }}";
                } else if constexpr (is_vector3_v<T>) {
                    _scratch << "{{ ";
                    print(value[0]);
                    _scratch << ", ";
                    print(value[1]);
                    _scratch << ", ";
                    print(value[2]);
                    _scratch << " }}";
                } else if constexpr (is_vector4_v<T>) {
                    _scratch << "{{ ";
                    print(value[0]);
                    _scratch << ", ";
                    print(value[1]);
                    _scratch << ", ";
                    print(value[2]);
                    _scratch << ", ";
                    print(value[3]);
                    _scratch << " }}";
                } else if constexpr (is_matrix2_v<T>) {
                    _scratch << "{{ ";
                    _scratch << "{{ ";
                    print(value[0].x);
                    _scratch << ", ";
                    print(value[0].y);
                    _scratch << " }}, ";
                    _scratch << "{{ ";
                    print(value[1].x);
                    _scratch << ", ";
                    print(value[1].y);
                    _scratch << " }}";
                    _scratch << " }}";
                } else if constexpr (is_matrix3_v<T>) {
                    _scratch << "{{ ";
                    _scratch << "{{ ";
                    print(value[0].x);
                    _scratch << ", ";
                    print(value[0].y);
                    _scratch << ", ";
                    print(value[0].z);
                    _scratch << " }}, ";
                    _scratch << "{{ ";
                    print(value[1].x);
                    _scratch << ", ";
                    print(value[1].y);
                    _scratch << ", ";
                    print(value[1].z);
                    _scratch << " }}, ";
                    _scratch << "{{ ";
                    print(value[2].x);
                    _scratch << ", ";
                    print(value[2].y);
                    _scratch << ", ";
                    print(value[2].z);
                    _scratch << " }}";
                    _scratch << " }}";
                } else if constexpr (is_matrix4_v<T>) {
                    _scratch << "{{ ";
                    _scratch << "{{ ";
                    print(value[0].x);
                    _scratch << ", ";
                    print(value[0].y);
                    _scratch << ", ";
                    print(value[0].z);
                    _scratch << ", ";
                    print(value[0].w);
                    _scratch << " }}, ";
                    _scratch << "{{ ";
                    print(value[1].x);
                    _scratch << ", ";
                    print(value[1].y);
                    _scratch << ", ";
                    print(value[1].z);
                    _scratch << ", ";
                    print(value[1].w);
                    _scratch << " }}, ";
                    _scratch << "{{ ";
                    print(value[2].x);
                    _scratch << ", ";
                    print(value[2].y);
                    _scratch << ", ";
                    print(value[2].z);
                    _scratch << ", ";
                    print(value[2].w);
                    _scratch << " }}, ";
                    _scratch << "{{ ";
                    print(value[3].x);
                    _scratch << ", ";
                    print(value[3].y);
                    _scratch << ", ";
                    print(value[3].z);
                    _scratch << ", ";
                    print(value[3].w);
                    _scratch << " }}";
                    _scratch << " }}";
                }
                _scratch << ", ";
            }
        },
        c.data.view());
    if (count > 0u) {
        _scratch.pop_back();
        _scratch.pop_back();
    }
    _scratch << "}};\n";
}

void ISPCCodegen::visit(const ConstantExpr *expr) {
    _scratch << "c" << hash_to_string(expr->data().hash());
}

void ISPCCodegen::visit(const ForStmt *stmt) {
    _scratch << "for (; ";
    stmt->condition()->accept(*this);
    _scratch << "; ";
    stmt->variable()->accept(*this);
    _scratch << " += ";
    stmt->step()->accept(*this);
    _scratch << ") ";
    stmt->body()->accept(*this);
}

void ISPCCodegen::visit(const CommentStmt *stmt) {
    _scratch << "/* " << stmt->comment() << " */";
}

void ISPCCodegen::visit(const MetaStmt *stmt) {
    _scratch << "\n";
    _emit_indent();
    _scratch << "// meta region begin: " << stmt->info();
    _emit_variable_declarations(stmt);
    for (auto s : stmt->scope()->statements()) {
        _scratch << "\n";
        _emit_indent();
        s->accept(*this);
    }
    _scratch << "\n";
    _emit_indent();
    _scratch << "// meta region end: " << stmt->info() << "\n";
}

void ISPCCodegen::_emit_variable_declarations(const MetaStmt *meta) noexcept {
    for (auto v : meta->variables()) {
        if (_function.variable_usage(v.uid()) != Usage::NONE) {
            _scratch << "\n";
            _emit_indent();
            _emit_variable_decl(v, false);
            _scratch << ";";
        }
    }
}

}// namespace luisa::compute::ispc
