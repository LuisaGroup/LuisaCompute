//
// Created by Mike Smith on 2021/3/25.
//

#import <span>

#import <core/hash.h>
#import <ast/type_registry.h>
#import <ast/function_builder.h>
#import <ast/constant_data.h>
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
    if (expr->is_swizzle()) {
        if (expr->swizzle_size() == 1u) {
            _scratch << "[" << expr->swizzle_index(0u) << "]";
        } else {
            static constexpr std::string_view xyzw[]{"x", "y", "z", "w"};
            _scratch << ".";
            for (auto i = 0u; i < expr->swizzle_size(); i++) {
                _scratch << xyzw[expr->swizzle_index(i)];
            }
        }
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
        if (std::isnan(v)) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Encountered with NaN."); }
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

    void operator()(float2x2 m) const noexcept {
        _s << "float2x2(";
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
                _s << ", ";
            }
        }
        _s.pop_back();
        _s.pop_back();
        _s << ")";
    }

    void operator()(const LiteralExpr::MetaValue &s) const noexcept {

    }
};

}// namespace detail

void MetalCodegen::visit(const LiteralExpr *expr) {
    std::visit(detail::LiteralPrinter{_scratch}, expr->value());
}

void MetalCodegen::visit(const RefExpr *expr) {
    _emit_variable_name(expr->variable());
}

void MetalCodegen::visit(const CallExpr *expr) {
    auto is_atomic_op = false;
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
        case CallOp::MOD: _scratch << "glsl_mod"; break;
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
        case CallOp::BLOCK_BARRIER: _scratch << "block_barrier"; break;
        case CallOp::DEVICE_BARRIER: _scratch << "device_barrier"; break;
        case CallOp::ALL_BARRIER: _scratch << "all_barrier"; break;
        case CallOp::ATOMIC_LOAD:
            _scratch << "atomic_load_explicit";
            is_atomic_op = true;
            break;
        case CallOp::ATOMIC_STORE:
            _scratch << "atomic_store_explicit";
            is_atomic_op = true;
            break;
        case CallOp::ATOMIC_EXCHANGE:
            _scratch << "atomic_exchange_explicit";
            is_atomic_op = true;
            break;
        case CallOp::ATOMIC_COMPARE_EXCHANGE:
            _scratch << "atomic_compare_exchange";
            is_atomic_op = true;
            break;
        case CallOp::ATOMIC_FETCH_ADD:
            _scratch << "atomic_fetch_add_explicit";
            is_atomic_op = true;
            break;
        case CallOp::ATOMIC_FETCH_SUB:
            _scratch << "atomic_fetch_sub_explicit";
            is_atomic_op = true;
            break;
        case CallOp::ATOMIC_FETCH_AND:
            _scratch << "atomic_fetch_and_explicit";
            is_atomic_op = true;
            break;
        case CallOp::ATOMIC_FETCH_OR:
            _scratch << "atomic_fetch_or_explicit";
            is_atomic_op = true;
            break;
        case CallOp::ATOMIC_FETCH_XOR:
            _scratch << "atomic_fetch_xor_explicit";
            is_atomic_op = true;
            break;
        case CallOp::ATOMIC_FETCH_MIN:
            _scratch << "atomic_fetch_min_explicit";
            is_atomic_op = true;
            break;
        case CallOp::ATOMIC_FETCH_MAX:
            _scratch << "atomic_fetch_max_explicit";
            is_atomic_op = true;
            break;
        case CallOp::TEXTURE_READ: _scratch << "texture_read"; break;
        case CallOp::TEXTURE_WRITE: _scratch << "texture_write"; break;
        case CallOp::TEXTURE_READ_LEVEL: _scratch << "texture_read_level"; break;
        case CallOp::TEXTURE_WRITE_LEVEL: _scratch << "texture_write_level"; break;
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
        case CallOp::BUFFER_HEAP_READ:
            _scratch << "buffer_heap_read<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        case CallOp::MAKE_BOOL2: _scratch << "bool2"; break;
        case CallOp::MAKE_BOOL3: _scratch << "bool3"; break;
        case CallOp::MAKE_BOOL4: _scratch << "bool4"; break;
        case CallOp::MAKE_INT2: _scratch << "int2"; break;
        case CallOp::MAKE_INT3: _scratch << "int3"; break;
        case CallOp::MAKE_INT4: _scratch << "int4"; break;
        case CallOp::MAKE_UINT2: _scratch << "uint2"; break;
        case CallOp::MAKE_UINT3: _scratch << "uint3"; break;
        case CallOp::MAKE_UINT4: _scratch << "uint4"; break;
        case CallOp::MAKE_FLOAT2: _scratch << "float2"; break;
        case CallOp::MAKE_FLOAT3: _scratch << "float3"; break;
        case CallOp::MAKE_FLOAT4: _scratch << "float4"; break;
        case CallOp::MAKE_FLOAT2X2: _scratch << "float2x2"; break;
        case CallOp::MAKE_FLOAT3X3: _scratch << "float3x3"; break;
        case CallOp::MAKE_FLOAT4X4: _scratch << "float4x4"; break;
        case CallOp::TRACE_CLOSEST: _scratch << "trace_closest"; break;
        case CallOp::TRACE_ANY: _scratch << "trace_any"; break;
    }

    _scratch << "(";
    if (is_atomic_op) {
        _scratch << "as_atomic(";
        auto args = expr->arguments();
        args[0]->accept(*this);
        _scratch << "), ";
        for (auto i = 1u; i < args.size(); i++) {
            args[i]->accept(*this);
            _scratch << ", ";
        }
        _scratch << "memory_order_relaxed";
    } else if (!expr->arguments().empty()) {
        auto arg_index = 0u;
        for (auto arg : expr->arguments()) {
            auto by_ref = !expr->is_builtin() &&
                          expr->custom().arguments()[arg_index].tag() == Variable::Tag::REFERENCE;
            if (by_ref) {
                if (arg->tag() == Expression::Tag::MEMBER &&
                    static_cast<const MemberExpr *>(arg)->is_swizzle()) {
                    // vector elements need special handling, since taking
                    // the address is not directly supported in Metal
                    auto vec_arg = static_cast<const MemberExpr *>(arg);
                    if (vec_arg->swizzle_size() != 1u) [[unlikely]] {
                        LUISA_ERROR_WITH_LOCATION("Invalid reference to vector swizzling.");
                    }
                    _scratch << "vector_element_ptr<" << vec_arg->swizzle_index(0u) << ">(";
                    vec_arg->self()->accept(*this);
                    _scratch << ")";
                } else {
                    _scratch << "address_of(";
                    arg->accept(*this);
                    _scratch << ")";
                }
            } else {
                arg->accept(*this);
            }
            _scratch << ", ";
            arg_index++;
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
            _scratch << "as_type<";
            _emit_type_name(expr->type());
            _scratch << ">(";
            break;
    }
    expr->expression()->accept(*this);
    _scratch << ")";
}

void MetalCodegen::visit(const BreakStmt *) {
    _scratch << "break;";
}

void MetalCodegen::visit(const ContinueStmt *) {
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

void MetalCodegen::visit(const LoopStmt *stmt) {
    _scratch << "for (;;) ";
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
    _emit_preamble(f);
    _emit_type_decl();
    _emit_function(f);
}

void MetalCodegen::_emit_function(Function f) noexcept {

    if (std::find(_generated_functions.cbegin(), _generated_functions.cend(), f) != _generated_functions.cend()) { return; }

    _generated_functions.emplace_back(f);
    for (auto callable : f.custom_callables()) {
        _emit_function(callable->function());
    }

    _function = f;
    _indent = 0u;

    // constants
    if (!f.constants().empty()) {
        for (auto c : f.constants()) { _emit_constant(c); }
        _scratch << "\n";
    }

    if (f.tag() == Function::Tag::KERNEL) {

        // function signature
        _scratch << "[[kernel]] // block_size = ("
                 << f.block_size().x << ", "
                 << f.block_size().y << ", "
                 << f.block_size().z << ")\n"
                 << "void kernel_" << hash_to_string(f.hash()) << "(";

        // arguments
        for (auto buffer : f.captured_buffers()) {
            _scratch << "\n    ";
            _emit_argument_decl(buffer.variable);
            _scratch << ",";
        }
        for (auto image : f.captured_textures()) {
            _scratch << "\n    ";
            _emit_argument_decl(image.variable);
            _scratch << ",";
        }
        for (auto heap : f.captured_heaps()) {
            _scratch << "\n    ";
            _emit_argument_decl(heap.variable);
            _scratch << ",";
        }
        for (auto accel : f.captured_accels()) {
            _scratch << "\n    ";
            _emit_argument_decl(accel.variable);
            _scratch << ",";
        }
        for (auto arg : f.arguments()) {
            _scratch << "\n    ";
            _emit_argument_decl(arg);
            _scratch << ",";
        }
        for (auto builtin : f.builtin_variables()) {
            _scratch << "\n    ";
            _emit_argument_decl(builtin);
            _scratch << ",";
        }
        _scratch.pop_back();
    } else if (f.tag() == Function::Tag::CALLABLE) {
        // emit templated access specifier for textures
        if (std::any_of(f.arguments().begin(), f.arguments().end(), [](auto v) noexcept {
                return v.tag() == Variable::Tag::TEXTURE || v.tag() == Variable::Tag::REFERENCE;
            })) {
            _scratch << "template<";
            for (auto arg : f.arguments()) {
                if (arg.tag() == Variable::Tag::TEXTURE) {
                    _scratch << "access a";
                    _emit_variable_name(arg);
                    _scratch << ", ";
                } else if (arg.tag() == Variable::Tag::REFERENCE) {
                    _scratch << "typename T" << arg.uid() << ", ";
                }
            }
            _scratch.pop_back();
            _scratch.pop_back();
            _scratch << ">\n";
        }
        if (f.return_type() != nullptr) {
            _scratch << "[[nodiscard]] ";
            _emit_type_name(f.return_type());
        } else {
            _scratch << "void";
        }
        _scratch << " custom_" << hash_to_string(f.hash()) << "(";
        for (auto arg : f.arguments()) {
            _scratch << "\n    ";
            _emit_argument_decl(arg);
            _scratch << ",";
        }
        if (!f.arguments().empty()) {
            _scratch.pop_back();
        }
    } else [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid function type.");
    }
    _scratch << ") {";

    // emit shared or "mutable" uniform variables for kernel
    if (f.tag() == Function::Tag::KERNEL) {
        auto has_mutable_args = false;
        for (auto v : f.arguments()) {
            if (v.tag() == Variable::Tag::LOCAL) {
                if (auto usage = f.variable_usage(v.uid());
                    usage == Usage::WRITE || usage == Usage::READ_WRITE) {
                    has_mutable_args = true;
                    _scratch << "\n  auto ";
                    _emit_variable_name(v);
                    _scratch << " = u";
                    _emit_variable_name(v);
                    _scratch << ";";
                }
            }
        }
        if (has_mutable_args) { _scratch << "\n"; }
    }
    // emit body
    _scratch << "\n";
    _emit_declarations(f.body());
    _scratch << "\n";
    _emit_statements(f.body()->scope()->statements());
    _scratch << "}\n\n";
}

void MetalCodegen::_emit_variable_name(Variable v) noexcept {
    switch (v.tag()) {
        case Variable::Tag::LOCAL: _scratch << "v" << v.uid(); break;
        case Variable::Tag::SHARED: _scratch << "s" << v.uid(); break;
        case Variable::Tag::REFERENCE: _scratch << "(*p" << v.uid() << ")"; break;
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

void MetalCodegen::_emit_type_decl() noexcept {
    Type::traverse(*this);
}

static constexpr std::string_view ray_type_desc = "struct<16,array<float,3>,float,array<float,3>,float>";
static constexpr std::string_view hit_type_desc = "struct<16,uint,uint,vector<float,2>>";

void MetalCodegen::visit(const Type *type) noexcept {
    if (type->is_structure()) {
        // skip ray or hit
        if (type->description() == ray_type_desc || type->description() == hit_type_desc) {
            return;
        }
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
        case Type::Tag::STRUCTURE:
            if (type->description() == ray_type_desc) {
                _scratch << "Ray";
            } else if (type->description() == hit_type_desc) {
                _scratch << "Hit";
            } else {
                _scratch << "S" << hash_to_string(type->hash());
            }
            break;
        default:
            LUISA_ERROR_WITH_LOCATION("Invalid type: {}.", type->description());
    }
}

void MetalCodegen::_emit_argument_decl(Variable v) noexcept {
    switch (v.tag()) {
        case Variable::Tag::LOCAL:
            if (_function.tag() == Function::Tag::KERNEL) {
                _scratch << "constant ";
                _emit_type_name(v.type());
                _scratch << " &";
                if (auto usage = _function.variable_usage(v.uid());
                    usage == Usage::WRITE || usage == Usage::READ_WRITE) {
                    _scratch << "u";
                }
            } else {
                if (auto usage = _function.variable_usage(v.uid());
                    usage == Usage::NONE || usage == Usage::READ) {
                    _scratch << "const ";
                }
                _emit_type_name(v.type());
                _scratch << " ";
            }
            _emit_variable_name(v);
            break;
        case Variable::Tag::REFERENCE:
            if (_function.tag() == Function::Tag::KERNEL) {
                LUISA_ERROR_WITH_LOCATION(
                    "Invalid reference argument in kernel.");
            }
            _scratch << "T" << v.uid() << " p" << v.uid();
            break;
        case Variable::Tag::BUFFER:
            _scratch << "device ";
            if (auto usage = _function.variable_usage(v.uid());
                usage == Usage::NONE || usage == Usage::READ) {
                _scratch << "const ";
            }
            _emit_type_name(v.type()->element());
            _scratch << " *__restrict__ ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::TEXTURE:
            _scratch << "texture" << v.type()->dimension() << "d<";
            _emit_type_name(v.type()->element());
            if (_function.tag() == Function::Tag::KERNEL) {
                if (auto usage = _function.variable_usage(v.uid());
                    usage == Usage::READ_WRITE) {
                    _scratch << ", access::read_write> ";
                } else if (usage == Usage::WRITE) {
                    _scratch << ", access::write> ";
                } else {
                    _scratch << ", access::read> ";
                }
            } else {
                _scratch << ", a";
                _emit_variable_name(v);
                _scratch << "> ";
            }
            _emit_variable_name(v);
            break;
        case Variable::Tag::HEAP:
            _scratch << "device const HeapItem *__restrict__ ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::ACCEL:
            _scratch << "instance_acceleration_structure ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::THREAD_ID:
            _scratch << "const uint3 ";
            _emit_variable_name(v);
            _scratch << " [[thread_position_in_threadgroup]]";
            break;
        case Variable::Tag::BLOCK_ID:
            _scratch << "const uint3 ";
            _emit_variable_name(v);
            _scratch << " [[threadgroup_position_in_grid]]";
            break;
        case Variable::Tag::DISPATCH_ID:
            _scratch << "const uint3 ";
            _emit_variable_name(v);
            _scratch << " [[thread_position_in_grid]]";
            break;
        case Variable::Tag::DISPATCH_SIZE:
            _scratch << "constant uint3 &";
            _emit_variable_name(v);
            break;
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Invalid argument type.");
    }
}

void MetalCodegen::_emit_indent() noexcept {
    for (auto i = 0u; i < _indent; i++) {
        _scratch << "  ";
    }
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
                  _generated_constants.cend(), c.data.hash()) != _generated_constants.cend()) { return; }
    _generated_constants.emplace_back(c.data.hash());

    _scratch << "constant ";
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

void MetalCodegen::visit(const ConstantExpr *expr) {
    _scratch << "c" << hash_to_string(expr->data().hash());
}

void MetalCodegen::visit(const ForStmt *stmt) {
    _scratch << "for (; ";
    stmt->condition()->accept(*this);
    _scratch << "; ";
    stmt->variable()->accept(*this);
    _scratch << " += ";
    stmt->step()->accept(*this);
    _scratch << ") ";
    stmt->body()->accept(*this);
}

void MetalCodegen::visit(const CommentStmt *stmt) {
    _scratch << "// ";
    for (auto c : stmt->comment()) {
        _scratch << std::string_view{&c, 1u};
        if (c == '\n') {
            _emit_indent();
            _scratch << "// ";
        }
    }
}

void MetalCodegen::visit(const MetaStmt *stmt) {
    // TODO: evaluate info
    LUISA_VERBOSE_WITH_LOCATION(
        "Generating code for meta statement body: {}.",
        stmt->info());
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

void MetalCodegen::_emit_declarations(const MetaStmt *meta) noexcept {
    for (auto v : meta->variables()) {
        _scratch << "\n  ";
        if (v.tag() == Variable::Tag::SHARED) {
            if (_function.tag() != Function::Tag::KERNEL) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Non-kernel functions are not allowed to have shared variables.");
            }
            _scratch << "threadgroup ";
        }
        _emit_type_name(v.type());
        _scratch << " ";
        _emit_variable_name(v);
        if (v.tag() == Variable::Tag::LOCAL) {
            _scratch << "{}";
        }
        _scratch << ";";
    }
    for (auto child : meta->children()) {
        _emit_declarations(child);
    }
}

void MetalCodegen::_emit_preamble(Function f) noexcept {

    _scratch << R"(#include <metal_stdlib>

using namespace metal;

template<typename T>
[[nodiscard, gnu::always_inline]] inline auto address_of(thread T &x) {
  return &x;
}

template<typename T>
[[nodiscard, gnu::always_inline]] inline auto address_of(threadgroup T &x) {
  return &x;
}

template<typename T>
[[nodiscard, gnu::always_inline]] inline auto address_of(device T &x) {
  return &x;
}

namespace detail {
  template<typename T>
  inline auto vector_element_impl(T v) { return v.x; }
}

template<typename T>
struct vector_element {
  using type = decltype(detail::vector_element_impl(T{}));
};

template<typename T>
using vector_element_t = typename vector_element<T>::type;

template<uint index, typename T>
[[nodiscard, gnu::always_inline]] inline auto vector_element_ptr(thread T &v) {
  return reinterpret_cast<thread vector_element_t<T> *>(&v) + index;
}

template<uint index, typename T>
[[nodiscard, gnu::always_inline]] inline auto vector_element_ptr(threadgroup T &v) {
  return reinterpret_cast<threadgroup vector_element_t<T> *>(&v) + index;
}

template<uint index, typename T>
[[nodiscard, gnu::always_inline]] inline auto vector_element_ptr(device T &v) {
  return reinterpret_cast<device vector_element_t<T> *>(&v) + index;
}

template<typename T>
[[nodiscard, gnu::always_inline]] inline auto none(T v) { return !any(v); }

template<typename T, access a>
[[nodiscard, gnu::always_inline]] inline auto texture_read(texture2d<T, a> t, uint2 uv) {
  if constexpr (a == access::read_write) { t.fence(); }
  return t.read(uv);
}

template<typename T, access a>
[[nodiscard, gnu::always_inline]] inline auto texture_read(texture3d<T, a> t, uint3 uvw) {
  if constexpr (a == access::read_write) { t.fence(); }
  return t.read(uvw);
}

template<typename T, access a, typename Value>
[[gnu::always_inline]] inline void texture_write(texture2d<T, a> t, uint2 uv, Value value) {
  t.write(value, uv);
}

template<typename T, access a, typename Value>
[[gnu::always_inline]] inline void texture_write(texture3d<T, a> t, uint3 uvw, Value value) {
  t.write(value, uvw);
}

template<typename T, access a, typename L>
[[nodiscard, gnu::always_inline]] inline auto texture_read_level(texture2d<T, a> t, uint2 uv, L l) {
  if constexpr (a == access::read_write) { t.fence(); }
  return t.read(uv, l);
}

template<typename T, access a, typename L>
[[nodiscard, gnu::always_inline]] inline auto texture_read(texture3d<T, a> t, uint3 uvw, L l) {
  if constexpr (a == access::read_write) { t.fence(); }
  return t.read(uvw, l);
}

template<typename T, access a, typename Value, typename L>
[[gnu::always_inline]] inline void texture_write(texture2d<T, a> t, uint2 uv, Value value, L l) {
  t.write(value, uv, l);
}

template<typename T, access a, typename Value, typename L>
[[gnu::always_inline]] inline void texture_write(texture3d<T, a> t, uint3 uvw, Value value, L l) {
  t.write(value, uvw, l);
}

template<typename T>
[[nodiscard, gnu::always_inline]] inline auto radians(T v) { return v * (M_PI_F / 180.0f); }

template<typename T>
[[nodiscard, gnu::always_inline]] inline auto degrees(T v) { return v * (180.0f * M_1_PI_F); }

[[nodiscard]] inline auto inverse(float2x2 m) {
  const auto one_over_determinant = 1.0f / (m[0][0] * m[1][1] - m[1][0] * m[0][1]);
  return float2x2(m[1][1] * one_over_determinant,
				- m[0][1] * one_over_determinant,
				- m[1][0] * one_over_determinant,
				+ m[0][0] * one_over_determinant);
}

[[nodiscard]] inline auto inverse(float3x3 m) {
  const auto one_over_determinant = 1.0f / (m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z)
                                          - m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z)
                                          + m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z));
  return float3x3(
    (m[1].y * m[2].z - m[2].y * m[1].z) * one_over_determinant,
    (m[2].y * m[0].z - m[0].y * m[2].z) * one_over_determinant,
    (m[0].y * m[1].z - m[1].y * m[0].z) * one_over_determinant,
    (m[2].x * m[1].z - m[1].x * m[2].z) * one_over_determinant,
    (m[0].x * m[2].z - m[2].x * m[0].z) * one_over_determinant,
    (m[1].x * m[0].z - m[0].x * m[1].z) * one_over_determinant,
    (m[1].x * m[2].y - m[2].x * m[1].y) * one_over_determinant,
    (m[2].x * m[0].y - m[0].x * m[2].y) * one_over_determinant,
    (m[0].x * m[1].y - m[1].x * m[0].y) * one_over_determinant);
}

[[nodiscard]] inline auto inverse(float4x4 m) {
  const auto coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
  const auto coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
  const auto coef03 = m[1].z * m[2].w - m[2].z * m[1].w;
  const auto coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
  const auto coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
  const auto coef07 = m[1].y * m[2].w - m[2].y * m[1].w;
  const auto coef08 = m[2].y * m[3].z - m[3].y * m[2].z;
  const auto coef10 = m[1].y * m[3].z - m[3].y * m[1].z;
  const auto coef11 = m[1].y * m[2].z - m[2].y * m[1].z;
  const auto coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
  const auto coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
  const auto coef15 = m[1].x * m[2].w - m[2].x * m[1].w;
  const auto coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
  const auto coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
  const auto coef19 = m[1].x * m[2].z - m[2].x * m[1].z;
  const auto coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
  const auto coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
  const auto coef23 = m[1].x * m[2].y - m[2].x * m[1].y;
  const auto fac0 = float4{coef00, coef00, coef02, coef03};
  const auto fac1 = float4{coef04, coef04, coef06, coef07};
  const auto fac2 = float4{coef08, coef08, coef10, coef11};
  const auto fac3 = float4{coef12, coef12, coef14, coef15};
  const auto fac4 = float4{coef16, coef16, coef18, coef19};
  const auto fac5 = float4{coef20, coef20, coef22, coef23};
  const auto Vec0 = float4{m[1].x, m[0].x, m[0].x, m[0].x};
  const auto Vec1 = float4{m[1].y, m[0].y, m[0].y, m[0].y};
  const auto Vec2 = float4{m[1].z, m[0].z, m[0].z, m[0].z};
  const auto Vec3 = float4{m[1].w, m[0].w, m[0].w, m[0].w};
  const auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
  const auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
  const auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
  const auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
  constexpr auto sign_a = float4{+1.0f, -1.0f, +1.0f, -1.0f};
  constexpr auto sign_b = float4{-1.0f, +1.0f, -1.0f, +1.0f};
  const auto inv_0 = inv0 * sign_a;
  const auto inv_1 = inv1 * sign_b;
  const auto inv_2 = inv2 * sign_a;
  const auto inv_3 = inv3 * sign_b;
  const auto dot0 = m[0] * float4{inv_0.x, inv_1.x, inv_2.x, inv_3.x};
  const auto dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
  const auto one_over_determinant = 1.0f / dot1;
  return float4x4(inv_0 * one_over_determinant,
                  inv_1 * one_over_determinant,
                  inv_2 * one_over_determinant,
                  inv_3 * one_over_determinant);
}

[[gnu::always_inline]] inline void block_barrier() {
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

[[gnu::always_inline]] inline void device_barrier() {
  threadgroup_barrier(mem_flags::mem_device);
}

[[gnu::always_inline]] inline void all_barrier() {
  block_barrier();
  device_barrier();
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(device int &a) {
  return reinterpret_cast<device atomic_int *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(device uint &a) {
  return reinterpret_cast<device atomic_uint *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(threadgroup int &a) {
  return reinterpret_cast<threadgroup atomic_int *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(threadgroup uint &a) {
  return reinterpret_cast<threadgroup atomic_uint *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(device const int &a) {
  return reinterpret_cast<device const atomic_int *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(device const uint &a) {
  return reinterpret_cast<device const atomic_uint *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(threadgroup const int &a) {
  return reinterpret_cast<threadgroup const atomic_int *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto as_atomic(threadgroup const uint &a) {
  return reinterpret_cast<threadgroup const atomic_uint *>(&a);
}

[[gnu::always_inline, nodiscard]] inline auto atomic_compare_exchange(device atomic_int *a, int cmp, int val, memory_order) {
  atomic_compare_exchange_weak_explicit(a, &cmp, val, memory_order_relaxed, memory_order_relaxed);
  return cmp;
}

[[gnu::always_inline, nodiscard]] inline auto atomic_compare_exchange(threadgroup atomic_int *a, int cmp, int val, memory_order) {
  atomic_compare_exchange_weak_explicit(a, &cmp, val, memory_order_relaxed, memory_order_relaxed);
  return cmp;
}

[[gnu::always_inline, nodiscard]] inline auto atomic_compare_exchange(device atomic_uint *a, uint cmp, uint val, memory_order) {
  atomic_compare_exchange_weak_explicit(a, &cmp, val, memory_order_relaxed, memory_order_relaxed);
  return cmp;
}

[[gnu::always_inline, nodiscard]] inline auto atomic_compare_exchange(threadgroup atomic_uint *a, uint cmp, uint val, memory_order) {
  atomic_compare_exchange_weak_explicit(a, &cmp, val, memory_order_relaxed, memory_order_relaxed);
  return cmp;
}

template<typename X, typename Y>
[[gnu::always_inline, nodiscard]] inline auto glsl_mod(X x, Y y) {
  return x - y * floor(x / y);
}

[[gnu::always_inline, nodiscard]] inline auto is_nan(float x) {
  auto u = as_type<uint>(x);
  return (u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu);
}

[[gnu::always_inline, nodiscard]] inline auto is_nan(float2 v) {
  return bool2(is_nan(v.x), is_nan(v.y));
}

[[gnu::always_inline, nodiscard]] inline auto is_nan(float3 v) {
  return bool3(is_nan(v.x), is_nan(v.y), is_nan(v.z));
}

[[gnu::always_inline, nodiscard]] inline auto is_nan(float4 v) {
  return bool4(is_nan(v.x), is_nan(v.y), is_nan(v.z), is_nan(v.w));
}

[[gnu::always_inline, nodiscard]] inline auto is_inf(float x) {
  auto u = as_type<uint>(x);
  return (u & 0x7F800000u) == 0x7F800000u && !(u & 0x7FFFFFu);
}

[[gnu::always_inline, nodiscard]] inline auto is_inf(float2 v) {
  return bool2(is_inf(v.x), is_inf(v.y));
}

[[gnu::always_inline, nodiscard]] inline auto is_inf(float3 v) {
  return bool3(is_inf(v.x), is_inf(v.y), is_inf(v.z));
}

[[gnu::always_inline, nodiscard]] inline auto is_inf(float4 v) {
  return bool4(is_inf(v.x), is_inf(v.y), is_inf(v.z), is_inf(v.w));
}

template<typename T>
[[gnu::always_inline, nodiscard]] inline auto select(T f, T t, bool b) {
  return b ? t : f;
}

struct alignas(16) HeapItem {
  metal::texture2d<float> handle2d;
  metal::texture3d<float> handle3d;
  metal::sampler sampler;
  device const void *buffer;
};

struct alignas(16) Ray {
  array<float, 3> m0;
  float m1;
  array<float, 3> m2;
  float m3;
};

struct alignas(16) Hit {
  uint m0;
  uint m1;
  float2 m2;
};

[[nodiscard]] auto texture_heap_sample2d(device const HeapItem *heap, uint index, float2 uv) {
  device const auto &t = heap[index];
  return t.handle2d.sample(t.sampler, uv);
}

[[nodiscard]] auto texture_heap_sample3d(device const HeapItem *heap, uint index, float3 uvw) {
  device const auto &t = heap[index];
  return t.handle3d.sample(t.sampler, uvw);
}

[[nodiscard]] auto texture_heap_sample2d_level(device const HeapItem *heap, uint index, float2 uv, float lod) {
  device const auto &t = heap[index];
  return t.handle2d.sample(t.sampler, uv, level(lod));
}

[[nodiscard]] auto texture_heap_sample3d_level(device const HeapItem *heap, uint index, float3 uvw, float lod) {
  device const auto &t = heap[index];
  return t.handle3d.sample(t.sampler, uvw, level(lod));
}

[[nodiscard]] auto texture_heap_sample2d_grad(device const HeapItem *heap, uint index, float2 uv, float2 dpdx, float2 dpdy) {
  device const auto &t = heap[index];
  return t.handle2d.sample(t.sampler, uv, gradient2d(dpdx, dpdy));
}

[[nodiscard]] auto texture_heap_sample3d_grad(device const HeapItem *heap, uint index, float3 uvw, float3 dpdx, float3 dpdy) {
  device const auto &t = heap[index];
  return t.handle3d.sample(t.sampler, uvw, gradient3d(dpdx, dpdy));
}

[[nodiscard]] auto texture_heap_size2d(device const HeapItem *heap, uint i) {
  return uint2(heap[i].handle2d.get_width(), heap[i].handle2d.get_height());
}

[[nodiscard]] auto texture_heap_size3d(device const HeapItem *heap, uint i) {
  return uint3(heap[i].handle3d.get_width(), heap[i].handle3d.get_height(), heap[i].handle3d.get_depth());
}

[[nodiscard]] auto texture_heap_size2d_level(device const HeapItem *heap, uint i, uint lv) {
  return uint2(heap[i].handle2d.get_width(lv), heap[i].handle2d.get_height(lv));
}

[[nodiscard]] auto texture_heap_size3d_level(device const HeapItem *heap, uint i, uint lv) {
  return uint3(heap[i].handle3d.get_width(lv), heap[i].handle3d.get_height(lv), heap[i].handle3d.get_depth(lv));
}

[[nodiscard]] auto texture_heap_read2d(device const HeapItem *heap, uint i, uint2 uv) {
  return heap[i].handle2d.read(uv);
}

[[nodiscard]] auto texture_heap_read3d(device const HeapItem *heap, uint i, uint3 uvw) {
  return heap[i].handle3d.read(uvw);
}

[[nodiscard]] auto texture_heap_read2d_level(device const HeapItem *heap, uint i, uint2 uv, uint lv) {
  return heap[i].handle2d.read(uv, lv);
}

[[nodiscard]] auto texture_heap_read3d_level(device const HeapItem *heap, uint i, uint3 uvw, uint lv) {
  return heap[i].handle3d.read(uvw, lv);
}

template<typename T>
[[nodiscard]] auto buffer_heap_read(device const HeapItem *heap, uint buffer_index, uint i) {
  return static_cast<device const T *>(heap[buffer_index].buffer)[i];
}

)";

    if (f.raytracing()) {
        _scratch << R"(using namespace metal::raytracing;

[[nodiscard]] constexpr auto intersector_closest() {
  intersector<triangle_data, instancing> i;
  i.assume_geometry_type(geometry_type::triangle);
  i.force_opacity(forced_opacity::opaque);
  i.accept_any_intersection(false);
  return i;
}

[[nodiscard]] constexpr auto intersector_any() {
  intersector<triangle_data, instancing> i;
  i.assume_geometry_type(geometry_type::triangle);
  i.force_opacity(forced_opacity::opaque);
  i.accept_any_intersection(true);
  return i;
}

[[nodiscard]] auto make_ray(Ray r_in) {
  auto o = float3(r_in.m0[0], r_in.m0[1], r_in.m0[2]);
  auto d = float3(r_in.m2[0], r_in.m2[1], r_in.m2[2]);
  return ray{o, d, r_in.m1, r_in.m3};
}

[[nodiscard]] auto trace_closest(instance_acceleration_structure accel, Ray r) {
  auto isect = intersector_closest().intersect(make_ray(r), accel);
  return isect.type == intersection_type::none ?
    Hit{0xffffffffu, 0xffffffffu, float2(0.0f)} :
    Hit{isect.instance_id, isect.primitive_id, isect.triangle_barycentric_coord};
}

[[nodiscard]] auto trace_any(instance_acceleration_structure accel, Ray r) {
  auto isect = intersector_any().intersect(make_ray(r), accel);
  return isect.type != intersection_type::none;
}

)";
    }
}

}
