#include <luisa/core/logging.h>
#include <luisa/core/magic_enum.h>
#include <luisa/runtime/rtx/ray.h>
#include <luisa/runtime/rtx/hit.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/runtime/dispatch_buffer.h>
#include "metal_builtin_embedded.h"
#include "metal_codegen_ast.h"

namespace luisa::compute::metal {

namespace detail {

class LiteralPrinter {

private:
    StringScratch &_s;

public:
    explicit LiteralPrinter(StringScratch &s) noexcept : _s{s} {}
    void operator()(bool v) const noexcept { _s << v; }
    void operator()(float v) const noexcept {
        if (luisa::isnan(v)) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Encountered with NaN."); }
        if (luisa::isinf(v)) {
            _s << (v < 0.0f ? "(-INFINITY)" : "(+INFINITY)");
        } else {
            _s << v << "f";
        }
    }
    void operator()(double v) const noexcept {
        if (std::isnan(v)) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Encountered with NaN."); }
        if (std::isinf(v)) {
            _s << (v < 0.0 ? "double(-INFINITY)" : "double(+INFINITY)");
        } else {
            _s << v;
        }
        LUISA_ERROR_WITH_LOCATION("Double literals are not supported.");
    }
    void operator()(half v) const noexcept {
        if (luisa::isnan(v)) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Encountered with NaN."); }
        if (luisa::isinf(v)) {
            _s << (v < 0.0f ? "half(-INFINITY)" : "half(+INFINITY)");
        } else {
            _s << static_cast<float>(v) << "h";
        }
    }
    void operator()(int v) const noexcept { _s << v; }
    void operator()(uint v) const noexcept { _s << v << "u"; }
    void operator()(short v) const noexcept { _s << luisa::format("ushort({})", v); }
    void operator()(ushort v) const noexcept { _s << luisa::format("short({})", v); }
    void operator()(slong v) const noexcept { _s << luisa::format("{}ll", v); }
    void operator()(ulong v) const noexcept { _s << luisa::format("{}ull", v); }

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
};

}// namespace detail

MetalCodegenAST::MetalCodegenAST(StringScratch &scratch) noexcept
    : _scratch{scratch},
      _ray_type{Type::of<Ray>()},
      _triangle_hit_type{Type::of<TriangleHit>()},
      _procedural_hit_type{Type::of<ProceduralHit>()},
      _committed_hit_type{Type::of<CommittedHit>()},
      _ray_query_all_type{Type::of<RayQueryAll>()},
      _ray_query_any_type{Type::of<RayQueryAny>()},
      _indirect_dispatch_buffer_type{Type::of<IndirectDispatchBuffer>()} {}

size_t MetalCodegenAST::type_size_bytes(const Type *type) noexcept {
    if (!type->is_custom()) { return type->size(); }
    LUISA_ERROR_WITH_LOCATION("Cannot get size of custom type.");
}

static void collect_types_in_function(Function f,
                                      luisa::unordered_set<const Type *> &types,
                                      luisa::unordered_set<Function> &visited) noexcept {

    // already visited
    if (!visited.emplace(f).second) { return; }

    // types from variables
    auto add = [&](auto &&self, auto t) noexcept -> void {
        if (t != nullptr && types.emplace(t).second) {
            if (t->is_array() || t->is_buffer()) {
                self(self, t->element());
            } else if (t->is_structure()) {
                for (auto m : t->members()) {
                    self(self, m);
                }
            }
        }
    };
    for (auto &&a : f.arguments()) { add(add, a.type()); }
    for (auto &&l : f.local_variables()) { add(add, l.type()); }
    traverse_expressions<true>(
        f.body(),
        [&add](auto expr) noexcept {
            if (auto type = expr->type()) {
                add(add, type);
            }
        },
        [](auto) noexcept {},
        [](auto) noexcept {});
    add(add, f.return_type());

    // types from called callables
    for (auto &&c : f.custom_callables()) {
        collect_types_in_function(
            Function{c.get()}, types, visited);
    }
}

void MetalCodegenAST::_emit_type_decls(Function kernel) noexcept {

    // collect used types in the kernel
    luisa::unordered_set<const Type *> types;
    luisa::unordered_set<Function> visited;
    collect_types_in_function(kernel, types, visited);

    // sort types by name so the generated
    // source is identical across runs
    luisa::vector<const Type *> sorted;
    sorted.reserve(types.size());
    std::copy(types.cbegin(), types.cend(),
              std::back_inserter(sorted));
    std::sort(sorted.begin(), sorted.end(), [](auto a, auto b) noexcept {
        return a->hash() < b->hash();
    });

    auto do_emit = [this](const Type *type) noexcept {
        if (type->is_structure() &&
            type != _ray_type &&
            type != _triangle_hit_type &&
            type != _procedural_hit_type &&
            type != _committed_hit_type &&
            type != _ray_query_all_type &&
            type != _ray_query_any_type &&
            type != _indirect_dispatch_buffer_type) {
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
        if (type->is_structure()) {
            // lc_zero and lc_one
            auto lc_make_value = [&](luisa::string_view name) noexcept {
                _scratch << "template<> inline auto " << name << "<";
                _emit_type_name(type);
                _scratch << ">() {\n"
                         << "  return ";
                _emit_type_name(type);
                _scratch << "{\n";
                for (auto i = 0u; i < type->members().size(); i++) {
                    _scratch << "    " << name << "<";
                    _emit_type_name(type->members()[i]);
                    _scratch << ">(),\n";
                }
                _scratch << "  };\n"
                         << "}\n\n";
            };
            lc_make_value("lc_zero");
            lc_make_value("lc_one");
            // lc_accumulate_grad
            _scratch << "inline void lc_accumulate_grad(thread ";
            _emit_type_name(type);
            _scratch << " *dst, ";
            _emit_type_name(type);
            _scratch << " grad) {\n";
            for (auto i = 0u; i < type->members().size(); i++) {
                _scratch << "  lc_accumulate_grad(&dst->m" << i << ", grad.m" << i << ");\n";
            }
            _scratch << "}\n\n";
        }
    };

    // process types in topological order
    types.clear();
    auto emit = [&](auto &&self, auto type) noexcept -> void {
        if (types.emplace(type).second) {
            if (type->is_array() || type->is_buffer()) {
                self(self, type->element());
            } else if (type->is_structure()) {
                for (auto m : type->members()) {
                    self(self, m);
                }
            }
            do_emit(type);
        }
    };

    _scratch << "/* user-defined structures begin */\n\n";
    for (auto t : sorted) { emit(emit, t); }
    _scratch << "/* user-defined structures end */\n\n";
}

void MetalCodegenAST::_emit_type_name(const Type *type, Usage usage) noexcept {

    if (type == nullptr) {
        _scratch << "void";
        return;
    }

    switch (type->tag()) {
        case Type::Tag::BOOL: _scratch << "bool"; break;
        case Type::Tag::FLOAT16: _scratch << "half"; break;
        case Type::Tag::FLOAT32: _scratch << "float"; break;
        case Type::Tag::FLOAT64: _scratch << "double"; break;
        case Type::Tag::INT8: _scratch << "char"; break;
        case Type::Tag::UINT8: _scratch << "uchar"; break;
        case Type::Tag::INT16: _scratch << "short"; break;
        case Type::Tag::UINT16: _scratch << "ushort"; break;
        case Type::Tag::INT32: _scratch << "int"; break;
        case Type::Tag::UINT32: _scratch << "uint"; break;
        case Type::Tag::INT64: _scratch << "long"; break;
        case Type::Tag::UINT64: _scratch << "ulong"; break;
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
        case Type::Tag::STRUCTURE: {
            if (type == _ray_type) {
                _scratch << "LCRay";
            } else if (type == _triangle_hit_type) {
                _scratch << "LCTriangleHit";
            } else if (type == _procedural_hit_type) {
                _scratch << "LCProceduralHit";
            } else if (type == _committed_hit_type) {
                _scratch << "LCCommittedHit";
            } else {
                _scratch << "S" << hash_to_string(type->hash());
            }
            break;
        }
        case Type::Tag::BUFFER:
            _scratch << "LCBuffer<";
            if (usage == Usage::NONE || usage == Usage::READ) { _scratch << "const "; }
            if (auto elem = type->element()) {
                _emit_type_name(elem);
            } else {
                _scratch << "lc_byte";
            }
            _scratch << ">";
            break;
        case Type::Tag::TEXTURE: {
            _scratch << "texture" << type->dimension() << "d<";
            auto elem = type->element();
            if (elem->is_vector()) { elem = elem->element(); }
            LUISA_ASSERT(elem->is_int32() || elem->is_uint32() || elem->is_float32(),
                         "Invalid texture element: {}.", elem->description());
            _emit_type_name(elem);
            _scratch << ", access::";
            if (usage == Usage::READ_WRITE) {
                _scratch << "read_write>";
            } else if (usage == Usage::WRITE) {
                _scratch << "write>";
            } else {
                _scratch << "read>";
            }
            break;
        }
        case Type::Tag::BINDLESS_ARRAY:
            _scratch << "LCBindlessArray";
            break;
        case Type::Tag::ACCEL:
            _scratch << "LCAccel";
            break;
        case Type::Tag::CUSTOM: {
            if (type == _ray_query_all_type ||
                type == _ray_query_any_type) {
                _scratch << "LCRayQuery";
            } else if (type == _indirect_dispatch_buffer_type) {
                _scratch << "LCIndirectDispatchBuffer";
            } else {
                LUISA_ERROR_WITH_LOCATION(
                    "Unsupported custom type: {}.",
                    type->description());
            }
            break;
        }
    }
}

void MetalCodegenAST::_emit_variable_name(Variable v) noexcept {
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
        case Variable::Tag::DISPATCH_SIZE: _scratch << "ds"; break;
        case Variable::Tag::KERNEL_ID: _scratch << "kid"; break;
        case Variable::Tag::WARP_LANE_COUNT: _scratch << "ws"; break;
        case Variable::Tag::WARP_LANE_ID: _scratch << "lid"; break;
        case Variable::Tag::OBJECT_ID: LUISA_NOT_IMPLEMENTED();
    }
}

void MetalCodegenAST::_emit_indention() noexcept {
    for (auto i = 0u; i < _indention; i++) { _scratch << "  "; }
}

void MetalCodegenAST::_emit_function() noexcept {

    LUISA_ASSERT(_function.tag() == Function::Tag::KERNEL ||
                     _function.tag() == Function::Tag::CALLABLE,
                 "Invalid function type '{}'",
                 luisa::to_string(_function.tag()));

    if (_function.tag() == Function::Tag::KERNEL) {

        // emit argument buffer struct
        _scratch << "struct alignas(16) Arguments {\n";
        for (auto arg : _function.arguments()) {
            _scratch << "  alignas(16) ";
            _emit_type_name(arg.type(), _function.variable_usage(arg.uid()));
            _scratch << " ";
            _emit_variable_name(arg);
            _scratch << ";\n";
        }
        if (_uses_printing) {
            _scratch << "  alignas(16) LCPrinterBuffer print_buffer;\n";
        }
        _scratch << "};\n\n";

        // emit function signature and prelude
        _scratch << "void kernel_main_impl(\n"
                 << "    constant Arguments &args,\n"
                 << "    uint3 tid, uint3 bid, uint3 did,\n"
                 << "    uint3 bs, uint ws, uint lid,\n"
                 << "    uint3 ds, uint kid";
        for (auto s : _function.shared_variables()) {
            _scratch << ", threadgroup ";
            _emit_type_name(s.type());
            _scratch << " &";
            _emit_variable_name(s);
        }
        _scratch << ") {\n"
                 << "  lc_assume("
                 << "bs.x == " << _function.block_size().x << " && "
                 << "bs.y == " << _function.block_size().y << " && "
                 << "bs.z == " << _function.block_size().z << ");\n"
                 << "  if (!all(did < ds)) { return; }\n\n"
                 << "  /* kernel arguments */\n";
        for (auto arg : _function.arguments()) {
            _scratch << "  auto ";
            _emit_variable_name(arg);
            _scratch << " = args.";
            _emit_variable_name(arg);
            _scratch << ";\n";
        }
        if (_uses_printing) {
            _scratch << "  auto print_buffer = args.print_buffer;\n";
        }
    } else {
        auto texture_count = std::count_if(
            _function.arguments().cbegin(), _function.arguments().cend(),
            [](auto arg) { return arg.type()->is_texture(); });
        if (texture_count > 0) {
            _scratch << "template<";
            for (auto i = 0u; i < texture_count; i++) {
                _scratch << "typename T" << i << ", ";
            }
            _scratch.pop_back();
            _scratch.pop_back();
            _scratch << ">\n";
        }
        _emit_type_name(_function.return_type());
        _scratch << " callable_" << hash_to_string(_function.hash()) << "(";
        auto emitted_texture_count = 0u;
        if (!_function.arguments().empty() || _uses_printing) {
            for (auto arg : _function.arguments()) {
                auto is_mut_ref = arg.is_reference() &&
                                  (to_underlying(_function.variable_usage(arg.uid())) &
                                   to_underlying(Usage::WRITE));
                if (is_mut_ref) { _scratch << "thread "; }
                if (arg.type()->is_texture()) {
                    _scratch << "T" << emitted_texture_count++;
                } else {
                    _emit_type_name(arg.type(), _function.variable_usage(arg.uid()));
                }
                _scratch << " ";
                if (is_mut_ref) { _scratch << "&"; }
                _emit_variable_name(arg);
                _scratch << ", ";
            }
            if (_uses_printing) {
                _scratch << "LCPrinterBuffer print_buffer";
            } else {
                _scratch.pop_back();
                _scratch.pop_back();
            }
        }
        _scratch << ") {\n";
    }

    // emit local variables
    _scratch << "\n  /* local variables */\n";
    for (auto local : _function.local_variables()) {
        _scratch << "  ";
        _emit_type_name(local.type(), _function.variable_usage(local.uid()));
        _scratch << " ";
        _emit_variable_name(local);
        _scratch << "{};\n";

        // create a shadow variable for ray query
        if (local.type() == _ray_query_any_type ||
            local.type() == _ray_query_all_type) {
            _scratch << "  LC_RAY_QUERY_SHADOW_VARIABLE(";
            _emit_variable_name(local);
            _scratch << ");\n";
        }
    }

    // emit gradient variables for autodiff
    luisa::unordered_set<Variable> gradient_variables;
    traverse_expressions<true>(
        _function.body(),
        [&](auto expr) noexcept {
            if (expr->tag() == Expression::Tag::CALL) {
                if (auto call = static_cast<const CallExpr *>(expr);
                    call->op() == CallOp::GRADIENT ||
                    call->op() == CallOp::GRADIENT_MARKER ||
                    call->op() == CallOp::REQUIRES_GRADIENT) {
                    LUISA_ASSERT(!call->arguments().empty() &&
                                     call->arguments().front()->tag() == Expression::Tag::REF,
                                 "Invalid gradient function call.");
                    auto v = static_cast<const RefExpr *>(call->arguments().front())->variable();
                    if (gradient_variables.emplace(v).second) {
                        _scratch << "  LC_GRAD_SHADOW_VARIABLE(";
                        _emit_variable_name(v);
                        _scratch << ");\n";
                    }
                }
            }
        },
        [](auto) noexcept {},
        [](auto) noexcept {});

    // emit function body
    _scratch << "\n  /* function body begin */\n";
    _indention = 1u;
    for (auto s : _function.body()->statements()) { s->accept(*this); }
    _scratch << "\n  /* function body end */\n";
    _scratch << "}\n\n";

    auto emit_shared_variable_decls = [&] {
        if (_function.tag() == Function::Tag::KERNEL &&
            !_function.shared_variables().empty()) {
            _scratch << "\n  /* shared variables */\n";
            for (auto shared : _function.shared_variables()) {
                _scratch << "  threadgroup ";
                _emit_type_name(shared.type());
                _scratch << " ";
                _emit_variable_name(shared);
                _scratch << ";\n";
            }
        }
    };

    // emit direct and indirect specializations
    if (_function.tag() == Function::Tag::KERNEL) {
        // direct dispatch
        _scratch << "[[kernel]] /* direct kernel dispatch entry */\n"
                 << "void kernel_main(\n"
                 << "    constant Arguments &args,\n"
                 << "    constant uint3 &ds,\n"
                 << "    uint3 tid [[thread_position_in_threadgroup]],\n"
                 << "    uint3 bid [[threadgroup_position_in_grid]],\n"
                 << "    uint3 did [[thread_position_in_grid]],\n"
                 << "    uint3 bs [[threads_per_threadgroup]],\n"
                 << "    uint ws [[threads_per_simdgroup]],\n"
                 << "    uint lid [[thread_index_in_simdgroup]]) {\n";
        emit_shared_variable_decls();
        _scratch << "  kernel_main_impl(args, tid, bid, did, bs, ws, lid, ds, 0u";
        for (auto s : _function.shared_variables()) {
            _scratch << ", ";
            _emit_variable_name(s);
        }
        _scratch << ");\n"
                 << "}\n\n";
        // indirect dispatch
        _scratch << "[[kernel]] /* indirect kernel dispatch entry */\n"
                 << "void kernel_main_indirect(\n"
                 << "    constant Arguments &args,\n"
                 << "    device uint4 &ds_kid,\n"
                 << "    uint3 tid [[thread_position_in_threadgroup]],\n"
                 << "    uint3 bid [[threadgroup_position_in_grid]],\n"
                 << "    uint3 did [[thread_position_in_grid]],\n"
                 << "    uint3 bs [[threads_per_threadgroup]],\n"
                 << "    uint ws [[threads_per_simdgroup]],\n"
                 << "    uint lid [[thread_index_in_simdgroup]]) {\n";
        emit_shared_variable_decls();
        _scratch << "  kernel_main_impl(args, tid, bid, did, bs, ws, lid, ds_kid.xyz, ds_kid.w";
        for (auto s : _function.shared_variables()) {
            _scratch << ", ";
            _emit_variable_name(s);
        }
        _scratch << ");\n"
                 << "}\n\n";
    }
}

class MetalConstantPrinter final : public ConstantDecoder {

private:
    MetalCodegenAST *_codegen;

public:
    explicit MetalConstantPrinter(MetalCodegenAST *codegen) noexcept
        : _codegen{codegen} {}

protected:
    void _decode_bool(bool x) noexcept override { _codegen->_scratch << (x ? "true" : "false"); }
    void _decode_char(char x) noexcept override { _codegen->_scratch << luisa::format("char({})", static_cast<int>(x)); }
    void _decode_uchar(uchar x) noexcept override { _codegen->_scratch << luisa::format("uchar({})", static_cast<uint>(x)); }
    void _decode_short(short x) noexcept override { _codegen->_scratch << luisa::format("short({})", x); }
    void _decode_ushort(ushort x) noexcept override { _codegen->_scratch << luisa::format("ushort({})", x); }
    void _decode_int(int x) noexcept override { _codegen->_scratch << luisa::format("int({})", x); }
    void _decode_uint(uint x) noexcept override { _codegen->_scratch << luisa::format("uint({})", x); }
    void _decode_long(slong x) noexcept override { _codegen->_scratch << luisa::format("long({})", x); }
    void _decode_ulong(ulong x) noexcept override { _codegen->_scratch << luisa::format("ulong({})", x); }
    void _decode_float(float x) noexcept override {
        _codegen->_scratch << "float(";
        detail::LiteralPrinter p{_codegen->_scratch};
        p(x);
        _codegen->_scratch << ")";
    }
    void _decode_half(half x) noexcept override {
        _codegen->_scratch << "half(";
        detail::LiteralPrinter p{_codegen->_scratch};
        p(x);
        _codegen->_scratch << ")";
    }
    void _decode_double(double x) noexcept override {
        _codegen->_scratch << "double(";
        detail::LiteralPrinter p{_codegen->_scratch};
        p(x);
        _codegen->_scratch << ")";
    }
    void _vector_separator(const Type *type, uint index) noexcept override {
        auto n = type->dimension();
        if (index == 0u) {
            _codegen->_emit_type_name(type);
            _codegen->_scratch << "(";
        } else if (index == n) {
            _codegen->_scratch << ")";
        } else {
            _codegen->_scratch << ", ";
        }
    }
    void _matrix_separator(const Type *type, uint index) noexcept override {
        auto n = type->dimension();
        if (index == 0u) {
            _codegen->_emit_type_name(type);
            _codegen->_scratch << "(";
        } else if (index == n) {
            _codegen->_scratch << ")";
        } else {
            _codegen->_scratch << ", ";
        }
    }
    void _struct_separator(const Type *type, uint index) noexcept override {
        auto n = type->members().size();
        if (index == 0u) {
            _codegen->_emit_type_name(type);
            _codegen->_scratch << "{";
        } else if (index == n) {
            _codegen->_scratch << "}";
        } else {
            _codegen->_scratch << ", ";
        }
    }
    void _array_separator(const Type *type, uint index) noexcept override {
        auto n = type->dimension();
        if (index == 0u) {
            _codegen->_emit_type_name(type);
            _codegen->_scratch << "{";
        } else if (index == n) {
            _codegen->_scratch << "}";
        } else {
            _codegen->_scratch << ", ";
        }
    }
};

void MetalCodegenAST::_emit_constant(const Function::Constant &c) noexcept {
    _scratch << "constant constexpr ";
    _emit_type_name(c.type());
    _scratch << " c" << hash_to_string(c.hash()) << " = ";
    MetalConstantPrinter printer{this};
    c.decode(printer);
    _scratch << ";\n\n";
}

void MetalCodegenAST::emit(Function kernel, luisa::string_view native_include) noexcept {

    _uses_printing = kernel.requires_printing();

    // ray query
    if (kernel.propagated_builtin_callables().uses_ray_query()) {
        _scratch << "#define LUISA_ENABLE_RAY_QUERY\n";
    }

    // motion blur
    if (kernel.requires_motion_blur()) {
        _scratch << "#define LUISA_ENABLE_MOTION_BLUR\n";
    }

    // curve basis
    if (auto bases = kernel.required_curve_bases(); bases.any()) {
        _scratch << "#define LUISA_ENABLE_CURVE\n";
        if (bases.count() > 1u) {
            _scratch << "#define LUISA_ENABLE_CURVE_MULTIPLE\n";
        }
        for (auto i = 0u; i < curve_basis_count; i++) {
            if (auto basis = static_cast<CurveBasis>(i); bases.test(basis)) {
                _scratch << "#define LUISA_ENABLE_CURVE_" << luisa::to_string(basis) << "\n";
            }
        }
        _scratch << "\n";
    }

    // emit device library
    _scratch << luisa::string_view{luisa_metal_builtin_metal_device_lib,
                                   sizeof(luisa_metal_builtin_metal_device_lib)}
             << "\n";

    // emit types
    _emit_type_decls(kernel);

    // emit native include
    if (!native_include.empty()) {
        _scratch << "\n/* native include begin */\n\n"
                 << native_include
                 << "\n/* native include end */\n\n";
    }

    // collect functions
    luisa::vector<Function> functions;
    {
        auto collect_functions = [&functions, collected = luisa::unordered_set<uint64_t>{}](
                                     auto &&self, Function function) mutable noexcept -> void {
            if (collected.emplace(function.hash()).second) {
                for (auto &&c : function.custom_callables()) { self(self, c->function()); }
                functions.emplace_back(function);
            }
        };
        collect_functions(collect_functions, kernel);
    }

    // collect constants
    {
        luisa::unordered_set<uint64_t> collected_constants;
        for (auto &&f : functions) {
            for (auto &&c : f.constants()) {
                if (collected_constants.emplace(c.hash()).second) {
                    _emit_constant(c);
                }
            }
        }
    }

    // emit functions
    for (auto f : functions) {
        _function = f;
        _emit_function();
    }
}

void MetalCodegenAST::visit(const UnaryExpr *expr) noexcept {
    switch (expr->op()) {
        case UnaryOp::PLUS: _scratch << "+"; break;
        case UnaryOp::MINUS: _scratch << "-"; break;
        case UnaryOp::NOT: _scratch << "!"; break;
        case UnaryOp::BIT_NOT: _scratch << "~"; break;
    }
    _scratch << "(";
    expr->operand()->accept(*this);
    _scratch << ")";
}

void MetalCodegenAST::visit(const BinaryExpr *expr) noexcept {
    _scratch << "(";
    expr->lhs()->accept(*this);
    _scratch << ")";
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
    _scratch << "(";
    expr->rhs()->accept(*this);
    _scratch << ")";
}

void MetalCodegenAST::visit(const MemberExpr *expr) noexcept {
    if (expr->is_swizzle()) {
        if (expr->swizzle_size() == 1u) {
            _scratch << "vector_element_ref(";
            expr->self()->accept(*this);
            _scratch << ", " << expr->swizzle_index(0u) << ")";
        } else {
            static constexpr std::string_view xyzw[]{"x", "y", "z", "w"};
            _scratch << "(";
            expr->self()->accept(*this);
            _scratch << ").";
            for (auto i = 0u; i < expr->swizzle_size(); i++) {
                _scratch << xyzw[expr->swizzle_index(i)];
            }
        }
    } else {
        _scratch << "(";
        expr->self()->accept(*this);
        _scratch << ").m" << expr->member_index();
    }
}

void MetalCodegenAST::visit(const AccessExpr *expr) noexcept {
    if (expr->range()->type()->is_vector()) {
        _scratch << "vector_element_ref(";
        expr->range()->accept(*this);
        _scratch << ", ";
        expr->index()->accept(*this);
        _scratch << ")";
    } else {
        _scratch << "(";
        expr->range()->accept(*this);
        _scratch << ")";
        _scratch << "[";
        expr->index()->accept(*this);
        _scratch << "]";
    }
}

void MetalCodegenAST::visit(const LiteralExpr *expr) noexcept {
    luisa::visit(detail::LiteralPrinter{_scratch}, expr->value());
}

void MetalCodegenAST::visit(const RefExpr *expr) noexcept {
    _emit_variable_name(expr->variable());
}

void MetalCodegenAST::_emit_access_chain(luisa::span<const Expression *const> chain) noexcept {
    auto type = chain.front()->type();
    _scratch << "vector_element_ref(";
    auto any_vector = false;
    chain.front()->accept(*this);
    for (auto index : chain.subspan(1u)) {
        switch (type->tag()) {
            case Type::Tag::VECTOR: {
                _scratch << ", ";
                index->accept(*this);
                _scratch << ")";
                type = type->element();
                any_vector = true;
                break;
            }
            case Type::Tag::ARRAY: {
                type = type->element();
                _scratch << "[";
                index->accept(*this);
                _scratch << "]";
                break;
            }
            case Type::Tag::MATRIX: {
                type = Type::vector(type->element(),
                                    type->dimension());
                _scratch << "[";
                index->accept(*this);
                _scratch << "]";
                break;
            }
            case Type::Tag::STRUCTURE: {
                LUISA_ASSERT(index->tag() == Expression::Tag::LITERAL,
                             "Indexing structure with non-constant "
                             "index is not supported.");
                auto literal = static_cast<const LiteralExpr *>(index)->value();
                auto i = luisa::holds_alternative<int>(literal) ?
                             static_cast<uint>(luisa::get<int>(literal)) :
                             luisa::get<uint>(literal);
                LUISA_ASSERT(i < type->members().size(),
                             "Index out of range.");
                type = type->members()[i];
                _scratch << ".m" << i;
                break;
            }
            case Type::Tag::BUFFER: {
                type = type->element();
                _scratch << ".data[";
                index->accept(*this);
                _scratch << "]";
                break;
            }
            default: LUISA_ERROR_WITH_LOCATION(
                "Invalid node type '{}' in access chain.",
                type->description());
        }
    }
    if (!any_vector) { _scratch << ")"; }
}

void MetalCodegenAST::visit(const CallExpr *expr) noexcept {

    switch (expr->op()) {
        case CallOp::CUSTOM: _scratch << "callable_" << hash_to_string(expr->custom().hash()); break;
        case CallOp::EXTERNAL: _scratch << expr->external()->name(); break;
        case CallOp::ALL: _scratch << "all"; break;
        case CallOp::ANY: _scratch << "any"; break;
        case CallOp::SELECT: _scratch << "lc_select"; break;
        case CallOp::CLAMP: _scratch << "clamp"; break;
        case CallOp::SATURATE: _scratch << "saturate"; break;
        case CallOp::LERP: _scratch << "mix"; break;
        case CallOp::STEP: _scratch << "step"; break;
        case CallOp::SMOOTHSTEP: _scratch << "smoothstep"; break;
        case CallOp::ABS: _scratch << "abs"; break;
        case CallOp::MIN: _scratch << "min"; break;
        case CallOp::MAX: _scratch << "max"; break;
        case CallOp::CLZ: _scratch << "clz"; break;
        case CallOp::CTZ: _scratch << "ctz"; break;
        case CallOp::POPCOUNT: _scratch << "popcount"; break;
        case CallOp::REVERSE: _scratch << "reverse_bits"; break;
        case CallOp::ISINF: _scratch << "lc_isinf"; break;
        case CallOp::ISNAN: _scratch << "lc_isnan"; break;
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
        case CallOp::REFLECT: _scratch << "reflect"; break;
        case CallOp::DETERMINANT: _scratch << "determinant"; break;
        case CallOp::TRANSPOSE: _scratch << "transpose"; break;
        case CallOp::INVERSE: _scratch << "inverse"; break;
        case CallOp::SYNCHRONIZE_BLOCK: _scratch << "block_barrier"; break;
        case CallOp::ATOMIC_EXCHANGE: _scratch << "lc_atomic_exchange"; break;
        case CallOp::ATOMIC_COMPARE_EXCHANGE: _scratch << "lc_atomic_compare_exchange"; break;
        case CallOp::ATOMIC_FETCH_ADD: _scratch << "lc_atomic_fetch_add"; break;
        case CallOp::ATOMIC_FETCH_SUB: _scratch << "lc_atomic_fetch_sub"; break;
        case CallOp::ATOMIC_FETCH_AND: _scratch << "lc_atomic_fetch_and"; break;
        case CallOp::ATOMIC_FETCH_OR: _scratch << "lc_atomic_fetch_or"; break;
        case CallOp::ATOMIC_FETCH_XOR: _scratch << "lc_atomic_fetch_xor"; break;
        case CallOp::ATOMIC_FETCH_MIN: _scratch << "lc_atomic_fetch_min"; break;
        case CallOp::ATOMIC_FETCH_MAX: _scratch << "lc_atomic_fetch_max"; break;
        case CallOp::BUFFER_READ: _scratch << "buffer_read"; break;
        case CallOp::BUFFER_WRITE: _scratch << "buffer_write"; break;
        case CallOp::BUFFER_SIZE: _scratch << "buffer_size"; break;
        case CallOp::BYTE_BUFFER_READ: _scratch << "byte_buffer_read"; break;
        case CallOp::BYTE_BUFFER_WRITE: _scratch << "byte_buffer_write"; break;
        case CallOp::BYTE_BUFFER_SIZE: _scratch << "byte_buffer_size"; break;
        case CallOp::TEXTURE_READ: _scratch << "texture_read"; break;
        case CallOp::TEXTURE_WRITE: _scratch << "texture_write"; break;
        case CallOp::TEXTURE_SIZE: _scratch << "texture_size"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE: _scratch << "bindless_texture_sample2d"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: _scratch << "bindless_texture_sample2d_level"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: _scratch << "bindless_texture_sample2d_grad"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: _scratch << "bindless_texture_sample2d_grad_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE: _scratch << "bindless_texture_sample3d"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: _scratch << "bindless_texture_sample3d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: _scratch << "bindless_texture_sample3d_grad"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: _scratch << "bindless_texture_sample3d_grad_level"; break;
        case CallOp::BINDLESS_TEXTURE2D_READ: _scratch << "bindless_texture_read2d"; break;
        case CallOp::BINDLESS_TEXTURE3D_READ: _scratch << "bindless_texture_read3d"; break;
        case CallOp::BINDLESS_TEXTURE2D_READ_LEVEL: _scratch << "bindless_texture_read2d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_READ_LEVEL: _scratch << "bindless_texture_read3d_level"; break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE: _scratch << "bindless_texture_size2d"; break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE: _scratch << "bindless_texture_size3d"; break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: _scratch << "bindless_texture_size2d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: _scratch << "bindless_texture_size3d_level"; break;
        case CallOp::BINDLESS_BUFFER_READ: {
            _scratch << "bindless_buffer_read<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::BINDLESS_BUFFER_WRITE: {
            _scratch << "bindless_buffer_write<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::BINDLESS_BYTE_BUFFER_READ: {
            _scratch << "bindless_byte_address_buffer_read<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::BINDLESS_BUFFER_SIZE: _scratch << "bindless_buffer_size"; break;
        case CallOp::BINDLESS_BUFFER_TYPE: _scratch << "bindless_buffer_type"; break;
#define LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(type, tag)        \
    case CallOp::MAKE_##tag##2: _scratch << #type "2"; break; \
    case CallOp::MAKE_##tag##3: _scratch << #type "3"; break; \
    case CallOp::MAKE_##tag##4: _scratch << #type "4"; break;
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(bool, BOOL)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(short, SHORT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(ushort, USHORT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(char, BYTE)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(uchar, UBYTE)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(int, INT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(uint, UINT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(long, LONG)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(ulong, ULONG)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(float, FLOAT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(half, HALF)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(double, DOUBLE)
#undef LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL
        case CallOp::MAKE_FLOAT2X2: _scratch << "float2x2"; break;
        case CallOp::MAKE_FLOAT3X3: _scratch << "float3x3"; break;
        case CallOp::MAKE_FLOAT4X4: _scratch << "float4x4"; break;
        case CallOp::ASSERT: _scratch << "lc_assert"; break;
        case CallOp::ASSUME: _scratch << "lc_assume"; break;
        case CallOp::UNREACHABLE: {
            _scratch << "lc_unreachable";
            if (auto type = expr->type()) {
                _scratch << "<";
                _emit_type_name(type);
                _scratch << ">";
            }
            break;
        }
        case CallOp::ZERO: {
            _scratch << "lc_zero<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::ONE: {
            _scratch << "lc_one<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::PACK: _scratch << "lc_pack"; break;
        case CallOp::UNPACK: {
            _scratch << "lc_unpack<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::RAY_TRACING_INSTANCE_TRANSFORM: _scratch << "accel_instance_transform"; break;
        case CallOp::RAY_TRACING_INSTANCE_USER_ID: _scratch << "accel_instance_user_id"; break;
        case CallOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: _scratch << "accel_instance_visibility_mask"; break;
        case CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM: _scratch << "accel_set_instance_transform"; break;
        case CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY: _scratch << "accel_set_instance_visibility"; break;
        case CallOp::RAY_TRACING_SET_INSTANCE_OPACITY: _scratch << "accel_set_instance_opacity"; break;
        case CallOp::RAY_TRACING_SET_INSTANCE_USER_ID: _scratch << "accel_set_instance_user_id"; break;
        case CallOp::RAY_TRACING_TRACE_CLOSEST: _scratch << "accel_trace_closest"; break;
        case CallOp::RAY_TRACING_TRACE_ANY: _scratch << "accel_trace_any"; break;
        case CallOp::RAY_TRACING_QUERY_ALL: _scratch << "accel_query_all"; break;
        case CallOp::RAY_TRACING_QUERY_ANY: _scratch << "accel_query_any"; break;
        case CallOp::RAY_QUERY_WORLD_SPACE_RAY: _scratch << "ray_query_world_ray"; break;
        case CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT: _scratch << "ray_query_procedural_candidate"; break;
        case CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT: _scratch << "ray_query_triangle_candidate"; break;
        case CallOp::RAY_QUERY_COMMITTED_HIT: _scratch << "ray_query_committed_hit"; break;
        case CallOp::RAY_QUERY_COMMIT_TRIANGLE: _scratch << "ray_query_commit_triangle"; break;
        case CallOp::RAY_QUERY_COMMIT_PROCEDURAL: _scratch << "ray_query_commit_procedural"; break;
        case CallOp::RAY_QUERY_TERMINATE: _scratch << "ray_query_terminate"; break;
        case CallOp::RAY_QUERY_PROCEED: _scratch << "ray_query_next"; break;
        case CallOp::RAY_QUERY_IS_TRIANGLE_CANDIDATE: _scratch << "ray_query_is_triangle_candidate"; break;
        case CallOp::RAY_QUERY_IS_PROCEDURAL_CANDIDATE: _scratch << "!ray_query_is_triangle_candidate"; break;
        case CallOp::REDUCE_SUM: _scratch << "lc_reduce_sum"; break;
        case CallOp::REDUCE_PRODUCT: _scratch << "lc_reduce_prod"; break;
        case CallOp::REDUCE_MIN: _scratch << "lc_reduce_min"; break;
        case CallOp::REDUCE_MAX: _scratch << "lc_reduce_max"; break;
        case CallOp::OUTER_PRODUCT: _scratch << "lc_outer_product"; break;
        case CallOp::MATRIX_COMPONENT_WISE_MULTIPLICATION: _scratch << "lc_mat_comp_mul"; break;
        case CallOp::REQUIRES_GRADIENT: _scratch << "LC_REQUIRES_GRAD"; break;
        case CallOp::GRADIENT: _scratch << "LC_GRAD"; break;
        case CallOp::GRADIENT_MARKER: _scratch << "LC_MARK_GRAD"; break;
        case CallOp::ACCUMULATE_GRADIENT: _scratch << "LC_ACCUM_GRAD"; break;
        case CallOp::BACKWARD: LUISA_ERROR_WITH_LOCATION("autodiff::backward() should have been lowered."); break;
        case CallOp::DETACH: {
            _scratch << "static_cast<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::RASTER_DISCARD: _scratch << "discard_fragment"; break;
        case CallOp::INDIRECT_SET_DISPATCH_COUNT: _scratch << "lc_indirect_dispatch_set_count"; break;
        case CallOp::INDIRECT_SET_DISPATCH_KERNEL: _scratch << "lc_indirect_dispatch_set_kernel"; break;
        case CallOp::DDX: _scratch << "dfdx"; break;
        case CallOp::DDY: _scratch << "dfdy"; break;
        case CallOp::WARP_IS_FIRST_ACTIVE_LANE: _scratch << "lc_warp_is_first_active_lane"; break;
        case CallOp::WARP_FIRST_ACTIVE_LANE: _scratch << "lc_warp_first_active_lane"; break;
        case CallOp::WARP_ACTIVE_ALL_EQUAL: _scratch << "lc_warp_active_all_equal"; break;
        case CallOp::WARP_ACTIVE_BIT_AND: _scratch << "lc_warp_active_bit_and"; break;
        case CallOp::WARP_ACTIVE_BIT_OR: _scratch << "lc_warp_active_bit_or"; break;
        case CallOp::WARP_ACTIVE_BIT_XOR: _scratch << "lc_warp_active_bit_xor"; break;
        case CallOp::WARP_ACTIVE_COUNT_BITS: _scratch << "lc_warp_active_count_bits"; break;
        case CallOp::WARP_ACTIVE_MAX: _scratch << "lc_warp_active_max"; break;
        case CallOp::WARP_ACTIVE_MIN: _scratch << "lc_warp_active_min"; break;
        case CallOp::WARP_ACTIVE_PRODUCT: _scratch << "lc_warp_active_product"; break;
        case CallOp::WARP_ACTIVE_SUM: _scratch << "lc_warp_active_sum"; break;
        case CallOp::WARP_ACTIVE_ALL: _scratch << "lc_warp_active_all"; break;
        case CallOp::WARP_ACTIVE_ANY: _scratch << "lc_warp_active_any"; break;
        case CallOp::WARP_ACTIVE_BIT_MASK: _scratch << "lc_warp_active_bit_mask"; break;
        case CallOp::WARP_PREFIX_COUNT_BITS: _scratch << "lc_warp_prefix_count_bits"; break;
        case CallOp::WARP_PREFIX_SUM: _scratch << "lc_warp_prefix_sum"; break;
        case CallOp::WARP_PREFIX_PRODUCT: _scratch << "lc_warp_prefix_product"; break;
        case CallOp::WARP_READ_LANE: _scratch << "lc_warp_read_lane"; break;
        case CallOp::WARP_READ_FIRST_ACTIVE_LANE: _scratch << "lc_warp_read_first_active_lane"; break;
        case CallOp::SHADER_EXECUTION_REORDER: _scratch << "lc_shader_execution_reorder"; break;
        case CallOp::ADDRESS_OF: _scratch << "lc_address_of"; break;
        case CallOp::BUFFER_ADDRESS: _scratch << "buffer_address"; break;
        case CallOp::BINDLESS_BUFFER_ADDRESS: _scratch << "bindless_buffer_address"; break;

        case CallOp::RAY_TRACING_INSTANCE_MOTION_MATRIX: break;
        case CallOp::RAY_TRACING_INSTANCE_MOTION_SRT: break;
        case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX: break;
        case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT: break;

        case CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: _scratch << "accel_trace_closest_motion_blur"; break;
        case CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: _scratch << "accel_trace_any_motion_blur"; break;
        case CallOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: _scratch << "accel_query_all_motion_blur"; break;
        case CallOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: _scratch << "accel_query_all_motion_blur"; break;

        case CallOp::TEXTURE2D_SAMPLE: LUISA_NOT_IMPLEMENTED();
        case CallOp::TEXTURE2D_SAMPLE_LEVEL: LUISA_NOT_IMPLEMENTED();
        case CallOp::TEXTURE2D_SAMPLE_GRAD: LUISA_NOT_IMPLEMENTED();
        case CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: LUISA_NOT_IMPLEMENTED();
        case CallOp::TEXTURE3D_SAMPLE: LUISA_NOT_IMPLEMENTED();
        case CallOp::TEXTURE3D_SAMPLE_LEVEL: LUISA_NOT_IMPLEMENTED();
        case CallOp::TEXTURE3D_SAMPLE_GRAD: LUISA_NOT_IMPLEMENTED();
        case CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: LUISA_NOT_IMPLEMENTED();
    }
    _scratch << "(";
    if (auto op = expr->op(); is_atomic_operation(op)) {
        // lower access chain to atomic operation
        auto args = expr->arguments();
        auto access_chain = args.subspan(
            0u,
            op == CallOp::ATOMIC_COMPARE_EXCHANGE ?
                args.size() - 2u :
                args.size() - 1u);
        _scratch << "as_ref(";
        _emit_access_chain(access_chain);
        _scratch << ")";
        for (auto extra : args.subspan(access_chain.size())) {
            _scratch << ", ";
            extra->accept(*this);
        }
    } else {
        auto trailing_comma = false;
        for (auto i = 0u; i < expr->arguments().size(); i++) {
            auto arg = expr->arguments()[i];
            trailing_comma = true;
            arg->accept(*this);
            _scratch << ", ";
        }
        if (expr->is_custom() && _uses_printing) {
            _scratch << "print_buffer";
            trailing_comma = false;
        }
        if (trailing_comma) {
            _scratch.pop_back();
            _scratch.pop_back();
        }
    }
    _scratch << ")";
}

void MetalCodegenAST::visit(const TypeIDExpr *expr) noexcept {
    _scratch << "static_cast<";
    _emit_type_name(expr->type());
    _scratch << ">(0ull)";
    // TODO: use expr->data_type() to generate correct type
}

void MetalCodegenAST::visit(const StringIDExpr *expr) noexcept {
    _scratch << "(static_cast<";
    _emit_type_name(expr->type());
    _scratch << luisa::format(">(0x{:016x}ull))", luisa::hash_value(expr->data()));
}

void MetalCodegenAST::visit(const CastExpr *expr) noexcept {
    switch (expr->op()) {
        case CastOp::STATIC: _scratch << "static_cast<"; break;
        case CastOp::BITWISE: _scratch << "bitcast<"; break;
    }
    _emit_type_name(expr->type());
    _scratch << ">(";
    expr->expression()->accept(*this);
    _scratch << ")";
}

void MetalCodegenAST::visit(const ConstantExpr *expr) noexcept {
    _scratch << "c" << hash_to_string(expr->data().hash());
}

void MetalCodegenAST::visit(const CpuCustomOpExpr *expr) noexcept {
    LUISA_ERROR_WITH_LOCATION("MetalCodegenAST: CpuCustomOpExpr not supported.");
}

void MetalCodegenAST::visit(const GpuCustomOpExpr *expr) noexcept {
    LUISA_ERROR_WITH_LOCATION("MetalCodegenAST: GpuCustomOpExpr not supported.");
}

void MetalCodegenAST::visit(const BreakStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "break;\n";
}

void MetalCodegenAST::visit(const ContinueStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "continue;\n";
}

void MetalCodegenAST::visit(const ReturnStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "return";
    if (auto expr = stmt->expression()) {
        _scratch << " ";
        expr->accept(*this);
    }
    _scratch << ";\n";
}

void MetalCodegenAST::visit(const ScopeStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "{\n";
    _indention++;
    for (auto s : stmt->statements()) { s->accept(*this); }
    _indention--;
    _emit_indention();
    _scratch << "}\n";
}

void MetalCodegenAST::visit(const IfStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "if (";
    stmt->condition()->accept(*this);
    _scratch << ") {\n";
    _indention++;
    for (auto s : stmt->true_branch()->statements()) {
        s->accept(*this);
    }
    _indention--;
    _emit_indention();
    _scratch << "}";
    if (auto &&fb = stmt->false_branch()->statements(); !fb.empty()) {
        _scratch << " else {\n";
        _indention++;
        for (auto s : fb) { s->accept(*this); }
        _indention--;
        _emit_indention();
        _scratch << "}";
    }
    _scratch << "\n";
}

void MetalCodegenAST::visit(const LoopStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "for (;;) {\n";
    _indention++;
    for (auto s : stmt->body()->statements()) {
        s->accept(*this);
    }
    _indention--;
    _emit_indention();
    _scratch << "}\n";
}

void MetalCodegenAST::visit(const ExprStmt *stmt) noexcept {
    _emit_indention();
    if (stmt->expression()->type() != nullptr) {
        _scratch << "static_cast<void>(";
    }
    stmt->expression()->accept(*this);
    if (stmt->expression()->type() != nullptr) {
        _scratch << ")";
    }
    _scratch << ";\n";
}

void MetalCodegenAST::visit(const SwitchStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "switch (";
    stmt->expression()->accept(*this);
    _scratch << ") {\n";
    _indention++;
    for (auto s : stmt->body()->statements()) {
        s->accept(*this);
    }
    _indention--;
    _emit_indention();
    _scratch << "}\n";
}

void MetalCodegenAST::visit(const SwitchCaseStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "case ";
    stmt->expression()->accept(*this);
    _scratch << ": {\n";
    _indention++;
    auto has_break = false;
    for (auto s : stmt->body()->statements()) {
        s->accept(*this);
        if (s->tag() == Statement::Tag::BREAK) {
            has_break = true;
            break;
        }
    }
    if (!has_break) {
        _emit_indention();
        _scratch << "break;\n";
    }
    _indention--;
    _emit_indention();
    _scratch << "}\n";
}

void MetalCodegenAST::visit(const SwitchDefaultStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "default: {\n";
    _indention++;
    auto has_break = false;
    for (auto s : stmt->body()->statements()) {
        s->accept(*this);
        if (s->tag() == Statement::Tag::BREAK) {
            has_break = true;
            break;
        }
    }
    if (!has_break) {
        _emit_indention();
        _scratch << "break;\n";
    }
    _indention--;
    _emit_indention();
    _scratch << "}\n";
}

void MetalCodegenAST::visit(const AssignStmt *stmt) noexcept {
    _emit_indention();
    stmt->lhs()->accept(*this);
    _scratch << " = ";
    stmt->rhs()->accept(*this);
    _scratch << ";\n";
}

void MetalCodegenAST::visit(const ForStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "for (; ";
    stmt->condition()->accept(*this);
    _scratch << "; ";
    stmt->variable()->accept(*this);
    _scratch << " += ";
    stmt->step()->accept(*this);
    _scratch << ") {\n";
    _indention++;
    for (auto s : stmt->body()->statements()) {
        s->accept(*this);
    }
    _indention--;
    _emit_indention();
    _scratch << "}\n";
}

void MetalCodegenAST::visit(const CommentStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "// ";
    for (auto c : stmt->comment()) {
        _scratch << std::string_view{&c, 1u};
        if (c == '\n') {
            _emit_indention();
            _scratch << "// ";
        }
    }
    _scratch << "\n";
}

void MetalCodegenAST::visit(const PrintStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "{// print\n";
    _indention++;
    _emit_indention();
    _scratch << "struct __lc_print_args_t {\n";
    _indention++;
    _emit_indention();
    _scratch << "uint size;\n";
    _emit_indention();
    _scratch << "uint token;\n";
    for (auto i = 0u; i < stmt->arguments().size(); i++) {
        _emit_indention();
        _emit_type_name(stmt->arguments()[i]->type(), Usage::READ);
        _scratch << " m" << i << ";\n";
    }
    _indention--;
    _emit_indention();
    _scratch << "};\n";
    auto [token, type] = [stmt, this] {
        luisa::vector<const Type *> arg_pack;
        arg_pack.reserve(stmt->arguments().size() + 2u);
        arg_pack.emplace_back(Type::of<uint>());// size
        arg_pack.emplace_back(Type::of<uint>());// fmt id
        for (auto a : stmt->arguments()) { arg_pack.emplace_back(a->type()); }
        auto s = Type::structure(arg_pack);
        for (auto i = 0u; i < _print_formats.size(); i++) {
            if (auto &&[f, t] = _print_formats[i];
                f == stmt->format() && t == s) {
                return std::make_pair(i, t);
            }
        }
        auto index = static_cast<uint>(_print_formats.size());
        _print_formats.emplace_back(stmt->format(), s);
        return std::make_pair(index, s);
    }();
    _emit_indention();
    _scratch << "static_assert(sizeof(__lc_print_args_t) == " << type->size() << "u);\n";
    _emit_indention();
    _scratch << "static_assert(alignof(__lc_print_args_t) == " << type->alignment() << "u);\n";
    _emit_indention();
    _scratch << "__lc_print_args_t print_args{"// TODO
             << ".size = " << type->size() << ", "
             << ".token = " << token << ", ";
    for (auto i = 0u; i < stmt->arguments().size(); i++) {
        _scratch << ".m" << i << " = ";
        stmt->arguments()[i]->accept(*this);
        _scratch << ", ";
    }
    _scratch.pop_back();
    _scratch.pop_back();
    _scratch << "};\n";
    _emit_indention();
    _scratch << "lc_print_impl(print_buffer, print_args);\n";
    _indention--;
    _emit_indention();
    _scratch << "}\n";
}

void MetalCodegenAST::visit(const RayQueryStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "/* ray query begin */\n";
    _emit_indention();
    if (stmt->on_procedural_candidate()->statements().empty()) {
        _scratch << "LC_RAY_QUERY_INIT_NO_PROCEDURAL(";
    } else {
        _scratch << "LC_RAY_QUERY_INIT(";
    }
    stmt->query()->accept(*this);
    _scratch << ");\n";
    _emit_indention();
    _scratch << "while (ray_query_next(";
    stmt->query()->accept(*this);
    _scratch << ")) {\n";
    _indention++;
    _emit_indention();
    _scratch << "if (ray_query_is_triangle_candidate(";
    stmt->query()->accept(*this);
    _scratch << ")) {\n";
    _indention++;
    _emit_indention();
    _scratch << "/* ray query triangle branch */\n";
    for (auto s : stmt->on_triangle_candidate()->statements()) {
        s->accept(*this);
    }
    _indention--;
    _emit_indention();
    _scratch << "} else {\n";
    _indention++;
    _emit_indention();
    _scratch << "/* ray query procedural branch */\n";
    for (auto s : stmt->on_procedural_candidate()->statements()) {
        s->accept(*this);
    }
    _indention--;
    _emit_indention();
    _scratch << "}\n";
    _indention--;
    _emit_indention();
    _scratch << "}\n";
    _emit_indention();
    _scratch << "/* ray query end */\n";
}

void MetalCodegenAST::visit(const AutoDiffStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "/* autodiff begin */\n";
    stmt->body()->accept(*this);
    _emit_indention();
    _scratch << "/* autodiff end */\n";
}

}// namespace luisa::compute::metal
