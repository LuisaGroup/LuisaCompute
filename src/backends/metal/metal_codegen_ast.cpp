#include <cstdlib>
#include <functional>
#include <luisa/core/logging.h>
#include <luisa/core/magic_enum.h>
#include <luisa/runtime/rtx/ray.h>
#include <luisa/runtime/rtx/hit.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include "metal_builtin_embedded.hpp"
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
        if (luisa::isnan(v)) [[unlikely]] {
            _s << luisa::format("as_type<float>(0x{:08x}u)", luisa::bit_cast<uint>(v));
        } else if (luisa::isinf(v)) {
            _s << (v < 0.0f ? "(-INFINITY)" : "(+INFINITY)");
        } else {
            _s << v << "f";
        }
    }
    void operator()(double v) const noexcept {
        if (std::isnan(v)) [[unlikely]] {
            _s << luisa::format("as_type<double>(0x{:016x}ull)", luisa::bit_cast<ulong>(v));
        } else if (std::isinf(v)) {
            _s << (v < 0.0 ? "-INFINITY" : "+INFINITY");
        } else {
            _s << v;
        }
    }
    void operator()(half v) const noexcept {
        if (luisa::isnan(v)) [[unlikely]] {
            _s << luisa::format("as_type<half>(ushort(0x{:04x}u))", luisa::bit_cast<ushort>(v));
        } else if (luisa::isinf(v)) {
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

[[nodiscard]] static auto sorted_custom_callables(Function function) noexcept {
    luisa::vector<Function> callables;
    callables.reserve(function.custom_callables().size());
    for (auto &&callable : function.custom_callables()) {
        callables.emplace_back(callable->function());
    }
    luisa::sort(
        callables.begin(), callables.end(),
        [](Function lhs, Function rhs) noexcept {
            return lhs.hash() < rhs.hash();
        });
    return callables;
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
    for (auto callable : sorted_custom_callables(f)) {
        collect_types_in_function(callable, types, visited);
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
    luisa::sort(sorted.begin(), sorted.end(), [](auto a, auto b) noexcept {
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
        if (type != nullptr && types.emplace(type).second) {
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

void MetalCodegenAST::_emit_type_name(const Type *type, Usage usage, bool sampled) noexcept {

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
// FP8 / I4 / FP4 have no Metal scalar type.  They are represented as
// byte storage (char / uchar) on the host-device boundary; the tile
// lowering handles pack/unpack.  These placeholders keep the type
// switch exhaustive so the Metal backend compiles cleanly even though
// standalone sub-byte typed buffers are not yet exercised on Metal.
case Type::Tag::FLOAT8_E4M3:
case Type::Tag::FLOAT8_E5M2:
case Type::Tag::FP4_E2M1: _scratch << "uchar"; break;
case Type::Tag::INT4: _scratch << "char"; break;
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
            if (sampled) {
                _scratch << "sample>";
            } else if ((to_underlying(usage) & to_underlying(Usage::READ_WRITE)) ==
                       to_underlying(Usage::READ_WRITE)) {
                _scratch << "read_write>";
            } else if ((to_underlying(usage) & to_underlying(Usage::WRITE)) != 0u) {
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
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Unsupported type: {}.",
                type->description());
            break;
    }
}

void MetalCodegenAST::_emit_variable_name(Variable v, bool sampled) noexcept {
    switch (v.tag()) {
        case Variable::Tag::LOCAL: _scratch << "v" << v.uid(); break;
        case Variable::Tag::SHARED: _scratch << "s" << v.uid(); break;
        case Variable::Tag::REFERENCE: _scratch << "r" << v.uid(); break;
        case Variable::Tag::BUFFER: _scratch << "b" << v.uid(); break;
        case Variable::Tag::TEXTURE:
            _scratch << "i" << v.uid();
            if (sampled) { _scratch << "_sampled"; }
            break;
        case Variable::Tag::BINDLESS_ARRAY: _scratch << "h" << v.uid(); break;
        case Variable::Tag::ACCEL: _scratch << "a" << v.uid(); break;
        case Variable::Tag::THREAD_ID: _scratch << "tid"; break;
        case Variable::Tag::BLOCK_ID: _scratch << "bid"; break;
        case Variable::Tag::DISPATCH_ID: _scratch << "did"; break;
        case Variable::Tag::DISPATCH_SIZE: _scratch << "ds"; break;
        case Variable::Tag::KERNEL_ID: _scratch << "kid"; break;
        case Variable::Tag::WARP_LANE_COUNT: _scratch << "ws"; break;
        case Variable::Tag::WARP_LANE_ID: _scratch << "lid"; break;
        case Variable::Tag::RASTER_BARYCENTRICS:
        case Variable::Tag::RASTER_OBJECT_ID:
        case Variable::Tag::RASTER_FRONT_FACING:
        case Variable::Tag::RASTER_BASE_INSTANCE: LUISA_NOT_IMPLEMENTED();
    }
}

void MetalCodegenAST::_emit_indention() noexcept {
    for (auto i = 0u; i < _indention; i++) { _scratch << "  "; }
}

bool MetalCodegenAST::_is_texture_sampled(Function function, Variable variable) const noexcept {
    if (auto iter = _sampled_texture_variables.find(function.hash());
        iter != _sampled_texture_variables.end()) {
        return iter->second.contains(variable.uid());
    }
    return false;
}

bool MetalCodegenAST::_is_texture_sampled(Variable variable) const noexcept {
    return _is_texture_sampled(_function, variable);
}

bool MetalCodegenAST::_is_texture_written(Function function, Variable variable) const noexcept {
    return (to_underlying(function.variable_usage(variable.uid())) &
            to_underlying(Usage::WRITE)) != 0u;
}

void MetalCodegenAST::_analyze_sampled_textures(Function function) noexcept {
    auto [iter, inserted] = _sampled_texture_variables.try_emplace(function.hash());
    if (!inserted) { return; }
    for (auto callable : sorted_custom_callables(function)) {
        _analyze_sampled_textures(callable);
    }
    auto &&sampled = iter->second;
    auto mark_argument = [&sampled](const Expression *expression) noexcept {
        if (expression->tag() == Expression::Tag::REF) {
            auto variable = static_cast<const RefExpr *>(expression)->variable();
            if (variable.type()->is_texture()) { sampled.emplace(variable.uid()); }
        }
    };
    traverse_expressions<true>(
        function.body(),
        [&](const Expression *expression) noexcept {
            if (expression->tag() != Expression::Tag::CALL) { return; }
            auto call = static_cast<const CallExpr *>(expression);
            switch (call->op()) {
                case CallOp::TEXTURE2D_SAMPLE:
                case CallOp::TEXTURE2D_SAMPLE_LEVEL:
                case CallOp::TEXTURE2D_SAMPLE_GRAD:
                case CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
                case CallOp::TEXTURE3D_SAMPLE:
                case CallOp::TEXTURE3D_SAMPLE_LEVEL:
                case CallOp::TEXTURE3D_SAMPLE_GRAD:
                case CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
                    mark_argument(call->arguments().front());
                    break;
                case CallOp::CUSTOM: {
                    auto callable = call->custom();
                    for (auto i = 0u; i < callable.arguments().size(); i++) {
                        if (_is_texture_sampled(callable, callable.arguments()[i])) {
                            mark_argument(call->arguments()[i]);
                        }
                    }
                    break;
                }
                default: break;
            }
        },
        [](auto) noexcept {},
        [](auto) noexcept {});
}

void MetalCodegenAST::_emit_call_argument(const Expression *argument, bool sampled) noexcept {
    if (sampled && argument->tag() == Expression::Tag::REF) {
        auto variable = static_cast<const RefExpr *>(argument)->variable();
        if (variable.type()->is_texture() &&
            _is_texture_sampled(variable) &&
            _is_texture_written(_function, variable)) {
            _emit_variable_name(variable, true);
            return;
        }
    }
    argument->accept(*this);
}

void MetalCodegenAST::_analyze_local_variables() noexcept {
    _local_variable_emissions.clear();
    _scope_local_variables.clear();
    _local_variable_initializers.clear();
    _gradient_variables.clear();

    auto root = _function.body();
    struct ScopeInfo {
        const ScopeStmt *parent{nullptr};
        uint depth{0u};
    };
    luisa::unordered_map<const ScopeStmt *, ScopeInfo> scope_info;
    luisa::unordered_set<const ScopeStmt *> switch_bodies;
    luisa::vector<const ScopeStmt *> scope_stack;

    const auto common_scope = [&scope_info](const ScopeStmt *lhs,
                                           const ScopeStmt *rhs) noexcept {
        auto lhs_depth = scope_info.at(lhs).depth;
        auto rhs_depth = scope_info.at(rhs).depth;
        while (lhs_depth > rhs_depth) {
            lhs = scope_info.at(lhs).parent;
            lhs_depth--;
        }
        while (rhs_depth > lhs_depth) {
            rhs = scope_info.at(rhs).parent;
            rhs_depth--;
        }
        while (lhs != rhs) {
            lhs = scope_info.at(lhs).parent;
            rhs = scope_info.at(rhs).parent;
        }
        return lhs;
    };

    traverse_expressions<true>(
        root,
        [&](const Expression *expression) noexcept {
            if (expression->tag() == Expression::Tag::REF) {
                auto variable = static_cast<const RefExpr *>(expression)->variable();
                if (variable.tag() == Variable::Tag::LOCAL) {
                    LUISA_ASSERT(!scope_stack.empty(),
                                 "A Metal local variable reference has no lexical scope.");
                    auto &&emission = _local_variable_emissions[variable.uid()];
                    emission.scope = emission.scope == nullptr ?
                                         scope_stack.back() :
                                         common_scope(emission.scope, scope_stack.back());
                }
            } else if (expression->tag() == Expression::Tag::CALL) {
                auto call = static_cast<const CallExpr *>(expression);
                if (call->op() == CallOp::GRADIENT ||
                    call->op() == CallOp::GRADIENT_MARKER ||
                    call->op() == CallOp::REQUIRES_GRADIENT) {
                    LUISA_ASSERT(!call->arguments().empty() &&
                                     call->arguments().front()->tag() == Expression::Tag::REF,
                                 "Invalid gradient function call.");
                    _gradient_variables.emplace(
                        static_cast<const RefExpr *>(
                            call->arguments().front())->variable());
                }
            }
        },
        [&](const Statement *statement) noexcept {
            if (statement->tag() == Statement::Tag::SWITCH) {
                switch_bodies.emplace(
                    static_cast<const SwitchStmt *>(statement)->body());
            }
            if (statement->tag() == Statement::Tag::SCOPE) {
                auto scope = static_cast<const ScopeStmt *>(statement);
                auto parent = scope_stack.empty() ? nullptr : scope_stack.back();
                scope_info.try_emplace(
                    scope, ScopeInfo{.parent = parent,
                                     .depth = parent == nullptr ?
                                                  0u :
                                                  scope_info.at(parent).depth + 1u});
                scope_stack.emplace_back(scope);
            }
        },
        [&](const Statement *statement) noexcept {
            if (statement->tag() == Statement::Tag::SCOPE) {
                LUISA_ASSERT(!scope_stack.empty() &&
                                 scope_stack.back() == statement,
                             "Metal local-scope traversal is unbalanced.");
                scope_stack.pop_back();
            }
        });

    for (auto variable : _gradient_variables) {
        if (auto iter = _local_variable_emissions.find(variable.uid());
            iter != _local_variable_emissions.end()) {
            iter->second.scope = root;
        }
    }

    for (auto local : _function.local_variables()) {
        auto iter = _local_variable_emissions.find(local.uid());
        if (iter == _local_variable_emissions.end()) {
            // FunctionBuilder may retain an unused local after an earlier
            // optimization. No emitted expression can name it, so Metal does
            // not need a declaration either.
            continue;
        }
        auto scope = iter->second.scope;
        // A declaration directly inside a switch body can be bypassed by a
        // case label. Lift it until ordinary structured scope rules apply.
        while (switch_bodies.contains(scope)) {
            scope = scope_info.at(scope).parent;
            LUISA_ASSERT(scope != nullptr,
                         "A Metal switch body has no enclosing declaration scope.");
        }
        iter->second.scope = scope;
    }

    struct ScopeTraversal {
        const ScopeStmt *scope;
        const Statement *direct_child{nullptr};
    };
    luisa::vector<ScopeTraversal> traversal_scopes;
    luisa::vector<const Statement *> statement_stack;
    luisa::unordered_map<uint, const Statement *> first_direct_statement;
    traverse_expressions<true>(
        root,
        [&](const Expression *expression) noexcept {
            if (expression->tag() != Expression::Tag::REF) { return; }
            auto variable = static_cast<const RefExpr *>(expression)->variable();
            if (variable.tag() != Variable::Tag::LOCAL ||
                first_direct_statement.contains(variable.uid())) {
                return;
            }
            auto emission = _local_variable_emissions.find(variable.uid());
            if (emission == _local_variable_emissions.end()) { return; }
            for (auto iter = traversal_scopes.rbegin();
                 iter != traversal_scopes.rend(); ++iter) {
                if (iter->scope == emission->second.scope) {
                    first_direct_statement.try_emplace(
                        variable.uid(), iter->direct_child);
                    return;
                }
            }
            LUISA_ERROR_WITH_LOCATION(
                "Failed to locate a Metal local variable declaration scope.");
        },
        [&](const Statement *statement) noexcept {
            if (!statement_stack.empty() &&
                statement_stack.back()->tag() == Statement::Tag::SCOPE) {
                LUISA_ASSERT(!traversal_scopes.empty() &&
                                 traversal_scopes.back().scope ==
                                     statement_stack.back(),
                             "Metal direct-child traversal is unbalanced.");
                traversal_scopes.back().direct_child = statement;
            }
            statement_stack.emplace_back(statement);
            if (statement->tag() == Statement::Tag::SCOPE) {
                traversal_scopes.emplace_back(ScopeTraversal{
                    .scope = static_cast<const ScopeStmt *>(statement)});
            }
        },
        [&](const Statement *statement) noexcept {
            if (statement->tag() == Statement::Tag::SCOPE) {
                LUISA_ASSERT(!traversal_scopes.empty() &&
                                 traversal_scopes.back().scope == statement,
                             "Metal direct-child scope traversal is unbalanced.");
                traversal_scopes.pop_back();
            }
            LUISA_ASSERT(!statement_stack.empty() &&
                             statement_stack.back() == statement,
                         "Metal direct-child statement traversal is unbalanced.");
            statement_stack.pop_back();
        });

    const auto expression_references = [](const Expression *expression,
                                          uint uid) noexcept {
        auto found = false;
        traverse_subexpressions(
            expression,
            [&](const Expression *candidate) noexcept {
                if (candidate->tag() == Expression::Tag::REF &&
                    static_cast<const RefExpr *>(candidate)->variable().uid() == uid) {
                    found = true;
                }
            },
            [](const Expression *) noexcept {});
        return found;
    };
    for (auto local : _function.local_variables()) {
        auto emission = _local_variable_emissions.find(local.uid());
        auto first = first_direct_statement.find(local.uid());
        if (emission == _local_variable_emissions.end() ||
            first == first_direct_statement.end() ||
            first->second == nullptr ||
            first->second->tag() != Statement::Tag::ASSIGN ||
            _gradient_variables.contains(local) ||
            local.type() == _ray_query_any_type ||
            local.type() == _ray_query_all_type) {
            continue;
        }
        auto assign = static_cast<const AssignStmt *>(first->second);
        if (assign->lhs()->tag() != Expression::Tag::REF ||
            static_cast<const RefExpr *>(assign->lhs())->variable() != local ||
            expression_references(assign->rhs(), local.uid())) {
            continue;
        }
        emission->second.initializer = assign;
        _local_variable_initializers.try_emplace(assign, local);
    }

    for (auto local : _function.local_variables()) {
        auto emission = _local_variable_emissions.find(local.uid());
        if (emission == _local_variable_emissions.end()) { continue; }
        // A declaration may move into a repeatedly entered branch/loop only
        // when its first direct statement initializes it before every read.
        // Otherwise retain the historical function-lifetime zero value: a
        // narrower `{}` declaration would reset state on every re-entry.
        if (emission->second.initializer == nullptr) {
            emission->second.scope = root;
        }
        _scope_local_variables[emission->second.scope].emplace_back(local);
    }
}

void MetalCodegenAST::_emit_scope_local_variables(
    const ScopeStmt *scope) noexcept {
    auto iter = _scope_local_variables.find(scope);
    if (iter == _scope_local_variables.end()) { return; }
    for (auto local : iter->second) {
        if (auto emission = _local_variable_emissions.find(local.uid());
            emission != _local_variable_emissions.end() &&
            emission->second.initializer != nullptr) {
            continue;
        }
        _emit_indention();
        _emit_type_name(local.type(), _function.variable_usage(local.uid()));
        _scratch << " ";
        _emit_variable_name(local);
        _scratch << "{};\n";
        if (local.type() == _ray_query_any_type ||
            local.type() == _ray_query_all_type) {
            if (!_ift_only_ray_query_uids.contains(luisa::hash_combine(
                    {static_cast<uint64_t>(_function.hash()),
                     static_cast<uint64_t>(local.uid())}))) {
                _emit_indention();
                _scratch << "LC_RAY_QUERY_SHADOW_VARIABLE(";
                _emit_variable_name(local);
                _scratch << ");\n";
            }
        }
    }
}

void MetalCodegenAST::_emit_assignment_lhs(
    const AssignStmt *stmt) noexcept {
    if (auto iter = _local_variable_initializers.find(stmt);
        iter != _local_variable_initializers.end()) {
        auto local = iter->second;
        _emit_type_name(local.type(), _function.variable_usage(local.uid()));
        _scratch << " ";
        _emit_variable_name(local);
    } else {
        stmt->lhs()->accept(*this);
    }
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
            auto usage = _function.variable_usage(arg.uid());
            auto sampled = arg.type()->is_texture() && _is_texture_sampled(arg);
            auto split = sampled && _is_texture_written(_function, arg);
            _scratch << "  alignas(16) ";
            _emit_type_name(arg.type(), usage, sampled && !split);
            _scratch << " ";
            _emit_variable_name(arg);
            _scratch << ";\n";
            if (split) {
                _scratch << "  alignas(16) ";
                _emit_type_name(arg.type(), Usage::READ, true);
                _scratch << " ";
                _emit_variable_name(arg, true);
                _scratch << ";\n";
            }
        }
        if (_uses_printing) {
            _scratch << "  alignas(16) LCPrinterBuffer print_buffer;\n";
        }
        for (auto i = 0u; i < _intersection_functions.size(); i++) {
            _scratch << "  alignas(16) intersection_function_table<triangle_data, instancing LC_EXTENDED_LIMITS_TAG> lc_ift_" << i << ";\n";
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
            if (arg.type()->is_texture() && _is_texture_sampled(arg) &&
                _is_texture_written(_function, arg)) {
                _scratch << "  auto ";
                _emit_variable_name(arg, true);
                _scratch << " = args.";
                _emit_variable_name(arg, true);
                _scratch << ";\n";
            }
        }
        if (_uses_printing) {
            _scratch << "  auto print_buffer = args.print_buffer;\n";
        }
        for (auto i = 0u; i < _intersection_functions.size(); i++) {
            _scratch << "  auto lc_ift_" << i << " = args.lc_ift_" << i << ";\n";
        }
    } else {
        auto needs_ift = _functions_using_ift.contains(_function.hash());
        LUISA_ASSERT(_function.shared_variables().empty(), "Shared memory declaration not allowed in callable.");
        auto texture_count = 0u;
        for (auto arg : _function.arguments()) {
            if (arg.type()->is_texture()) {
                texture_count++;
                if (_is_texture_sampled(arg) && _is_texture_written(_function, arg)) {
                    texture_count++;
                }
            }
        }
        if (texture_count > 0) {
            _scratch << "template<";
            for (auto i = 0u; i < texture_count; i++) {
                _scratch << "typename T" << i << ", ";
            }
            _scratch.pop_back();
            _scratch.pop_back();
            _scratch << ">\n";
        }
        if (_function.requires_noinline()) {
            _scratch << "__attribute__((noinline)) ";
        }
        _emit_type_name(_function.return_type());
        _scratch << " callable_" << hash_to_string(_function.hash()) << "(";
        auto emitted_texture_count = 0u;
        if (!_function.arguments().empty() || _uses_printing || needs_ift) {
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
                if (arg.type()->is_texture() && _is_texture_sampled(arg) &&
                    _is_texture_written(_function, arg)) {
                    _scratch << "T" << emitted_texture_count++ << " ";
                    _emit_variable_name(arg, true);
                    _scratch << ", ";
                }
            }
            if (_uses_printing) {
                _scratch << "LCPrinterBuffer print_buffer, ";
            }
            if (needs_ift) {
                for (auto i = 0u; i < _intersection_functions.size(); i++) {
                    _scratch << "intersection_function_table<triangle_data, instancing LC_EXTENDED_LIMITS_TAG> lc_ift_" << i << ", ";
                }
            }
            _scratch.pop_back();
            _scratch.pop_back();
        }
        _scratch << ") {\n";
    }

    _analyze_local_variables();

    // Emit only locals whose smallest safe scope is the function body here.
    // Nested scopes declare their own locals when the body is emitted below.
    _scratch << "\n  /* local variables */\n";
    _indention = 1u;
    _emit_scope_local_variables(_function.body());

    // emit gradient variables for autodiff
    for (auto variable : _gradient_variables) {
        _scratch << "  LC_GRAD_SHADOW_VARIABLE(";
        _emit_variable_name(variable);
        _scratch << ");\n";
    }

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
    void _decode_bfloat16(bfloat16) noexcept override {
        LUISA_NOT_IMPLEMENTED("Metal bfloat16 constants are not implemented.");
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
    _analyze_sampled_textures(kernel);
    _argument_sampled.clear();
    _argument_sampled.reserve(kernel.arguments().size());
    for (auto argument : kernel.arguments()) {
        _argument_sampled.emplace_back(
            argument.type()->is_texture() && _is_texture_sampled(kernel, argument));
    }

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
    _scratch << luisa::string_view{luisa_compute_metal_device_lib, luisa_compute_metal_device_lib_size}
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
                for (auto callable : sorted_custom_callables(function)) {
                    self(self, callable);
                }
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

    _analyze_ray_queries(functions);
    _emit_lowered_ray_query_functions();

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
    if (expr->op() == BinaryOp::MOD &&
        expr->type()->is_float_or_float_vector()) {
        _scratch << "lc_fmod(";
        expr->lhs()->accept(*this);
        _scratch << ", ";
        expr->rhs()->accept(*this);
        _scratch << ")";
        return;
    }
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
            if (expr->self()->tag() == Expression::Tag::MEMBER &&
                static_cast<const MemberExpr *>(expr->self())->is_swizzle() &&
                static_cast<const MemberExpr *>(expr->self())->swizzle_size() > 1u) {
                _scratch << "static_cast<const ";
                _emit_type_name(expr->self()->type());
                _scratch << ">(";
                expr->self()->accept(*this);
                _scratch << ")";
            } else {
                expr->self()->accept(*this);
            }
            _scratch << ", " << expr->swizzle_index(0u) << ")";
        } else {
            static constexpr luisa::string_view xyzw[]{"x", "y", "z", "w"};
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
        if (expr->range()->tag() == Expression::Tag::MEMBER &&
            static_cast<const MemberExpr *>(expr->range())->is_swizzle() &&
            static_cast<const MemberExpr *>(expr->range())->swizzle_size() > 1u) {
            // Metal treats a multi-component swizzle as a vector element
            // expression. Force the const overload of vector_element_ref for
            // read-only indexing; mutable references are handled at call
            // sites by _emit_swizzle_reference_temporaries.
            _scratch << "static_cast<const ";
            _emit_type_name(expr->range()->type());
            _scratch << ">(";
            expr->range()->accept(*this);
            _scratch << ")";
        } else {
            expr->range()->accept(*this);
        }
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
        case CallOp::ERF: _scratch << "erf"; break;
        case CallOp::ERFC: _scratch << "erfc"; break;
        case CallOp::EXPM1: _scratch << "expm1"; break;
        case CallOp::LDEXP: _scratch << "ldexp"; break;
        case CallOp::SQRT: _scratch << "sqrt"; break;
        case CallOp::RSQRT: _scratch << "rsqrt"; break;
        case CallOp::CEIL: _scratch << "ceil"; break;
        case CallOp::FLOOR: _scratch << "floor"; break;
        case CallOp::FRACT: _scratch << "fract"; break;
        case CallOp::TRUNC: _scratch << "trunc"; break;
        case CallOp::ROUND: _scratch << "round"; break;
        case CallOp::RINT: _scratch << "rint"; break;
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
        case CallOp::BUFFER_VOLATILE_READ: _scratch << "buffer_read_volatile"; break;
        case CallOp::BUFFER_VOLATILE_WRITE: _scratch << "buffer_write_volatile"; break;
        case CallOp::BUFFER_SIZE: _scratch << "buffer_size"; break;
        case CallOp::BYTE_BUFFER_READ: {
            _scratch << "byte_buffer_read<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::BYTE_BUFFER_WRITE: _scratch << "byte_buffer_write"; break;
        case CallOp::BYTE_BUFFER_VOLATILE_READ: {
            _scratch << "byte_buffer_read_volatile<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::BYTE_BUFFER_VOLATILE_WRITE: _scratch << "byte_buffer_write_volatile"; break;
        case CallOp::BYTE_BUFFER_SIZE: _scratch << "byte_buffer_size"; break;
        case CallOp::TEXTURE_READ: _scratch << "texture_read"; break;
        case CallOp::TEXTURE_WRITE: _scratch << "texture_write"; break;
        case CallOp::TEXTURE_SIZE: _scratch << "texture_size"; break;
#define LUISA_METAL_BINDLESS_VARIANTS(op)              \
    case CallOp::BINDLESS_##op:                        \
    case CallOp::UNIFORM_BINDLESS_##op:                \
    case CallOp::TYPED_UNIFORM_BINDLESS_##op:          \
    case CallOp::TYPED_BINDLESS_##op
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE2D_SAMPLE): _scratch << "bindless_texture_sample2d"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE2D_SAMPLE_LEVEL): _scratch << "bindless_texture_sample2d_level"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE2D_SAMPLE_GRAD): _scratch << "bindless_texture_sample2d_grad"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE2D_SAMPLE_GRAD_LEVEL): _scratch << "bindless_texture_sample2d_grad_level"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE3D_SAMPLE): _scratch << "bindless_texture_sample3d"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE3D_SAMPLE_LEVEL): _scratch << "bindless_texture_sample3d_level"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE3D_SAMPLE_GRAD): _scratch << "bindless_texture_sample3d_grad"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE3D_SAMPLE_GRAD_LEVEL): _scratch << "bindless_texture_sample3d_grad_level"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE2D_READ): _scratch << "bindless_texture_read2d"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE3D_READ): _scratch << "bindless_texture_read3d"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE2D_READ_LEVEL): _scratch << "bindless_texture_read2d_level"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE3D_READ_LEVEL): _scratch << "bindless_texture_read3d_level"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE2D_SIZE): _scratch << "bindless_texture_size2d"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE3D_SIZE): _scratch << "bindless_texture_size3d"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE2D_SIZE_LEVEL): _scratch << "bindless_texture_size2d_level"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE3D_SIZE_LEVEL): _scratch << "bindless_texture_size3d_level"; break;
            LUISA_METAL_BINDLESS_VARIANTS(BUFFER_READ): {
            _scratch << "bindless_buffer_read<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
            LUISA_METAL_BINDLESS_VARIANTS(BUFFER_WRITE): {
            _scratch << "bindless_buffer_write<";
            _emit_type_name(expr->arguments().back()->type());
            _scratch << ">";
            break;
        }
            LUISA_METAL_BINDLESS_VARIANTS(BYTE_BUFFER_READ): {
            _scratch << "bindless_byte_address_buffer_read<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
            LUISA_METAL_BINDLESS_VARIANTS(BUFFER_SIZE): _scratch << "bindless_buffer_size"; break;
            LUISA_METAL_BINDLESS_VARIANTS(BUFFER_TYPE): _scratch << "bindless_buffer_type"; break;
            LUISA_METAL_BINDLESS_VARIANTS(BUFFER_ADDRESS): _scratch << "bindless_buffer_address"; break;
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
        case CallOp::FLATTEN:
        case CallOp::BRANCH:
        case CallOp::FORCE_CASE:
            break;
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
        case CallOp::UNDEFINED: {
            // Zero is a legal concrete refinement for direct AST codegen.
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
        case CallOp::RAY_QUERY_OBJECT_SPACE_RAY: _scratch << "ray_query_object_ray"; break;
        case CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT: _scratch << "ray_query_procedural_candidate"; break;
        case CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT:
            if (_active_lowered_ray_query != nullptr) {
                _scratch << "LCTriangleHit{_lc_inst, _lc_prim, _lc_bary, _lc_dist}";
                return;
            }
            _scratch << "ray_query_triangle_candidate";
            break;
        case CallOp::RAY_QUERY_COMMITTED_HIT: _scratch << "ray_query_committed_hit"; break;
        case CallOp::RAY_QUERY_COMMIT_TRIANGLE:
            if (_active_lowered_ray_query != nullptr) {
                _scratch << "(_lc_accept = true)";
                return;
            }
            _scratch << "ray_query_commit_triangle";
            break;
        case CallOp::RAY_QUERY_COMMIT_PROCEDURAL: _scratch << "ray_query_commit_procedural"; break;
        case CallOp::RAY_QUERY_TERMINATE:
            if (_active_lowered_ray_query != nullptr) {
                _scratch << "(_lc_continue = false)";
                return;
            }
            _scratch << "ray_query_terminate";
            break;
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
        case CallOp::RASTER_SET_Z_DEPTH: [[fallthrough]];
        case CallOp::RASTER_SET_Z_DEPTH_GREATER_EQUAL: [[fallthrough]];
        case CallOp::RASTER_SET_Z_DEPTH_LESS_EQUAL: [[fallthrough]];
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

        case CallOp::RAY_TRACING_INSTANCE_MOTION_MATRIX: break;
        case CallOp::RAY_TRACING_INSTANCE_MOTION_SRT: break;
        case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX: break;
        case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT: break;

        case CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: _scratch << "accel_trace_closest_motion_blur"; break;
        case CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: _scratch << "accel_trace_any_motion_blur"; break;
        case CallOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: _scratch << "accel_query_all_motion_blur"; break;
        case CallOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: _scratch << "accel_query_all_motion_blur"; break;

        case CallOp::TEXTURE2D_SAMPLE:
        case CallOp::TEXTURE3D_SAMPLE:
            _scratch << "texture_sample";
            break;
        case CallOp::TEXTURE2D_SAMPLE_LEVEL:
        case CallOp::TEXTURE3D_SAMPLE_LEVEL:
            _scratch << "texture_sample_level";
            break;
        case CallOp::TEXTURE2D_SAMPLE_GRAD:
        case CallOp::TEXTURE3D_SAMPLE_GRAD:
            _scratch << "texture_sample_grad";
            break;
        case CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
            _scratch << "texture_sample_grad_level";
            break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE2D_SAMPLE_SAMPLER): _scratch << "bindless_texture_sample2d_sample"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE2D_SAMPLE_LEVEL_SAMPLER): _scratch << "bindless_texture_sample2d_level_sample"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE2D_SAMPLE_GRAD_SAMPLER): _scratch << "bindless_texture_sample2d_grad_sample"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER): _scratch << "bindless_texture_sample2d_grad_level_sample"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE3D_SAMPLE_SAMPLER): _scratch << "bindless_texture_sample3d_sample"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE3D_SAMPLE_LEVEL_SAMPLER): _scratch << "bindless_texture_sample3d_level_sample"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE3D_SAMPLE_GRAD_SAMPLER): _scratch << "bindless_texture_sample3d_grad_sample"; break;
            LUISA_METAL_BINDLESS_VARIANTS(TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER): _scratch << "bindless_texture_sample3d_grad_level_sample"; break;
#undef LUISA_METAL_BINDLESS_VARIANTS
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Unsupported call operation '{}' in Metal AST codegen.",
                luisa::to_string(expr->op()));
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
        auto emitted_argument = false;
        auto emit_argument = [&](const Expression *argument, bool sampled) noexcept {
            if (emitted_argument) { _scratch << ", "; }
            emitted_argument = true;
            _emit_call_argument(argument, sampled);
        };
        for (auto i = 0u; i < expr->arguments().size(); i++) {
            auto arg = expr->arguments()[i];
            auto replacement = std::find_if(
                _reference_temporaries.begin(), _reference_temporaries.end(),
                [arg](auto &&binding) noexcept { return binding.expression == arg; });
            if (replacement != _reference_temporaries.end()) {
                if (emitted_argument) { _scratch << ", "; }
                emitted_argument = true;
                _scratch << replacement->name;
            } else {
                auto sampled = false;
                auto split = false;
                switch (expr->op()) {
                    case CallOp::TEXTURE2D_SAMPLE:
                    case CallOp::TEXTURE2D_SAMPLE_LEVEL:
                    case CallOp::TEXTURE2D_SAMPLE_GRAD:
                    case CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
                    case CallOp::TEXTURE3D_SAMPLE:
                    case CallOp::TEXTURE3D_SAMPLE_LEVEL:
                    case CallOp::TEXTURE3D_SAMPLE_GRAD:
                    case CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
                        sampled = i == 0u;
                        break;
                    case CallOp::CUSTOM: {
                        auto callable = expr->custom();
                        auto parameter = callable.arguments()[i];
                        sampled = parameter.type()->is_texture() &&
                                  _is_texture_sampled(callable, parameter);
                        split = sampled && _is_texture_written(callable, parameter);
                        break;
                    }
                    default: break;
                }
                emit_argument(arg, sampled && !split);
                if (split) { emit_argument(arg, true); }
            }
        }
        if (expr->is_custom() && _uses_printing) {
            if (emitted_argument) { _scratch << ", "; }
            emitted_argument = true;
            _scratch << "print_buffer";
        }
        if (expr->is_custom() && _functions_using_ift.contains(expr->custom().hash())) {
            for (auto i = 0u; i < _intersection_functions.size(); i++) {
                if (emitted_argument) { _scratch << ", "; }
                emitted_argument = true;
                _scratch << "lc_ift_" << i;
            }
        }
    }
    _scratch << ")";
}

void MetalCodegenAST::_emit_swizzle_reference_temporaries(const Expression *expression) noexcept {
    traverse_subexpressions(
        expression,
        [&](const Expression *subexpression) noexcept {
            if (subexpression->tag() != Expression::Tag::CALL) { return; }
            auto call = static_cast<const CallExpr *>(subexpression);
            if (!call->is_custom()) { return; }
            auto parameters = call->custom().arguments();
            for (auto i = 0u; i < call->arguments().size() && i < parameters.size(); i++) {
                auto argument = call->arguments()[i];
                auto parameter = parameters[i];
                auto is_mutable_reference = parameter.is_reference() &&
                                            (to_underlying(call->custom().variable_usage(parameter.uid())) &
                                             to_underlying(Usage::WRITE)) != 0u;
                if (!is_mutable_reference || argument->tag() != Expression::Tag::MEMBER) { continue; }
                auto member = static_cast<const MemberExpr *>(argument);
                if (!member->is_swizzle() || member->swizzle_size() <= 1u) { continue; }
                auto name = luisa::format("__lc_swizzle_tmp_{}", _next_reference_temporary++);
                _emit_indention();
                _scratch << "auto " << name << " = ";
                argument->accept(*this);
                _scratch << ";\n";
                _reference_temporaries.emplace_back(ReferenceTemporary{argument, std::move(name)});
            }
        },
        [](auto) noexcept {});
}

void MetalCodegenAST::_emit_swizzle_reference_writebacks(size_t first) noexcept {
    for (auto i = first; i < _reference_temporaries.size(); i++) {
        _emit_indention();
        _reference_temporaries[i].expression->accept(*this);
        _scratch << " = " << _reference_temporaries[i].name << ";\n";
    }
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
    if (_active_lowered_ray_query != nullptr &&
        _active_lowered_ray_query->is_wrapped_callable) {
        for (const auto &cap : _active_lowered_ray_query->captures) {
            if (cap.is_written) {
                _emit_indention();
                _scratch << "_lc_payload.";
                _emit_variable_name(cap.callee_variable);
                _scratch << " = ";
                _emit_variable_name(cap.callee_variable);
                _scratch << ";\n";
            }
        }
        _emit_indention();
        _scratch << "return LCTriangleIsectResult{_lc_accept, _lc_continue};\n";
        return;
    }
    auto previous_temporary_count = _reference_temporaries.size();
    if (auto expression = stmt->expression()) {
        _emit_swizzle_reference_temporaries(expression);
        if (_reference_temporaries.size() != previous_temporary_count) {
            auto result = luisa::format("__lc_call_result_{}", _next_reference_temporary++);
            _emit_indention();
            _scratch << "auto " << result << " = ";
            expression->accept(*this);
            _scratch << ";\n";
            _emit_swizzle_reference_writebacks(previous_temporary_count);
            _emit_indention();
            _scratch << "return " << result << ";\n";
            _reference_temporaries.resize(previous_temporary_count);
            return;
        }
    }
    _emit_indention();
    _scratch << "return";
    if (auto expr = stmt->expression()) {
        _scratch << " ";
        expr->accept(*this);
    }
    _scratch << ";\n";
    _reference_temporaries.resize(previous_temporary_count);
}

void MetalCodegenAST::visit(const ScopeStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "{\n";
    _indention++;
    _emit_scope_local_variables(stmt);
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
    _emit_scope_local_variables(stmt->true_branch());
    for (auto s : stmt->true_branch()->statements()) {
        s->accept(*this);
    }
    _indention--;
    _emit_indention();
    _scratch << "}";
    if (auto &&fb = stmt->false_branch()->statements(); !fb.empty()) {
        _scratch << " else {\n";
        _indention++;
        _emit_scope_local_variables(stmt->false_branch());
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
    _emit_scope_local_variables(stmt->body());
    for (auto s : stmt->body()->statements()) {
        s->accept(*this);
    }
    _indention--;
    _emit_indention();
    _scratch << "}\n";
}

void MetalCodegenAST::visit(const ExprStmt *stmt) noexcept {
    auto previous_temporary_count = _reference_temporaries.size();
    _emit_swizzle_reference_temporaries(stmt->expression());
    _emit_indention();
    if (stmt->expression()->type() != nullptr) {
        _scratch << "static_cast<void>(";
    }
    stmt->expression()->accept(*this);
    if (stmt->expression()->type() != nullptr) {
        _scratch << ")";
    }
    _scratch << ";\n";
    _emit_swizzle_reference_writebacks(previous_temporary_count);
    _reference_temporaries.resize(previous_temporary_count);
}

void MetalCodegenAST::visit(const SwitchStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "switch (";
    stmt->expression()->accept(*this);
    _scratch << ") {\n";
    _indention++;
    _emit_scope_local_variables(stmt->body());
    for (auto s : stmt->body()->statements()) {
        s->accept(*this);
    }
    _indention--;
    _emit_indention();
    _scratch << "}\n";
}

void MetalCodegenAST::visit(const SwitchCaseStmt *stmt) noexcept {
    _emit_indention();
    for (auto expression : stmt->expressions()) {
        _scratch << "case ";
        expression->accept(*this);
        _scratch << ": ";
    }
    _scratch << "{\n";
    _indention++;
    _emit_scope_local_variables(stmt->body());
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
    _emit_scope_local_variables(stmt->body());
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
    if (is_local_undefined_lifetime_seed(stmt)) {
        // The scope analysis can select the lifetime seed as the declaration
        // site. Drop its undefined assignment, not the declaration itself.
        if (_local_variable_initializers.contains(stmt)) {
            _emit_indention();
            _emit_assignment_lhs(stmt);
            _scratch << ";\n";
        }
        return;
    }
    auto previous_temporary_count = _reference_temporaries.size();
    _emit_swizzle_reference_temporaries(stmt->rhs());
    if (_reference_temporaries.size() != previous_temporary_count) {
        auto result = luisa::format("__lc_call_result_{}", _next_reference_temporary++);
        _emit_indention();
        _scratch << "auto " << result << " = ";
        stmt->rhs()->accept(*this);
        _scratch << ";\n";
        _emit_swizzle_reference_writebacks(previous_temporary_count);
        _emit_indention();
        _emit_assignment_lhs(stmt);
        _scratch << " = " << result << ";\n";
        _reference_temporaries.resize(previous_temporary_count);
        return;
    }
    _emit_indention();
    _emit_assignment_lhs(stmt);
    _scratch << " = ";
    stmt->rhs()->accept(*this);
    _scratch << ";\n";
    _reference_temporaries.resize(previous_temporary_count);
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
    _emit_scope_local_variables(stmt->body());
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
        _scratch << luisa::string_view{&c, 1u};
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

void MetalCodegenAST::_analyze_ray_queries(
    luisa::span<const Function> functions) noexcept {

    _intersection_functions.clear();
    _lowered_ray_queries.clear();
    _lowered_ray_query_map.clear();
    _ift_only_ray_query_uids.clear();
    _functions_using_ift.clear();
    _active_lowered_ray_query = nullptr;

    static const auto disabled = [] {
        auto v = std::getenv("LUISA_METAL_DISABLE_RAY_QUERY_IFT");
        return v != nullptr && v[0] != '\0' && (v[0] != '0' || v[1] != '\0');
    }();
    if (disabled || functions.empty()) { return; }

    const auto &kernel = functions.back();
    if (kernel.requires_motion_blur() || kernel.required_curve_bases().any()) {
        return;
    }

    auto make_rq_key = [](uint64_t fn_hash, uint uid) noexcept {
        return luisa::hash_combine({fn_hash, static_cast<uint64_t>(uid)});
    };

    auto is_forbidden_call = [](CallOp op) noexcept {
        if (is_atomic_operation(op)) { return true; }
        switch (op) {
            case CallOp::CUSTOM:
            case CallOp::EXTERNAL:
            case CallOp::SYNCHRONIZE_BLOCK:
            case CallOp::TEXTURE_READ:
            case CallOp::TEXTURE_WRITE:
            case CallOp::TEXTURE_SIZE:
            case CallOp::TEXTURE2D_SAMPLE:
            case CallOp::TEXTURE2D_SAMPLE_LEVEL:
            case CallOp::TEXTURE2D_SAMPLE_GRAD:
            case CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
            case CallOp::TEXTURE3D_SAMPLE:
            case CallOp::TEXTURE3D_SAMPLE_LEVEL:
            case CallOp::TEXTURE3D_SAMPLE_GRAD:
            case CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
            case CallOp::RAY_TRACING_INSTANCE_TRANSFORM:
            case CallOp::RAY_TRACING_INSTANCE_USER_ID:
            case CallOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK:
            case CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM:
            case CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY:
            case CallOp::RAY_TRACING_SET_INSTANCE_OPACITY:
            case CallOp::RAY_TRACING_SET_INSTANCE_USER_ID:
            case CallOp::RAY_TRACING_TRACE_CLOSEST:
            case CallOp::RAY_TRACING_TRACE_ANY:
            case CallOp::RAY_TRACING_QUERY_ALL:
            case CallOp::RAY_TRACING_QUERY_ANY:
            case CallOp::RAY_TRACING_INSTANCE_MOTION_MATRIX:
            case CallOp::RAY_TRACING_INSTANCE_MOTION_SRT:
            case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX:
            case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT:
            case CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
            case CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
            case CallOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
            case CallOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR:
            case CallOp::RAY_QUERY_WORLD_SPACE_RAY:
            case CallOp::RAY_QUERY_OBJECT_SPACE_RAY:
            case CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT:
            case CallOp::RAY_QUERY_COMMIT_PROCEDURAL:
            case CallOp::RAY_QUERY_COMMITTED_HIT:
            case CallOp::RAY_QUERY_PROCEED:
            case CallOp::RAY_QUERY_IS_TRIANGLE_CANDIDATE:
            case CallOp::RAY_QUERY_IS_PROCEDURAL_CANDIDATE:
            case CallOp::REQUIRES_GRADIENT:
            case CallOp::GRADIENT:
            case CallOp::GRADIENT_MARKER:
            case CallOp::ACCUMULATE_GRADIENT:
            case CallOp::BACKWARD:
            case CallOp::RASTER_SET_Z_DEPTH:
            case CallOp::RASTER_SET_Z_DEPTH_GREATER_EQUAL:
            case CallOp::RASTER_SET_Z_DEPTH_LESS_EQUAL:
            case CallOp::RASTER_DISCARD:
            case CallOp::INDIRECT_SET_DISPATCH_COUNT:
            case CallOp::INDIRECT_SET_DISPATCH_KERNEL:
            case CallOp::DDX:
            case CallOp::DDY:
            case CallOp::WARP_IS_FIRST_ACTIVE_LANE:
            case CallOp::WARP_FIRST_ACTIVE_LANE:
            case CallOp::WARP_ACTIVE_ALL_EQUAL:
            case CallOp::WARP_ACTIVE_BIT_AND:
            case CallOp::WARP_ACTIVE_BIT_OR:
            case CallOp::WARP_ACTIVE_BIT_XOR:
            case CallOp::WARP_ACTIVE_COUNT_BITS:
            case CallOp::WARP_ACTIVE_MAX:
            case CallOp::WARP_ACTIVE_MIN:
            case CallOp::WARP_ACTIVE_PRODUCT:
            case CallOp::WARP_ACTIVE_SUM:
            case CallOp::WARP_ACTIVE_ALL:
            case CallOp::WARP_ACTIVE_ANY:
            case CallOp::WARP_ACTIVE_BIT_MASK:
            case CallOp::WARP_PREFIX_COUNT_BITS:
            case CallOp::WARP_PREFIX_SUM:
            case CallOp::WARP_PREFIX_PRODUCT:
            case CallOp::WARP_READ_LANE:
            case CallOp::WARP_READ_FIRST_ACTIVE_LANE:
            case CallOp::SHADER_EXECUTION_REORDER:
                return true;
            default: {
                const auto v = luisa::to_underlying(op);
                return v >= luisa::to_underlying(CallOp::BINDLESS_TEXTURE2D_SAMPLE) &&
                       v <= luisa::to_underlying(CallOp::TYPED_BINDLESS_BUFFER_ADDRESS);
            }
        }
    };

    auto is_forbidden_var = [](Variable v) noexcept {
        if (v.is_shared() || v.is_builtin()) { return true; }
        switch (v.tag()) {
            case Variable::Tag::TEXTURE:
            case Variable::Tag::BINDLESS_ARRAY:
            case Variable::Tag::ACCEL:
                return true;
            default:
                return false;
        }
    };

    auto capture_size_align = [](const Type *type) noexcept -> std::pair<size_t, size_t> {
        if (type->is_buffer()) { return {16u, 8u}; }
        return {std::max<size_t>(type_size_bytes(type), 1u),
                std::max<size_t>(type->alignment(), 1u)};
    };

    struct WrappedCandidateCallable {
        Function callee;
        luisa::vector<Variable> caller_args;
        const Expression *q_arg_expr{nullptr};
    };
    auto match_wrapped_candidate_callable = [](const ScopeStmt *scope, uint q_uid) noexcept
        -> luisa::optional<WrappedCandidateCallable> {
        const CallExpr *found_call = nullptr;
        for (auto s : scope->statements()) {
            if (s->tag() == Statement::Tag::COMMENT) { continue; }
            if (found_call != nullptr || s->tag() != Statement::Tag::EXPR) {
                return luisa::nullopt;
            }
            auto e = static_cast<const ExprStmt *>(s)->expression();
            if (e == nullptr || e->tag() != Expression::Tag::CALL) {
                return luisa::nullopt;
            }
            found_call = static_cast<const CallExpr *>(e);
        }
        if (found_call == nullptr ||
            found_call->op() != CallOp::CUSTOM ||
            found_call->type() != nullptr) {
            return luisa::nullopt;
        }
        auto callee = found_call->custom();
        auto all_args_unbound = std::all_of(
            callee.bound_arguments().begin(), callee.bound_arguments().end(),
            [](const Function::Binding &b) noexcept {
                return luisa::holds_alternative<luisa::monostate>(b);
            });
        if (callee.return_type() != nullptr ||
            !all_args_unbound ||
            found_call->arguments().size() != callee.arguments().size() ||
            callee.arguments().empty()) {
            return luisa::nullopt;
        }
        auto arg0 = found_call->arguments().front();
        if (arg0->tag() != Expression::Tag::REF ||
            static_cast<const RefExpr *>(arg0)->variable().uid() != q_uid) {
            return luisa::nullopt;
        }
        luisa::vector<Variable> caller_args;
        caller_args.reserve(callee.arguments().size());
        luisa::unordered_set<uint> seen_ref_caller_uids;
        for (size_t k = 0u; k < callee.arguments().size(); k++) {
            auto arg = found_call->arguments()[k];
            if (arg->tag() != Expression::Tag::REF) { return luisa::nullopt; }
            auto v = static_cast<const RefExpr *>(arg)->variable();
            if (k > 0u &&
                callee.arguments()[k].tag() == Variable::Tag::REFERENCE &&
                !seen_ref_caller_uids.emplace(v.uid()).second) {
                return luisa::nullopt;
            }
            caller_args.emplace_back(v);
        }
        return WrappedCandidateCallable{callee, std::move(caller_args), arg0};
    };
    auto is_effectively_empty_scope = [&](const ScopeStmt *scope, uint q_uid) noexcept {
        bool has_non_comment = false;
        for (auto s : scope->statements()) {
            if (s->tag() != Statement::Tag::COMMENT) {
                has_non_comment = true;
                break;
            }
        }
        if (!has_non_comment) { return true; }
        auto wrapped = match_wrapped_candidate_callable(scope, q_uid);
        if (!wrapped) { return false; }
        for (auto s : wrapped->callee.body()->statements()) {
            if (s->tag() == Statement::Tag::COMMENT) { continue; }
            if (s->tag() == Statement::Tag::RETURN &&
                static_cast<const ReturnStmt *>(s)->expression() == nullptr) {
                continue;
            }
            return false;
        }
        return true;
    };

    for (auto f : functions) {
        if (!f.direct_builtin_callables().uses_ray_query() &&
            !f.propagated_builtin_callables().uses_ray_query()) {
            continue;
        }
        luisa::vector<const RayQueryStmt *> rqs;
        traverse_expressions<false>(
            f.body(),
            [](const Expression *) noexcept {},
            [&](const Statement *s) noexcept {
                if (s->tag() == Statement::Tag::RAY_QUERY) {
                    rqs.emplace_back(static_cast<const RayQueryStmt *>(s));
                }
            },
            [](const Statement *) noexcept {});
        if (rqs.empty()) { continue; }

        // Check ray-query variable outside uses across f.body().
        luisa::unordered_set<uint> invalid_rq_uids;
        luisa::unordered_set<uint> rq_uids;
        for (auto *rq : rqs) {
            if (rq->query() == nullptr ||
                rq->query()->tag() != Expression::Tag::REF) {
                continue;
            }
            auto qv = rq->query()->variable();
            if (qv.tag() == Variable::Tag::LOCAL &&
                (qv.type() == _ray_query_all_type || qv.type() == _ray_query_any_type)) {
                rq_uids.emplace(qv.uid());
            } else {
                invalid_rq_uids.emplace(qv.uid());
            }
        }
        for (auto arg : f.arguments()) {
            if (arg.type() == _ray_query_all_type || arg.type() == _ray_query_any_type) {
                invalid_rq_uids.emplace(arg.uid());
            }
        }
        auto is_query_construct_assign = [](const Statement *s, uint uid) noexcept {
            if (s->tag() != Statement::Tag::ASSIGN) { return false; }
            auto a = static_cast<const AssignStmt *>(s);
            if (a->lhs()->tag() != Expression::Tag::REF ||
                static_cast<const RefExpr *>(a->lhs())->variable().uid() != uid ||
                a->rhs()->tag() != Expression::Tag::CALL) {
                return false;
            }
            auto op = static_cast<const CallExpr *>(a->rhs())->op();
            return op == CallOp::RAY_TRACING_QUERY_ALL ||
                   op == CallOp::RAY_TRACING_QUERY_ANY;
        };
        luisa::unordered_set<const Expression *> allowed_rq_refs;
        traverse_expressions<true>(
            f.body(),
            [&](const Expression *e) noexcept {
                if (e->tag() == Expression::Tag::CALL) {
                    auto call = static_cast<const CallExpr *>(e);
                    if ((call->op() == CallOp::RAY_QUERY_COMMITTED_HIT ||
                         call->op() == CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT ||
                         call->op() == CallOp::RAY_QUERY_COMMIT_TRIANGLE ||
                         call->op() == CallOp::RAY_QUERY_TERMINATE) &&
                        !call->arguments().empty() &&
                        call->arguments().front()->tag() == Expression::Tag::REF) {
                        auto v = static_cast<const RefExpr *>(call->arguments().front())->variable();
                        if (rq_uids.contains(v.uid())) {
                            allowed_rq_refs.emplace(call->arguments().front());
                        }
                    }
                } else if (e->tag() == Expression::Tag::REF) {
                    auto v = static_cast<const RefExpr *>(e)->variable();
                    if ((v.type() == _ray_query_all_type || v.type() == _ray_query_any_type) &&
                        !allowed_rq_refs.contains(e)) {
                        invalid_rq_uids.emplace(v.uid());
                    }
                }
            },
            [&](const Statement *s) noexcept {
                if (s->tag() == Statement::Tag::RAY_QUERY) {
                    auto rqs_stmt = static_cast<const RayQueryStmt *>(s);
                    allowed_rq_refs.emplace(rqs_stmt->query());
                    if (rqs_stmt->query() != nullptr &&
                        rqs_stmt->query()->tag() == Expression::Tag::REF) {
                        auto q_uid = static_cast<const RefExpr *>(rqs_stmt->query())->variable().uid();
                        if (auto w = match_wrapped_candidate_callable(rqs_stmt->on_triangle_candidate(), q_uid)) {
                            allowed_rq_refs.emplace(w->q_arg_expr);
                        }
                        if (auto w = match_wrapped_candidate_callable(rqs_stmt->on_procedural_candidate(), q_uid)) {
                            allowed_rq_refs.emplace(w->q_arg_expr);
                        }
                    }
                } else if (s->tag() == Statement::Tag::ASSIGN) {
                    auto a = static_cast<const AssignStmt *>(s);
                    if (a->lhs()->tag() == Expression::Tag::REF) {
                        auto v = static_cast<const RefExpr *>(a->lhs())->variable();
                        if (rq_uids.contains(v.uid()) && is_query_construct_assign(s, v.uid())) {
                            allowed_rq_refs.emplace(a->lhs());
                        }
                    }
                }
            },
            [](const Statement *) noexcept {});

        luisa::unordered_map<uint, luisa::vector<size_t>> lowered_by_q;
        luisa::unordered_set<uint> failed_q;

        for (auto *rq : rqs) {
            if (rq->query() == nullptr || rq->query()->tag() != Expression::Tag::REF) { continue; }
            auto q_var = rq->query()->variable();
            if (!is_effectively_empty_scope(rq->on_procedural_candidate(), q_var.uid()) ||
                invalid_rq_uids.contains(q_var.uid()) ||
                !rq_uids.contains(q_var.uid())) {
                failed_q.emplace(q_var.uid());
                continue;
            }

            auto wrapped_tri = match_wrapped_candidate_callable(rq->on_triangle_candidate(), q_var.uid());
            const bool is_wrapped_callable = wrapped_tri.has_value();
            Function tri_fn = is_wrapped_callable ? wrapped_tri->callee : f;
            const ScopeStmt *tri_body = is_wrapped_callable ? wrapped_tri->callee.body() : rq->on_triangle_candidate();
            const uint tri_q_uid = is_wrapped_callable ? wrapped_tri->callee.arguments().front().uid() : q_var.uid();
            luisa::unordered_map<uint, std::pair<Variable, Variable>> callee_arg_map;
            if (is_wrapped_callable) {
                for (size_t k = 1u; k < wrapped_tri->callee.arguments().size(); k++) {
                    auto callee_arg = wrapped_tri->callee.arguments()[k];
                    auto caller_arg = wrapped_tri->caller_args[k];
                    callee_arg_map.try_emplace(callee_arg.uid(), std::pair{caller_arg, callee_arg});
                }
            }

            bool ok = true;
            bool uses_buffers = false;
            bool has_loops_or_switches = false;
            uint nested_rq_depth = 0u;
            luisa::unordered_set<uint> inner_vars;
            luisa::vector<Variable> inner_var_order;
            luisa::unordered_set<uint> written_non_buffer;

            auto mark_lvalue_written = [&](const Expression *lhs) noexcept {
                while (lhs != nullptr) {
                    if (lhs->tag() == Expression::Tag::MEMBER) {
                        lhs = static_cast<const MemberExpr *>(lhs)->self();
                    } else if (lhs->tag() == Expression::Tag::ACCESS) {
                        auto acc = static_cast<const AccessExpr *>(lhs);
                        if (acc->range()->type()->is_buffer()) {
                            uses_buffers = true;
                            return;
                        }
                        lhs = acc->range();
                    } else {
                        break;
                    }
                }
                if (lhs != nullptr && lhs->tag() == Expression::Tag::REF) {
                    auto v = static_cast<const RefExpr *>(lhs)->variable();
                    if (v.type()->is_buffer()) {
                        uses_buffers = true;
                    } else {
                        written_non_buffer.emplace(v.uid());
                    }
                }
            };

            luisa::unordered_set<const Expression *> rq_self_refs;
            traverse_expressions<true>(
                tri_body,
                [&](const Expression *e) noexcept {
                    if (!ok) { return; }
                    if (e->tag() == Expression::Tag::CALL) {
                        auto call = static_cast<const CallExpr *>(e);
                        if (is_forbidden_call(call->op())) {
                            ok = false;
                            return;
                        }
                        if (call->op() == CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT ||
                            call->op() == CallOp::RAY_QUERY_COMMIT_TRIANGLE ||
                            call->op() == CallOp::RAY_QUERY_TERMINATE) {
                            if (!call->arguments().empty() &&
                                call->arguments().front()->tag() == Expression::Tag::REF &&
                                static_cast<const RefExpr *>(call->arguments().front())->variable().uid() == tri_q_uid) {
                                rq_self_refs.emplace(call->arguments().front());
                            } else {
                                ok = false;
                            }
                        } else if (call->op() == CallOp::BUFFER_READ ||
                                   call->op() == CallOp::BUFFER_WRITE ||
                                   call->op() == CallOp::BUFFER_VOLATILE_READ ||
                                   call->op() == CallOp::BUFFER_VOLATILE_WRITE ||
                                   call->op() == CallOp::BUFFER_SIZE ||
                                   call->op() == CallOp::BYTE_BUFFER_READ ||
                                   call->op() == CallOp::BYTE_BUFFER_WRITE ||
                                   call->op() == CallOp::BYTE_BUFFER_VOLATILE_READ ||
                                   call->op() == CallOp::BYTE_BUFFER_VOLATILE_WRITE ||
                                   call->op() == CallOp::BYTE_BUFFER_SIZE) {
                            uses_buffers = true;
                        }
                    } else if (e->tag() == Expression::Tag::CPUCUSTOM ||
                               e->tag() == Expression::Tag::GPUCUSTOM) {
                        ok = false;
                    } else if (e->tag() == Expression::Tag::REF) {
                        if (rq_self_refs.contains(e)) { return; }
                        auto v = static_cast<const RefExpr *>(e)->variable();
                        if (v.uid() == tri_q_uid || is_forbidden_var(v)) {
                            ok = false;
                            return;
                        }
                        if (v.type()->is_buffer()) { uses_buffers = true; }
                        if (inner_vars.emplace(v.uid()).second) {
                            inner_var_order.emplace_back(v);
                        }
                    }
                },
                [&](const Statement *s) noexcept {
                    if (!ok) { return; }
                    switch (s->tag()) {
                        case Statement::Tag::RETURN:
                            if (!is_wrapped_callable ||
                                static_cast<const ReturnStmt *>(s)->expression() != nullptr) {
                                ok = false;
                            }
                            break;
                        case Statement::Tag::AUTO_DIFF:
                        case Statement::Tag::PRINT:
                        case Statement::Tag::RAY_QUERY:
                        case Statement::Tag::SUSPEND:
                            ok = false;
                            nested_rq_depth++;
                            break;
                        case Statement::Tag::LOOP:
                        case Statement::Tag::FOR:
                        case Statement::Tag::SWITCH:
                            has_loops_or_switches = true;
                            break;
                        case Statement::Tag::ASSIGN:
                            mark_lvalue_written(static_cast<const AssignStmt *>(s)->lhs());
                            break;
                        default: break;
                    }
                },
                [](const Statement *) noexcept {});

            if (!ok || nested_rq_depth != 0u) {
                failed_q.emplace(q_var.uid());
                continue;
            }

            // Collect variables referenced outside rq->on_triangle_candidate().
            luisa::unordered_set<uint> outside_vars;
            if (!is_wrapped_callable) {
                for (auto arg : f.arguments()) { outside_vars.emplace(arg.uid()); }
                uint skip_depth = 0u;
                traverse_expressions<true>(
                    f.body(),
                    [&](const Expression *e) noexcept {
                        if (skip_depth > 0u) { return; }
                        if (e->tag() == Expression::Tag::REF) {
                            outside_vars.emplace(static_cast<const RefExpr *>(e)->variable().uid());
                        }
                    },
                    [&](const Statement *s) noexcept {
                        if (s == rq->on_triangle_candidate()) { skip_depth++; }
                    },
                    [&](const Statement *s) noexcept {
                        if (s == rq->on_triangle_candidate()) { skip_depth--; }
                    });
            }

            luisa::vector<LoweredRayQueryCapture> captures;
            luisa::vector<Variable> inner_locals;
            size_t payload_offset = 0u;
            size_t payload_max_align = 1u;
            for (auto v : inner_var_order) {
                if (is_wrapped_callable) {
                    if (auto iter = callee_arg_map.find(v.uid());
                        iter != callee_arg_map.end()) {
                        auto caller_v = iter->second.first;
                        auto callee_v = iter->second.second;
                        if (callee_v.type() == _ray_query_all_type ||
                            callee_v.type() == _ray_query_any_type ||
                            callee_v.type() == _indirect_dispatch_buffer_type ||
                            callee_v.type()->is_texture() ||
                            callee_v.type()->is_accel() ||
                            callee_v.type()->is_bindless_array()) {
                            ok = false;
                            break;
                        }
                        auto [sz, align] = capture_size_align(callee_v.type());
                        payload_offset = luisa::align(payload_offset, align) + sz;
                        payload_max_align = std::max(payload_max_align, align);
                        const bool is_written =
                            callee_v.tag() == Variable::Tag::REFERENCE &&
                            !callee_v.type()->is_buffer() &&
                            written_non_buffer.contains(callee_v.uid());
                        captures.emplace_back(LoweredRayQueryCapture{
                            .caller_variable = caller_v,
                            .callee_variable = callee_v,
                            .is_written = is_written,
                            .is_buffer = callee_v.type()->is_buffer()});
                    } else {
                        if (!v.is_local()) {
                            ok = false;
                            break;
                        }
                        inner_locals.emplace_back(v);
                    }
                } else if (outside_vars.contains(v.uid())) {
                    if (v.type() == _ray_query_all_type ||
                        v.type() == _ray_query_any_type ||
                        v.type() == _indirect_dispatch_buffer_type ||
                        v.type()->is_texture() ||
                        v.type()->is_accel() ||
                        v.type()->is_bindless_array()) {
                        ok = false;
                        break;
                    }
                    auto [sz, align] = capture_size_align(v.type());
                    payload_offset = luisa::align(payload_offset, align) + sz;
                    payload_max_align = std::max(payload_max_align, align);
                    captures.emplace_back(LoweredRayQueryCapture{
                        .caller_variable = v,
                        .callee_variable = v,
                        .is_written = !v.type()->is_buffer() &&
                                      written_non_buffer.contains(v.uid()),
                        .is_buffer = v.type()->is_buffer()});
                } else {
                    if (!v.is_local()) {
                        ok = false;
                        break;
                    }
                    inner_locals.emplace_back(v);
                }
            }
            const auto payload_bytes =
                captures.empty() ? 1u : luisa::align(payload_offset, payload_max_align);
            if (!ok || payload_bytes > 128u) {
                failed_q.emplace(q_var.uid());
                continue;
            }

            // Symbolic dynamic force_opacity(forced_opacity::opaque) check.
            bool has_dynamic_opaque = false;
            luisa::vector<Variable> guard_vars;
            const bool no_written_captures =
                std::none_of(captures.begin(), captures.end(),
                             [](const auto &c) noexcept { return c.is_written; });
            if (no_written_captures && !uses_buffers && !has_loops_or_switches) {
                luisa::unordered_map<uint, Variable> captured_read_to_caller;
                for (const auto &c : captures) {
                    captured_read_to_caller.try_emplace(c.callee_variable.uid(), c.caller_variable);
                }
                luisa::unordered_set<uint> local_hit_vars;
                luisa::unordered_set<uint> local_hit_inst_vars;
                luisa::unordered_map<uint, luisa::optional<bool>> local_const_bools;

                auto is_hit_expr = [&](const Expression *e) noexcept {
                    if (e->tag() == Expression::Tag::CALL &&
                        static_cast<const CallExpr *>(e)->op() == CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT) {
                        return true;
                    }
                    return e->tag() == Expression::Tag::REF &&
                           local_hit_vars.contains(static_cast<const RefExpr *>(e)->variable().uid());
                };
                auto is_hit_inst_expr = [&](const Expression *e) noexcept {
                    if (e->tag() == Expression::Tag::MEMBER) {
                        auto m = static_cast<const MemberExpr *>(e);
                        return !m->is_swizzle() && m->member_index() == 0u &&
                               is_hit_expr(m->self());
                    }
                    return e->tag() == Expression::Tag::REF &&
                           local_hit_inst_vars.contains(static_cast<const RefExpr *>(e)->variable().uid());
                };
                auto add_guard_var = [&](Variable v) noexcept {
                    if (std::find(guard_vars.begin(), guard_vars.end(), v) == guard_vars.end()) {
                        guard_vars.emplace_back(v);
                    }
                };
                auto is_guard_uint_var = [&](const Expression *e, Variable &out_caller_v) noexcept {
                    if (e->tag() != Expression::Tag::REF) { return false; }
                    auto v = static_cast<const RefExpr *>(e)->variable();
                    if (!v.type()->is_uint32()) { return false; }
                    auto it = captured_read_to_caller.find(v.uid());
                    if (it == captured_read_to_caller.end()) { return false; }
                    out_caller_v = it->second;
                    return true;
                };
                std::function<luisa::optional<bool>(const Expression *)> eval_bool;
                eval_bool = [&](const Expression *e) noexcept -> luisa::optional<bool> {
                    if (e->tag() == Expression::Tag::LITERAL) {
                        auto lit = static_cast<const LiteralExpr *>(e)->value();
                        if (luisa::holds_alternative<bool>(lit)) {
                            return luisa::get<bool>(lit);
                        }
                        return luisa::nullopt;
                    }
                    if (e->tag() == Expression::Tag::REF) {
                        auto v = static_cast<const RefExpr *>(e)->variable();
                        if (auto it = local_const_bools.find(v.uid());
                            it != local_const_bools.end()) {
                            return it->second;
                        }
                        return luisa::nullopt;
                    }
                    if (e->tag() == Expression::Tag::UNARY) {
                        auto u = static_cast<const UnaryExpr *>(e);
                        if (u->op() == UnaryOp::NOT) {
                            if (auto sub = eval_bool(u->operand())) { return !*sub; }
                        }
                        return luisa::nullopt;
                    }
                    if (e->tag() == Expression::Tag::BINARY) {
                        auto b = static_cast<const BinaryExpr *>(e);
                        if (b->op() == BinaryOp::AND || b->op() == BinaryOp::BIT_AND) {
                            auto l = eval_bool(b->lhs());
                            if (l.has_value() && !*l) { return false; }
                            auto r = eval_bool(b->rhs());
                            if (r.has_value() && !*r) { return false; }
                            if (l.has_value() && r.has_value()) { return *l && *r; }
                            return luisa::nullopt;
                        }
                        if (b->op() == BinaryOp::OR || b->op() == BinaryOp::BIT_OR) {
                            auto l = eval_bool(b->lhs());
                            if (l.has_value() && *l) { return true; }
                            auto r = eval_bool(b->rhs());
                            if (r.has_value() && *r) { return true; }
                            if (l.has_value() && r.has_value()) { return *l || *r; }
                            return luisa::nullopt;
                        }
                        if (b->op() == BinaryOp::EQUAL || b->op() == BinaryOp::NOT_EQUAL) {
                            Variable gv;
                            if ((is_hit_inst_expr(b->lhs()) && is_guard_uint_var(b->rhs(), gv)) ||
                                (is_hit_inst_expr(b->rhs()) && is_guard_uint_var(b->lhs(), gv))) {
                                add_guard_var(gv);
                                return b->op() == BinaryOp::NOT_EQUAL;
                            }
                        }
                        return luisa::nullopt;
                    }
                    if (e->tag() == Expression::Tag::CALL) {
                        auto c = static_cast<const CallExpr *>(e);
                        if (c->op() == CallOp::SELECT && c->arguments().size() == 3u) {
                            if (auto cond = eval_bool(c->arguments()[2])) {
                                return eval_bool(*cond ? c->arguments()[1] : c->arguments()[0]);
                            }
                        }
                    }
                    return luisa::nullopt;
                };

                struct PathOutcome {
                    bool must_commit{false};
                    bool may_terminate{false};
                    bool valid{true};
                };
                std::function<PathOutcome(const ScopeStmt *)> eval_scope;
                eval_scope = [&](const ScopeStmt *scope) noexcept -> PathOutcome {
                    PathOutcome out{};
                    for (auto s : scope->statements()) {
                        if (is_wrapped_callable && s->tag() == Statement::Tag::RETURN) {
                            break;
                        }
                        if (s->tag() == Statement::Tag::SCOPE) {
                            auto sub = eval_scope(static_cast<const ScopeStmt *>(s));
                            if (!sub.valid) { out.valid = false; return out; }
                            out.must_commit |= sub.must_commit;
                            out.may_terminate |= sub.may_terminate;
                        } else if (s->tag() == Statement::Tag::ASSIGN) {
                            auto a = static_cast<const AssignStmt *>(s);
                            if (a->lhs()->tag() == Expression::Tag::REF) {
                                auto lv = static_cast<const RefExpr *>(a->lhs())->variable();
                                if (is_hit_expr(a->rhs())) {
                                    local_hit_vars.emplace(lv.uid());
                                } else if (is_hit_inst_expr(a->rhs()) && lv.type()->is_uint32()) {
                                    local_hit_inst_vars.emplace(lv.uid());
                                } else if (lv.type()->is_bool()) {
                                    local_const_bools[lv.uid()] = eval_bool(a->rhs());
                                }
                            }
                        } else if (s->tag() == Statement::Tag::EXPR) {
                            auto e = static_cast<const ExprStmt *>(s)->expression();
                            if (e->tag() == Expression::Tag::CALL) {
                                auto op = static_cast<const CallExpr *>(e)->op();
                                if (op == CallOp::RAY_QUERY_COMMIT_TRIANGLE) {
                                    out.must_commit = true;
                                } else if (op == CallOp::RAY_QUERY_TERMINATE) {
                                    out.may_terminate = true;
                                }
                            }
                        } else if (s->tag() == Statement::Tag::IF) {
                            auto ifs = static_cast<const IfStmt *>(s);
                            if (auto c = eval_bool(ifs->condition())) {
                                auto sub = eval_scope(*c ? ifs->true_branch() : ifs->false_branch());
                                if (!sub.valid) { out.valid = false; return out; }
                                out.must_commit |= sub.must_commit;
                                out.may_terminate |= sub.may_terminate;
                            } else {
                                auto saved_bools = local_const_bools;
                                auto tb = eval_scope(ifs->true_branch());
                                local_const_bools = saved_bools;
                                auto fb = eval_scope(ifs->false_branch());
                                local_const_bools.clear();
                                if (!tb.valid || !fb.valid) { out.valid = false; return out; }
                                out.must_commit |= (tb.must_commit && fb.must_commit);
                                out.may_terminate |= (tb.may_terminate || fb.may_terminate);
                            }
                        } else if (s->tag() != Statement::Tag::COMMENT) {
                            out.valid = false;
                            return out;
                        }
                    }
                    return out;
                };
                auto outcome = eval_scope(tri_body);
                const bool is_any = q_var.type() == _ray_query_any_type;
                if (outcome.valid && outcome.must_commit &&
                    (is_any || !outcome.may_terminate)) {
                    has_dynamic_opaque = true;
                } else {
                    guard_vars.clear();
                }
            }

            const bool needs_ift = !(has_dynamic_opaque && guard_vars.empty());
            size_t ift_idx = 0u;
            luisa::string fn_name;
            if (needs_ift) {
                ift_idx = _intersection_functions.size();
                fn_name = luisa::format("lc_rq_isect_{}", ift_idx);
                _intersection_functions.emplace_back(fn_name);
            }
            const auto info_index = _lowered_ray_queries.size();
            _lowered_ray_queries.emplace_back(LoweredRayQueryInfo{
                .stmt = rq,
                .owner_function = f,
                .handler_function = tri_fn,
                .handler_body = tri_body,
                .is_wrapped_callable = is_wrapped_callable,
                .ift_index = ift_idx,
                .function_name = std::move(fn_name),
                .needs_ift = needs_ift,
                .has_dynamic_opaque_fast_path = has_dynamic_opaque,
                .dynamic_opaque_guards = std::move(guard_vars),
                .captures = std::move(captures),
                .inner_locals = std::move(inner_locals)});
            _lowered_ray_query_map.try_emplace(rq, info_index);
            lowered_by_q[q_var.uid()].emplace_back(info_index);
        }

        for (const auto &[q_uid, indices] : lowered_by_q) {
            if (!failed_q.contains(q_uid)) {
                _ift_only_ray_query_uids.emplace(make_rq_key(f.hash(), q_uid));
            }
        }
    }

    // Mark functions containing or transitively calling lowered IFT queries.
    for (const auto &info : _lowered_ray_queries) {
        if (info.needs_ift) {
            _functions_using_ift.emplace(info.owner_function.hash());
        }
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto f : functions) {
            if (_functions_using_ift.contains(f.hash())) { continue; }
            for (auto callee : sorted_custom_callables(f)) {
                if (_functions_using_ift.contains(callee.hash())) {
                    _functions_using_ift.emplace(f.hash());
                    changed = true;
                    break;
                }
            }
        }
    }
}

void MetalCodegenAST::_emit_lowered_ray_query_functions() noexcept {
    if (_lowered_ray_queries.empty()) { return; }
    for (const auto &info : _lowered_ray_queries) {
        if (!info.needs_ift) { continue; }
        _function = info.handler_function;
        _analyze_local_variables();

        _scratch << "struct alignas(16) LCRayQueryPayload_" << info.ift_index << " {\n";
        if (info.captures.empty()) {
            _scratch << "  uchar _dummy;\n";
        } else {
            for (const auto &cap : info.captures) {
                _scratch << "  ";
                _emit_type_name(cap.callee_variable.type(),
                                info.handler_function.variable_usage(cap.callee_variable.uid()));
                _scratch << " ";
                _emit_variable_name(cap.callee_variable);
                _scratch << ";\n";
            }
        }
        _scratch << "};\n\n";

        _scratch << "[[intersection(triangle, triangle_data, instancing LC_EXTENDED_LIMITS_TAG)]]\n"
                 << "LCTriangleIsectResult " << info.function_name << "(\n"
                 << "    uint _lc_inst [[instance_id]],\n"
                 << "    uint _lc_prim [[primitive_id]],\n"
                 << "    float2 _lc_bary [[barycentric_coord]],\n"
                 << "    float _lc_dist [[distance]],\n"
                 << "    ray_data LCRayQueryPayload_" << info.ift_index << " &_lc_payload [[payload]]) {\n"
                 << "  bool _lc_accept = false;\n"
                 << "  bool _lc_continue = true;\n";
        for (const auto &cap : info.captures) {
            _scratch << "  ";
            _emit_type_name(cap.callee_variable.type(),
                            info.handler_function.variable_usage(cap.callee_variable.uid()));
            _scratch << " ";
            _emit_variable_name(cap.callee_variable);
            _scratch << " = _lc_payload.";
            _emit_variable_name(cap.callee_variable);
            _scratch << ";\n";
        }
        for (auto u : info.inner_locals) {
            if (auto iter = _local_variable_emissions.find(u.uid());
                iter != _local_variable_emissions.end() &&
                iter->second.initializer == nullptr) {
                _scratch << "  ";
                _emit_type_name(u.type(), info.handler_function.variable_usage(u.uid()));
                _scratch << " ";
                _emit_variable_name(u);
                _scratch << "{};\n";
            }
        }
        _active_lowered_ray_query = &info;
        _indention = 1u;
        for (auto s : info.handler_body->statements()) {
            s->accept(*this);
        }
        _active_lowered_ray_query = nullptr;
        for (const auto &cap : info.captures) {
            if (cap.is_written) {
                _scratch << "  _lc_payload.";
                _emit_variable_name(cap.callee_variable);
                _scratch << " = ";
                _emit_variable_name(cap.callee_variable);
                _scratch << ";\n";
            }
        }
        _scratch << "  return LCTriangleIsectResult{_lc_accept, _lc_continue};\n"
                 << "}\n\n";
    }
}

void MetalCodegenAST::visit(const RayQueryStmt *stmt) noexcept {
    if (auto iter = _lowered_ray_query_map.find(stmt);
        iter != _lowered_ray_query_map.end()) {
        const auto &info = _lowered_ray_queries[iter->second];
        auto emit_opaque = [&] {
            _emit_indention();
            _scratch << "_lc_isect.force_opacity(forced_opacity::opaque);\n";
            _emit_indention();
            _scratch << "auto _lc_res = _lc_isect.intersect(";
            stmt->query()->accept(*this);
            _scratch << ".ray, ";
            stmt->query()->accept(*this);
            _scratch << ".accel, ";
            stmt->query()->accept(*this);
            _scratch << ".mask);\n";
            _emit_indention();
            stmt->query()->accept(*this);
            _scratch << ".ift_hit = lc_committed_hit_from_triangle_result(_lc_res);\n";
            _emit_indention();
            stmt->query()->accept(*this);
            _scratch << ".has_ift_hit = true;\n";
        };
        auto emit_ift = [&] {
            _emit_indention();
            _scratch << "_lc_isect.force_opacity(forced_opacity::non_opaque);\n";
            _emit_indention();
            _scratch << "LCRayQueryPayload_" << info.ift_index << " _lc_payload{};\n";
            for (const auto &cap : info.captures) {
                _emit_indention();
                _scratch << "_lc_payload.";
                _emit_variable_name(cap.callee_variable);
                _scratch << " = ";
                _emit_variable_name(cap.caller_variable);
                _scratch << ";\n";
            }
            _emit_indention();
            _scratch << "auto _lc_res = _lc_isect.intersect(";
            stmt->query()->accept(*this);
            _scratch << ".ray, ";
            stmt->query()->accept(*this);
            _scratch << ".accel, ";
            stmt->query()->accept(*this);
            _scratch << ".mask, lc_ift_" << info.ift_index << ", _lc_payload);\n";
            for (const auto &cap : info.captures) {
                if (cap.is_written) {
                    _emit_indention();
                    _emit_variable_name(cap.caller_variable);
                    _scratch << " = _lc_payload.";
                    _emit_variable_name(cap.callee_variable);
                    _scratch << ";\n";
                }
            }
            _emit_indention();
            stmt->query()->accept(*this);
            _scratch << ".ift_hit = lc_committed_hit_from_triangle_result(_lc_res);\n";
            _emit_indention();
            stmt->query()->accept(*this);
            _scratch << ".has_ift_hit = true;\n";
        };

        _emit_indention();
        _scratch << "/* ray query ift begin */\n";
        _emit_indention();
        _scratch << "{\n";
        _indention++;
        _emit_indention();
        _scratch << "intersector<triangle_data, instancing LC_EXTENDED_LIMITS_TAG> _lc_isect;\n";
        _emit_indention();
        _scratch << "_lc_isect.assume_geometry_type(geometry_type::triangle);\n";
        _emit_indention();
        _scratch << "_lc_isect.accept_any_intersection(";
        stmt->query()->accept(*this);
        _scratch << ".terminate_on_first_hit);\n";
        if (!info.needs_ift) {
            emit_opaque();
        } else if (info.has_dynamic_opaque_fast_path && !info.dynamic_opaque_guards.empty()) {
            _emit_indention();
            _scratch << "if (";
            for (auto i = 0u; i < info.dynamic_opaque_guards.size(); i++) {
                if (i != 0u) { _scratch << " && "; }
                _emit_variable_name(info.dynamic_opaque_guards[i]);
                _scratch << " == 0xffffffffu";
            }
            _scratch << ") {\n";
            _indention++;
            emit_opaque();
            _indention--;
            _emit_indention();
            _scratch << "} else {\n";
            _indention++;
            emit_ift();
            _indention--;
            _emit_indention();
            _scratch << "}\n";
        } else {
            emit_ift();
        }
        _indention--;
        _emit_indention();
        _scratch << "}\n";
        _emit_indention();
        _scratch << "/* ray query ift end */\n";
        return;
    }

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
    _emit_scope_local_variables(stmt->on_triangle_candidate());
    _emit_indention();
    _scratch << "/* ray query triangle branch */\n";
    for (auto s : stmt->on_triangle_candidate()->statements()) {
        s->accept(*this);
    }
    _indention--;
    _emit_indention();
    _scratch << "} else {\n";
    _indention++;
    _emit_scope_local_variables(stmt->on_procedural_candidate());
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

void MetalCodegenAST::visit(const SuspendStmt *) noexcept {
}

void MetalCodegenAST::visit(const AutoDiffStmt *stmt) noexcept {
    _emit_indention();
    _scratch << "/* autodiff begin */\n";
    stmt->body()->accept(*this);
    _emit_indention();
    _scratch << "/* autodiff end */\n";
}

}// namespace luisa::compute::metal
