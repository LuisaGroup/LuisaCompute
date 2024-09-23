#include <string_view>

#include <luisa/core/stl/algorithm.h>
#include <luisa/core/logging.h>
#include <luisa/ast/type_registry.h>
#include <luisa/ast/constant_data.h>
#include <luisa/runtime/rtx/ray.h>
#include <luisa/runtime/rtx/hit.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/dsl/rtx/ray_query.h>

#include "cuda_texture.h"
#include "cuda_codegen_ast.h"
#include "../common/cast.h"

namespace luisa::compute::cuda {

namespace detail {

[[nodiscard]] static auto glob_variables_with_grad(Function f) noexcept {
    luisa::unordered_set<Variable> gradient_variables;
    traverse_expressions<true>(
        f.body(),
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
                    gradient_variables.emplace(v);
                }
            }
        },
        [](auto) noexcept {},
        [](auto) noexcept {});
    return gradient_variables;
}

}// namespace detail

class CUDACodegenAST::RayQueryLowering {

public:
    struct CapturedElement {
        Variable base_variable;
        const Type *element_type;
        luisa::vector<uint> access_indices;
    };

    struct OutlineInfo {
        uint index;
        luisa::vector<Variable> captured_resources;
        luisa::vector<CapturedElement> captured_elements;
    };

    struct FunctionResource {
        Function f;
        Variable v;
        [[nodiscard]] auto operator==(const FunctionResource &rhs) const noexcept {
            return f.builder() == rhs.f.builder() && v.uid() == rhs.v.uid();
        }
        [[nodiscard]] auto hash() const noexcept {
            return luisa::hash_combine({f.hash(), v.hash()});
        }
    };

    struct FunctionResourceHash {
        using is_avalanching = void;
        [[nodiscard]] auto operator()(const FunctionResource &x) const noexcept {
            return x.hash();
        }
    };

private:
    CUDACodegenAST *_codegen;
    luisa::unordered_map<const RayQueryStmt *, Function> _ray_query_statements;
    luisa::unordered_map<const RayQueryStmt *, OutlineInfo> _outline_infos;
    luisa::unordered_map<FunctionResource,
                         luisa::unique_ptr<luisa::unordered_set<Variable>>,
                         FunctionResourceHash>
        _root_resources;

private:
    void _collect_ray_query_statements(Function f) noexcept {
        traverse_expressions<true>(
            f.body(),
            [this](auto expr) noexcept {
                if (expr->tag() == Expression::Tag::CALL) {
                    auto call_expr = static_cast<const CallExpr *>(expr);
                    if (!call_expr->is_builtin()) {
                        _collect_ray_query_statements(call_expr->custom());
                    }
                }
            },
            [this, f](auto s) noexcept {
                if (s->tag() == Statement::Tag::RAY_QUERY) {
                    auto rq_stmt = static_cast<const RayQueryStmt *>(s);
                    _ray_query_statements.emplace(rq_stmt, f);
                }
            },
            [](auto) noexcept {});
    }

    void _glob_variables(luisa::unordered_set<Variable> &within_scope,
                         luisa::unordered_set<Variable> &without_scope,
                         const ScopeStmt *scope,
                         luisa::span<const ScopeStmt *const> target_scopes) noexcept {
        luisa::vector<bool> inside_scope_stack{false};
        traverse_expressions<true>(
            scope,
            [&](auto expr) noexcept {
                if (expr->tag() == Expression::Tag::REF) {
                    auto v = static_cast<const RefExpr *>(expr)->variable();
                    if (inside_scope_stack.back()) {
                        within_scope.emplace(v);
                    } else {
                        without_scope.emplace(v);
                    }
                }
            },
            [&inside_scope_stack, target_scopes](auto s) noexcept {
                if (s->tag() == Statement::Tag::SCOPE) {
                    auto inside_targets = inside_scope_stack.back() ||
                                          std::find(target_scopes.begin(), target_scopes.end(), s) !=
                                              target_scopes.end();
                    inside_scope_stack.emplace_back(inside_targets);
                }
            },
            [&inside_scope_stack](auto s) noexcept {
                if (s->tag() == Statement::Tag::SCOPE) {
                    inside_scope_stack.pop_back();
                }
            });
    }

    void _find_root_resources(Function f, luisa::span<const luisa::unordered_set<Variable> *> root_resources) noexcept {
        // collect root resources
        auto root_index = 0u;
        for (auto v : f.arguments()) {
            if (v.is_resource()) {
                LUISA_ASSERT(root_index < root_resources.size(),
                             "Root resource index {} is out of bound.",
                             root_index);
                auto set = _root_resources.try_emplace(
                                              FunctionResource{f, v},
                                              luisa::make_unique<luisa::unordered_set<Variable>>())
                               .first->second.get();
                for (auto r : *root_resources[root_index]) { set->emplace(r); }
                root_index++;
            }
        }
        LUISA_ASSERT(root_index == root_resources.size(),
                     "Root resource index and size mismatches.");
        // pass on root resources
        traverse_expressions<true>(
            f.body(),
            [this, f](auto expr) noexcept {
                if (expr->tag() == Expression::Tag::CALL) {
                    auto call_expr = static_cast<const CallExpr *>(expr);
                    if (!call_expr->is_builtin()) {// custom callables
                        // prepare root resource list
                        luisa::fixed_vector<const luisa::unordered_set<Variable> *, 16u> root_resources;
                        for (auto arg : call_expr->arguments()) {
                            if (arg->tag() == Expression::Tag::REF) {
                                if (auto v = static_cast<const RefExpr *>(arg)->variable();
                                    v.is_resource()) {
                                    auto iter = _root_resources.find(FunctionResource{f, v});
                                    LUISA_ASSERT(iter != _root_resources.cend(),
                                                 "Failed to find root resource.");
                                    root_resources.emplace_back(iter->second.get());
                                }
                            }
                        }
                        // pass on to the callee
                        _find_root_resources(
                            call_expr->custom(), root_resources);
                    }
                }
            },
            [](auto) noexcept {},
            [](auto) noexcept {});
    }

    void _find_root_resources(Function f) noexcept {
        LUISA_ASSERT(f.tag() == Function::Tag::KERNEL, "Invalid root.");
        using set_type = luisa::unordered_set<Variable>;
        luisa::vector<set_type> root_resources;
        root_resources.reserve(f.arguments().size());
        for (auto a : f.arguments()) {
            if (a.is_resource()) {
                root_resources.emplace_back(set_type{a});
            }
        }
        luisa::vector<const set_type *> views;
        views.reserve(root_resources.size());
        for (auto &s : root_resources) { views.emplace_back(&s); }
        _find_root_resources(f, views);
    }

    void _emit_captured_element(Variable base, luisa::span<const uint> indices) const noexcept {
        _codegen->_emit_variable_name(base);
        auto type = base.type();
        for (auto i : indices) {
            switch (type->tag()) {
                case Type::Tag::VECTOR: {
                    LUISA_ASSERT(i < type->dimension(),
                                 "Invalid access index {} for vector type {}.",
                                 i, type->description());
                    std::array elem{"x", "y", "z", "w"};
                    _codegen->_scratch << "." << elem[i];
                    type = type->element();
                    break;
                }
                case Type::Tag::MATRIX: {
                    LUISA_ASSERT(i < type->dimension(),
                                 "Invalid access index {} for matrix type {}.",
                                 i, type->description());
                    _codegen->_scratch << "[" << i << "]";
                    type = Type::vector(type->element(), type->dimension());
                    break;
                }
                case Type::Tag::ARRAY: {
                    LUISA_ASSERT(i < type->dimension(),
                                 "Invalid access index {} for array type {}.",
                                 i, type->description());
                    _codegen->_scratch << "[" << i << "]";
                    type = type->element();
                    break;
                }
                case Type::Tag::STRUCTURE: {
                    LUISA_ASSERT(i < type->members().size(),
                                 "Invalid access index {} for structure type {}.",
                                 i, type->description());
                    _codegen->_scratch << ".m" << i;
                    type = type->members()[i];
                    break;
                }
                default:
                    LUISA_ERROR_WITH_LOCATION(
                        "Invalid type {} for access chain.",
                        type->description());
            }
        }
    }

    void _emit_outline_context_member_name(Variable base, luisa::span<const uint> indices) const noexcept {
        _codegen->_emit_variable_name(base);
        for (auto i : indices) { _codegen->_scratch << "_" << i; }
    }

    void _create_outline_definitions(Function f, const RayQueryStmt *s) noexcept {
        // check if the ray query is already outlined
        if (_outline_infos.contains(s)) { return; }

        // sort variables used in the ray query
        std::array target_scopes{s->on_triangle_candidate(),
                                 s->on_procedural_candidate()};
        luisa::unordered_set<Variable> within_scope_variables;
        luisa::unordered_set<Variable> without_scope_variables;
        _glob_variables(within_scope_variables,
                        without_scope_variables,
                        f.body(), target_scopes);
        // find local and captured variables
        luisa::unordered_set<Variable> local_variable_set;// V(local) = V(all) - V(function - scope)
        for (auto v : f.local_variables()) {
            if (!without_scope_variables.contains(v) &&
                !v.is_builtin() &&
                v.type() != _codegen->_ray_query_all_type &&
                v.type() != _codegen->_ray_query_any_type) {
                local_variable_set.emplace(v);
            }
        }

        luisa::vector<Variable> uniquely_identified_resources;
        luisa::vector<Variable> captured_resources;
        luisa::vector<CapturedElement> captured_elements;
        {
            luisa::vector<Variable> captured_variables;// V(captured) = V(scope) - V(local)
            for (auto v : within_scope_variables) {
                if (!local_variable_set.contains(v) &&
                    !v.is_builtin() &&
                    v.type() != _codegen->_ray_query_all_type &&
                    v.type() != _codegen->_ray_query_any_type) {
                    captured_variables.emplace_back(v);
                }
            }

            // if the resources can be uniquely identified in the __constant__ params,
            // then no need for passing via stack with I/O on local memory
            {
                auto iter = std::partition(
                    captured_variables.begin(), captured_variables.end(),
                    [this, f](auto v) noexcept {
                        LUISA_ASSERT(!v.is_resource() ||
                                         _root_resources.contains({f, v}),
                                     "Invalid variable.");
                        return !v.is_resource() ||
                               _root_resources.at({f, v})->size() > 1u;
                    });

                uniquely_identified_resources.reserve(
                    std::distance(iter, captured_variables.end()));
                for (auto i = iter; i != captured_variables.end(); i++) {
                    uniquely_identified_resources.emplace_back(*i);
                }
                captured_variables.erase(iter, captured_variables.cend());
            }

            // find the captured resources
            {
                auto iter = std::partition(
                    captured_variables.begin(), captured_variables.end(),
                    [](auto v) noexcept { return !v.is_resource(); });
                captured_resources.reserve(
                    std::distance(iter, captured_variables.end()));
                for (auto i = iter; i != captured_variables.end(); i++) {
                    captured_resources.emplace_back(*i);
                }
                captured_variables.erase(iter, captured_variables.cend());
            }

            // pack the variables into a tight struct to minimize the stack size
            {
                luisa::fixed_vector<uint, 8u> indices;
                auto glob_elements = [&indices, &captured_elements](auto &&self, Variable base, const Type *type) noexcept -> void {
                    if (type->is_scalar()) {
                        CapturedElement element{
                            base, type, {indices.cbegin(), indices.cend()}};
                        captured_elements.emplace_back(std::move(element));
                        return;
                    }
                    switch (type->tag()) {
                        case Type::Tag::VECTOR:
                        case Type::Tag::ARRAY: {
                            auto n = type->dimension();
                            auto elem = type->element();
                            for (auto i = 0u; i < n; i++) {
                                indices.emplace_back(i);
                                self(self, base, elem);
                                indices.pop_back();
                            }
                            break;
                        }
                        case Type::Tag::MATRIX: {
                            auto n = type->dimension();
                            auto elem = Type::vector(type->element(), n);
                            for (auto i = 0u; i < n; i++) {
                                indices.emplace_back(i);
                                self(self, base, elem);
                                indices.pop_back();
                            }
                            break;
                        }
                        case Type::Tag::STRUCTURE: {
                            auto members = type->members();
                            for (auto i = 0u; i < members.size(); i++) {
                                indices.emplace_back(i);
                                self(self, base, members[i]);
                                indices.pop_back();
                            }
                            break;
                        }
                        default: LUISA_ERROR_WITH_LOCATION(
                            "Invalid type {}.", type->description());
                    }
                };

                captured_elements.reserve(captured_variables.size() * 4u);
                for (auto v : captured_variables) {
                    glob_elements(glob_elements, v, v.type());
                }

                // sort the members to minimize the stack size
                std::stable_sort(
                    captured_elements.begin(), captured_elements.end(),
                    [](auto lhs, auto rhs) noexcept {
                        return lhs.element_type->alignment() > rhs.element_type->alignment();
                    });
            }
        }

        // create outline struct
        // TODO: we may pass the values directly through
        //  OptiX registers if they are small enough
        auto rq_index = static_cast<uint>(_outline_infos.size());
        if (!captured_elements.empty() ||
            !captured_resources.empty()) {
            _codegen->_scratch << "struct LCRayQueryCtx" << rq_index << " {";
            for (auto &&v : captured_resources) {
                _codegen->_scratch << "\n  ";
                _codegen->_emit_variable_decl(f, v, false);
                _codegen->_scratch << ";";
            }
            for (auto &&v : captured_elements) {
                _codegen->_scratch << "\n  ";
                _codegen->_emit_type_name(v.element_type);
                _codegen->_scratch << " ";
                _emit_outline_context_member_name(
                    v.base_variable, v.access_indices);
                if (v.element_type == Type::of<bool>()) {
                    _codegen->_scratch << " : 1";
                }
                _codegen->_scratch << ";";
            }
            _codegen->_scratch << "\n};\n\n";
        }

        auto grad_variables = detail::glob_variables_with_grad(f);
        auto generate_intersection_body = [&](const ScopeStmt *stmt) noexcept {
            // corner case: emit nothing if empty handler
            if (stmt->statements().empty()) { return; }

            // emit the code
            auto indent = _codegen->_indent;
            _codegen->_indent = 1;
            // built-in variables
            _codegen->_emit_builtin_variables();
            _codegen->_scratch << "\n";
            // obtain the uniquely-identified resources
            for (auto v : uniquely_identified_resources) {
                auto r = _root_resources.at({f, v}).get();
                LUISA_ASSERT(r->size() == 1u, "Invalid root resource.");
                _codegen->_emit_indent();
                _codegen->_emit_variable_decl(f, v, false);
                _codegen->_scratch << " = params.";
                _codegen->_emit_variable_name(*r->cbegin());
                _codegen->_scratch << ";\n";
                // inform the compiler of the underlying storage if the resource is captured
                // Note: this is O(n^2) but we should not have that many resources
                for (auto i = 0u; i < f.bound_arguments().size(); i++) {
                    auto binding = f.bound_arguments()[i];
                    if (auto b = luisa::get_if<Function::TextureBinding>(&binding);
                        b != nullptr && f.arguments()[i] == v) {
                        auto surf = reinterpret_cast<CUDATexture *>(b->handle)->binding(b->level);
                        _codegen->_emit_indent();
                        _codegen->_scratch << "lc_assume(";
                        _codegen->_emit_variable_name(v);
                        _codegen->_scratch << ".surface.storage == " << surf.storage << ");\n";
                    }
                }
            }
            // captured variables through stack
            if (!captured_elements.empty() ||
                !captured_resources.empty()) {
                // get ctx
                _codegen->_scratch << "  lc_assume(__isLocal(ctx_in));\n"
                                   << "  auto ctx = *static_cast<LCRayQueryCtx"
                                   << rq_index << " *>(ctx_in);\n";
                // copy captured resources
                for (auto &&v : captured_resources) {
                    _codegen->_emit_indent();
                    _codegen->_emit_variable_decl(f, v, false);
                    _codegen->_scratch << " = ctx.";
                    _codegen->_emit_variable_name(v);
                    _codegen->_scratch << ";\n";
                }
                // copy captured variables
                luisa::unordered_set<Variable> emitted_variables;
                for (auto &&v : captured_elements) {
                    if (emitted_variables.emplace(v.base_variable).second) {
                        _codegen->_emit_indent();
                        _codegen->_emit_type_name(v.base_variable.type());
                        _codegen->_scratch << " ";
                        _codegen->_emit_variable_name(v.base_variable);
                        _codegen->_scratch << "{};\n";
                    }
                    _codegen->_emit_indent();
                    _emit_captured_element(
                        v.base_variable, v.access_indices);
                    _codegen->_scratch << " = ctx.";
                    _emit_outline_context_member_name(
                        v.base_variable, v.access_indices);
                    _codegen->_scratch << ";\n";
                }
            }
            // declare local variables
            for (auto v : local_variable_set) {
                _codegen->_emit_indent();
                _codegen->_emit_variable_decl(f, v, false);
                _codegen->_scratch << "{};\n";
                if (grad_variables.contains(v)) {
                    _codegen->_emit_indent();
                    _codegen->_scratch << "LC_GRAD_SHADOW_VARIABLE(";
                    _codegen->_emit_variable_name(v);
                    _codegen->_scratch << ");\n";
                }
            }
            // emit body
            _codegen->_emit_indent();
            _codegen->_scratch << "{ // intersection handling body\n";
            _codegen->_indent++;
            for (auto s : stmt->statements()) {
                _codegen->_emit_indent();
                s->accept(*_codegen);
                _codegen->_scratch << "\n";
            }
            _codegen->_indent--;
            _codegen->_emit_indent();
            _codegen->_scratch << "} // intersection handling body\n";
            // copy back captured variables
            if (!captured_elements.empty()) {
                for (auto v : captured_elements) {
                    _codegen->_emit_indent();
                    _codegen->_scratch << "ctx.";
                    _emit_outline_context_member_name(
                        v.base_variable, v.access_indices);
                    _codegen->_scratch << " = ";
                    _emit_captured_element(
                        v.base_variable, v.access_indices);
                    _codegen->_scratch << ";\n";
                }
                _codegen->_emit_indent();
                _codegen->_scratch << "lc_assume(__isLocal(ctx_in));\n";
                _codegen->_emit_indent();
                _codegen->_scratch << "*static_cast<LCRayQueryCtx"
                                   << rq_index << " *>(ctx_in) = ctx;\n";
            }
            _codegen->_indent = indent;
        };
        // create outlined triangle function
        _codegen->_scratch << "LUISA_DECL_RAY_QUERY_TRIANGLE_IMPL(" << rq_index << ") {\n"
                           << "  LCIntersectionResult result{};\n";
        generate_intersection_body(s->on_triangle_candidate());
        _codegen->_scratch << "  return result;\n"
                              "}\n\n";

        // create outlined procedural function
        _codegen->_scratch << "LUISA_DECL_RAY_QUERY_PROCEDURAL_IMPL(" << rq_index << ") {\n"
                           << "  LCIntersectionResult result{};\n";
        generate_intersection_body(s->on_procedural_candidate());
        _codegen->_scratch << "  return result;\n"
                              "}\n\n";

        // record the outline function
        OutlineInfo info{rq_index,
                         std::move(captured_resources),
                         std::move(captured_elements)};
        _outline_infos.emplace(s, std::move(info));
    }

public:
    explicit RayQueryLowering(CUDACodegenAST *codegen) noexcept
        : _codegen{codegen} {}

    void preprocess(Function f) noexcept {
        _find_root_resources(f);
        _collect_ray_query_statements(f);
        _codegen->_scratch << "#define LUISA_RAY_QUERY_IMPL_COUNT "
                           << _ray_query_statements.size() << "\n";
    }

    void outline(Function f) noexcept {
        for (auto [rq, func] : _ray_query_statements) {
            if (func == f) { _create_outline_definitions(func, rq); }
        }
    }

    void lower(const RayQueryStmt *stmt) noexcept {
        LUISA_ASSERT(_outline_infos.contains(stmt),
                     "Ray query statement not outlined.");
        auto &&[rq_index, captured_resources, captured_elements] = _outline_infos.at(stmt);
        // create ray query context
        _codegen->_scratch << "\n";
        _codegen->_emit_indent();
        _codegen->_scratch << "{ // ray query #" << rq_index << "\n";
        _codegen->_indent++;
        // copy captured variables if any
        if (!captured_resources.empty() ||
            !captured_elements.empty()) {
            _codegen->_emit_indent();
            _codegen->_scratch << "LCRayQueryCtx" << rq_index << " ctx{\n";
            // copy captured variables
            _codegen->_indent++;
            for (auto &&v : captured_resources) {
                _codegen->_emit_indent();
                _codegen->_emit_variable_name(v);
                _codegen->_scratch << ",\n";
            }
            for (auto &&v : captured_elements) {
                _codegen->_emit_indent();
                _emit_captured_element(
                    v.base_variable, v.access_indices);
                _codegen->_scratch << ",\n";
            }
            _codegen->_indent--;
            _codegen->_emit_indent();
            _codegen->_scratch << "};\n";
        }
        // emit ray query call
        _codegen->_emit_indent();
        _codegen->_scratch << "lc_ray_query_trace(";
        stmt->query()->accept(*_codegen);
        if (captured_elements.empty() && captured_resources.empty()) {
            _codegen->_scratch << ", " << rq_index << ", nullptr);\n";
        } else {
            _codegen->_scratch << ", " << rq_index << ", &ctx);\n";
        }
        // copy back captured variables
        for (auto v : captured_elements) {
            if (!v.base_variable.is_resource()) {
                _codegen->_emit_indent();
                _emit_captured_element(
                    v.base_variable, v.access_indices);
                _codegen->_scratch << " = ctx.";
                _emit_outline_context_member_name(
                    v.base_variable, v.access_indices);
                _codegen->_scratch << ";\n";
            }
        }
        _codegen->_indent--;
        _codegen->_emit_indent();
        _codegen->_scratch << "} // ray query #" << rq_index << "\n";
    }
};

void CUDACodegenAST::visit(const UnaryExpr *expr) {
    switch (expr->op()) {
        case UnaryOp::PLUS: _scratch << "+"; break;
        case UnaryOp::MINUS: _scratch << "-"; break;
        case UnaryOp::NOT: _scratch << "!"; break;
        case UnaryOp::BIT_NOT: _scratch << "~"; break;
        default: break;
    }
    _scratch << "(";
    expr->operand()->accept(*this);
    _scratch << ")";
}

void CUDACodegenAST::visit(const BinaryExpr *expr) {
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

void CUDACodegenAST::visit(const MemberExpr *expr) {
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
                case Type::Tag::INT32: _scratch << "int"; break;
                case Type::Tag::UINT32: _scratch << "uint"; break;
                case Type::Tag::FLOAT32: _scratch << "float"; break;
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

void CUDACodegenAST::visit(const AccessExpr *expr) {
    expr->range()->accept(*this);
    _scratch << "[";
    expr->index()->accept(*this);
    _scratch << "]";
}

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
            _s << (v < 0.0f ? "(-lc_infinity_float())" : "(lc_infinity_float())");
        } else {
            _s << v << "f";
        }
    }
    void operator()(half v) const noexcept {
        if (luisa::isnan(v)) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Encountered with NaN."); }
        _s << luisa::format("lc_half({})", static_cast<float>(v));
    }
    void operator()(double v) const noexcept {
        if (std::isnan(v)) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Encountered with NaN."); }
        if (std::isinf(v)) {
            _s << (v < 0.0 ? "(-lc_infinity_double())" : "(lc_infinity_double())");
        } else {
            _s << v;
        }
    }
    void operator()(int v) const noexcept { _s << v; }
    void operator()(uint v) const noexcept { _s << v << "u"; }
    void operator()(short v) const noexcept { _s << luisa::format("lc_ushort({})", v); }
    void operator()(ushort v) const noexcept { _s << luisa::format("lc_short({})", v); }
    void operator()(slong v) const noexcept { _s << luisa::format("{}ll", v); }
    void operator()(ulong v) const noexcept { _s << luisa::format("{}ull", v); }
    void operator()(byte v) const noexcept { _s << luisa::format("lc_byte({})", v); }
    void operator()(ubyte v) const noexcept { _s << luisa::format("lc_ubyte({})", v); }
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

void CUDACodegenAST::visit(const LiteralExpr *expr) {
    luisa::visit(detail::LiteralPrinter{_scratch}, expr->value());
}

void CUDACodegenAST::visit(const RefExpr *expr) {
    _emit_variable_name(expr->variable());
}

void CUDACodegenAST::visit(const CallExpr *expr) {

    switch (expr->op()) {
        case CallOp::PACK: _scratch << "lc_pack_to"; break;
        case CallOp::UNPACK: {
            _scratch << "lc_unpack_from<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::CUSTOM: _scratch << "custom_" << hash_to_string(expr->custom().hash()); break;
        case CallOp::EXTERNAL: _scratch << expr->external()->name(); break;
        case CallOp::ALL: _scratch << "lc_all"; break;
        case CallOp::ANY: _scratch << "lc_any"; break;
        case CallOp::SELECT: _scratch << "lc_select"; break;
        case CallOp::CLAMP: _scratch << "lc_clamp"; break;
        case CallOp::SATURATE: _scratch << "lc_saturate"; break;
        case CallOp::LERP: _scratch << "lc_lerp"; break;
        case CallOp::STEP: _scratch << "lc_step"; break;
        case CallOp::SMOOTHSTEP: _scratch << "lc_smoothstep"; break;
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
        case CallOp::REFLECT: _scratch << "lc_reflect"; break;
        case CallOp::DETERMINANT: _scratch << "lc_determinant"; break;
        case CallOp::TRANSPOSE: _scratch << "lc_transpose"; break;
        case CallOp::INVERSE: _scratch << "lc_inverse"; break;
        case CallOp::SYNCHRONIZE_BLOCK: _scratch << "lc_synchronize_block"; break;
        case CallOp::ADDRESS_OF: _scratch << "lc_address_of"; break;
        case CallOp::ATOMIC_EXCHANGE: _scratch << "lc_atomic_exchange"; break;
        case CallOp::ATOMIC_COMPARE_EXCHANGE: _scratch << "lc_atomic_compare_exchange"; break;
        case CallOp::ATOMIC_FETCH_ADD: _scratch << "lc_atomic_fetch_add"; break;
        case CallOp::ATOMIC_FETCH_SUB: _scratch << "lc_atomic_fetch_sub"; break;
        case CallOp::ATOMIC_FETCH_AND: _scratch << "lc_atomic_fetch_and"; break;
        case CallOp::ATOMIC_FETCH_OR: _scratch << "lc_atomic_fetch_or"; break;
        case CallOp::ATOMIC_FETCH_XOR: _scratch << "lc_atomic_fetch_xor"; break;
        case CallOp::ATOMIC_FETCH_MIN: _scratch << "lc_atomic_fetch_min"; break;
        case CallOp::ATOMIC_FETCH_MAX: _scratch << "lc_atomic_fetch_max"; break;
        case CallOp::BUFFER_READ: _scratch << "lc_buffer_read"; break;
        case CallOp::BUFFER_WRITE: _scratch << "lc_buffer_write"; break;
        case CallOp::BUFFER_SIZE: _scratch << "lc_buffer_size"; break;
        case CallOp::BUFFER_ADDRESS: _scratch << "lc_buffer_address"; break;
        case CallOp::BYTE_BUFFER_READ: {
            _scratch << "lc_byte_buffer_read<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::BYTE_BUFFER_WRITE: _scratch << "lc_byte_buffer_write"; break;
        case CallOp::BYTE_BUFFER_SIZE: _scratch << "lc_byte_buffer_size"; break;
        case CallOp::TEXTURE_READ: _scratch << "lc_texture_read"; break;
        case CallOp::TEXTURE_WRITE: _scratch << "lc_texture_write"; break;
        case CallOp::TEXTURE_SIZE: _scratch << "lc_texture_size"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE: _scratch << "lc_bindless_texture_sample2d"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: _scratch << "lc_bindless_texture_sample2d_level"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: _scratch << "lc_bindless_texture_sample2d_grad"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: LUISA_NOT_IMPLEMENTED(); break;// TODO
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE: _scratch << "lc_bindless_texture_sample3d"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: _scratch << "lc_bindless_texture_sample3d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: _scratch << "lc_bindless_texture_sample3d_grad"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: LUISA_NOT_IMPLEMENTED(); break;// TODO
        case CallOp::BINDLESS_TEXTURE2D_READ: _scratch << "lc_bindless_texture_read2d"; break;
        case CallOp::BINDLESS_TEXTURE3D_READ: _scratch << "lc_bindless_texture_read3d"; break;
        case CallOp::BINDLESS_TEXTURE2D_READ_LEVEL: _scratch << "lc_bindless_texture_read2d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_READ_LEVEL: _scratch << "lc_bindless_texture_read3d_level"; break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE: _scratch << "lc_bindless_texture_size2d"; break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE: _scratch << "lc_bindless_texture_size3d"; break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: _scratch << "lc_bindless_texture_size2d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: _scratch << "lc_bindless_texture_size3d_level"; break;
        case CallOp::BINDLESS_BUFFER_READ: {
            _scratch << "lc_bindless_buffer_read<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::BINDLESS_BUFFER_WRITE: {
            _scratch << "lc_bindless_buffer_write";
            break;
        }
        case CallOp::BINDLESS_BYTE_BUFFER_READ: {
            _scratch << "lc_bindless_byte_buffer_read<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::BINDLESS_BUFFER_SIZE: _scratch << "lc_bindless_buffer_size"; break;
        case CallOp::BINDLESS_BUFFER_TYPE: _scratch << "lc_bindless_buffer_type"; break;
        case CallOp::BINDLESS_BUFFER_ADDRESS: _scratch << "lc_bindless_buffer_address"; break;
#define LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(type, tag)                      \
    case CallOp::MAKE_##tag##2: _scratch << "lc_make_" << #type "2"; break; \
    case CallOp::MAKE_##tag##3: _scratch << "lc_make_" << #type "3"; break; \
    case CallOp::MAKE_##tag##4: _scratch << "lc_make_" << #type "4"; break;
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(bool, BOOL)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(byte, BYTE)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(ubyte, UBYTE)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(short, SHORT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(ushort, USHORT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(int, INT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(uint, UINT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(long, LONG)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(ulong, ULONG)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(half, HALF)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(float, FLOAT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(double, DOUBLE)
#undef LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL
        case CallOp::MAKE_FLOAT2X2: _scratch << "lc_make_float2x2"; break;
        case CallOp::MAKE_FLOAT3X3: _scratch << "lc_make_float3x3"; break;
        case CallOp::MAKE_FLOAT4X4: _scratch << "lc_make_float4x4"; break;
        case CallOp::ASSERT: {
            if (expr->arguments().size() == 1u) {
                _scratch << "lc_assert";
            } else {
                _scratch << "lc_assert_with_message";
            }
            break;
        }
        case CallOp::ASSUME: _scratch << "lc_assume"; break;
        case CallOp::UNREACHABLE:
            if (expr->arguments().empty()) {
                _scratch << "lc_unreachable<";
                _emit_type_name(expr->type());
                _scratch << ">";
            } else {
                _scratch << "lc_unreachable_with_message<";
                _emit_type_name(expr->type());
                _scratch << ">";
            }
            break;
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
        case CallOp::RAY_TRACING_INSTANCE_TRANSFORM: _scratch << "lc_accel_instance_transform"; break;
        case CallOp::RAY_TRACING_INSTANCE_USER_ID: _scratch << "lc_accel_instance_user_id"; break;
        case CallOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: _scratch << "lc_accel_instance_visibility_mask"; break;
        case CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM: _scratch << "lc_accel_set_instance_transform"; break;
        case CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY: _scratch << "lc_accel_set_instance_visibility"; break;
        case CallOp::RAY_TRACING_SET_INSTANCE_OPACITY: _scratch << "lc_accel_set_instance_opacity"; break;
        case CallOp::RAY_TRACING_SET_INSTANCE_USER_ID: _scratch << "lc_accel_set_instance_user_id"; break;
        case CallOp::RAY_TRACING_TRACE_CLOSEST: _scratch << "lc_accel_trace_closest"; break;
        case CallOp::RAY_TRACING_TRACE_ANY: _scratch << "lc_accel_trace_any"; break;
        case CallOp::RAY_TRACING_QUERY_ALL: _scratch << "lc_accel_query_all"; break;
        case CallOp::RAY_TRACING_QUERY_ANY: _scratch << "lc_accel_query_any"; break;
        case CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: _scratch << "lc_accel_trace_closest_motion_blur"; break;
        case CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: _scratch << "lc_accel_trace_any_motion_blur"; break;
        case CallOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: _scratch << "lc_accel_query_all_motion_blur"; break;
        case CallOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: _scratch << "lc_accel_query_any_motion_blur"; break;
        case CallOp::RAY_QUERY_WORLD_SPACE_RAY: _scratch << "LC_RAY_QUERY_WORLD_RAY"; break;
        case CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT: _scratch << "LC_RAY_QUERY_PROCEDURAL_CANDIDATE_HIT"; break;
        case CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT: _scratch << "LC_RAY_QUERY_TRIANGLE_CANDIDATE_HIT"; break;
        case CallOp::RAY_QUERY_COMMITTED_HIT: _scratch << "lc_ray_query_committed_hit"; break;
        case CallOp::RAY_QUERY_COMMIT_TRIANGLE: _scratch << "LC_RAY_QUERY_COMMIT_TRIANGLE"; break;
        case CallOp::RAY_QUERY_COMMIT_PROCEDURAL: _scratch << "LC_RAY_QUERY_COMMIT_PROCEDURAL"; break;
        case CallOp::RAY_QUERY_TERMINATE: _scratch << "LC_RAY_QUERY_TERMINATE"; break;
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
        case CallOp::RASTER_DISCARD: LUISA_NOT_IMPLEMENTED(); break;
        case CallOp::INDIRECT_SET_DISPATCH_KERNEL: _scratch << "lc_indirect_set_dispatch_kernel"; break;
        case CallOp::INDIRECT_SET_DISPATCH_COUNT: _scratch << "lc_indirect_set_dispatch_count"; break;
        case CallOp::DDX: LUISA_NOT_IMPLEMENTED(); break;
        case CallOp::DDY: LUISA_NOT_IMPLEMENTED(); break;
        case CallOp::WARP_FIRST_ACTIVE_LANE: _scratch << "lc_warp_first_active_lane"; break;
        case CallOp::WARP_IS_FIRST_ACTIVE_LANE: _scratch << "lc_warp_is_first_active_lane"; break;
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
        case CallOp::RAY_TRACING_INSTANCE_MOTION_MATRIX: _scratch << "lc_accel_instance_motion_matrix"; break;
        case CallOp::RAY_TRACING_INSTANCE_MOTION_SRT: _scratch << "lc_accel_instance_motion_srt"; break;
        case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX: _scratch << "lc_accel_set_instance_motion_matrix"; break;
        case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT: _scratch << "lc_accel_set_instance_motion_srt"; break;

        // not supported
        case CallOp::RAY_QUERY_PROCEED: [[fallthrough]];
        case CallOp::RAY_QUERY_IS_TRIANGLE_CANDIDATE: [[fallthrough]];
        case CallOp::RAY_QUERY_IS_PROCEDURAL_CANDIDATE: [[fallthrough]];
        case CallOp::TEXTURE2D_SAMPLE: [[fallthrough]];
        case CallOp::TEXTURE2D_SAMPLE_LEVEL: [[fallthrough]];
        case CallOp::TEXTURE2D_SAMPLE_GRAD: [[fallthrough]];
        case CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: [[fallthrough]];
        case CallOp::TEXTURE3D_SAMPLE: [[fallthrough]];
        case CallOp::TEXTURE3D_SAMPLE_LEVEL: [[fallthrough]];
        case CallOp::TEXTURE3D_SAMPLE_GRAD: [[fallthrough]];
        case CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
            LUISA_NOT_IMPLEMENTED();
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
        _emit_access_chain(access_chain);
        for (auto extra : args.subspan(access_chain.size())) {
            _scratch << ", ";
            extra->accept(*this);
        }
    } else {
        if (op == CallOp::UNREACHABLE) {
            _scratch << "__FILE__, __LINE__";
            if (!expr->arguments().empty()) {
                _scratch << ", LC_DECODE_STRING_FROM_ID(";
                expr->arguments().front()->accept(*this);
                _scratch << ")";
            }
        } else if (op == CallOp::ASSERT) {
            auto args = expr->arguments();
            args[0]->accept(*this);
            if (args.size() > 1u) {
                _scratch << ", LC_DECODE_STRING_FROM_ID(";
                args[1]->accept(*this);
                _scratch << ")";
            }
        } else {
            auto trailing_comma = false;
            for (auto arg : expr->arguments()) {
                trailing_comma = true;
                arg->accept(*this);
                _scratch << ", ";
            }
            if (op == CallOp::CUSTOM && _requires_printing && !_requires_optix) {
                _scratch << "print_buffer, ";
                trailing_comma = true;
            }
            if (trailing_comma) {
                _scratch.pop_back();
                _scratch.pop_back();
            }
        }
    }
    _scratch << ")";
}

void CUDACodegenAST::_emit_access_chain(luisa::span<const Expression *const> chain) noexcept {
    auto type = chain.front()->type();
    _scratch << "(";
    chain.front()->accept(*this);
    for (auto index : chain.subspan(1u)) {
        switch (type->tag()) {
            case Type::Tag::VECTOR: [[fallthrough]];
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
                _scratch << ".ptr[";
                index->accept(*this);
                _scratch << "]";
                break;
            }
            default: LUISA_ERROR_WITH_LOCATION(
                "Invalid node type '{}' in access chain.",
                type->description());
        }
    }
    _scratch << ")";
}

void CUDACodegenAST::visit(const CastExpr *expr) {
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

void CUDACodegenAST::visit(const TypeIDExpr *expr) {
    _scratch << "static_cast<";
    _emit_type_name(expr->type());
    _scratch << ">(0ull)";
    // TODO: use type id
}

void CUDACodegenAST::visit(const StringIDExpr *expr) {
    _scratch << "static_cast<";
    _emit_type_name(expr->type());
    _scratch << luisa::format(">({})", _string_ids.at(expr->data()));
}

void CUDACodegenAST::visit(const BreakStmt *) {
    _scratch << "break;";
}

void CUDACodegenAST::visit(const ContinueStmt *) {
    _scratch << "continue;";
}

void CUDACodegenAST::visit(const ReturnStmt *stmt) {
    _scratch << "return";
    if (auto expr = stmt->expression(); expr != nullptr) {
        _scratch << " ";
        expr->accept(*this);
    }
    _scratch << ";";
}

void CUDACodegenAST::visit(const ScopeStmt *stmt) {
    _scratch << "{";
    _emit_statements(stmt->statements());
    _scratch << "}";
}

void CUDACodegenAST::visit(const IfStmt *stmt) {
    _scratch << "if (";
    stmt->condition()->accept(*this);
    _scratch << ") ";
    stmt->true_branch()->accept(*this);
    if (auto fb = stmt->false_branch(); fb != nullptr && !fb->statements().empty()) {
        _scratch << " else ";
        if (auto elif = ast_cast_to<IfStmt>(fb->statements().front());
            fb->statements().size() == 1u && elif != nullptr) {
            elif->accept(*this);
        } else {
            fb->accept(*this);
        }
    }
}

void CUDACodegenAST::visit(const LoopStmt *stmt) {
    _scratch << "for (;;) ";
    stmt->body()->accept(*this);
}

void CUDACodegenAST::visit(const ExprStmt *stmt) {
    stmt->expression()->accept(*this);
    _scratch << ";";
}

void CUDACodegenAST::visit(const SwitchStmt *stmt) {
    _scratch << "switch (";
    stmt->expression()->accept(*this);
    _scratch << ") ";
    stmt->body()->accept(*this);
}

void CUDACodegenAST::visit(const SwitchCaseStmt *stmt) {
    _scratch << "case ";
    stmt->expression()->accept(*this);
    _scratch << ": ";
    stmt->body()->accept(*this);
    if (std::none_of(stmt->body()->statements().begin(),
                     stmt->body()->statements().end(),
                     [](const auto &s) noexcept { return s->tag() == Statement::Tag::BREAK; })) {
        _scratch << " break;";
    }
}

void CUDACodegenAST::visit(const SwitchDefaultStmt *stmt) {
    _scratch << "default: ";
    stmt->body()->accept(*this);
    if (std::none_of(stmt->body()->statements().begin(),
                     stmt->body()->statements().end(),
                     [](const auto &s) noexcept { return s->tag() == Statement::Tag::BREAK; })) {
        _scratch << " break;";
    }
}

void CUDACodegenAST::visit(const AssignStmt *stmt) {
    stmt->lhs()->accept(*this);
    _scratch << " = ";
    stmt->rhs()->accept(*this);
    _scratch << ";";
}

void CUDACodegenAST::visit(const RayQueryStmt *stmt) {
    _ray_query_lowering->lower(stmt);
}

void CUDACodegenAST::visit(const AutoDiffStmt *stmt) {
    stmt->body()->accept(*this);
}

void CUDACodegenAST::emit(Function f,
                          luisa::string_view device_lib,
                          luisa::string_view native_include) {

    _requires_printing = f.requires_printing();
    _requires_optix = f.requires_raytracing();

    if (f.requires_raytracing()) {
        _scratch << "#define LUISA_ENABLE_OPTIX\n";
        if (f.required_curve_bases().any()) {
            _scratch << "#define LUISA_ENABLE_OPTIX_CURVE\n";
        }
        if (f.propagated_builtin_callables().test(CallOp::RAY_TRACING_TRACE_CLOSEST)) {
            _scratch << "#define LUISA_ENABLE_OPTIX_TRACE_CLOSEST\n";
        }
        if (f.propagated_builtin_callables().test(CallOp::RAY_TRACING_TRACE_ANY)) {
            _scratch << "#define LUISA_ENABLE_OPTIX_TRACE_ANY\n";
        }
        if (f.propagated_builtin_callables().test(CallOp::RAY_TRACING_QUERY_ALL) ||
            f.propagated_builtin_callables().test(CallOp::RAY_TRACING_QUERY_ANY)) {
            _scratch << "#define LUISA_ENABLE_OPTIX_RAY_QUERY\n";
            _ray_query_lowering->preprocess(f);
        }
    }
    _scratch << "#define LC_BLOCK_SIZE lc_make_uint3("
             << f.block_size().x << ", "
             << f.block_size().y << ", "
             << f.block_size().z << ")\n"
             << "\n/* built-in device library begin */\n"
             << device_lib
             << "\n/* built-in device library end */\n\n";

    _emit_type_decl(f);

    if (!native_include.empty()) {
        _scratch << "\n/* native include begin */\n\n"
                 << native_include
                 << "\n/* native include end */\n\n";
    }

    _emit_string_ids(f);
    _emit_function(f);
}

void CUDACodegenAST::_emit_function(Function f) noexcept {

    if (auto iter = std::find_if(
            _generated_functions.cbegin(),
            _generated_functions.cend(),
            [&](auto &&other) noexcept { return other == f.hash(); });
        iter != _generated_functions.cend()) { return; }
    _generated_functions.emplace_back(f.hash());

    // ray tracing kernels use __constant__ args
    // note: this must go before any other
    if (f.tag() == Function::Tag::KERNEL) {
        _scratch << "struct alignas(16) Params {";
        for (auto arg : f.arguments()) {
            _scratch << "\n  alignas(16) ";
            _emit_variable_decl(f, arg, !arg.type()->is_buffer());
            _scratch << "{};";
        }
        if (_requires_printing) {
            _scratch << "\n  alignas(16) LCPrintBuffer print_buffer{};";
        }
        _scratch << "\n  alignas(16) lc_uint4 ls_kid;";
        _scratch << "\n};\n\n";
        if (f.requires_raytracing()) {
            _scratch << "extern \"C\" { __constant__ Params params; }\n\n";
        }
    }

    // process dependent callables if any
    for (auto &&callable : f.custom_callables()) {
        _emit_function(callable->function());
    }

    _indent = 0u;
    _function = f;

    // constants
    if (!f.constants().empty()) {
        for (auto c : f.constants()) { _emit_constant(c); }
        _scratch << "\n";
    }

    // detect if there is a RayQueryStmt
    auto has_ray_query = false;
    traverse_expressions<false>(
        f.body(),
        [](auto) noexcept {},
        [&](auto stmt) noexcept {
            if (stmt->tag() == Statement::Tag::RAY_QUERY) {
                has_ray_query = true;
            }
        },
        [](auto) noexcept {});

    // outline ray query functions
    if (has_ray_query) { _ray_query_lowering->outline(f); }

    // signature
    if (f.tag() == Function::Tag::KERNEL) {
        _scratch << "extern \"C\" __global__ void "
                 << (f.requires_raytracing() ?
                         "__raygen__main" :
                         "kernel_main");
    } else if (f.tag() == Function::Tag::CALLABLE) {
        _scratch << "__forceinline__ __device__ ";
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
    if (f.tag() == Function::Tag::KERNEL) {
        if (!f.requires_raytracing()) {
            _scratch << "const Params params";
        }
        _scratch << ") {";
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
        if (!_requires_optix && _requires_printing) {
            _scratch << "\n  const auto &print_buffer = params.print_buffer;";
        }
        for (auto i = 0u; i < f.bound_arguments().size(); i++) {
            auto binding = f.bound_arguments()[i];
            if (auto b = luisa::get_if<Function::TextureBinding>(&binding)) {
                auto surface = reinterpret_cast<CUDATexture *>(b->handle)->binding(b->level);
                // inform the compiler of the underlying storage
                _scratch << "\n  lc_assume(";
                _emit_variable_name(f.arguments()[i]);
                _scratch << ".surface.storage == " << surface.storage << ");";
            }
        }
    } else {
        auto any_arg = false;
        for (auto arg : f.arguments()) {
            _scratch << "\n    ";
            _emit_variable_decl(f, arg, false);
            _scratch << ",";
            any_arg = true;
        }
        // we have to pass the print buffer to all callables
        if (!_requires_optix && _requires_printing) {
            _scratch << "\n    LCPrintBuffer print_buffer,";
            any_arg = true;
        }
        if (any_arg) { _scratch.pop_back(); }
        _scratch << ") noexcept {";
    }
    // emit built-in variables
    if (f.tag() == Function::Tag::KERNEL) {
        _emit_builtin_variables();
        if (!f.requires_raytracing()) {
            _scratch << "\n  if (lc_any(did >= ls)) { return; }";
        }
    }
    _indent = 1;
    _emit_variable_declarations(f);
    _indent = 0;
    _emit_statements(f.body()->statements());
    _scratch << "}\n\n";

    if (_allow_indirect_dispatch) {
        // generate meta-function that launches the kernel with dynamic parallelism
        if (f.tag() == Function::Tag::KERNEL && !f.requires_raytracing()) {
            _scratch << "extern \"C\" __global__ void kernel_launcher(Params params, const LCIndirectBuffer indirect) {\n"
                     << "  auto i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                     << "  auto n = min(indirect.header()->size, indirect.capacity - indirect.offset);\n"
                     << "  if (i < n) {\n"
                     << "    auto args = params;\n"
                     << "    auto d = indirect.dispatches()[i + indirect.offset];\n"
                     << "    args.ls_kid = d.dispatch_size_and_kernel_id;\n"
                     << "    auto block_size = lc_block_size();\n"
                     << "#ifdef LUISA_DEBUG\n"
                     << "    lc_assert(lc_all(block_size == d.block_size));\n"
                     << "#endif\n"
                     << "    auto dispatch_size = lc_make_uint3(d.dispatch_size_and_kernel_id);\n"
                     << "    if (lc_all(dispatch_size > 0u)) {\n"
                     << "      auto block_count = (dispatch_size + block_size - 1u) / block_size;\n"
                     << "      auto nb = dim3(block_count.x, block_count.y, block_count.z);\n"
                     << "      auto bs = dim3(block_size.x, block_size.y, block_size.z);\n"
                     << "      kernel_main<<<nb, bs>>>(args);\n"
                     << "    }\n"
                     << "  }\n"
                     << "}\n\n";
        }
    }
}

void CUDACodegenAST::_emit_builtin_variables() noexcept {
    _scratch
        // block size
        << "\n  constexpr auto bs = lc_block_size();"
        // launch size
        << "\n  const auto ls = lc_dispatch_size();"
        // dispatch id
        << "\n  const auto did = lc_dispatch_id();"
        // thread id
        << "\n  const auto tid = lc_thread_id();"
        // block id
        << "\n  const auto bid = lc_block_id();"
        // kernel id
        << "\n  const auto kid = lc_kernel_id();"
        // warp size
        << "\n  const auto ws = lc_warp_size();"
        // warp lane id
        << "\n  const auto lid = lc_warp_lane_id();";
}

void CUDACodegenAST::_emit_variable_name(Variable v) noexcept {
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
        case Variable::Tag::KERNEL_ID: _scratch << "kid"; break;
        case Variable::Tag::WARP_LANE_COUNT: _scratch << "ws"; break;
        case Variable::Tag::WARP_LANE_ID: _scratch << "lid"; break;
        default: LUISA_ERROR_WITH_LOCATION("Not implemented.");
    }
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

void CUDACodegenAST::_emit_type_decl(Function kernel) noexcept {

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

    // process types in topological order
    types.clear();
    auto emit = [&](auto &&self, auto type) noexcept -> void {
        if (type == Type::of<void>()) { return; }
        if (types.emplace(type).second) {
            if (type->is_array() || type->is_buffer()) {
                self(self, type->element());
            } else if (type->is_structure()) {
                for (auto m : type->members()) {
                    self(self, m);
                }
            }
            this->visit(type);
        }
    };
    for (auto t : sorted) { emit(emit, t); }

    // collect print args
    visited.clear();
    auto collect_print_args = [this, &visited](auto &&self, Function f) noexcept {
        if (!visited.emplace(f).second) { return; }
        for (auto &&c : f.custom_callables()) {
            self(self, Function{c.get()});
        }
        traverse_expressions<false>(
            f.body(),
            [](auto) noexcept {},
            [this](auto stmt) noexcept {
                if (stmt->tag() == Statement::Tag::PRINT) {
                    auto p = static_cast<const PrintStmt *>(stmt);
                    luisa::vector<const Type *> args;
                    args.reserve(p->arguments().size() + 2u);
                    args.emplace_back(Type::of<uint>());// arg size
                    args.emplace_back(Type::of<uint>());// fmt id
                    for (auto arg : p->arguments()) {
                        args.emplace_back(arg->type());
                    }
                    auto s = Type::structure(args);
                    this->_print_stmt_types.emplace(p, s);
                }
            },
            [](auto) noexcept {});
    };
    collect_print_args(collect_print_args, kernel);

    // sort print args by name so the generated
    // source is identical across runs
    sorted.clear();
    sorted.reserve(_print_stmt_types.size());
    for (auto [_, s] : _print_stmt_types) { sorted.emplace_back(s); }
    luisa::sort(sorted.begin(), sorted.end(), [](auto a, auto b) noexcept {
        return a->hash() < b->hash();
    });
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    // generate print args
    for (auto t : sorted) {
        _scratch << "struct LCPrintArgs_" << hash_to_string(t->hash()) << " {\n";
        for (auto i = 0u; i < t->members().size(); i++) {
            _scratch << "  ";
            _emit_type_name(t->members()[i]);
            _scratch << " m" << i << ";\n";
        }
        _scratch << "};\n\n";
    }
}

void CUDACodegenAST::visit(const Type *type) noexcept {
    if (type->is_structure() &&
        type != _ray_type &&
        type != _triangle_hit_type &&
        type != _procedural_hit_type &&
        type != _committed_hit_type &&
        type != _ray_query_all_type &&
        type != _ray_query_any_type &&
        type != _motion_srt_type) {

        auto emit_decl = [type, this](bool hack_float_to_int) noexcept {
            _scratch << "struct alignas(" << type->alignment() << ") ";
            _emit_type_name(type, hack_float_to_int);
            _scratch << " {\n";
            for (auto i = 0u; i < type->members().size(); i++) {
                _scratch << "  ";
                _emit_type_name(type->members()[i], hack_float_to_int);
                _scratch << " m" << i << "{};\n";
            }
            _scratch << "};\n\n";
        };
        emit_decl(false);
        emit_decl(true);
    }
    if (type->is_structure()) {
        // lc_zero and lc_one
        auto lc_make_value = [&](luisa::string_view name) noexcept {
            _scratch << "template<> __device__ inline auto " << name << "<";
            _emit_type_name(type);
            _scratch << ">() noexcept {\n"
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
        _scratch << "__device__ inline void lc_accumulate_grad(";
        _emit_type_name(type);
        _scratch << " *dst, ";
        _emit_type_name(type);
        _scratch << " grad) noexcept {\n";
        for (auto i = 0u; i < type->members().size(); i++) {
            _scratch << "  lc_accumulate_grad(&dst->m" << i << ", grad.m" << i << ");\n";
        }
        _scratch << "}\n\n";
    }
}

void CUDACodegenAST::_emit_type_name(const Type *type, bool hack_float_to_int) noexcept {
    if (type == nullptr) {
        _scratch << "void";
        return;
    }
    switch (type->tag()) {
        case Type::Tag::BOOL: _scratch << "lc_bool"; break;
        case Type::Tag::FLOAT16: _scratch << (hack_float_to_int ? "lc_ushort" : "lc_half"); break;
        case Type::Tag::FLOAT32: _scratch << (hack_float_to_int ? "lc_uint" : "lc_float"); break;
        case Type::Tag::FLOAT64: _scratch << (hack_float_to_int ? "lc_ulong" : "lc_double"); break;
        case Type::Tag::INT8: _scratch << "lc_byte"; break;
        case Type::Tag::UINT8: _scratch << "lc_ubyte"; break;
        case Type::Tag::INT16: _scratch << "lc_short"; break;
        case Type::Tag::UINT16: _scratch << "lc_ushort"; break;
        case Type::Tag::INT32: _scratch << "lc_int"; break;
        case Type::Tag::UINT32: _scratch << "lc_uint"; break;
        case Type::Tag::INT64: _scratch << "lc_long"; break;
        case Type::Tag::UINT64: _scratch << "lc_ulong"; break;
        case Type::Tag::VECTOR:
            _emit_type_name(type->element(), hack_float_to_int);
            _scratch << type->dimension();
            break;
        case Type::Tag::MATRIX:
            if (hack_float_to_int) {
                _scratch << "lc_array<lc_uint"
                         << type->dimension()
                         << ", "
                         << type->dimension()
                         << ">";
            } else {
                _scratch << "lc_float"
                         << type->dimension()
                         << "x"
                         << type->dimension();
            }
            break;
        case Type::Tag::ARRAY:
            _scratch << "lc_array<";
            _emit_type_name(type->element(), hack_float_to_int);
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
            } else if (type == _motion_srt_type) {
                _scratch << "LCMotionSRT";
            } else {
                _scratch << "S" << hash_to_string(type->hash());
                if (hack_float_to_int) { _scratch << "_int"; }
            }
            break;
        }
        case Type::Tag::CUSTOM: {
            if (type == _ray_query_all_type) {
                _scratch << "LCRayQueryAll";
            } else if (type == _ray_query_any_type) {
                _scratch << "LCRayQueryAny";
            } else if (type == _indirect_buffer_type) {
                _scratch << "LCIndirectBuffer";
            } else {
                LUISA_ERROR_WITH_LOCATION(
                    "Unsupported custom type: {}.",
                    type->description());
            }
            break;
        }
        default: break;
    }
}

void CUDACodegenAST::_emit_variable_decl(Function f, Variable v, bool force_const) noexcept {
    auto usage = f.variable_usage(v.uid());
    auto readonly = usage == Usage::NONE || usage == Usage::READ;
    switch (v.tag()) {
        case Variable::Tag::SHARED: {
            LUISA_ASSERT(v.type()->is_array(),
                         "Shared variable must be an array.");
            _scratch << "__shared__ lc_aligned_storage<"
                     << v.type()->alignment() << ", "
                     << v.type()->size() << ">  _";
            _emit_variable_name(v);
            break;
        }
        case Variable::Tag::REFERENCE:
            if (readonly || force_const) {
                _scratch << "const ";
                _emit_type_name(v.type());
                _scratch << " &";
            } else {
                _emit_type_name(v.type());
                _scratch << " &";
            }
            _emit_variable_name(v);
            break;
        case Variable::Tag::BUFFER:
            if (v.type() == _indirect_buffer_type) {
                _scratch << "LCIndirectBuffer ";
            } else {
                _scratch << "const LCBuffer<";
                if (readonly || force_const) { _scratch << "const "; }
                if (auto elem = v.type()->element()) {
                    _emit_type_name(elem);
                } else {// void type marks a buffer of bytes
                    _scratch << "lc_ubyte";
                }
                _scratch << "> ";
            }
            _emit_variable_name(v);
            break;
        case Variable::Tag::TEXTURE:
            _scratch << "const LCTexture"
                     << v.type()->dimension()
                     << "D<";
            _emit_type_name(v.type()->element());
            _scratch << "> ";
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

void CUDACodegenAST::_emit_indent() noexcept {
    for (auto i = 0u; i < _indent; i++) { _scratch << "  "; }
}

void CUDACodegenAST::_emit_statements(luisa::span<const Statement *const> stmts) noexcept {
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

// TODO: This would have trouble with infinities and NaNs
class CUDAConstantPrinter final : public ConstantDecoder {

private:
    CUDACodegenAST *_codegen;

public:
    explicit CUDAConstantPrinter(CUDACodegenAST *codegen) noexcept
        : _codegen{codegen} {}

protected:
    void _decode_bool(bool x) noexcept override { _codegen->_scratch << (x ? "true" : "false"); }
    void _decode_char(char x) noexcept override { _codegen->_scratch << luisa::format("lc_byte({})", static_cast<int>(x)); }
    void _decode_uchar(ubyte x) noexcept override { _codegen->_scratch << luisa::format("lc_ubyte({})", static_cast<uint>(x)); }
    void _decode_short(short x) noexcept override { _codegen->_scratch << luisa::format("lc_short({})", x); }
    void _decode_ushort(ushort x) noexcept override { _codegen->_scratch << luisa::format("lc_ushort({})", x); }
    void _decode_int(int x) noexcept override { _codegen->_scratch << luisa::format("lc_int({})", x); }
    void _decode_uint(uint x) noexcept override { _codegen->_scratch << luisa::format("lc_uint({})", x); }
    void _decode_long(slong x) noexcept override { _codegen->_scratch << luisa::format("lc_long({})", x); }
    void _decode_ulong(ulong x) noexcept override { _codegen->_scratch << luisa::format("lc_ulong({})", x); }
    void _decode_half(half x) noexcept override {
        _codegen->_scratch << luisa::format("lc_ushort({})", luisa::bit_cast<ushort>(x));
    }
    void _decode_float(float x) noexcept override {
        _codegen->_scratch << luisa::format("lc_uint({})", luisa::bit_cast<uint>(x));
    }
    void _decode_double(double x) noexcept override {
        _codegen->_scratch << luisa::format("lc_ulong({})", luisa::bit_cast<ulong>(x));
    }
    void _vector_separator(const Type *type, uint index) noexcept override {
        auto n = type->dimension();
        if (index == 0u) {
            _codegen->_emit_type_name(type, true);
            _codegen->_scratch << "{";
        } else if (index == n) {
            _codegen->_scratch << "}";
        } else {
            _codegen->_scratch << ", ";
        }
    }
    void _matrix_separator(const Type *type, uint index) noexcept override {
        auto n = type->dimension();
        if (index == 0u) {
            _codegen->_emit_type_name(type, true);
            _codegen->_scratch << "{";
        } else if (index == n) {
            _codegen->_scratch << "}";
        } else {
            _codegen->_scratch << ", ";
        }
    }
    void _struct_separator(const Type *type, uint index) noexcept override {
        auto n = type->members().size();
        if (index == 0u) {
            _codegen->_emit_type_name(type, true);
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
            _codegen->_emit_type_name(type, true);
            _codegen->_scratch << "{";
        } else if (index == n) {
            _codegen->_scratch << "}";
        } else {
            _codegen->_scratch << ", ";
        }
    }
};

void CUDACodegenAST::_emit_constant(Function::Constant c) noexcept {

    if (auto iter = std::find(_generated_constants.cbegin(),
                              _generated_constants.cend(), c.hash());
        iter != _generated_constants.cend()) { return; }
    _generated_constants.emplace_back(c.hash());
    _scratch << "__constant__ LC_CONSTANT auto c"
             << hash_to_string(c.hash())
             << " = ";
    CUDAConstantPrinter printer{this};
    c.decode(printer);
    _scratch << ";\n";
}

void CUDACodegenAST::_emit_string_ids(Function f) noexcept {

    auto collect_string_ids = [this, visited = luisa::unordered_set<Function>{}](
                                  auto &&self, Function f) mutable noexcept {
        if (!visited.emplace(f).second) { return; }
        traverse_expressions<true>(
            f.body(),
            [this, &self](const Expression *expr) noexcept {
                if (expr->tag() == Expression::Tag::CALL) {
                    auto call = static_cast<const CallExpr *>(expr);
                    if (call->is_custom()) {
                        auto custom = call->custom();
                        self(self, custom);
                    }
                } else if (expr->tag() == Expression::Tag::STRING_ID) {
                    auto s = static_cast<const StringIDExpr *>(expr);
                    auto n = static_cast<uint>(this->_string_ids.size());
                    this->_string_ids.try_emplace(s->data(), n);
                }
            },
            [](auto) noexcept {},
            [](auto) noexcept {});
    };

    collect_string_ids(collect_string_ids, f);

    if (!_string_ids.empty()) {
        luisa::vector<luisa::string_view> strings;
        strings.resize(_string_ids.size());
        auto total_size = static_cast<size_t>(0u);
        for (auto &&[s, i] : _string_ids) {
            LUISA_ASSERT(i < strings.size(), "String ID out of range.");
            strings[i] = s;
            total_size += s.size() + 1u /* trailing zero */;
        }

        // generate string offsets
        _scratch << "__constant__ LC_CONSTANT const lc_uint lc_string_offsets[] {\n  ";
        luisa::vector<char> string_data;
        string_data.resize(total_size);
        auto offset = static_cast<size_t>(0u);
        for (auto s : strings) {
            _scratch << offset << "u, ";
            std::memcpy(string_data.data() + offset, s.data(), s.size() + 1u);
            offset += s.size() + 1u;
        }
        _scratch << "\n};\n\n";

        // generate string data
        _scratch << "static const char lc_string_data[] {";
        for (auto i = 0u; i < string_data.size(); i++) {
            if (i % 32u == 0u) { _scratch << "\n  "; }
            _scratch << luisa::format("0x{:02x}, ", static_cast<int>(string_data[i]));
        }
        _scratch << "\n};\n\n";
    }
}

void CUDACodegenAST::visit(const ConstantExpr *expr) {
    _scratch << "(*reinterpret_cast<const ";
    _emit_type_name(expr->type());
    _scratch << " *>(&c" << hash_to_string(expr->data().hash()) << "))";
}

void CUDACodegenAST::visit(const ForStmt *stmt) {
    _scratch << "for (; ";
    stmt->condition()->accept(*this);
    _scratch << "; ";
    stmt->variable()->accept(*this);
    _scratch << " += ";
    stmt->step()->accept(*this);
    _scratch << ") ";
    stmt->body()->accept(*this);
}

void CUDACodegenAST::visit(const CommentStmt *stmt) {
    _scratch << "// ";
    for (auto c : stmt->comment()) {
        if (c == '\n') {
            _scratch << "\n";
            _emit_indent();
            _scratch << "// ";
        } else {
            char s[] = {c, '\0'};
            _scratch << s;
        }
    }
}

void CUDACodegenAST::_emit_variable_declarations(Function f) noexcept {
    for (auto v : f.shared_variables()) {
        if (_function.variable_usage(v.uid()) != Usage::NONE) {
            _scratch << "\n";
            _emit_indent();
            _emit_variable_decl(f, v, false);
            _scratch << ";\n";
            _emit_indent();
            _scratch << "auto &";
            _emit_variable_name(v);
            _scratch << " = *reinterpret_cast<";
            _emit_type_name(v.type());
            _scratch << " *>(&_";
            _emit_variable_name(v);
            _scratch << ");";
        }
    }
    auto grad_vars = detail::glob_variables_with_grad(f);
    for (auto v : f.local_variables()) {
        if (_function.variable_usage(v.uid()) != Usage::NONE) {
            _scratch << "\n";
            _emit_indent();
            _emit_variable_decl(f, v, false);
            _scratch << "{};";
        }
    }
    for (auto v : grad_vars) {
        _scratch << "\n";
        _emit_indent();
        _scratch << "LC_GRAD_SHADOW_VARIABLE(";
        _emit_variable_name(v);
        _scratch << ");";
    }
}

void CUDACodegenAST::visit(const PrintStmt *stmt) {
    //_scratch << "struct LCPrintArgs_" << hash_to_string(t->hash())
    auto iter = _print_stmt_types.find(stmt);
    LUISA_ASSERT(iter != _print_stmt_types.end(),
                 "Print statement type not found.");
    auto t = iter->second;
    auto fmt_id = [this, fmt = stmt->format(), t] {
        for (auto i = 0u; i < _print_formats.size(); i++) {
            if (auto &&[ff, tt] = _print_formats[i];
                ff == fmt && tt == t) { return i; }
        }
        auto index = static_cast<uint>(_print_formats.size());
        _print_formats.emplace_back(fmt, t);
        return index;
    }();
    _scratch << "lc_print_impl(LC_PRINT_BUFFER, LCPrintArgs_"
             << hash_to_string(t->hash())
             << "{" << t->size() << "u, " << fmt_id << "u";
    for (auto a : stmt->arguments()) {
        _scratch << ", ";
        a->accept(*this);
    }
    _scratch << "});";
}

void CUDACodegenAST::visit(const CpuCustomOpExpr *expr) {
    LUISA_ERROR_WITH_LOCATION(
        "CudaCodegen: CpuCustomOpExpr is not supported in CUDA backend.");
}

void CUDACodegenAST::visit(const GpuCustomOpExpr *expr) {
    LUISA_ERROR_WITH_LOCATION(
        "CudaCodegen: GpuCustomOpExpr is not supported in CUDA backend.");
}

CUDACodegenAST::CUDACodegenAST(StringScratch &scratch, bool allow_indirect) noexcept
    : _scratch{scratch},
      _ray_query_lowering{luisa::make_unique<RayQueryLowering>(this)},
      _allow_indirect_dispatch{allow_indirect},
      _ray_type{Type::of<Ray>()},
      _triangle_hit_type{Type::of<TriangleHit>()},
      _procedural_hit_type{Type::of<ProceduralHit>()},
      _committed_hit_type{Type::of<CommittedHit>()},
      _ray_query_all_type{Type::of<RayQueryAll>()},
      _ray_query_any_type{Type::of<RayQueryAny>()},
      _indirect_buffer_type{Type::of<IndirectDispatchBuffer>()},
      _motion_srt_type{Type::of<MotionInstanceTransformSRT>()} {}

CUDACodegenAST::~CUDACodegenAST() noexcept = default;

}// namespace luisa::compute::cuda
