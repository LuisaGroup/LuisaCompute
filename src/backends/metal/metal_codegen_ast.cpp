//
// Created by Mike Smith on 2023/4/15.
//

#include <core/logging.h>
#include <core/magic_enum.h>
#include <runtime/rtx/ray.h>
#include <runtime/rtx/hit.h>
#include <dsl/rtx/ray_query.h>
#include <backends/metal/metal_builtin_embedded.h>
#include <backends/metal/metal_codegen_ast.h>

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
};

}// namespace detail

MetalCodegenAST::MetalCodegenAST(StringScratch &scratch) noexcept
    : _scratch{scratch},
      _ray_type{Type::of<Ray>()},
      _triangle_hit_type{Type::of<TriangleHit>()},
      _procedural_hit_type{Type::of<ProceduralHit>()},
      _committed_hit_type{Type::of<CommittedHit>()},
      _ray_query_all_type{Type::of<RayQueryAll>()},
      _ray_query_any_type{Type::of<RayQueryAny>()} {}

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
            type != _ray_query_any_type) {
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

    if (type == nullptr) { _scratch << "void"; }

    switch (type->tag()) {
        case Type::Tag::BOOL: _scratch << "bool"; break;
        case Type::Tag::FLOAT16: _scratch << "half"; break;
        case Type::Tag::FLOAT32: _scratch << "float"; break;
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
            _emit_type_name(type->element());
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
            if (type == _ray_query_all_type) {
                _scratch << "LCRayQueryAll";
            } else if (type == _ray_query_any_type) {
                _scratch << "LCRayQueryAny";
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
        case Variable::Tag::DISPATCH_SIZE: _scratch << "ls"; break;
        default: LUISA_ERROR_WITH_LOCATION("Not implemented.");
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
        _scratch << "  alignas(16) uint3 ds;\n"
                 << "};\n\n";

        // emit function signature and prelude
        _scratch << "[[kernel]]\n"
                 << "void kernel_main(constant Arguments &args,\n"
                 << "                 uint3 tid [[thread_position_in_threadgroup]],\n"
                 << "                 uint3 bid [[threadgroup_position_in_grid]],\n"
                 << "                 uint3 did [[thread_position_in_grid]],\n"
                 << "                 uint3 bs [[threads_per_threadgroup]]) {\n\n"
                 << "  lc_assume("
                 << "bs.x == " << _function.block_size().x << " && "
                 << "bs.y == " << _function.block_size().y << " && "
                 << "bs.z == " << _function.block_size().z << ");\n"
                 << "  auto ds = args.ds;\n"
                 << "  if (!all(did < ds)) { return; }\n\n"
                 << "  /* kernel arguments */\n";
        for (auto arg : _function.arguments()) {
            _scratch << "  auto ";
            _emit_variable_name(arg);
            _scratch << " = args.";
            _emit_variable_name(arg);
            _scratch << ";\n";
        }
    } else {
        auto is_mut_ref = [f = _function](auto arg) noexcept {
            return arg.is_reference() &&
                   (to_underlying(f.variable_usage(arg.uid())) &
                    to_underlying(Usage::WRITE));
        };
        auto any_mut_ref = std::any_of(_function.arguments().cbegin(),
                                       _function.arguments().cend(), is_mut_ref);
        if (any_mut_ref) {
            _scratch << "template<";
            for (auto arg : _function.arguments()) {
                if (is_mut_ref(arg)) {
                    _scratch << "typename T" << arg.uid() << ", ";
                }
            }
            _scratch.pop_back();
            _scratch.pop_back();
            _scratch << ">\n";
        }
        _emit_type_name(_function.return_type());
        _scratch << " callable_" << hash_to_string(_function.hash()) << "(";
        if (!_function.arguments().empty()) {
            for (auto arg : _function.arguments()) {
                if (is_mut_ref(arg)) {
                    _scratch << "T" << arg.uid();
                } else {
                    _emit_type_name(arg.type(), _function.variable_usage(arg.uid()));
                }
                _scratch << " ";
                _emit_variable_name(arg);
                _scratch << ", ";
            }
            _scratch.pop_back();
            _scratch.pop_back();
        }
        _scratch << ") {\n";
    }

    // emit shared variables
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

    // emit local variables
    _scratch << "\n  /* local variables */\n";
    for (auto local : _function.local_variables()) {
        _scratch << "  ";
        _emit_type_name(local.type(), _function.variable_usage(local.uid()));
        _scratch << " ";
        _emit_variable_name(local);
        _scratch << "{};\n";
    }

    // emit function body
    _scratch << "\n  /* function body begin */\n";
    _indention = 1u;
    for (auto s : _function.body()->statements()) { s->accept(*this); }
    _scratch << "\n  /* function body end */\n";
    _scratch << "}\n\n";
}

void MetalCodegenAST::_emit_constant(const Function::Constant &c) noexcept {
    _scratch << "constant ";
    _emit_type_name(c.type);
    _scratch << " c" << hash_to_string(c.data.hash()) << "{";
    auto count = c.type->dimension();
    static constexpr auto wrap = 16u;
    using namespace std::string_view_literals;
    luisa::visit([count, this](auto ptr) {
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
    _scratch << "};\n\n";
}

void MetalCodegenAST::emit(Function kernel) noexcept {

    // emit device library
    _scratch << luisa::string_view{luisa_metal_builtin_metal_device_lib,
                                   sizeof(luisa_metal_builtin_metal_device_lib)}
             << "\n";

    // emit types
    _emit_type_decls(kernel);

    // collect functions
    luisa::vector<Function> functions;
    {
        auto collect_functions = [&functions, collected = luisa::unordered_set<Function>{}](
                                     auto &&self, Function function) mutable noexcept -> void {
            if (collected.emplace(function).second) {
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
    _scratch << "(";
    expr->self()->accept(*this);
    _scratch << ")";
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

void MetalCodegenAST::visit(const AccessExpr *expr) noexcept {
    expr->range()->accept(*this);
    _scratch << "[";
    expr->index()->accept(*this);
    _scratch << "]";
}

void MetalCodegenAST::visit(const LiteralExpr *expr) noexcept {
    luisa::visit(detail::LiteralPrinter{_scratch}, expr->value());
}

void MetalCodegenAST::visit(const RefExpr *expr) noexcept {
    if (auto v = expr->variable();
        v.is_reference() &&
        (to_underlying(_function.variable_usage(v.uid())) &
         to_underlying(Usage::WRITE))) {
        _scratch << "(*";
        _emit_variable_name(v);
        _scratch << ")";
    } else {
        _emit_variable_name(v);
    }
}

void MetalCodegenAST::visit(const CallExpr *expr) noexcept {
}

void MetalCodegenAST::visit(const CastExpr *expr) noexcept {
    switch (expr->op()) {
        case CastOp::STATIC: _scratch << "static_cast<"; break;
        case CastOp::BITWISE: _scratch << "as_type<"; break;
    }
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
        _scratch << " else {";
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
    _scratch << "static_cast<void>(";
    stmt->expression()->accept(*this);
    _scratch << ");\n";
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

void MetalCodegenAST::visit(const RayQueryStmt *stmt) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

}// namespace luisa::compute::metal
