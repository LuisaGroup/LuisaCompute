#include <luisa/core/logging.h>
#include <luisa/ast/variable.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/function_builder.h>

namespace luisa::compute {

Expression::Expression(Expression::Tag tag, const Type *type) noexcept
    : _type{type}, _builder{detail::FunctionBuilder::current()}, _tag{tag} {}

void Expression::mark(Usage usage) const noexcept {
    if (auto a = to_underlying(_usage), u = a | to_underlying(usage); a != u) {
        _usage = static_cast<Usage>(u);
        _mark(usage);
    }
}

uint64_t Expression::hash() const noexcept {
    if (!_hash_computed) {
        using namespace std::string_view_literals;
        static auto seed = hash_value("__hash_expression"sv);
        _hash = hash_combine(
            {static_cast<uint64_t>(_tag),
             _compute_hash(),
             _type ? _type->hash() : 0ull},
            seed);
        _hash_computed = true;
    }
    return _hash;
}

void RefExpr::_mark(Usage usage) const noexcept {
    if (auto fb = detail::FunctionBuilder::current(); fb == builder()) {
        fb->mark_variable_usage(_variable.uid(), usage);
    }
}

uint64_t RefExpr::_compute_hash() const noexcept {
    return hash_value(_variable);
}

void CallExpr::_mark() const noexcept {
    if (is_builtin()) {
        switch (_op) {
            case CallOp::BUFFER_WRITE:
            case CallOp::BINDLESS_BUFFER_WRITE:
            case CallOp::BYTE_BUFFER_WRITE:
            case CallOp::TEXTURE_WRITE:
            case CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM:
            case CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY:
            case CallOp::RAY_TRACING_SET_INSTANCE_OPACITY:
            case CallOp::RAY_TRACING_SET_INSTANCE_USER_ID:
            case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX:
            case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT:
            case CallOp::RAY_QUERY_COMMIT_TRIANGLE:
            case CallOp::RAY_QUERY_COMMIT_PROCEDURAL:
            case CallOp::RAY_QUERY_TERMINATE:
            case CallOp::RAY_QUERY_PROCEED:
            case CallOp::GRADIENT_MARKER:
            case CallOp::ACCUMULATE_GRADIENT:
            case CallOp::ATOMIC_EXCHANGE:
            case CallOp::ATOMIC_COMPARE_EXCHANGE:
            case CallOp::ATOMIC_FETCH_ADD:
            case CallOp::ATOMIC_FETCH_SUB:
            case CallOp::ATOMIC_FETCH_AND:
            case CallOp::ATOMIC_FETCH_OR:
            case CallOp::ATOMIC_FETCH_XOR:
            case CallOp::ATOMIC_FETCH_MIN:
            case CallOp::ATOMIC_FETCH_MAX:
            case CallOp::INDIRECT_SET_DISPATCH_KERNEL:
            case CallOp::INDIRECT_SET_DISPATCH_COUNT:
                _arguments[0]->mark(Usage::WRITE);
                for (auto i = 1u; i < _arguments.size(); i++) {
                    _arguments[i]->mark(Usage::READ);
                }
                break;
            default:
                for (auto arg : _arguments) {
                    arg->mark(Usage::READ);
                }
        }
    } else if (is_external()) {
        auto f = external();
        for (auto i = 0u; i < _arguments.size(); i++) {
            _arguments[i]->mark(f->argument_usages()[i]);
        }
    } else {
        auto args = custom().arguments();
        for (auto i = 0u; i < args.size(); i++) {
            auto arg = args[i];
            _arguments[i]->mark(
                arg.is_reference() || arg.is_resource() ?
                    custom().variable_usage(arg.uid()) :
                    Usage::READ);
        }
    }
}

uint64_t CallExpr::_compute_hash() const noexcept {
    auto hash = hash_value(_op);
    for (auto &&a : _arguments) {
        hash = hash_value(a->hash(), hash);
    }
    if (_op == CallOp::CUSTOM) {
        if (custom().hash_computed()) {
            hash = hash_value(custom().hash(), hash);
        } else {
            // recursive
            hash = hash_value(custom().builder(), hash);
        }
    } else if (_op == CallOp::EXTERNAL) {
        hash = hash_value(external()->hash(), hash);
    }
    return hash;
}

CallExpr::CallExpr(const Type *type, CallOp builtin, CallExpr::ArgumentList args) noexcept
    : Expression{Tag::CALL, type},
      _arguments{std::move(args)},
      _op{builtin} { _mark(); }

CallExpr::CallExpr(const Type *type, Function callable, CallExpr::ArgumentList args) noexcept
    : Expression{Tag::CALL, type},
      _arguments{std::move(args)},
      _op{CallOp::CUSTOM},
      _func{callable.builder()} { _mark(); }

CallExpr::CallExpr(const Type *type, const ExternalFunction *external, ArgumentList args) noexcept
    : Expression{Tag::CALL, type},
      _arguments{std::move(args)},
      _op{CallOp::EXTERNAL},
      _func{external} { _mark(); }

Function CallExpr::custom() const noexcept {
    LUISA_ASSERT(is_custom(), "Not a custom function.");
    return Function{luisa::get<CustomCallee>(_func)};
}

const ExternalFunction *CallExpr::external() const noexcept {
    LUISA_ASSERT(is_external(), "Not an external function.");
    return luisa::get<ExternalCallee>(_func);
}

void CallExpr::_unsafe_set_custom(CallExpr::CustomCallee callee) const noexcept {
    auto f = luisa::get_if<CustomCallee>(&_func);
    LUISA_ASSERT(f != nullptr && (*f)->hash() == callee->hash(),
                 "Not a custom function with hash {}.",
                 callee->hash());
    const_cast<Callee &>(_func) = callee;
}

uint64_t UnaryExpr::_compute_hash() const noexcept {
    return hash_combine({static_cast<uint64_t>(_op), _operand->hash()});
}

uint64_t BinaryExpr::_compute_hash() const noexcept {
    return hash_combine({static_cast<uint64_t>(_op),
                         _lhs->hash(),
                         _rhs->hash()});
}

uint64_t AccessExpr::_compute_hash() const noexcept {
    return hash_combine({_index->hash(), _range->hash()});
}

uint64_t MemberExpr::_compute_hash() const noexcept {
    return hash_combine({(static_cast<uint64_t>(_swizzle_size) << 32u) |
                             _swizzle_code,
                         _self->hash()});
}

MemberExpr::MemberExpr(const Type *type,
                       const Expression *self,
                       uint member_index) noexcept
    : Expression{Tag::MEMBER, type}, _self{self},
      _swizzle_size{0u}, _swizzle_code{member_index} {}

MemberExpr::MemberExpr(const Type *type,
                       const Expression *self,
                       uint swizzle_size,
                       uint swizzle_code) noexcept
    : Expression{Tag::MEMBER, type}, _self{self},
      _swizzle_size{swizzle_size}, _swizzle_code{swizzle_code} {
    LUISA_ASSERT(_swizzle_size != 0u && _swizzle_size <= 4u,
                 "Swizzle size must be in [1, 4]");
}

uint MemberExpr::swizzle_size() const noexcept {
    LUISA_ASSERT(_swizzle_size != 0u && _swizzle_size <= 4u,
                 "Invalid swizzle size {}.", _swizzle_size);
    return _swizzle_size;
}

uint MemberExpr::swizzle_code() const noexcept {
    LUISA_ASSERT(is_swizzle(), "MemberExpr is not swizzled.");
    return _swizzle_code;
}

uint MemberExpr::swizzle_index(uint index) const noexcept {
    if (auto s = swizzle_size(); index >= s) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid swizzle index {} (count = {}).",
            index, s);
    }
    return (_swizzle_code >> (index * 4u)) & 0x0fu;
}

uint MemberExpr::member_index() const noexcept {
    LUISA_ASSERT(!is_swizzle(), "MemberExpr is not a member");
    return _swizzle_code;
}

uint64_t CastExpr::_compute_hash() const noexcept {
    return hash_combine({static_cast<uint64_t>(_op),
                         _source->hash()});
}

uint64_t LiteralExpr::_compute_hash() const noexcept {
    return luisa::visit([](auto &&v) noexcept { return hash_value(v); }, _value);
}

uint64_t ConstantExpr::_compute_hash() const noexcept {
    return hash_value(_data);
}

uint64_t TypeIDExpr::_compute_hash() const noexcept {
    return _data_type->hash();
}

uint64_t StringIDExpr::_compute_hash() const noexcept {
    return hash_value(_data);
}

void ExprVisitor::visit(const FuncRefExpr *) {
    LUISA_ERROR_WITH_LOCATION("Func ref op is not supported on this backend.");
}

void ExprVisitor::visit(const CpuCustomOpExpr *) {
    LUISA_ERROR_WITH_LOCATION("CPU custom op is not supported on this backend.");
}

void ExprVisitor::visit(const GpuCustomOpExpr *) {
    LUISA_ERROR_WITH_LOCATION("GPU custom op is not supported on this backend.");
}

uint64_t FuncRefExpr::_compute_hash() const noexcept {
    return _func->hash();
}
}// namespace luisa::compute
