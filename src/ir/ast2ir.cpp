//
// Created by Mike Smith on 2022/10/17.
//

#include <ir/ast2ir.h>
#include <ast/function_builder.h>

namespace luisa::compute {

ir::Module AST2IR::_convert_body() noexcept {
    for (auto v : _function.local_variables()) {
        static_cast<void>(_convert_local_variable(v));
    }
    // process body scope
    static_cast<void>(_convert(_function.body()));
    // finalize
    auto bb = ir::luisa_compute_ir_build_finish(*_current_builder());
    bb.set_root(true);
    return ir::Module{.kind = _function.tag() == Function::Tag::KERNEL ?
                                  ir::ModuleKind::Kernel :
                                  ir::ModuleKind::Function,
                      .entry = bb};
}

ir::Gc<ir::KernelModule> AST2IR::convert_kernel(Function function) noexcept {
    LUISA_ASSERT(function.tag() == Function::Tag::KERNEL,
                 "Invalid function tag.");
    LUISA_ASSERT(_struct_types.empty() && _constants.empty() &&
                     _variables.empty() && _builder_stack.empty() &&
                     !_function,
                 "Invalid state.");
    _function = function;
    return _with_builder([this](auto builder) noexcept {
        auto bindings = _function.builder()->argument_bindings();
        auto capture_count = std::count_if(
            bindings.cbegin(), bindings.cend(), [](auto &&b) {
                return luisa::holds_alternative<luisa::monostate>(b);
            });
        auto non_capture_count = bindings.size() - capture_count;
        auto captures = _boxed_slice<ir::Capture>(capture_count);
        auto non_captures = _boxed_slice<ir::NodeRef>(non_capture_count);
        auto capture_index = 0u;
        auto non_capture_index = 0u;
        // process arguments
        for (auto i = 0u; i < bindings.size(); i++) {
            using FB = detail::FunctionBuilder;
            auto binding = bindings[i];
            auto node = _convert_argument(_function.arguments()[i]);
            luisa::visit(
                luisa::overloaded{
                    [&](luisa::monostate) noexcept {
                        non_captures.ptr[non_capture_index++] = node;
                    },
                    [&](FB::BufferBinding b) noexcept {
                        captures.ptr[capture_index++] = {
                            .node = node,
                            .binding = {.tag = ir::Binding::Tag::Buffer,
                                        .buffer = {{.handle = b.handle,
                                                    .offset = b.offset_bytes,
                                                    .size = b.size_bytes}}}};
                    },
                    [&](FB::TextureBinding b) noexcept {
                        captures.ptr[capture_index++] = {
                            .node = node,
                            .binding = {.tag = ir::Binding::Tag::Texture,
                                        .texture = {{.handle = b.handle,
                                                     .level = b.level}}}};
                    },
                    [&](FB::BindlessArrayBinding b) noexcept {
                        captures.ptr[capture_index++] = {
                            .node = node,
                            .binding = {.tag = ir::Binding::Tag::BindlessArray,
                                        .bindless_array = {b.handle}}};
                    },
                    [&](FB::AccelBinding b) noexcept {
                        captures.ptr[capture_index++] = {
                            .node = node,
                            .binding = {.tag = ir::Binding::Tag::Accel,
                                        .bindless_array = {b.handle}}};
                    }},
                binding);
        }
        // process built-in variables
        for (auto v : _function.builtin_variables()) {
            static_cast<void>(_convert_builtin_variable(v));
        }
        // process shared memory
        auto shared = _boxed_slice<ir::NodeRef>(_function.shared_variables().size());
        for (auto i = 0u; i < _function.shared_variables().size(); i++) {
            shared.ptr[i] = _convert_shared_variable(_function.shared_variables()[i]);
        }
        auto module = _convert_body();
        return ir::Gc<ir::KernelModule>::from_raw(luisa_compute_ir_new_kernel_module(ir::KernelModule{.module = module,
                                                                                                      .captures = captures,
                                                                                                      .args = non_captures,
                                                                                                      .shared = shared}));
    });
}

ir::Gc<ir::CallableModule> AST2IR::convert_callable(Function function) noexcept {
    LUISA_ASSERT(function.tag() == Function::Tag::CALLABLE,
                 "Invalid function tag.");
    if (auto m = ir::luisa_compute_ir_get_symbol(function.hash())) { return ir::Gc<ir::CallableModule>::from_raw(m); }
    LUISA_ASSERT(_struct_types.empty() && _constants.empty() &&
                     _variables.empty() && _builder_stack.empty() &&
                     !_function,
                 "Invalid state.");
    _function = function;
    auto m = _with_builder([this](auto builder) noexcept {
        auto arg_count = _function.arguments().size();
        auto args = _boxed_slice<ir::NodeRef>(arg_count);
        for (auto i = 0u; i < arg_count; i++) {
            args.ptr[i] = _convert_argument(_function.arguments()[i]);
        }
        return ir::Gc<ir::CallableModule>::from_raw(
            ir::luisa_compute_ir_new_callable_module(ir::CallableModule{.module = _convert_body(), .args = args}));
    });
    ir::luisa_compute_ir_add_symbol(function.hash(), m);
    return m;
}

ir::NodeRef AST2IR::_convert_expr(const Expression *expr) noexcept {
    switch (expr->tag()) {
        case Expression::Tag::UNARY: return _convert(static_cast<const UnaryExpr *>(expr));
        case Expression::Tag::BINARY: return _convert(static_cast<const BinaryExpr *>(expr));
        case Expression::Tag::MEMBER: return _convert(static_cast<const MemberExpr *>(expr));
        case Expression::Tag::ACCESS: return _convert(static_cast<const AccessExpr *>(expr));
        case Expression::Tag::LITERAL: return _convert(static_cast<const LiteralExpr *>(expr));
        case Expression::Tag::REF: return _convert(static_cast<const RefExpr *>(expr));
        case Expression::Tag::CONSTANT: return _convert(static_cast<const ConstantExpr *>(expr));
        case Expression::Tag::CALL: return _convert(static_cast<const CallExpr *>(expr));
        case Expression::Tag::CAST: return _convert(static_cast<const CastExpr *>(expr));
        case Expression::Tag::CPUCUSTOM: return _convert(static_cast<const CpuCustomOpExpr *>(expr));
        case Expression::Tag::GPUCUSTOM: return _convert(static_cast<const GpuCustomOpExpr *>(expr));
    }
    LUISA_ERROR_WITH_LOCATION("Invalid expression tag.");
}

ir::NodeRef AST2IR::_convert_stmt(const Statement *stmt) noexcept {
    switch (stmt->tag()) {
        case Statement::Tag::BREAK: return _convert(static_cast<const BreakStmt *>(stmt)); break;
        case Statement::Tag::CONTINUE: return _convert(static_cast<const ContinueStmt *>(stmt)); break;
        case Statement::Tag::RETURN: return _convert(static_cast<const ReturnStmt *>(stmt)); break;
        case Statement::Tag::SCOPE: return _convert(static_cast<const ScopeStmt *>(stmt)); break;
        case Statement::Tag::IF: return _convert(static_cast<const IfStmt *>(stmt)); break;
        case Statement::Tag::LOOP: return _convert(static_cast<const LoopStmt *>(stmt)); break;
        case Statement::Tag::EXPR: return _convert(static_cast<const ExprStmt *>(stmt)); break;
        case Statement::Tag::SWITCH: return _convert(static_cast<const SwitchStmt *>(stmt)); break;
        case Statement::Tag::SWITCH_CASE: return _convert(static_cast<const SwitchCaseStmt *>(stmt)); break;
        case Statement::Tag::SWITCH_DEFAULT: return _convert(static_cast<const SwitchDefaultStmt *>(stmt)); break;
        case Statement::Tag::ASSIGN: return _convert(static_cast<const AssignStmt *>(stmt)); break;
        case Statement::Tag::FOR: return _convert(static_cast<const ForStmt *>(stmt)); break;
        case Statement::Tag::COMMENT: return _convert(static_cast<const CommentStmt *>(stmt)); break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid statement tag.");
}

ir::IrBuilder *AST2IR::_current_builder() noexcept {
    LUISA_ASSERT(!_builder_stack.empty(), "Builder stack is empty.");
    return _builder_stack.back();
}

ir::Gc<ir::Type> AST2IR::_convert_type(const Type *type) noexcept {
    auto register_type = [](ir::Type t) noexcept {
        return ir::Gc<ir::Type>::from_raw(ir::luisa_compute_ir_register_type(t));
    };
    // special handling for void
    if (type == nullptr) { return register_type({.tag = ir::Type::Tag::Void}); }
    // basic types
    switch (type->tag()) {
        case Type::Tag::BOOL: return register_type(
            {.tag = ir::Type::Tag::Primitive,
             .primitive = {ir::Primitive::Bool}});
        case Type::Tag::FLOAT: return register_type(
            {.tag = ir::Type::Tag::Primitive,
             .primitive = {ir::Primitive::Float32}});
        case Type::Tag::INT: return register_type(
            {.tag = ir::Type::Tag::Primitive,
             .primitive = {ir::Primitive::Int32}});
        case Type::Tag::UINT: return register_type(
            {.tag = ir::Type::Tag::Primitive,
             .primitive = {ir::Primitive::Uint32}});
        case Type::Tag::VECTOR: {
            auto dim = static_cast<uint>(type->dimension());
            switch (auto elem = type->element(); elem->tag()) {
                case Type::Tag::BOOL:
                    return register_type(
                        {.tag = ir::Type::Tag::Vector,
                         .vector = {{.element = {.tag = ir::VectorElementType::Tag::Scalar,
                                                 .scalar = {ir::Primitive::Bool}},
                                     .length = dim}}});
                case Type::Tag::FLOAT: return register_type(
                    {.tag = ir::Type::Tag::Vector,
                     .vector = {{.element = {.tag = ir::VectorElementType::Tag::Scalar,
                                             .scalar = {ir::Primitive::Float32}},
                                 .length = dim}}});
                case Type::Tag::INT: return register_type(
                    {.tag = ir::Type::Tag::Vector,
                     .vector = {{.element = {.tag = ir::VectorElementType::Tag::Scalar,
                                             .scalar = {ir::Primitive::Int32}},
                                 .length = dim}}});
                case Type::Tag::UINT: return register_type(
                    {.tag = ir::Type::Tag::Vector,
                     .vector = {{.element = {.tag = ir::VectorElementType::Tag::Scalar,
                                             .scalar = {ir::Primitive::Uint32}},
                                 .length = dim}}});
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Invalid vector type: {}.", type->description());
        }
        case Type::Tag::MATRIX: return register_type(
            {.tag = ir::Type::Tag::Matrix,
             .matrix = {{.element = {.tag = ir::VectorElementType::Tag::Scalar,
                                     .scalar = {ir::Primitive::Float32}},
                         .dimension = static_cast<uint>(type->dimension())}}});
        case Type::Tag::ARRAY: return register_type(
            {.tag = ir::Type::Tag::Array,
             .array = {{.element = _convert_type(type->element()),
                        .length = type->dimension()}}});
        case Type::Tag::STRUCTURE: {
            if (auto iter = _struct_types.find(type->hash());
                iter != _struct_types.end()) { return iter->second; }
            luisa::vector<ir::Gc<ir::Type>> members;
            members.reserve(type->members().size());
            for (auto member : type->members()) {
                members.emplace_back(_convert_type(member));
            }
            auto t = register_type(
                {.tag = ir::Type::Tag::Struct,
                 .struct_ = {{.fields = {members.data(), members.size()},
                              .alignment = type->alignment(),
                              .size = type->size()}}});
            _struct_types.emplace(type->hash(), t);
            return t;
        }
        case Type::Tag::BUFFER:
        case Type::Tag::TEXTURE:
        case Type::Tag::BINDLESS_ARRAY:
        case Type::Tag::ACCEL:
            LUISA_ERROR_WITH_LOCATION("AST2IR::_convert_type() should not "
                                      "be called for resource arguments.");
    }
    LUISA_ERROR_WITH_LOCATION("Invalid type: {}.", type->description());
}

ir::NodeRef AST2IR::_convert_constant(const ConstantData &data) noexcept {
    if (auto iter = _constants.find(data.hash()); iter != _constants.end()) {
        return iter->second;
    }
    auto b = _current_builder();
    auto c = ir::Const{
        .tag = ir::Const::Tag::Generic,
        .generic = luisa::visit(
            [this](auto view) noexcept {
                using T = typename decltype(view)::value_type;
                auto type = _convert_type(Type::from(luisa::format(
                    "array<{},{}>", Type::of<T>()->description(), view.size())));
                auto slice = _boxed_slice<uint8_t>(view.size_bytes());
                std::memcpy(slice.ptr, view.data(), view.size_bytes());
                return ir::Const::Generic_Body{slice, type};
            },
            data.view())};
    auto node = ir::luisa_compute_ir_build_const(b, c);
    _constants.emplace(data.hash(), node);
    return node;
}

ir::NodeRef AST2IR::_convert(const LiteralExpr *expr) noexcept {
    return luisa::visit(
        [&](auto x) noexcept {
            using T = std::decay_t<decltype(x)>;
            LUISA_ASSERT(*Type::of<T>() == *expr->type(),
                         "Type mismatch: '{}' vs. '{}'.",
                         Type::of<T>()->description(),
                         expr->type()->description());
            if constexpr (is_scalar_v<T>) {
                auto c = [&]() noexcept -> ir::Const {
                    if (x == static_cast<T>(0)) {
                        ir::Const cc{.tag = ir::Const::Tag::Zero};
                        cc.zero = {_convert_type(expr->type())};
                        return cc;
                    }
                    if constexpr (std::is_same_v<T, bool>) {
                        return {.tag = ir::Const::Tag::Bool, .bool_ = {x}};
                    } else if constexpr (std::is_same_v<T, float>) {
                        return {.tag = ir::Const::Tag::Float32, .float32 = {x}};
                    } else if constexpr (std::is_same_v<T, int>) {
                        return {.tag = ir::Const::Tag::Int32, .int32 = {x}};
                    } else if constexpr (std::is_same_v<T, uint>) {
                        return {.tag = ir::Const::Tag::Uint32, .uint32 = {x}};
                    } else {
                        static_assert(always_false_v<T>, "Unsupported scalar type.");
                    }
                }();
                auto b = _current_builder();
                return ir::luisa_compute_ir_build_const(b, c);
            } else {
                auto salt = luisa::hash64("__ast2ir_literal");
                auto hash = luisa::hash64(x, luisa::hash64(expr->type()->hash(), salt));
                if (auto iter = _constants.find(hash); iter != _constants.end()) { return iter->second; }
                auto slice = _boxed_slice<uint8_t>(sizeof(T));
                std::memcpy(slice.ptr, &x, sizeof(T));
                auto c = ir::Const{.tag = ir::Const::Tag::Generic};
                c.generic = {slice, _convert_type(expr->type())};
                auto b = _current_builder();
                auto node = ir::luisa_compute_ir_build_const(b, c);
                _constants.emplace(hash, node);
                return node;
            }
        },
        expr->value());
}

ir::NodeRef AST2IR::_convert(const UnaryExpr *expr) noexcept {
    auto x = _convert_expr(expr->operand());
    if (expr->op() == UnaryOp::PLUS) { return x; }
    auto tag = [expr] {
        switch (expr->op()) {
            case UnaryOp::MINUS: return ir::Func::Tag::Neg;
            case UnaryOp::NOT: return ir::Func::Tag::Not;
            case UnaryOp::BIT_NOT: return ir::Func::Tag::BitNot;
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION(
            "Unsupported unary operator: 0x{:02x}.",
            luisa::to_underlying(expr->op()));
    }();
    return ir::luisa_compute_ir_build_call(
        _current_builder(), {.tag = tag},
        {.ptr = &x, .len = 1u},
        _convert_type(expr->type()));
}

ir::NodeRef AST2IR::_convert(const BinaryExpr *expr) noexcept {
    auto tag = [expr] {
        switch (expr->op()) {
            case BinaryOp::ADD: return ir::Func::Tag::Add;
            case BinaryOp::SUB: return ir::Func::Tag::Sub;
            case BinaryOp::MUL: return ir::Func::Tag::Mul;
            case BinaryOp::DIV: return ir::Func::Tag::Div;
            case BinaryOp::MOD: return ir::Func::Tag::Rem;
            case BinaryOp::BIT_AND: return ir::Func::Tag::BitAnd;
            case BinaryOp::BIT_OR: return ir::Func::Tag::BitOr;
            case BinaryOp::BIT_XOR: return ir::Func::Tag::BitXor;
            case BinaryOp::SHL: return ir::Func::Tag::Shl;
            case BinaryOp::SHR: return ir::Func::Tag::Shr;
            case BinaryOp::LESS: return ir::Func::Tag::Lt;
            case BinaryOp::GREATER: return ir::Func::Tag::Gt;
            case BinaryOp::LESS_EQUAL: return ir::Func::Tag::Le;
            case BinaryOp::GREATER_EQUAL: return ir::Func::Tag::Ge;
            case BinaryOp::EQUAL: return ir::Func::Tag::Eq;
            case BinaryOp::NOT_EQUAL: return ir::Func::Tag::Ne;
            case BinaryOp::AND: return ir::Func::Tag::BitAnd;
            case BinaryOp::OR: return ir::Func::Tag::BitOr;
        }
        LUISA_ERROR_WITH_LOCATION(
            "Unsupported binary operator: 0x{:02x}.",
            luisa::to_underlying(expr->op()));
    }();
    auto lhs_type = expr->lhs()->type();
    auto rhs_type = expr->rhs()->type();
    auto lhs = _convert_expr(expr->lhs());
    auto rhs = _convert_expr(expr->rhs());
    if (is_relational(expr->op())) {

    } else if (is_logical(expr->op())) {

    } else {
        if (lhs_type->is_scalar() && rhs_type->is_scalar()) {
            // TODO: conversion
            lhs = _cast(expr->type(), expr->lhs()->type(), lhs);
            rhs = _cast(expr->type(), expr->rhs()->type(), rhs);
        }
    }
    std::array args{lhs, rhs};
    return ir::luisa_compute_ir_build_call(
        _current_builder(), {.tag = tag},
        {.ptr = args.data(), .len = args.size()},
        _convert_type(expr->type()));
}

ir::NodeRef AST2IR::_convert(const MemberExpr *expr) noexcept {
    auto self = _convert_expr(expr->self());
    auto b = _current_builder();
    if (expr->is_swizzle() && expr->swizzle_size() > 1u) {
        std::array<ir::NodeRef, 5u> args{self};
        for (auto i = 0u; i < expr->swizzle_size(); i++) {
            args[i + 1u] = _literal(Type::of<uint>(), expr->swizzle_index(i));
        }
        return ir::luisa_compute_ir_build_call(
            b, {.tag = ir::Func::Tag::Permute},
            {.ptr = args.data(), .len = expr->swizzle_size() + 1u},
            _convert_type(expr->type()));
    }
    auto index = expr->is_swizzle() ?
                     _literal(Type::of<uint>(), expr->swizzle_index(0u)) :
                     _literal(Type::of<uint>(), expr->member_index());
    std::array args{self, index};
    return ir::luisa_compute_ir_build_call(
        b, {.tag = ir::Func::Tag::GetElementPtr},
        {.ptr = args.data(), .len = args.size()},
        _convert_type(expr->type()));
}

ir::NodeRef AST2IR::_convert(const AccessExpr *expr) noexcept {
    auto self = _convert_expr(expr->range());
    auto index = _convert_expr(expr->index());
    auto b = _current_builder();
    std::array args{self, index};
    return ir::luisa_compute_ir_build_call(
        b, {.tag = ir::Func::Tag::GetElementPtr},
        {.ptr = args.data(), .len = args.size()},
        _convert_type(expr->type()));
}

ir::NodeRef AST2IR::_convert(const RefExpr *expr) noexcept {
    auto iter = _variables.find(expr->variable().uid());
    LUISA_ASSERT(iter != _variables.end(),
                 "Variable #{} not found.",
                 expr->variable().uid());
    return iter->second;
}

ir::NodeRef AST2IR::_convert(const ConstantExpr *expr) noexcept {
    return _convert_constant(expr->data());
}

ir::NodeRef AST2IR::_convert(const CallExpr *expr) noexcept {
    // custom callable
    if (!expr->is_builtin()) {
        AST2IR cvt;
        auto callable = expr->custom();
        static_cast<void>(cvt.convert_callable(callable));
        luisa::vector<ir::NodeRef> args;
        args.reserve(expr->arguments().size());
        for (auto i = 0u; i < expr->arguments().size(); i++) {
            auto t = callable.arguments()[i].type();
            auto arg = expr->arguments()[i];
            args.emplace_back(_cast(t, arg->type(), _convert_expr(arg)));
        }
        auto call = ir::luisa_compute_ir_build_call(
            _current_builder(),
            {.tag = ir::Func::Tag::Callable, .callable = {callable.hash()}},
            {.ptr = args.data(), .len = args.size()},
            _convert_type(callable.return_type()));
        return _cast(expr->type(), callable.return_type(), call);
    }
    // built-in
    auto tag = [expr] {
        switch (expr->op()) {
            case CallOp::ALL: return ir::Func::Tag::All;
            case CallOp::ANY: return ir::Func::Tag::Any;
            case CallOp::SELECT: return ir::Func::Tag::Select;
            case CallOp::CLAMP: return ir::Func::Tag::Clamp;
            case CallOp::LERP: return ir::Func::Tag::Lerp;
            case CallOp::STEP: return ir::Func::Tag::Step;
            case CallOp::ABS: return ir::Func::Tag::Abs;
            case CallOp::MIN: return ir::Func::Tag::Min;
            case CallOp::MAX: return ir::Func::Tag::Max;
            case CallOp::CLZ: return ir::Func::Tag::Clz;
            case CallOp::CTZ: return ir::Func::Tag::Ctz;
            case CallOp::POPCOUNT: return ir::Func::Tag::PopCount;
            case CallOp::REVERSE: return ir::Func::Tag::Reverse;
            case CallOp::ISINF: return ir::Func::Tag::IsInf;
            case CallOp::ISNAN: return ir::Func::Tag::IsNan;
            case CallOp::ACOS: return ir::Func::Tag::Acos;
            case CallOp::ACOSH: return ir::Func::Tag::Acosh;
            case CallOp::ASIN: return ir::Func::Tag::Asin;
            case CallOp::ASINH: return ir::Func::Tag::Asinh;
            case CallOp::ATAN: return ir::Func::Tag::Atan;
            case CallOp::ATAN2: return ir::Func::Tag::Atan2;
            case CallOp::ATANH: return ir::Func::Tag::Atanh;
            case CallOp::COS: return ir::Func::Tag::Cos;
            case CallOp::COSH: return ir::Func::Tag::Cosh;
            case CallOp::SIN: return ir::Func::Tag::Sin;
            case CallOp::SINH: return ir::Func::Tag::Sinh;
            case CallOp::TAN: return ir::Func::Tag::Tan;
            case CallOp::TANH: return ir::Func::Tag::Tanh;
            case CallOp::EXP: return ir::Func::Tag::Exp;
            case CallOp::EXP2: return ir::Func::Tag::Exp2;
            case CallOp::EXP10: return ir::Func::Tag::Exp10;
            case CallOp::LOG: return ir::Func::Tag::Log;
            case CallOp::LOG2: return ir::Func::Tag::Log2;
            case CallOp::LOG10: return ir::Func::Tag::Log10;
            case CallOp::POW: return ir::Func::Tag::Powf;
            case CallOp::SQRT: return ir::Func::Tag::Sqrt;
            case CallOp::RSQRT: return ir::Func::Tag::Rsqrt;
            case CallOp::CEIL: return ir::Func::Tag::Ceil;
            case CallOp::FLOOR: return ir::Func::Tag::Floor;
            case CallOp::FRACT: return ir::Func::Tag::Fract;
            case CallOp::TRUNC: return ir::Func::Tag::Trunc;
            case CallOp::ROUND: return ir::Func::Tag::Round;
            case CallOp::FMA: return ir::Func::Tag::Fma;
            case CallOp::COPYSIGN: return ir::Func::Tag::Copysign;
            case CallOp::CROSS: return ir::Func::Tag::Cross;
            case CallOp::DOT: return ir::Func::Tag::Dot;
            case CallOp::LENGTH: return ir::Func::Tag::Length;
            case CallOp::LENGTH_SQUARED: return ir::Func::Tag::LengthSquared;
            case CallOp::NORMALIZE: return ir::Func::Tag::Normalize;
            case CallOp::FACEFORWARD: return ir::Func::Tag::Faceforward;
            case CallOp::DETERMINANT: return ir::Func::Tag::Determinant;
            case CallOp::TRANSPOSE: return ir::Func::Tag::Transpose;
            case CallOp::INVERSE: return ir::Func::Tag::Inverse;
            case CallOp::SYNCHRONIZE_BLOCK: return ir::Func::Tag::SynchronizeBlock;
            case CallOp::ATOMIC_EXCHANGE: return ir::Func::Tag::AtomicExchange;
            case CallOp::ATOMIC_COMPARE_EXCHANGE: return ir::Func::Tag::AtomicCompareExchange;
            case CallOp::ATOMIC_FETCH_ADD: return ir::Func::Tag::AtomicFetchAdd;
            case CallOp::ATOMIC_FETCH_SUB: return ir::Func::Tag::AtomicFetchSub;
            case CallOp::ATOMIC_FETCH_AND: return ir::Func::Tag::AtomicFetchAnd;
            case CallOp::ATOMIC_FETCH_OR: return ir::Func::Tag::AtomicFetchOr;
            case CallOp::ATOMIC_FETCH_XOR: return ir::Func::Tag::AtomicFetchXor;
            case CallOp::ATOMIC_FETCH_MIN: return ir::Func::Tag::AtomicFetchMin;
            case CallOp::ATOMIC_FETCH_MAX: return ir::Func::Tag::AtomicFetchMax;
            case CallOp::BUFFER_READ: return ir::Func::Tag::BufferRead;
            case CallOp::BUFFER_WRITE: return ir::Func::Tag::BufferWrite;
            case CallOp::TEXTURE_READ: return ir::Func::Tag::TextureRead;
            case CallOp::TEXTURE_WRITE: return ir::Func::Tag::TextureWrite;
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE: return ir::Func::Tag::BindlessTexture2dSample;
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: return ir::Func::Tag::BindlessTexture2dSampleLevel;
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: return ir::Func::Tag::BindlessTexture2dSampleGrad;
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE: return ir::Func::Tag::BindlessTexture3dSample;
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: return ir::Func::Tag::BindlessTexture3dSampleLevel;
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: return ir::Func::Tag::BindlessTexture3dSampleGrad;
            case CallOp::BINDLESS_TEXTURE2D_READ: return ir::Func::Tag::BindlessTexture2dRead;
            case CallOp::BINDLESS_TEXTURE3D_READ: return ir::Func::Tag::BindlessTexture3dRead;
            case CallOp::BINDLESS_TEXTURE2D_READ_LEVEL: return ir::Func::Tag::BindlessTexture2dReadLevel;
            case CallOp::BINDLESS_TEXTURE3D_READ_LEVEL: return ir::Func::Tag::BindlessTexture3dReadLevel;
            case CallOp::BINDLESS_TEXTURE2D_SIZE: return ir::Func::Tag::BindlessTexture2dSize;
            case CallOp::BINDLESS_TEXTURE3D_SIZE: return ir::Func::Tag::BindlessTexture3dSize;
            case CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: return ir::Func::Tag::BindlessTexture2dSizeLevel;
            case CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: return ir::Func::Tag::BindlessTexture3dSizeLevel;
            case CallOp::BINDLESS_BUFFER_READ: return ir::Func::Tag::BindlessBufferRead;
            case CallOp::MAKE_BOOL2: return ir::Func::Tag::Vec2;
            case CallOp::MAKE_BOOL3: return ir::Func::Tag::Vec3;
            case CallOp::MAKE_BOOL4: return ir::Func::Tag::Vec4;
            case CallOp::MAKE_INT2: return ir::Func::Tag::Vec2;
            case CallOp::MAKE_INT3: return ir::Func::Tag::Vec3;
            case CallOp::MAKE_INT4: return ir::Func::Tag::Vec4;
            case CallOp::MAKE_UINT2: return ir::Func::Tag::Vec2;
            case CallOp::MAKE_UINT3: return ir::Func::Tag::Vec3;
            case CallOp::MAKE_UINT4: return ir::Func::Tag::Vec4;
            case CallOp::MAKE_FLOAT2: return ir::Func::Tag::Vec2;
            case CallOp::MAKE_FLOAT3: return ir::Func::Tag::Vec3;
            case CallOp::MAKE_FLOAT4: return ir::Func::Tag::Vec4;
            case CallOp::MAKE_FLOAT2X2: return ir::Func::Tag::Matrix2;
            case CallOp::MAKE_FLOAT3X3: return ir::Func::Tag::Matrix3;
            case CallOp::MAKE_FLOAT4X4: return ir::Func::Tag::Matrix4;
            case CallOp::ASSUME: return ir::Func::Tag::Assume;
            case CallOp::UNREACHABLE: return ir::Func::Tag::Unreachable;
            case CallOp::INSTANCE_TO_WORLD_MATRIX: return ir::Func::Tag::InstanceToWorldMatrix;
            case CallOp::SET_INSTANCE_TRANSFORM: return ir::Func::Tag::SetInstanceTransform;
            case CallOp::SET_INSTANCE_VISIBILITY: return ir::Func::Tag::SetInstanceVisibility;
            case CallOp::TRACE_CLOSEST: return ir::Func::Tag::TraceClosest;
            case CallOp::TRACE_ANY: return ir::Func::Tag::TraceAny;
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION(
            "Invalid CallOp: 0x{:02x}.",
            luisa::to_underlying(expr->op()));
    }();
    luisa::vector<ir::NodeRef> args;
    if (is_atomic_operation(expr->op())) {
        args.reserve(expr->arguments().size() + 1u);
        LUISA_ASSERT(expr->arguments().front()->tag() == Expression::Tag::ACCESS,
                     "First argument of atomic operation must be an AccessExpr.");
        auto access = static_cast<const AccessExpr *>(expr->arguments().front());
        args.emplace_back(_convert_expr(access->range()));
        args.emplace_back(_convert_expr(access->index()));
        for (auto arg : luisa::span{expr->arguments()}.subspan(1u)) {
            args.emplace_back(_convert_expr(arg));
        }
    } else if (is_vector_maker(expr->op())) {
        // resolve overloaded vector maker
        args.reserve(expr->type()->dimension());
        auto a = expr->arguments();
        if (a.size() == 1u) {
            if (auto t = a.front()->type(); t->is_scalar()) {
                // vector from a single scalar
                auto elem = _convert_expr(a.front());
                for (uint32_t i = 0u; i < expr->type()->dimension(); i++) {
                    args.emplace_back(elem);
                }
            } else {
                // vector from a vector
                LUISA_ASSERT(t->is_vector() && t->dimension() >= expr->type()->dimension(),
                             "Invalid {} vector maker from {}.",
                             expr->type()->description(),
                             a.front()->type()->description());
                auto v = _convert_expr(a.front());
                auto b = _current_builder();
                for (auto i = 0u; i < expr->type()->dimension(); i++) {
                    std::array extract_args{v, _literal(Type::of<uint>(), i)};
                    auto elem = ir::luisa_compute_ir_build_call(
                        b, {.tag = ir::Func::Tag::ExtractElement},
                        {.ptr = extract_args.data(), .len = extract_args.size()},
                        _convert_type(t->element()));
                    args.emplace_back(_cast(expr->type()->element(), t->element(), elem));
                }
            }
        } else {
            // vector from multiple scalars or vectors
            for (auto v : a) {
                if (v->type()->is_scalar()) {
                    args.emplace_back(_cast(expr->type()->element(), v->type(), _convert_expr(v)));
                } else {
                    auto vv = _convert_expr(v);
                    auto b = _current_builder();
                    for (auto i = 0u; i < v->type()->dimension(); i++) {
                        std::array extract_args{vv, _literal(Type::of<uint>(), i)};
                        auto elem = ir::luisa_compute_ir_build_call(
                            b, {.tag = ir::Func::Tag::ExtractElement},
                            {.ptr = extract_args.data(), .len = extract_args.size()},
                            _convert_type(v->type()->element()));
                        args.emplace_back(_cast(expr->type()->element(), v->type()->element(), elem));
                    }
                }
            }
            args.resize(expr->type()->dimension());
        }
    } else if (is_matrix_maker(expr->op())) {
        LUISA_ASSERT(expr->arguments().size() == expr->type()->dimension(),
                     "Invalid {} matrix maker from {} vector(s).",
                     expr->type()->description(),
                     expr->arguments().size());
        args.reserve(expr->type()->dimension() * expr->type()->dimension());
        for (auto v : expr->arguments()) {
            LUISA_ASSERT(v->type()->is_vector() &&
                             *v->type()->element() == *expr->type()->element() &&
                             v->type()->dimension() == expr->type()->dimension(),
                         "Invalid {} matrix maker from {}.",
                         expr->type()->description(),
                         v->type()->description());
            auto vv = _convert_expr(v);
            auto b = _current_builder();
            for (auto i = 0u; i < v->type()->dimension(); i++) {
                std::array extract_args{vv, _literal(Type::of<uint>(), i)};
                auto elem = ir::luisa_compute_ir_build_call(
                    b, {.tag = ir::Func::Tag::ExtractElement},
                    {.ptr = extract_args.data(), .len = extract_args.size()},
                    _convert_type(v->type()->element()));
                args.emplace_back(elem);
            }
        }
    } else {
        args.reserve(expr->arguments().size());
        for (auto arg : expr->arguments()) {
            args.emplace_back(_convert_expr(arg));
        }
    }
    return ir::luisa_compute_ir_build_call(
        _current_builder(), {.tag = tag},
        {.ptr = args.data(), .len = args.size()},
        _convert_type(expr->type()));
}

ir::NodeRef AST2IR::_convert(const CastExpr *expr) noexcept {
    auto src = _convert_expr(expr->expression());
    if (expr->op() == CastOp::STATIC) {
        return _cast(expr->type(), expr->expression()->type(), src, true);
    }
    return ir::luisa_compute_ir_build_call(
        _current_builder(), {.tag = ir::Func::Tag::Bitcast},
        {.ptr = &src, .len = 1u}, _convert_type(expr->type()));
}

ir::NodeRef AST2IR::_convert(const CpuCustomOpExpr *expr) noexcept {
    // TODO
    LUISA_ERROR_WITH_LOCATION("TODO: AST2IR::_convert(const CpuCustomOpExpr *expr)");
}

ir::NodeRef AST2IR::_convert(const GpuCustomOpExpr *expr) noexcept {
    // TODO
    LUISA_ERROR_WITH_LOCATION("TODO: AST2IR::_convert(const GpuCustomOpExpr *expr)");
}

ir::NodeRef AST2IR::_convert(const BreakStmt *stmt) noexcept {
    auto instr = ir::luisa_compute_ir_new_instruction(
        {.tag = ir::Instruction::Tag::Break});
    auto node = ir::luisa_compute_ir_new_node(
        {.type_ = _convert_type(nullptr), .instruction = instr});
    ir::luisa_compute_ir_append_node(_current_builder(), node);
    return node;
}

ir::NodeRef AST2IR::_convert(const ContinueStmt *stmt) noexcept {
    auto instr = ir::luisa_compute_ir_new_instruction(
        {.tag = ir::Instruction::Tag::Continue});
    auto void_type = _convert_type(nullptr);
    auto node = ir::luisa_compute_ir_new_node(
        {.type_ = void_type, .instruction = instr});
    ir::luisa_compute_ir_append_node(_current_builder(), node);
    return node;
}

ir::NodeRef AST2IR::_convert(const ReturnStmt *stmt) noexcept {
    // TODO: handle reference
    auto ret_type = _function.return_type();
    auto ret = ret_type ?
                   _cast(ret_type, stmt->expression()->type(),
                         _convert_expr(stmt->expression())) :
                   ir::INVALID_REF;
    auto instr = ir::luisa_compute_ir_new_instruction(
        {.tag = ir::Instruction::Tag::Return, .return_ = {ret}});
    auto node = ir::luisa_compute_ir_new_node(
        {.type_ = _convert_type(ret_type), .instruction = instr});
    ir::luisa_compute_ir_append_node(_current_builder(), node);
    return node;
}

ir::NodeRef AST2IR::_convert(const ScopeStmt *stmt) noexcept {
    for (auto s : stmt->statements()) {
        static_cast<void>(_convert_stmt(s));
        if (auto tag = s->tag();
            tag == Statement::Tag::RETURN ||
            tag == Statement::Tag::CONTINUE ||
            tag == Statement::Tag::BREAK) { break; }
    }
    return ir::INVALID_REF;
}

ir::NodeRef AST2IR::_convert(const IfStmt *stmt) noexcept {
    auto cond = _convert_expr(stmt->condition());
    auto true_block = _with_builder([this, stmt](auto b) noexcept {
        static_cast<void>(_convert(stmt->true_branch()));
        return ir::luisa_compute_ir_build_finish(*b);
    });
    auto false_block = _with_builder([this, stmt](auto b) noexcept {
        static_cast<void>(_convert(stmt->false_branch()));
        return ir::luisa_compute_ir_build_finish(*b);
    });
    auto instr = ir::luisa_compute_ir_new_instruction(
        {.tag = ir::Instruction::Tag::If,
         .if_ = {.cond = cond,
                 .true_branch = true_block,
                 .false_branch = false_block}});
    auto node = ir::luisa_compute_ir_new_node(
        {.type_ = _convert_type(nullptr),
         .instruction = instr});
    ir::luisa_compute_ir_append_node(_current_builder(), node);
    return node;
}

ir::NodeRef AST2IR::_convert(const LoopStmt *stmt) noexcept {
    auto cond = _literal(Type::of<bool>(), true);
    auto body = _with_builder([this, stmt](auto b) noexcept {
        static_cast<void>(_convert(stmt->body()));
        return ir::luisa_compute_ir_build_finish(*b);
    });
    auto instr = ir::luisa_compute_ir_new_instruction(
        {.tag = ir::Instruction::Tag::Loop,
         .loop = {.body = body, .cond = cond}});
    auto node = ir::luisa_compute_ir_new_node(
        {.type_ = _convert_type(nullptr),
         .instruction = instr});
    ir::luisa_compute_ir_append_node(_current_builder(), node);
    return node;
}

ir::NodeRef AST2IR::_convert(const ExprStmt *stmt) noexcept {
    return _convert_expr(stmt->expression());
}

ir::NodeRef AST2IR::_convert(const SwitchStmt *stmt) noexcept {
    LUISA_ASSERT(stmt->expression()->type()->tag() == Type::Tag::INT ||
                     stmt->expression()->type()->tag() == Type::Tag::UINT,
                 "Only integer type is supported in switch statement.");
    auto value = _convert_expr(stmt->expression());
    ir::Instruction switch_instr{.tag = ir::Instruction::Tag::Switch,
                                 .switch_ = {.value = value}};
    luisa::vector<ir::SwitchCase> case_blocks;
    luisa::optional<ir::Gc<ir::BasicBlock>> default_block;
    case_blocks.reserve(stmt->body()->statements().size());
    for (auto s : stmt->body()->statements()) {
        LUISA_ASSERT(s->tag() == Statement::Tag::SWITCH_CASE ||
                         s->tag() == Statement::Tag::SWITCH_DEFAULT,
                     "Only case and default statements are "
                     "allowed in switch body.");
        if (s->tag() == Statement::Tag::SWITCH_CASE) {
            auto case_stmt = static_cast<const SwitchCaseStmt *>(s);
            LUISA_ASSERT(case_stmt->expression()->tag() == Expression::Tag::LITERAL,
                         "Only literal expression is supported in case statement.");
            auto case_tag = _cast(stmt->expression()->type(), case_stmt->expression()->type(),
                                  _convert_expr(case_stmt->expression()));
            auto case_block = _with_builder([this, case_stmt](auto b) noexcept {
                static_cast<void>(_convert(case_stmt->body()));
                return ir::luisa_compute_ir_build_finish(*b);
            });
            case_blocks.emplace_back(ir::SwitchCase{
                .value = case_tag, .block = case_block});
        } else {
            LUISA_ASSERT(!default_block.has_value(),
                         "Only one default statement is "
                         "allowed in switch body.");
            default_block.emplace(_with_builder([this, s](auto b) noexcept {
                static_cast<void>(_convert(
                    static_cast<const SwitchDefaultStmt *>(s)));
                return ir::luisa_compute_ir_build_finish(*b);
            }));
        }
    }
    switch_instr.switch_.default_ = default_block.value_or(_with_builder([](auto b) noexcept {
        return ir::luisa_compute_ir_build_finish(*b);
    }));
    switch_instr.switch_.cases = _boxed_slice<ir::SwitchCase>(case_blocks.size());
    for (auto i = 0u; i < case_blocks.size(); i++) {
        switch_instr.switch_.cases.ptr[i] = case_blocks[i];
    }
    auto instr = ir::luisa_compute_ir_new_instruction(switch_instr);
    auto node = ir::luisa_compute_ir_new_node(
        {.type_ = _convert_type(nullptr), .instruction = instr});
    ir::luisa_compute_ir_append_node(_current_builder(), node);
    return node;
}

ir::NodeRef AST2IR::_convert(const SwitchCaseStmt *stmt) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "AST2IR::_convert(const SwitchCaseStmt *stmt) "
        "should not be called.");
}

ir::NodeRef AST2IR::_convert(const SwitchDefaultStmt *stmt) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "AST2IR::_convert(const SwitchDefaultStmt *stmt) "
        "should not be called.");
}

ir::NodeRef AST2IR::_convert(const AssignStmt *stmt) noexcept {
    auto lhs = _convert_expr(stmt->lhs());
    auto rhs = _cast(stmt->lhs()->type(), stmt->rhs()->type(),
                     _convert_expr(stmt->rhs()));
    auto instr = ir::luisa_compute_ir_new_instruction(
        {.tag = ir::Instruction::Tag::Update,
         .update = {.var = lhs, .value = rhs}});
    auto node = ir::luisa_compute_ir_new_node(
        {.type_ = _convert_type(nullptr),// TODO: check if UpdateNode returns void
         .instruction = instr});
    ir::luisa_compute_ir_append_node(_current_builder(), node);
    return node;
}

ir::NodeRef AST2IR::_convert(const ForStmt *stmt) noexcept {
    // for (; cond; var += update) { /* body */ }
    auto var = _convert_expr(stmt->variable());
    auto [cond, prepare] = _with_builder([this, stmt](auto b) noexcept {
        auto c = _cast(Type::of<bool>(), stmt->condition()->type(),
                       _convert_expr(stmt->condition()));
        auto p = ir::luisa_compute_ir_build_finish(*b);
        return std::make_pair(c, p);
    });
    auto body = _with_builder([this, stmt](auto b) noexcept {
        static_cast<void>(_convert(stmt->body()));
        return ir::luisa_compute_ir_build_finish(*b);
    });
    auto update = _with_builder([this, stmt, var](auto b) noexcept {
        // step
        auto step = _cast(stmt->variable()->type(), stmt->step()->type(),
                          _convert_expr(stmt->step()));
        // next = var + step
        std::array args{var, step};
        auto next = ir::luisa_compute_ir_build_call(
            b, {.tag = ir::Func::Tag::Add},
            {.ptr = args.data(), .len = args.size()},
            _convert_type(stmt->variable()->type()));
        // var = next
        auto instr = ir::luisa_compute_ir_new_instruction(
            {.tag = ir::Instruction::Tag::Update,
             .update = {.var = var, .value = next}});
        auto node = ir::luisa_compute_ir_new_node(
            {.type_ = _convert_type(nullptr),// TODO: check if UpdateNode returns void
             .instruction = instr});
        ir::luisa_compute_ir_append_node(b, node);
        // finish
        return ir::luisa_compute_ir_build_finish(*b);
    });
    auto instr = ir::luisa_compute_ir_new_instruction(
        {.tag = ir::Instruction::Tag::GenericLoop,
         .generic_loop = {.prepare = prepare,
                          .cond = cond,
                          .body = body,
                          .update = update}});
    auto node = ir::luisa_compute_ir_new_node(
        {.type_ = _convert_type(nullptr), .instruction = instr});
    ir::luisa_compute_ir_append_node(_current_builder(), node);
    return node;
}

ir::NodeRef AST2IR::_convert(const CommentStmt *stmt) noexcept {
    auto b = _current_builder();
    auto msg = _boxed_slice<uint8_t>(stmt->comment().size());
    std::memcpy(msg.ptr, stmt->comment().data(), stmt->comment().size());
    auto instr = ir::luisa_compute_ir_new_instruction(
        {.tag = ir::Instruction::Tag::Comment, .comment = {msg}});
    auto node = ir::luisa_compute_ir_new_node(
        {.type_ = _convert_type(nullptr), .instruction = instr});
    ir::luisa_compute_ir_append_node(b, node);
    return node;
}

ir::NodeRef AST2IR::_convert_argument(Variable v) noexcept {
    auto b = _current_builder();
    auto node = [&] {
        switch (v.tag()) {
            case Variable::Tag::REFERENCE:
                // TODO: lower reference to return value
                LUISA_ERROR_WITH_LOCATION("TODO");
            case Variable::Tag::BUFFER: {
                auto instr = ir::luisa_compute_ir_new_instruction(
                    {.tag = ir::Instruction::Tag::Buffer});
                return ir::luisa_compute_ir_new_node(
                    {.type_ = _convert_type(v.type()->element()),
                     .instruction = instr});
            }
            case Variable::Tag::TEXTURE: {
                auto instr = ir::luisa_compute_ir_new_instruction(
                    {.tag = v.type()->dimension() == 2u ?
                                ir::Instruction::Tag::Texture2D :
                                ir::Instruction::Tag::Texture3D});
                return ir::luisa_compute_ir_new_node(
                    {.type_ = _convert_type(v.type()->element()),
                     .instruction = instr});
            }
            case Variable::Tag::BINDLESS_ARRAY: {
                auto instr = ir::luisa_compute_ir_new_instruction(
                    {.tag = ir::Instruction::Tag::Bindless});
                return ir::luisa_compute_ir_new_node(
                    {.type_ = _convert_type(nullptr),
                     .instruction = instr});
            }
            case Variable::Tag::ACCEL: {
                auto instr = ir::luisa_compute_ir_new_instruction(
                    {.tag = ir::Instruction::Tag::Accel});
                return ir::luisa_compute_ir_new_node(
                    {.type_ = _convert_type(nullptr),
                     .instruction = instr});
            }
            default: {
                // TODO: callable?
                auto instr = ir::luisa_compute_ir_new_instruction(
                    {.tag = ir::Instruction::Tag::Uniform});
                return ir::luisa_compute_ir_new_node(
                    {.type_ = _convert_type(v.type()),
                     .instruction = instr});
            }
        }
        LUISA_ERROR_WITH_LOCATION("Invalid variable tag.");
    }();
    ir::luisa_compute_ir_append_node(b, node);
    _variables.emplace(v.uid(), node);
    return node;
}

ir::NodeRef AST2IR::_convert_shared_variable(Variable v) noexcept {
    auto b = _current_builder();
    auto type = _convert_type(v.type());
    auto instr = ir::luisa_compute_ir_new_instruction(
        {.tag = ir::Instruction::Tag::Shared});
    auto node = ir::luisa_compute_ir_new_node(
        {.type_ = type, .instruction = instr});
    ir::luisa_compute_ir_append_node(b, node);
    _variables.emplace(v.uid(), node);
    return node;
}

ir::NodeRef AST2IR::_convert_local_variable(Variable v) noexcept {
    auto b = _current_builder();
    auto type = _convert_type(v.type());
    auto n = ir::luisa_compute_ir_build_local_zero_init(b, type);
    _variables.emplace(v.uid(), n);
    return n;
}

ir::NodeRef AST2IR::_convert_builtin_variable(Variable v) noexcept {
    auto func = [tag = v.tag()] {
        switch (tag) {
            case Variable::Tag::THREAD_ID:
                return ir::Func::Tag::ThreadId;
            case Variable::Tag::BLOCK_ID:
                return ir::Func::Tag::BlockId;
            case Variable::Tag::DISPATCH_ID:
                return ir::Func::Tag::DispatchId;
            case Variable::Tag::DISPATCH_SIZE:
                return ir::Func::Tag::DispatchSize;
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION(
            "Invalid builtin variable tag.");
    }();
    auto b = _current_builder();
    auto type = _convert_type(v.type());
    auto node = ir::luisa_compute_ir_build_call(
        b, {.tag = func}, {}, type);
    _variables.emplace(v.uid(), node);
    return node;
}

ir::NodeRef AST2IR::_cast(const Type *type_dst, const Type *type_src, ir::NodeRef node_src, bool diagonal_matrix) noexcept {
    if (*type_dst == *type_src) { return node_src; }
    // scalar to scalar
    auto builder = _current_builder();
    if (type_dst->is_scalar() && type_src->is_scalar()) {
        return ir::luisa_compute_ir_build_call(
            builder, {.tag = ir::Func::Tag::Cast},
            {.ptr = &node_src, .len = 1u},
            _convert_type(type_dst));
    }
    // vector to vector
    if (type_dst->is_vector() && type_src->is_vector()) {
        LUISA_ASSERT(type_dst->dimension() == type_src->dimension(),
                     "Vector dimension mismatch: dst = {}, src = {}.",
                     type_dst->dimension(), type_src->dimension());
        return ir::luisa_compute_ir_build_call(
            builder, {.tag = ir::Func::Tag::Cast},
            {.ptr = &node_src, .len = 1u},
            _convert_type(type_dst));
    }
    // scalar to vector
    if (type_dst->is_vector() && type_src->is_scalar()) {
        auto elem = _cast(type_dst->element(), type_src, node_src);
        return ir::luisa_compute_ir_build_call(
            builder, {.tag = ir::Func::Tag::Vec},
            {.ptr = &elem, .len = 1u},
            _convert_type(type_dst));
    }
    // scalar to matrix
    if (type_dst->is_matrix() && type_src->is_scalar()) {
        LUISA_ASSERT(type_dst->element()->tag() == Type::Tag::FLOAT,
                     "Only float matrices are supported.");
        auto elem = _cast(Type::of<float>(), type_src, node_src);
        return ir::luisa_compute_ir_build_call(
            builder,
            {.tag = diagonal_matrix ?
                        ir::Func::Tag::Eye :
                        ir::Func::Tag::Full},
            {.ptr = &elem, .len = 1u},
            _convert_type(type_dst));
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid type cast: {} -> {}.",
        type_src->description(), type_dst->description());
}

ir::NodeRef AST2IR::_literal(const Type *type, LiteralExpr::Value value) noexcept {
    return luisa::visit(
        [&](auto x) noexcept {
            using T = std::decay_t<decltype(x)>;
            LUISA_ASSERT(*Type::of<T>() == *type,
                         "Type mismatch: '{}' vs. '{}'.",
                         Type::of<T>()->description(),
                         type->description());
            if constexpr (is_scalar_v<T>) {
                auto c = [&]() noexcept -> ir::Const {
                    if (x == static_cast<T>(0)) {
                        ir::Const cc{.tag = ir::Const::Tag::Zero};
                        cc.zero = {_convert_type(type)};
                        return cc;
                    }
                    if constexpr (std::is_same_v<T, bool>) {
                        return {.tag = ir::Const::Tag::Bool, .bool_ = {x}};
                    } else if constexpr (std::is_same_v<T, float>) {
                        return {.tag = ir::Const::Tag::Float32, .float32 = {x}};
                    } else if constexpr (std::is_same_v<T, int>) {
                        return {.tag = ir::Const::Tag::Int32, .int32 = {x}};
                    } else if constexpr (std::is_same_v<T, uint>) {
                        return {.tag = ir::Const::Tag::Uint32, .uint32 = {x}};
                    } else {
                        static_assert(always_false_v<T>, "Unsupported scalar type.");
                    }
                }();
                auto b = _current_builder();
                return ir::luisa_compute_ir_build_const(b, c);
            } else {
                auto salt = luisa::hash64("__ast2ir_literal");
                auto hash = luisa::hash64(x, luisa::hash64(type->hash(), salt));
                if (auto iter = _constants.find(hash); iter != _constants.end()) { return iter->second; }
                auto slice = _boxed_slice<uint8_t>(sizeof(T));
                std::memcpy(slice.ptr, &x, sizeof(T));
                auto c = ir::Const{.tag = ir::Const::Tag::Generic};
                c.generic = {slice, _convert_type(type)};
                auto b = _current_builder();
                auto node = ir::luisa_compute_ir_build_const(b, c);
                _constants.emplace(hash, node);
                return node;
            }
        },
        value);
}

}// namespace luisa::compute
