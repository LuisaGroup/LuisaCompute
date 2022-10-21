//
// Created by Mike Smith on 2022/10/17.
//

#include <ir/ast2ir.h>
#include <ast/function_builder.h>

namespace luisa::compute {

ir::Module AST2IR::_convert_body(Function function) noexcept {
    for (auto v : function.local_variables()) {
        static_cast<void>(_convert_local_variable(v));
    }
    // process body scope
    static_cast<void>(_convert(function.body()));
    // finalize
    auto bb = ir::luisa_compute_ir_build_finish(*_current_builder());
    return {.kind = function.tag() == Function::Tag::KERNEL ?
                        ir::ModuleKind::Kernel :
                        ir::ModuleKind::Function,
            .entry = bb};
}

ir::KernelModule AST2IR::convert_kernel(Function function) noexcept {
    LUISA_ASSERT(function.tag() == Function::Tag::KERNEL,
                 "Invalid function tag.");
    LUISA_ASSERT(_struct_types.empty() && _constants.empty() &&
                     _variables.empty() && _builder_stack.empty(),
                 "Invalid state.");
    return _with_builder([function, this](auto builder) noexcept {
        auto bindings = function.builder()->argument_bindings();
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
            auto node = _convert_argument(function.arguments()[i]);
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
        for (auto v : function.builtin_variables()) {
            static_cast<void>(_convert_builtin_variable(v));
        }
        // process shared memory
        auto shared = _boxed_slice<ir::NodeRef>(function.shared_variables().size());
        for (auto i = 0u; i < function.shared_variables().size(); i++) {
            shared.ptr[i] = _convert_shared_variable(function.shared_variables()[i]);
        }
        auto module = _convert_body(function);
        return ir::KernelModule{.module = module,
                                .captures = captures,
                                .args = non_captures,
                                .shared = shared};
    });
}

ir::CallableModule AST2IR::convert_callable(Function function) noexcept {
    LUISA_ASSERT(function.tag() == Function::Tag::CALLABLE,
                 "Invalid function tag.");
    if (auto m = ir::luisa_compute_ir_get_symbol(function.hash())) { return *m; }
    LUISA_ASSERT(_struct_types.empty() && _constants.empty() &&
                     _variables.empty() && _builder_stack.empty(),
                 "Invalid state.");
    auto m = _with_builder([function, this](auto builder) noexcept {
        auto arg_count = function.arguments().size();
        auto args = _boxed_slice<ir::NodeRef>(arg_count);
        for (auto i = 0u; i < arg_count; i++) {
            args.ptr[i] = _convert_argument(function.arguments()[i]);
        }
        return ir::CallableModule{.module = _convert_body(function), .args = args};
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
        case Expression::Tag::PHI: return _convert(static_cast<const PhiExpr *>(expr));
        case Expression::Tag::REPLACE_MEMBER: return _convert(static_cast<const ReplaceMemberExpr *>(expr));
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
        return ir::luisa_compute_ir_register_type(t);
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
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const BinaryExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const MemberExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const AccessExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const RefExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const ConstantExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const CallExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const CastExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const PhiExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const CpuCustomOpExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const GpuCustomOpExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const ReplaceMemberExpr *expr) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const BreakStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const ContinueStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const ReturnStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const ScopeStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const IfStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const LoopStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const ExprStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const SwitchStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const SwitchCaseStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const SwitchDefaultStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const AssignStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const ForStmt *stmt) noexcept {
    return ir::NodeRef();
}

ir::NodeRef AST2IR::_convert(const CommentStmt *stmt) noexcept {
    auto b = _current_builder();
    auto msg = _boxed_slice<uint8_t>(stmt->comment().size());
    std::memcpy(msg.ptr, stmt->comment().data(), stmt->comment().size());
    auto instr = ir::luisa_compute_ir_new_instruction({
        .tag = ir::Instruction::Tag::Comment, .comment = {msg}});
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
                LUISA_ERROR_WITH_LOCATION("TODO");
            case Variable::Tag::BUFFER: {
                auto instr = ir::luisa_compute_ir_new_instruction({
                    .tag = ir::Instruction::Tag::Buffer});
                return ir::luisa_compute_ir_new_node(
                    {.type_ = _convert_type(v.type()->element()),
                     .instruction = instr});
            }
            case Variable::Tag::TEXTURE: {
                auto instr = ir::luisa_compute_ir_new_instruction({
                    .tag = v.type()->dimension() == 2u ?
                               ir::Instruction::Tag::Texture2D :
                               ir::Instruction::Tag::Texture3D});
                return ir::luisa_compute_ir_new_node(
                    {.type_ = _convert_type(v.type()->element()),
                     .instruction = instr});
            }
            case Variable::Tag::BINDLESS_ARRAY: {
                auto instr = ir::luisa_compute_ir_new_instruction({
                    .tag = ir::Instruction::Tag::Bindless});
                return ir::luisa_compute_ir_new_node(
                    {.type_ = _convert_type(nullptr),
                     .instruction = instr});
            }
            case Variable::Tag::ACCEL: {
                auto instr = ir::luisa_compute_ir_new_instruction({
                    .tag = ir::Instruction::Tag::Accel});
                return ir::luisa_compute_ir_new_node(
                    {.type_ = _convert_type(nullptr),
                     .instruction = instr});
            }
            default: {
                auto instr = ir::luisa_compute_ir_new_instruction({
                    .tag = ir::Instruction::Tag::Uniform});
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
    auto instr = ir::luisa_compute_ir_new_instruction({
        .tag = ir::Instruction::Tag::Shared});
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

}// namespace luisa::compute
