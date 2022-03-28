//
// Created by Mike Smith on 2020/12/2.
//

#include <ast/function_builder.h>

namespace luisa::compute::detail {

luisa::vector<FunctionBuilder *> &FunctionBuilder::_function_stack() noexcept {
    static thread_local luisa::vector<FunctionBuilder *> stack;
    return stack;
}

void FunctionBuilder::push(FunctionBuilder *func) noexcept {
    if (func->tag() == Function::Tag::KERNEL && !_function_stack().empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Kernel definitions cannot be nested.");
    }
    _function_stack().emplace_back(func);
}

void FunctionBuilder::pop(FunctionBuilder *func) noexcept {
    if (_function_stack().empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid pop on empty function stack.");
    }
    auto f = _function_stack().back();
    _function_stack().pop_back();
    if (func != nullptr && f != func) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid function on stack top.");
    }
    if (f->tag() == Function::Tag::CALLABLE &&
        !f->builtin_variables().empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Custom callables may not use builtin "
            "variables. Pass them as arguments "
            "if you want to use their values.");
    }
    if (f->_raytracing &&
        (f->_using_shared_storage ||
         f->_used_builtin_callables.test(
             CallOp::SYNCHRONIZE_BLOCK))) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Raytracing functions are not allowed to "
            "use shared storage or call synchronize_block().");
    }
    if (f->_arguments.size() != f->_argument_bindings.size()) {
        LUISA_ERROR_WITH_LOCATION(
            "Arguments and their bindings have different sizes ({} and {}).",
            f->_arguments.size(), f->_argument_bindings.size());
    }

    // hash
    f->_compute_hash();

    // clear temporary data
    for (auto p : f->_temporary_data) {
        luisa::detail::allocator_deallocate(
            p.first, p.second);
    }
    f->_temporary_data.clear();
    f->_temporary_data.shrink_to_fit();
}

FunctionBuilder *FunctionBuilder::current() noexcept {
    LUISA_ASSERT(!_function_stack().empty(), "Empty function stack.");
    return _function_stack().back();
}

void FunctionBuilder::_append(const Statement *statement) noexcept {
    if (_scope_stack.empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Scope stack is empty.");
    }
    _scope_stack.back()->append(statement);
}

void FunctionBuilder::break_() noexcept {
    _create_and_append_statement<BreakStmt>();
}

void FunctionBuilder::continue_() noexcept {
    _create_and_append_statement<ContinueStmt>();
}

void FunctionBuilder::return_(const Expression *expr) noexcept {
    if (_ret != nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Multiple non-void return statements are not allowed.");
    }
    if (expr == nullptr) {
        static thread_local ReturnStmt null_return{nullptr};
        _append(&null_return);
    } else {
        _ret = expr->type();
        _create_and_append_statement<ReturnStmt>(expr);
    }
}

IfStmt *FunctionBuilder::if_(const Expression *cond) noexcept {
    return _create_and_append_statement<IfStmt>(cond);
}

LoopStmt *FunctionBuilder::loop_() noexcept {
    return _create_and_append_statement<LoopStmt>();
}

void FunctionBuilder::_void_expr(const Expression *expr) noexcept {
    _create_and_append_statement<ExprStmt>(expr);
}

SwitchStmt *FunctionBuilder::switch_(const Expression *expr) noexcept {
    return _create_and_append_statement<SwitchStmt>(expr);
}

SwitchCaseStmt *FunctionBuilder::case_(const Expression *expr) noexcept {
    return _create_and_append_statement<SwitchCaseStmt>(expr);
}

SwitchDefaultStmt *FunctionBuilder::default_() noexcept {
    return _create_and_append_statement<SwitchDefaultStmt>();
}

void FunctionBuilder::assign(const Expression *lhs, const Expression *rhs) noexcept {
    _create_and_append_statement<AssignStmt>(lhs, rhs);
}

const LiteralExpr *FunctionBuilder::literal(const Type *type, LiteralExpr::Value value) noexcept {
    return _create_expression<LiteralExpr>(type, value);
}

const RefExpr *FunctionBuilder::local(const Type *type) noexcept {
    Variable v{type, Variable::Tag::LOCAL, _next_variable_uid()};
    if (_meta_stack.empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Empty meta stack when adding local variable.");
    }
    _meta_stack.back()->add(v);
    return _ref(v);
}

const RefExpr *FunctionBuilder::shared(const Type *type) noexcept {
    Variable sv{type, Variable::Tag::SHARED, _next_variable_uid()};
    if (_meta_stack.empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Empty meta stack when adding shared variable.");
    }
    _meta_stack.back()->add(sv);
    _using_shared_storage = true;
    return _ref(sv);
}

uint32_t FunctionBuilder::_next_variable_uid() noexcept {
    auto uid = static_cast<uint32_t>(_variable_usages.size());
    _variable_usages.emplace_back(Usage::NONE);
    return uid;
}

const RefExpr *FunctionBuilder::thread_id() noexcept { return _builtin(Variable::Tag::THREAD_ID); }
const RefExpr *FunctionBuilder::block_id() noexcept { return _builtin(Variable::Tag::BLOCK_ID); }
const RefExpr *FunctionBuilder::dispatch_id() noexcept { return _builtin(Variable::Tag::DISPATCH_ID); }
const RefExpr *FunctionBuilder::dispatch_size() noexcept { return _builtin(Variable::Tag::DISPATCH_SIZE); }

const RefExpr *FunctionBuilder::_builtin(Variable::Tag tag) noexcept {
    if (auto iter = std::find_if(
            _builtin_variables.cbegin(),
            _builtin_variables.cend(),
            [tag](auto &&v) noexcept { return v.tag() == tag; });
        iter != _builtin_variables.cend()) {
        return _ref(*iter);
    }
    Variable v{Type::of<uint3>(), tag, _next_variable_uid()};
    _builtin_variables.emplace_back(v);
    return _ref(v);
}

const RefExpr *FunctionBuilder::argument(const Type *type) noexcept {
    Variable v{type, Variable::Tag::LOCAL, _next_variable_uid()};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back();
    return _ref(v);
}

const RefExpr *FunctionBuilder::buffer(const Type *type) noexcept {
    Variable v{type, Variable::Tag::BUFFER, _next_variable_uid()};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back();
    return _ref(v);
}

const RefExpr *FunctionBuilder::buffer_binding(const Type *type, uint64_t handle, size_t offset_bytes) noexcept {
    // find if already bound
    for (auto i = 0u; i < _arguments.size(); i++) {
        if (luisa::visit(
                [&]<typename T>(T binding) noexcept {
                    if constexpr (std::is_same_v<T, BufferBinding>) {
                        return *_arguments[i].type() == *type &&
                               binding.handle == handle &&
                               binding.offset_bytes == offset_bytes;
                    } else {
                        return false;
                    }
                },
                _argument_bindings[i])) {
            return _ref(_arguments[i]);
        }
    }
    Variable v{type, Variable::Tag::BUFFER, _next_variable_uid()};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back(BufferBinding{handle, offset_bytes});
    return _ref(v);
}

const UnaryExpr *FunctionBuilder::unary(const Type *type, UnaryOp op, const Expression *expr) noexcept {
    return _create_expression<UnaryExpr>(type, op, expr);
}

const BinaryExpr *FunctionBuilder::binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept {
    return _create_expression<BinaryExpr>(type, op, lhs, rhs);
}

const MemberExpr *FunctionBuilder::member(const Type *type, const Expression *self, size_t member_index) noexcept {
    return _create_expression<MemberExpr>(type, self, member_index);
}

const MemberExpr *FunctionBuilder::swizzle(const Type *type, const Expression *self, size_t swizzle_size, uint64_t swizzle_code) noexcept {
    if (self->tag() == Expression::Tag::LITERAL) {
        auto v = local(self->type());
        assign(v, self);
        self = v;
    }
    return _create_expression<MemberExpr>(type, self, swizzle_size, swizzle_code);
}

const AccessExpr *FunctionBuilder::access(const Type *type, const Expression *range, const Expression *index) noexcept {
    if (range->tag() == Expression::Tag::LITERAL) {
        auto v = local(range->type());
        assign(v, range);
        range = v;
    }
    return _create_expression<AccessExpr>(type, range, index);
}

const CastExpr *FunctionBuilder::cast(const Type *type, CastOp op, const Expression *expr) noexcept {
    return _create_expression<CastExpr>(type, op, expr);
}

const RefExpr *FunctionBuilder::_ref(Variable v) noexcept {
    return _create_expression<RefExpr>(v);
}

const ConstantExpr *FunctionBuilder::constant(const Type *type, ConstantData data) noexcept {
    if (!type->is_array()) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Constant data must be array."); }
    _captured_constants.emplace_back(Constant{type, data});
    return _create_expression<ConstantExpr>(type, data);
}

void FunctionBuilder::push_scope(ScopeStmt *s) noexcept {
    _scope_stack.emplace_back(s);
}

void FunctionBuilder::pop_scope(const ScopeStmt *s) noexcept {
    if (_scope_stack.empty() ||
        (s != nullptr && _scope_stack.back() != s)) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid scope stack pop.");
    }
    _scope_stack.pop_back();
}

ForStmt *FunctionBuilder::for_(const Expression *var, const Expression *condition, const Expression *update) noexcept {
    return _create_and_append_statement<ForStmt>(var, condition, update);
}

void FunctionBuilder::mark_variable_usage(uint32_t uid, Usage usage) noexcept {
    auto old_usage = to_underlying(_variable_usages[uid]);
    auto u = static_cast<Usage>(old_usage | to_underlying(usage));
    _variable_usages[uid] = u;
}

FunctionBuilder::FunctionBuilder(FunctionBuilder::Tag tag) noexcept
    : _body{"__function_body"}, _hash{0ul}, _tag{tag} {}

const RefExpr *FunctionBuilder::texture(const Type *type) noexcept {
    Variable v{type, Variable::Tag::TEXTURE, _next_variable_uid()};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back();
    return _ref(v);
}

const RefExpr *FunctionBuilder::texture_binding(const Type *type, uint64_t handle, uint32_t level) noexcept {
    for (auto i = 0u; i < _arguments.size(); i++) {
        if (luisa::visit(
                [&]<typename T>(T binding) noexcept {
                    if constexpr (std::is_same_v<T, TextureBinding>) {
                        return *_arguments[i].type() == *type &&
                               binding.handle == handle &&
                               binding.level == level;
                    } else {
                        return false;
                    }
                },
                _argument_bindings[i])) {
            return _ref(_arguments[i]);
        }
    }
    Variable v{type, Variable::Tag::TEXTURE, _next_variable_uid()};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back(TextureBinding{handle, level});
    return _ref(v);
}

const CallExpr *FunctionBuilder::call(const Type *type, CallOp call_op, std::initializer_list<const Expression *> args) noexcept {
    luisa::vector<const Expression *> arg_list{args};
    return call(type, call_op, arg_list);
}

const CallExpr *FunctionBuilder::call(const Type *type, Function custom, std::initializer_list<const Expression *> args) noexcept {
    luisa::vector<const Expression *> arg_list{args};
    return call(type, custom, arg_list);
}

void FunctionBuilder::call(CallOp call_op, std::initializer_list<const Expression *> args) noexcept {
    _void_expr(call(nullptr, call_op, args));
}

void FunctionBuilder::call(Function custom, std::initializer_list<const Expression *> args) noexcept {
    _void_expr(call(nullptr, custom, args));
}

void FunctionBuilder::_compute_hash() noexcept {// FIXME: seems not good
    _hash = hash64(_body.hash(), hash64(_tag, hash64("__hash_function")));
    if (_ret != nullptr) { _hash = hash64(_ret->hash(), _hash); }
    for (auto &&arg : _arguments) { _hash = hash64(arg.hash(), _hash); }
    for (auto &&c : _captured_constants) { _hash = hash64(c.hash(), _hash); }
    _hash = hash64(_block_size, _hash);
}

const RefExpr *FunctionBuilder::bindless_array_binding(uint64_t handle) noexcept {
    for (auto i = 0u; i < _arguments.size(); i++) {
        if (luisa::visit(
                [&]<typename T>(T binding) noexcept {
                    if constexpr (std::is_same_v<T, BindlessArrayBinding>) {
                        return binding.handle == handle;
                    } else {
                        return false;
                    }
                },
                _argument_bindings[i])) {
            return _ref(_arguments[i]);
        }
    }
    Variable v{Type::of<BindlessArray>(), Variable::Tag::BINDLESS_ARRAY, _next_variable_uid()};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back(BindlessArrayBinding{handle});
    return _ref(v);
}

const RefExpr *FunctionBuilder::bindless_array() noexcept {
    Variable v{Type::of<BindlessArray>(), Variable::Tag::BINDLESS_ARRAY, _next_variable_uid()};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back();
    return _ref(v);
}

const RefExpr *FunctionBuilder::accel_binding(uint64_t handle) noexcept {
    for (auto i = 0u; i < _arguments.size(); i++) {
        if (luisa::visit(
                [&]<typename T>(T binding) noexcept {
                    if constexpr (std::is_same_v<T, AccelBinding>) {
                        return binding.handle == handle;
                    } else {
                        return false;
                    }
                },
                _argument_bindings[i])) {
            return _ref(_arguments[i]);
        }
    }
    Variable v{Type::of<Accel>(), Variable::Tag::ACCEL, _next_variable_uid()};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back(AccelBinding{handle});
    return _ref(v);
}

const RefExpr *FunctionBuilder::accel() noexcept {
    Variable v{Type::of<Accel>(), Variable::Tag::ACCEL, _next_variable_uid()};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back();
    return _ref(v);
}

const CallExpr *FunctionBuilder::call(const Type *type, CallOp call_op, luisa::span<const Expression *const> args) noexcept {
    if (call_op == CallOp::CUSTOM) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Custom functions are not allowed to "
            "be called with enum CallOp.");
    }
    if (call_op == CallOp::TRACE_CLOSEST || call_op == CallOp::TRACE_ANY) {
        _raytracing = true;
    }
    _used_builtin_callables.mark(call_op);
    return _create_expression<CallExpr>(
        type, call_op,
        CallExpr::ArgumentList{
            args.begin(),
            args.end()});
}

const CallExpr *FunctionBuilder::call(const Type *type, Function custom, luisa::span<const Expression *const> args) noexcept {
    if (custom.tag() != Function::Tag::CALLABLE) {
        LUISA_ERROR_WITH_LOCATION(
            "Calling non-callable function in device code.");
    }
    auto f = custom.builder();
    if (f->raytracing()) { _raytracing = true; }
    CallExpr::ArgumentList call_args(f->_arguments.size(), nullptr);
    auto in_iter = args.begin();
    for (auto i = 0u; i < f->_arguments.size(); i++) {
        call_args[i] = luisa::visit(
            [&]<typename T>(T binding) noexcept -> const Expression * {
                if constexpr (std::is_same_v<T, BufferBinding>) {
                    return buffer_binding(f->_arguments[i].type(), binding.handle, binding.offset_bytes);
                } else if constexpr (std::is_same_v<T, TextureBinding>) {
                    return texture_binding(f->_arguments[i].type(), binding.handle, binding.level);
                } else if constexpr (std::is_same_v<T, BindlessArrayBinding>) {
                    return bindless_array_binding(binding.handle);
                } else if constexpr (std::is_same_v<T, AccelBinding>) {
                    return accel_binding(binding.handle);
                } else {
                    return *(in_iter++);
                }
            },
            f->_argument_bindings[i]);
    }
    if (in_iter != args.end()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid call arguments for custom callable #{}.",
            custom.hash());
    }
    auto expr = _create_expression<CallExpr>(type, custom, std::move(call_args));
    if (auto iter = std::find_if(
            _used_custom_callables.cbegin(),
            _used_custom_callables.cend(),
            [c = custom.builder()](auto &&p) noexcept { return c == p.get(); });
        iter == _used_custom_callables.cend()) {
        _used_custom_callables.emplace_back(custom.shared_builder());
    }
    return expr;
}

void FunctionBuilder::call(CallOp call_op, luisa::span<const Expression *const> args) noexcept {
    _void_expr(call(nullptr, call_op, args));
}

void FunctionBuilder::call(Function custom, luisa::span<const Expression *const> args) noexcept {
    _void_expr(call(nullptr, custom, args));
}

const RefExpr *FunctionBuilder::reference(const Type *type) noexcept {
    Variable v{type, Variable::Tag::REFERENCE, _next_variable_uid()};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back();
    return _ref(v);
}

void FunctionBuilder::comment_(luisa::string comment) noexcept {
    _create_and_append_statement<CommentStmt>(std::move(comment));
}

MetaStmt *FunctionBuilder::meta(luisa::string info) noexcept {
    auto meta = _create_and_append_statement<MetaStmt>(std::move(info));
    if (_meta_stack.empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid meta stack state.");
    }
    _meta_stack.back()->add(meta);
    return meta;
}

void FunctionBuilder::push_meta(MetaStmt *meta) noexcept {
    _meta_stack.emplace_back(meta);
    push_scope(meta->scope());
}

void FunctionBuilder::pop_meta(const MetaStmt *meta) noexcept {
    if (meta != nullptr) {// checks required
        pop_scope(meta->scope());
        if (_meta_stack.empty() || _meta_stack.back() != meta) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Invalid meta stack pop.");
        }
    }
    _meta_stack.pop_back();
}

}// namespace luisa::compute::detail
