//
// Created by Mike Smith on 2020/12/2.
//

#include <core/logging.h>
#include <ast/function_builder.h>

namespace luisa::compute::detail {

luisa::vector<FunctionBuilder *> &FunctionBuilder::_function_stack() noexcept {
    static thread_local luisa::vector<FunctionBuilder *> stack;
    return stack;
}

void FunctionBuilder::push(FunctionBuilder *func) noexcept {
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
    if (f->tag() == Tag::KERNEL &&
        f->_return_type.value_or(nullptr) != nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Kernels cannot have non-void return types.");
    }
    if (f->tag() != Tag::KERNEL &&
        (!f->shared_variables().empty() ||
         f->propagated_builtin_callables().test(CallOp::SYNCHRONIZE_BLOCK))) {
        LUISA_ERROR_WITH_LOCATION("Shared variables and block synchronization "
                                  "are only allowed in kernels.");
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
    LUISA_ASSERT(_tag != Tag::KERNEL || expr == nullptr,
                 "Kernels cannot return non-void values.");
    if (_return_type) {// multiple return statements, check if they match
        if (*_return_type == nullptr) {
            LUISA_ASSERT(expr != nullptr,
                         "Mismatched return types: {} and previously void.",
                         expr->type()->description());
        } else {
            LUISA_ASSERT(expr != nullptr && *_return_type.value() == *expr->type(),
                         "Mismatched return types: {} and previously {}.",
                         expr == nullptr ? "void" : expr->type()->description(),
                         _return_type.value()->description());
        }
    }
    if (expr == nullptr) {// returning void
        static thread_local ReturnStmt null_return{nullptr};
        _append(&null_return);
    } else {// returning a non-void value
        _return_type.emplace(expr->type());
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
    if (expr != nullptr) { _create_and_append_statement<ExprStmt>(expr); }
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
    value = luisa::visit([type](auto x) noexcept -> LiteralExpr::Value {
        if constexpr (luisa::is_scalar_v<decltype(x)>) {
            switch (type->tag()) {
                case Type::Tag::BOOL: return bool(x);
                case Type::Tag::FLOAT32: return float(x);
                case Type::Tag::INT32: return int(x);
                case Type::Tag::UINT32: return uint(x);
                default: LUISA_ERROR_WITH_LOCATION("Invalid type for LiteralExpr: {}", type->description());
            }
        } else {
            return x;
        }
    },
                         value);
    return _create_expression<LiteralExpr>(type, value);
}

const RefExpr *FunctionBuilder::local(const Type *type) noexcept {
    Variable v{type, Variable::Tag::LOCAL, _next_variable_uid()};
    _local_variables.emplace_back(v);
    return _ref(v);
}

const RefExpr *FunctionBuilder::shared(const Type *type) noexcept {
    Variable sv{type, Variable::Tag::SHARED, _next_variable_uid()};
    _shared_variables.emplace_back(sv);
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
const RefExpr *FunctionBuilder::kernel_id() noexcept { return _builtin(Variable::Tag::KERNEL_ID); }
const RefExpr *FunctionBuilder::object_id() noexcept { return _builtin(Variable::Tag::OBJECT_ID); }

inline const RefExpr *FunctionBuilder::_builtin(Variable::Tag tag) noexcept {
    if (auto iter = std::find_if(
            _builtin_variables.cbegin(), _builtin_variables.cend(),
            [tag](auto &&v) noexcept { return v.tag() == tag; });
        iter != _builtin_variables.cend()) {
        return _ref(*iter);
    }
    Variable v{Type::of<uint3>(), tag, _next_variable_uid()};
    _builtin_variables.emplace_back(v);
    // for callables, builtin variables are treated like arguments
    if (_tag != Function::Tag::KERNEL) [[unlikely]] {
        _arguments.emplace_back(v);
        _argument_bindings.emplace_back();
    }
    return _ref(v);
}

const RefExpr *FunctionBuilder::argument(const Type *type) noexcept {
    Variable v{type, Variable::Tag::LOCAL, _next_variable_uid(), true};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back();
    return _ref(v);
}

const RefExpr *FunctionBuilder::buffer(const Type *type) noexcept {
    Variable v{type, Variable::Tag::BUFFER, _next_variable_uid(), true};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back();
    return _ref(v);
}

const RefExpr *FunctionBuilder::buffer_binding(const Type *type, uint64_t handle, size_t offset_bytes, size_t size_bytes) noexcept {
    // find if already bound
    for (auto i = 0u; i < _arguments.size(); i++) {
        if (luisa::visit(
                [&]<typename T>(T binding) noexcept {
                    if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                        return *_arguments[i].type() == *type &&
                               binding.handle == handle &&
                               binding.offset_bytes == offset_bytes;
                    } else {
                        return false;
                    }
                },
                _argument_bindings[i])) {
            auto &binding = luisa::get<Function::BufferBinding>(_argument_bindings[i]);
            binding.size_bytes = std::max(binding.size_bytes, size_bytes);
            return _ref(_arguments[i]);
        }
    }
    Variable v{type, Variable::Tag::BUFFER, _next_variable_uid(), true};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back(Function::BufferBinding{handle, offset_bytes, size_bytes});
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

const Expression *FunctionBuilder::swizzle(const Type *type, const Expression *self, size_t swizzle_size, uint64_t swizzle_code) noexcept {
    // special handling for literal swizzles
    if (self->tag() == Expression::Tag::LITERAL) {
        auto element = luisa::visit(
            [&](auto &&v) noexcept -> const Expression * {
                using TVec = std::decay_t<decltype(v)>;
                if constexpr (is_vector_v<TVec>) {
                    using TElem = vector_element_t<TVec>;
                    if (swizzle_size == 1u) {
                        auto i = swizzle_code & 0b11u;
                        return literal(Type::of<TElem>(), v[i]);
                    } else if (swizzle_size == 2u) {
                        auto i = (swizzle_code >> 0u) & 0b11u;
                        auto j = (swizzle_code >> 4u) & 0b11u;
                        return literal(Type::of<TVec>(),
                                       Vector<TElem, 2u>{v[i], v[j]});
                    } else if (swizzle_size == 3u) {
                        auto i = (swizzle_code >> 0u) & 0b11u;
                        auto j = (swizzle_code >> 4u) & 0b11u;
                        auto k = (swizzle_code >> 8u) & 0b11u;
                        return literal(Type::of<TVec>(),
                                       Vector<TElem, 3u>{v[i], v[j], v[k]});
                    } else if (swizzle_size == 4u) {
                        auto i = (swizzle_code >> 0u) & 0b11u;
                        auto j = (swizzle_code >> 4u) & 0b11u;
                        auto k = (swizzle_code >> 8u) & 0b11u;
                        auto l = (swizzle_code >> 12u) & 0b11u;
                        return literal(Type::of<TVec>(),
                                       Vector<TElem, 4u>{v[i], v[j], v[k], v[l]});
                    }
                    LUISA_ERROR_WITH_LOCATION("Invalid swizzle size.");
                } else {
                    LUISA_ERROR_WITH_LOCATION("Swizzle must be a vector but got '{}'.",
                                              self->type()->description());
                }
            },
            static_cast<const LiteralExpr *>(self)->value());
        return element;
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
    LUISA_ASSERT(type->is_array(),
                 "Constant must be array but got '{}'.",
                 type->description());
    if (auto iter = std::find_if(
            _captured_constants.begin(), _captured_constants.end(),
            [data](auto &&c) noexcept { return c.data.hash() == data.hash(); });
        iter == _captured_constants.end()) {
        _captured_constants.emplace_back(Constant{type, data});
    }
    return _create_expression<ConstantExpr>(type, data);
}

const Statement *FunctionBuilder::pop_stmt() noexcept {
    return _scope_stack.back()->pop();
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

FunctionBuilder::~FunctionBuilder() noexcept {
    LUISA_VERBOSE("FunctionBuilder destructor called");
}
FunctionBuilder::FunctionBuilder(FunctionBuilder::Tag tag) noexcept
    : _hash{0ul}, _tag{tag} {
    LUISA_VERBOSE("FunctionBuilder constructor called");
}

const RefExpr *FunctionBuilder::texture(const Type *type) noexcept {
    Variable v{type, Variable::Tag::TEXTURE, _next_variable_uid(), true};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back();
    return _ref(v);
}

const RefExpr *FunctionBuilder::texture_binding(const Type *type, uint64_t handle, uint32_t level) noexcept {
    for (auto i = 0u; i < _arguments.size(); i++) {
        if (luisa::visit(
                [&]<typename T>(T binding) noexcept {
                    if constexpr (std::is_same_v<T, Function::TextureBinding>) {
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
    Variable v{type, Variable::Tag::TEXTURE, _next_variable_uid(), true};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back(Function::TextureBinding{handle, level});
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
    static_cast<void>(call(nullptr, call_op, args));
}

void FunctionBuilder::call(Function custom, std::initializer_list<const Expression *> args) noexcept {
    static_cast<void>(call(nullptr, custom, args));
}

void FunctionBuilder::_compute_hash() noexcept {
    using namespace std::string_view_literals;
    static auto seed = hash_value("__hash_function"sv);
    luisa::vector<uint64_t> hashes;
    hashes.reserve(2u /* body and tag */ +
                   1u /* return type */ +
                   _arguments.size() +
                   _captured_constants.size() +
                   1u /* block size */);
    hashes.emplace_back(hash_value(_tag));
    hashes.emplace_back(_body.hash());
    hashes.emplace_back(_return_type ? hash_value(*_return_type.value()) : hash_value("void"sv));
    for (auto &&arg : _arguments) { hashes.emplace_back(hash_value(arg)); }
    for (auto &&c : _captured_constants) { hashes.emplace_back(hash_value(c)); }
    hashes.emplace_back(hash_value(_block_size));
    _hash = hash64(hashes.data(), hashes.size() * sizeof(uint64_t), seed);
}

const RefExpr *FunctionBuilder::bindless_array_binding(uint64_t handle) noexcept {
    for (auto i = 0u; i < _arguments.size(); i++) {
        if (luisa::visit(
                [&]<typename T>(T binding) noexcept {
                    if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                        return binding.handle == handle;
                    } else {
                        return false;
                    }
                },
                _argument_bindings[i])) {
            return _ref(_arguments[i]);
        }
    }
    Variable v{Type::of<BindlessArray>(), Variable::Tag::BINDLESS_ARRAY, _next_variable_uid(), true};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back(Function::BindlessArrayBinding{handle});
    return _ref(v);
}

const RefExpr *FunctionBuilder::bindless_array() noexcept {
    Variable v{Type::of<BindlessArray>(), Variable::Tag::BINDLESS_ARRAY, _next_variable_uid(), true};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back();
    return _ref(v);
}

const RefExpr *FunctionBuilder::accel_binding(uint64_t handle) noexcept {
    for (auto i = 0u; i < _arguments.size(); i++) {
        if (luisa::visit(
                [&]<typename T>(T binding) noexcept {
                    if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                        return binding.handle == handle;
                    } else {
                        return false;
                    }
                },
                _argument_bindings[i])) {
            return _ref(_arguments[i]);
        }
    }
    Variable v{Type::of<Accel>(), Variable::Tag::ACCEL, _next_variable_uid(), true};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back(Function::AccelBinding{handle});
    return _ref(v);
}

const RefExpr *FunctionBuilder::accel() noexcept {
    Variable v{Type::of<Accel>(), Variable::Tag::ACCEL, _next_variable_uid(), true};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back();
    return _ref(v);
}

// call builtin functions
const CallExpr *FunctionBuilder::call(const Type *type, CallOp call_op, luisa::span<const Expression *const> args) noexcept {
    if (call_op == CallOp::CUSTOM) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Custom functions are not allowed to "
            "be called with enum CallOp.");
    }
    _direct_builtin_callables.mark(call_op);
    _propagated_builtin_callables.mark(call_op);
    if (is_atomic_operation(call_op)) {
        LUISA_ASSERT(!args.empty(), "First argument of a call must be a buffer.");
        if (args.front()->type()->element()->tag() == Type::Tag::FLOAT32) {
            _requires_atomic_float = true;
        }
    }
    auto expr = _create_expression<CallExpr>(
        type, call_op, CallExpr::ArgumentList{args.begin(), args.end()});
    if (type == nullptr) {
        _void_expr(expr);
        return nullptr;
    }
    return expr;
}

// call custom functions
const CallExpr *FunctionBuilder::call(const Type *type, Function custom, luisa::span<const Expression *const> args) noexcept {
    if (custom.tag() != Function::Tag::CALLABLE) {
        LUISA_ERROR_WITH_LOCATION(
            "Calling non-callable function in device code.");
    }
    auto f = custom.builder();
    CallExpr::ArgumentList call_args(f->_arguments.size(), nullptr);
    auto in_iter = args.begin();
    for (auto i = 0u; i < f->_arguments.size(); i++) {
        if (auto arg = f->_arguments[i]; arg.is_builtin()) {
            call_args[i] = _builtin(arg.tag());
        } else {
            call_args[i] = luisa::visit(
                [&]<typename T>(T binding) noexcept -> const Expression * {
                    if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                        return buffer_binding(f->_arguments[i].type(), binding.handle, binding.offset_bytes, binding.size_bytes);
                    } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                        return texture_binding(f->_arguments[i].type(), binding.handle, binding.level);
                    } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                        return bindless_array_binding(binding.handle);
                    } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                        return accel_binding(binding.handle);
                    } else {
                        return *(in_iter++);
                    }
                },
                f->_argument_bindings[i]);
        }
    }
    if (in_iter != args.end()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid call arguments for custom callable #{:016x}.",
            custom.hash());
    }
    auto expr = _create_expression<CallExpr>(type, custom, std::move(call_args));
    if (auto iter = std::find_if(
            _used_custom_callables.cbegin(), _used_custom_callables.cend(),
            [&](auto &&p) noexcept { return f == p.get(); });
        iter == _used_custom_callables.cend()) {
        _used_custom_callables.emplace_back(custom.shared_builder());
        // propagate used builtin/custom callables and constants
        _propagated_builtin_callables.propagate(f->_propagated_builtin_callables);
        _requires_atomic_float |= f->_requires_atomic_float;
    }
    if (type == nullptr) {
        _void_expr(expr);
        return nullptr;
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
    Variable v{type, Variable::Tag::REFERENCE, _next_variable_uid(), true};
    _arguments.emplace_back(v);
    _argument_bindings.emplace_back();
    return _ref(v);
}

void FunctionBuilder::comment_(luisa::string comment) noexcept {
    _create_and_append_statement<CommentStmt>(std::move(comment));
}

void FunctionBuilder::set_block_size(uint3 size) noexcept {
    if (_tag == Tag::KERNEL) {
        _block_size = size;
    } else {
        LUISA_WARNING_WITH_LOCATION(
            "Setting block size is not valid in Callables. "
            "Ignoring the `set_block_size({}, {}, {})` call.",
            size.x, size.y, size.z);
    }
}

bool FunctionBuilder::requires_raytracing() const noexcept {
    return _propagated_builtin_callables.uses_raytracing();
}

bool FunctionBuilder::requires_atomic() const noexcept {
    return _propagated_builtin_callables.uses_atomic();
}

bool FunctionBuilder::requires_atomic_float() const noexcept {
    return _requires_atomic_float;
}

}// namespace luisa::compute::detail
