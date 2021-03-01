//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include "function.h"
#include <vector>
#include <fmt/format.h>

#include <core/memory.h>
#include <runtime/buffer.h>
#include <ast/statement.h>
#include <ast/variable.h>
#include <ast/expression.h>
#include <ast/type_registry.h>

namespace luisa::compute {

struct Statement;
struct Expression;

class FunctionBuilder {

public:
    using Tag = Function::Tag;
    using ConstantData = Function::ConstantData;
    using BufferBinding = Function::BufferBinding;
    using TextureBinding = Function::TextureBinding;

private:
    Arena _arena;
    const ScopeStmt *_body;
    ArenaVector<ArenaVector<const Statement *>> _scope_stack;
    ArenaVector<Variable> _builtin_variables;
    ArenaVector<Variable> _shared_variables;
    ArenaVector<ConstantData> _constant_variables;
    ArenaVector<BufferBinding> _captured_buffers;
    ArenaVector<TextureBinding> _captured_textures;
    ArenaVector<Variable> _arguments;
    ArenaVector<uint32_t> _used_custom_callables;
    ArenaVector<std::string_view> _used_builtin_callables;
    Tag _tag;
    uint32_t _uid;
    uint32_t _variable_counter{0u};

private:
    [[nodiscard]] static std::vector<FunctionBuilder *> &_function_stack() noexcept;
    [[nodiscard]] static std::vector<std::unique_ptr<FunctionBuilder>> &_function_registry() noexcept;
    [[nodiscard]] uint32_t _next_variable_uid() noexcept;

    static void _push(FunctionBuilder *func) noexcept;
    static FunctionBuilder *_pop() noexcept;

    void _append(const Statement *statement) noexcept;

    [[nodiscard]] const Expression *_literal(const Type *type, LiteralExpr::Value value) noexcept;
    [[nodiscard]] const Expression *_constant(const Type *type, const void *data) noexcept;
    [[nodiscard]] const Expression *_builtin(Variable::Tag tag) noexcept;
    [[nodiscard]] const Expression *_buffer_binding(const Type *type, uint64_t handle, size_t offset_bytes) noexcept;
    [[nodiscard]] const Expression *_texture_binding(const Type *type, uint64_t handle) noexcept;
    [[nodiscard]] const Expression *_ref(Variable v) noexcept;

private:
    explicit FunctionBuilder(Tag tag, uint32_t uid) noexcept
        : _body{nullptr},
          _scope_stack{_arena},
          _builtin_variables{_arena},
          _shared_variables{_arena},
          _constant_variables{_arena},
          _captured_buffers{_arena},
          _captured_textures{_arena},
          _arguments{_arena},
          _used_custom_callables{_arena},
          _used_builtin_callables{_arena},
          _tag{tag},
          _uid{uid} {}

    template<typename Def>
    static auto _define(Function::Tag tag, Def &&def) noexcept {
        auto f_uid = static_cast<uint32_t>(_function_registry().size());
        auto f = _function_registry().emplace_back(new FunctionBuilder{tag, f_uid}).get();
        _push(f);
        f->_body = f->scope(std::forward<Def>(def));
        if (_pop() != f) { LUISA_ERROR_WITH_LOCATION("Invalid function on stack top."); }
        return Function{*f};
    }

public:
    [[nodiscard]] static FunctionBuilder *current() noexcept;

    // interfaces for class Function
    [[nodiscard]] auto builtin_variables() const noexcept { return std::span{_builtin_variables.data(), _builtin_variables.size()}; }
    [[nodiscard]] auto shared_variables() const noexcept { return std::span{_shared_variables.data(), _shared_variables.size()}; }
    [[nodiscard]] auto constant_variables() const noexcept { return std::span{_constant_variables.data(), _constant_variables.size()}; }
    [[nodiscard]] auto captured_buffers() const noexcept { return std::span{_captured_buffers.data(), _captured_buffers.size()}; }
    [[nodiscard]] auto captured_textures() const noexcept { return std::span{_captured_textures.data(), _captured_textures.size()}; }
    [[nodiscard]] auto arguments() const noexcept { return std::span{_arguments.data(), _arguments.size()}; }
    [[nodiscard]] auto custom_callables() const noexcept { return std::span{_used_custom_callables.data(), _used_custom_callables.size()}; }
    [[nodiscard]] auto builtin_callables() const noexcept { return std::span{_used_builtin_callables.data(), _used_builtin_callables.size()}; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto body() const noexcept { return _body; }
    [[nodiscard]] auto uid() const noexcept { return _uid; }
    [[nodiscard]] static Function custom_callable(size_t uid) noexcept;

    // build primitives
    template<typename Def>
    static auto define_kernel(Def &&def) noexcept { return _define(Function::Tag::KERNEL, std::forward<Def>(def)); }

    template<typename Def>
    static auto define_callable(Def &&def) noexcept {
        auto f = _define(Function::Tag::CALLABLE, std::forward<Def>(def));
        if (!f.builtin_variables().empty() ||
            !f.shared_variables().empty() ||
            !f.captured_buffers().empty() ||
            !f.captured_textures().empty()) {
            LUISA_ERROR_WITH_LOCATION("Custom callables may not have builtin, shared or captured variables.");
        }
        return f;
    }

    [[nodiscard]] const Expression *thread_id() noexcept;
    [[nodiscard]] const Expression *block_id() noexcept;
    [[nodiscard]] const Expression *dispatch_id() noexcept;

    // variables
    [[nodiscard]] const Expression *local(const Type *type, std::span<const Expression *> init) noexcept;
    [[nodiscard]] const Expression *local(const Type *type, std::initializer_list<const Expression *> init) noexcept;
    [[nodiscard]] const Expression *shared(const Type *type) noexcept;

    template<concepts::Container U>
    requires concepts::Native<typename std::remove_cvref_t<U>::value_type> [[nodiscard]] auto constant(U &&data) noexcept {
        using T = typename std::remove_cvref_t<U>::value_type;
        auto type = Type::from(fmt::format("array<{},{}>", Type::of<T>()->description(), data.size()));
        auto bytes = _arena.allocate<T>(data.size());
        std::uninitialized_copy_n(data.cbegin(), data.size(), bytes);
        return _constant(type, bytes);
    }

    template<typename T>
    [[nodiscard]] auto constant(std::initializer_list<T> data) noexcept { return constant<std::initializer_list<T>>(data); }
    
    template<typename T>
    [[nodiscard]] auto buffer_binding(BufferView<T> bv) noexcept {
        return _buffer_binding(Type::of<BufferView<T>>(), bv.handle(), bv.offset_bytes());
    }

    [[nodiscard]] const Expression *texture_binding(const Type *type, uint64_t handle) noexcept;

    // explicit arguments
    [[nodiscard]] const Expression *uniform(const Type *type) noexcept;
    [[nodiscard]] const Expression *buffer(const Type *type) noexcept;
    [[nodiscard]] const Expression *texture(const Type *type) noexcept;

    // expressions
    template<typename T>
    requires concepts::Native<T> [[nodiscard]] auto literal(T value) noexcept { return _literal(Type::of(value), value); }
    [[nodiscard]] const Expression *unary(const Type *type, UnaryOp op, const Expression *expr) noexcept;
    [[nodiscard]] const Expression *binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept;
    [[nodiscard]] const Expression *member(const Type *type, const Expression *self, size_t member_index) noexcept;
    [[nodiscard]] const Expression *access(const Type *type, const Expression *range, const Expression *index) noexcept;
    [[nodiscard]] const Expression *call(const Type *type /* nullptr for void */, std::string_view func, std::span<const Expression *> args) noexcept;
    [[nodiscard]] const Expression *call(const Type *type /* nullptr for void */, std::string_view func, std::initializer_list<const Expression *> args) noexcept;
    [[nodiscard]] const Expression *cast(const Type *type, CastOp op, const Expression *expr) noexcept;

    // statements
    void break_() noexcept;
    void continue_() noexcept;
    void return_(const Expression *expr = nullptr /* nullptr for void */) noexcept;

    template<typename Body>
    const ScopeStmt *scope(Body &&body) noexcept {
        _scope_stack.emplace_back(ArenaVector<const Statement *>(_arena));
        body();
        auto stmt = _arena.create<ScopeStmt>(_scope_stack.back());
        _scope_stack.pop_back();
        return stmt;
    }

    void if_(const Expression *cond, const Statement *true_branch) noexcept;
    void if_(const Expression *cond, const Statement *true_branch, const Statement *false_branch) noexcept;
    void while_(const Expression *cond, const Statement *body) noexcept;
    void void_(const Expression *expr) noexcept;
    void switch_(const Expression *expr, const Statement *body) noexcept;
    void case_(const Expression *expr, const Statement *body) noexcept;
    void default_(const Statement *body) noexcept;
    void assign(AssignOp op, const Expression *lhs, const Expression *rhs) noexcept;
};

}// namespace luisa::compute
