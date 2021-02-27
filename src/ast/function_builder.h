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
    using UniformBinding = Function::UniformBinding;

private:
    Arena _arena;
    ScopeStmt *_body;
    ArenaVector<ArenaVector<const Statement *>> _scope_stack;
    ArenaVector<Variable> _builtin_variables;
    ArenaVector<Variable> _shared_variables;
    ArenaVector<ConstantData> _constant_variables;
    ArenaVector<BufferBinding> _captured_buffers;
    ArenaVector<TextureBinding> _captured_textures;
    ArenaVector<UniformBinding> _captured_uniforms;
    ArenaVector<Variable> _arguments;
    Tag _tag;
    uint32_t _variable_counter{0u};

private:
    [[nodiscard]] static std::vector<FunctionBuilder *> &_function_stack() noexcept;
    [[nodiscard]] uint32_t _next_variable_uid() noexcept;

    static void _push(FunctionBuilder *func) noexcept;
    static FunctionBuilder *_pop() noexcept;

    void _add(const Statement *statement) noexcept;

    [[nodiscard]] const Expression *_literal(const Type *type, LiteralExpr::Value value) noexcept;
    [[nodiscard]] Variable _constant(const Type *type, const void *data) noexcept;
    [[nodiscard]] Variable _builtin(Variable::Tag tag) noexcept;
    [[nodiscard]] Variable _uniform_binding(const Type *type, const void *data) noexcept;
    [[nodiscard]] Variable _buffer_binding(const Type *type, uint64_t handle, size_t offset_bytes) noexcept;
    [[nodiscard]] Variable _texture_binding(const Type *type, uint64_t handle) noexcept;

public:
    explicit FunctionBuilder(Tag tag) noexcept
        : _body{nullptr},
          _scope_stack{_arena},
          _builtin_variables{_arena},
          _shared_variables{_arena},
          _constant_variables{_arena},
          _captured_buffers{_arena},
          _captured_textures{_arena},
          _captured_uniforms{_arena},
          _arguments{_arena},
          _tag{tag} {}

    [[nodiscard]] static FunctionBuilder *current() noexcept;

    template<typename Def>
    void define(Def &&def) noexcept {
        if (_body != nullptr) { LUISA_ERROR_WITH_LOCATION("Multiple definition."); }
        _push(this);
        _body = scope(std::forward<Def>(def));
        if (_pop() != this) { LUISA_ERROR_WITH_LOCATION("Invalid function on stack top."); }
    }

    [[nodiscard]] auto builtin_variables() const noexcept { return std::span{_builtin_variables.data(), _builtin_variables.size()}; }
    [[nodiscard]] auto shared_variables() const noexcept { return std::span{_shared_variables.data(), _shared_variables.size()}; }
    [[nodiscard]] auto constant_variables() const noexcept { return std::span{_constant_variables.data(), _constant_variables.size()}; }
    [[nodiscard]] auto captured_buffers() const noexcept { return std::span{_captured_buffers.data(), _captured_buffers.size()}; }
    [[nodiscard]] auto captured_textures() const noexcept { return std::span{_captured_textures.data(), _captured_textures.size()}; }
    [[nodiscard]] auto captured_uniforms() const noexcept { return std::span{_captured_uniforms.data(), _captured_uniforms.size()}; }
    [[nodiscard]] auto arguments() const noexcept { return std::span{_arguments.data(), _arguments.size()}; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto body() const noexcept { return _body; }

    [[nodiscard]] Variable thread_id() noexcept;
    [[nodiscard]] Variable block_id() noexcept;
    [[nodiscard]] Variable dispatch_id() noexcept;

    // variables
    [[nodiscard]] Variable local(const Type *type, std::span<const Expression *> init) noexcept;
    [[nodiscard]] Variable local(const Type *type, std::initializer_list<const Expression *> init) noexcept;
    [[nodiscard]] Variable shared(const Type *type) noexcept;

    template<typename U>
    requires concepts::container_type<U> &&concepts::core_data_type<typename std::remove_cvref_t<U>::value_type> [[nodiscard]] auto constant(U &&data) noexcept {
        using T = typename std::remove_cvref_t<U>::value_type;
        auto type = Type::from(fmt::format("array<{},{}>", Type::of<T>()->description(), data.size()));
        auto bytes = _arena.allocate<T>(data.size());
        std::uninitialized_copy_n(data.cbegin(), data.size(), bytes);
        return _constant(type, bytes);
    }

    template<typename T>
    [[nodiscard]] auto constant(std::initializer_list<T> data) noexcept { return constant<std::initializer_list<T>>(data); }

    // implicit arguments
    template<typename T>
    [[nodiscard]] Variable uniform_binding(std::span<T> u) noexcept {
        auto type = Type::from(fmt::format("array<{},{}>", Type::of<T>()->description(), u.size()));
        return _uniform_binding(type, u.data());
    }

    template<typename T>
    [[nodiscard]] Variable uniform_binding(const T *data) noexcept { return _uniform_binding(Type::of<T>(), data); }

    template<typename T>
    [[nodiscard]] Variable buffer_binding(BufferView<T> bv) noexcept {
        return _buffer_binding(Type::of<BufferView<T>>(), bv.handle(), bv.offset_bytes());
    }

    [[nodiscard]] Variable texture_binding(const Type *type, uint64_t handle) noexcept;

    // explicit arguments
    [[nodiscard]] Variable uniform(const Type *type) noexcept;
    [[nodiscard]] Variable buffer(const Type *type) noexcept;
    [[nodiscard]] Variable texture(const Type *type) noexcept;

    // expressions
    template<typename T>
    requires concepts::core_data_type<T> [[nodiscard]] auto literal(T value) noexcept { return _literal(Type::of(value), value); }
    [[nodiscard]] const Expression *ref(Variable v) noexcept;
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
    const Statement *scope(Body &&body) noexcept {
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
