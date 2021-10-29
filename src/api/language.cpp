//
// Created by Mike Smith on 2021/10/27.
//

#include <ast/function_builder.h>
#include <api/language.h>

using namespace luisa;
using namespace luisa::compute;

using luisa::compute::detail::FunctionBuilder;

void *luisa_compute_ast_begin_kernel() LUISA_NOEXCEPT {
    auto shared_f = luisa::make_shared<FunctionBuilder>(Function::Tag::KERNEL);
    auto f = shared_f.get();
    FunctionBuilder::push(f);
    f->push_meta(f->body());
    auto gid = f->dispatch_id();
    auto gs = f->dispatch_size();
    auto less = f->binary(Type::of<bool3>(), BinaryOp::LESS, gid, gs);
    auto cond = f->call(Type::of<bool>(), CallOp::ALL, {less});
    auto ret_cond = f->unary(Type::of<bool>(), UnaryOp::NOT, cond);
    auto if_stmt = f->if_(ret_cond);
    f->with(if_stmt->true_branch(), [f] { f->return_(); });
    return new_with_allocator<luisa::shared_ptr<FunctionBuilder>>(std::move(shared_f));
}

void luisa_compute_ast_end_kernel(void *kernel) LUISA_NOEXCEPT {
    auto pf = static_cast<luisa::shared_ptr<FunctionBuilder> *>(kernel);
    auto f = pf->get();
    f->pop_meta(f->body());
    FunctionBuilder::pop(f);
}

void luisa_compute_ast_destroy_function(void *function) LUISA_NOEXCEPT {
    delete_with_allocator(static_cast<luisa::shared_ptr<FunctionBuilder> *>(function));
}

void *luisa_compute_ast_begin_callable() LUISA_NOEXCEPT {
    auto shared_f = luisa::make_shared<FunctionBuilder>(Function::Tag::CALLABLE);
    auto f = shared_f.get();
    FunctionBuilder::push(f);
    f->push_meta(f->body());
    return new_with_allocator<luisa::shared_ptr<FunctionBuilder>>(std::move(shared_f));
}

void luisa_compute_ast_end_callable(void *callable) LUISA_NOEXCEPT {
    auto pf = static_cast<luisa::shared_ptr<FunctionBuilder> *>(callable);
    auto f = pf->get();
    f->pop_meta(f->body());
    FunctionBuilder::pop(f);
}

void *luisa_compute_ast_create_constant_data(const void *t, const void *data, size_t n) LUISA_NOEXCEPT {
    auto view = [type = static_cast<const Type *>(t), data, n]() noexcept -> ConstantData::View {
        switch (type->tag()) {
            case Type::Tag::BOOL:
                return std::span{static_cast<const bool *>(data), n};
            case Type::Tag::FLOAT:
                return std::span{static_cast<const float *>(data), n};
            case Type::Tag::INT:
                return std::span{static_cast<const int *>(data), n};
            case Type::Tag::UINT:
                return std::span{static_cast<const uint *>(data), n};
            case Type::Tag::VECTOR:
                return [dim = type->dimension(), elem = type->element(), data, n]() noexcept -> ConstantData::View {
                    auto vector_view = [dim, data, n]<typename T>() noexcept -> ConstantData::View {
                        if (dim == 2) {
                            return std::span{static_cast<const Vector<T, 2> *>(data), n};
                        } else if (dim == 3) {
                            return std::span{static_cast<const Vector<T, 3> *>(data), n};
                        } else if (dim == 4) {
                            return std::span{static_cast<const Vector<T, 4> *>(data), n};
                        }
                        LUISA_ERROR_WITH_LOCATION(
                            "Invalid matrix dimension: {}.", dim);
                    };
                    switch (elem->tag()) {
                        case Type::Tag::BOOL: return vector_view.operator()<bool>();
                        case Type::Tag::FLOAT: return vector_view.operator()<float>();
                        case Type::Tag::INT: return vector_view.operator()<int>();
                        case Type::Tag::UINT: return vector_view.operator()<uint>();
                        default: [[unlikely]] LUISA_ERROR_WITH_LOCATION(
                            "Invalid vector element in constant data: {}",
                            elem->description());
                    }
                }();
            case Type::Tag::MATRIX:
                return [dim = type->dimension(), data, n]() noexcept -> ConstantData::View {
                    if (dim == 2) {
                        return std::span{static_cast<const float2x2 *>(data), n};
                    } else if (dim == 3) {
                        return std::span{static_cast<const float3x3 *>(data), n};
                    } else if (dim == 4) {
                        return std::span{static_cast<const float4x4 *>(data), n};
                    }
                    LUISA_ERROR_WITH_LOCATION(
                        "Invalid matrix dimension: {}.", dim);
                }();
            default: [[unlikely]] LUISA_ERROR_WITH_LOCATION(
                "Invalid constant data type: {}",
                type->description());
        }
    }();
    return new_with_allocator<ConstantData>(ConstantData::create(view));
}

void luisa_compute_ast_destroy_constant_data(void *data) LUISA_NOEXCEPT {
    return delete_with_allocator(static_cast<ConstantData *>(data));
}

void luisa_compute_ast_set_block_size(uint32_t sx, uint32_t sy, uint32_t sz) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    f->set_block_size(make_uint3(sx, sy, sz));
}

const void *luisa_compute_ast_thread_id() LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->thread_id();
}

const void *luisa_compute_ast_block_id() LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->block_id();
}

const void *luisa_compute_ast_dispatch_id() LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->dispatch_id();
}

const void *luisa_compute_ast_dispatch_size() LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->dispatch_size();
}

const void *luisa_compute_ast_local_variable(const void *t) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->local(static_cast<const Type *>(t));
}

const void *luisa_compute_ast_shared_variable(const void *t) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->shared(static_cast<const Type *>(t));
}

const void *luisa_compute_ast_constant_variable(const void *t, const void *data) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->constant(static_cast<const Type *>(t), *static_cast<const ConstantData *>(data));
}

const void *luisa_compute_ast_buffer_binding(const void *elem_t, uint64_t buffer, size_t offset_bytes) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->buffer_binding(static_cast<const Type *>(elem_t), buffer, offset_bytes);
}

const void *luisa_compute_ast_texture_binding(const void *t, uint64_t texture) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->texture_binding(static_cast<const Type *>(t), texture);
}

const void *luisa_compute_ast_heap_binding(uint64_t heap) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->heap_binding(heap);
}

const void *luisa_compute_ast_accel_binding(uint64_t accel) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->accel_binding(accel);
}

const void *luisa_compute_ast_value_argument(const void *t) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->argument(static_cast<const Type *>(t));
}

const void *luisa_compute_ast_reference_argument(const void *t) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->reference(static_cast<const Type *>(t));
}

const void *luisa_compute_ast_buffer_argument(const void *t) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->buffer(static_cast<const Type *>(t));
}

const void *luisa_compute_ast_texture_argument(const void *t) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->texture(static_cast<const Type *>(t));
}

const void *luisa_compute_ast_heap_argument() LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->heap();
}

const void *luisa_compute_ast_accel_argument() LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->accel();
}

const void *luisa_compute_ast_literal_expr(const void *t, const void *value) LUISA_NOEXCEPT {
    auto v = [type = static_cast<const Type *>(t), value]() noexcept -> LiteralExpr::Value {
        switch (type->tag()) {
            case Type::Tag::BOOL:
                return *static_cast<const bool *>(value);
            case Type::Tag::FLOAT:
                return *static_cast<const float *>(value);
            case Type::Tag::INT:
                return *static_cast<const int *>(value);
            case Type::Tag::UINT:
                return *static_cast<const uint *>(value);
            case Type::Tag::VECTOR:
                return [dim = type->dimension(), elem = type->element(), value]() noexcept -> LiteralExpr::Value {
                    auto vector_view = [dim, value]<typename T>() noexcept -> LiteralExpr::Value {
                        if (dim == 2) {
                            return *static_cast<const Vector<T, 2> *>(value);
                        } else if (dim == 3) {
                            return *static_cast<const Vector<T, 3> *>(value);
                        } else if (dim == 4) {
                            return *static_cast<const Vector<T, 4> *>(value);
                        }
                        LUISA_ERROR_WITH_LOCATION(
                            "Invalid matrix dimension: {}.", dim);
                    };
                    switch (elem->tag()) {
                        case Type::Tag::BOOL: return vector_view.operator()<bool>();
                        case Type::Tag::FLOAT: return vector_view.operator()<float>();
                        case Type::Tag::INT: return vector_view.operator()<int>();
                        case Type::Tag::UINT: return vector_view.operator()<uint>();
                        default: [[unlikely]] LUISA_ERROR_WITH_LOCATION(
                            "Invalid vector element in constant data: {}",
                            elem->description());
                    }
                }();
            case Type::Tag::MATRIX:
                return [dim = type->dimension(), value]() noexcept -> LiteralExpr::Value {
                    if (dim == 2) {
                        return *static_cast<const float2x2 *>(value);
                    } else if (dim == 3) {
                        return *static_cast<const float3x3 *>(value);
                    } else if (dim == 4) {
                        return *static_cast<const float4x4 *>(value);
                    }
                    LUISA_ERROR_WITH_LOCATION(
                        "Invalid matrix dimension: {}.", dim);
                }();
            default: [[unlikely]] LUISA_ERROR_WITH_LOCATION(
                "Invalid constant data type: {}",
                type->description());
        }
    }();
    auto f = FunctionBuilder::current();
    return f->literal(static_cast<const Type *>(t), v);
}

const void *luisa_compute_ast_unary_expr(const void *t, uint32_t op, const void *expr) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->unary(
        static_cast<const Type *>(t),
        static_cast<UnaryOp>(op),
        static_cast<const Expression *>(expr));
}

const void *luisa_compute_ast_binary_expr(const void *t, uint32_t op, const void *lhs, const void *rhs) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->binary(
        static_cast<const Type *>(t),
        static_cast<BinaryOp>(op),
        static_cast<const Expression *>(lhs),
        static_cast<const Expression *>(rhs));
}

const void *luisa_compute_ast_member_expr(const void *t, const void *self, size_t member_id) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->member(
        static_cast<const Type *>(t),
        static_cast<const Expression *>(self),
        member_id);
}

const void *luisa_compute_ast_swizzle_expr(const void *t, const void *self, size_t swizzle_size, uint64_t swizzle_code) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->swizzle(
        static_cast<const Type *>(t),
        static_cast<const Expression *>(self),
        swizzle_size, swizzle_code);
}

const void *luisa_compute_ast_access_expr(const void *t, const void *range, const void *index) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->access(
        static_cast<const Type *>(t),
        static_cast<const Expression *>(range),
        static_cast<const Expression *>(index));
}

const void *luisa_compute_ast_cast_expr(const void *t, uint32_t op, const void *expr) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return f->cast(
        static_cast<const Type *>(t),
        static_cast<CastOp>(op),
        static_cast<const Expression *>(expr));
}

const void *luisa_compute_ast_call_expr(const void *t, uint32_t call_op, const void *custom_callable, const void *args, size_t arg_count) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    auto ret = static_cast<const Type *>(t);
    auto op = static_cast<CallOp>(call_op);
    std::span a{static_cast<Expression const *const *>(args), arg_count};
    if (op == CallOp::CUSTOM) {
        auto callable = Function{static_cast<const luisa::shared_ptr<FunctionBuilder> *>(custom_callable)->get()};
        if (ret != nullptr) {
            return f->call(ret, callable, a);
        }
        f->call(callable, a);
        return nullptr;
    }
    if (ret != nullptr) { return f->call(ret, op, a); }
    f->call(op, a);
    return nullptr;
}

void luisa_compute_ast_break_stmt() LUISA_NOEXCEPT {
    FunctionBuilder::current()->break_();
}

void luisa_compute_ast_continue_stmt() LUISA_NOEXCEPT {
    FunctionBuilder::current()->continue_();
}

void luisa_compute_ast_return_stmt(const void *expr) LUISA_NOEXCEPT {
    FunctionBuilder::current()->return_(static_cast<const Expression *>(expr));
}

void *luisa_compute_ast_if_stmt(const void *cond) LUISA_NOEXCEPT {
    return FunctionBuilder::current()->if_(static_cast<const Expression *>(cond));
}

void *luisa_compute_ast_loop_stmt() LUISA_NOEXCEPT {
    return FunctionBuilder::current()->loop_();
}

void *luisa_compute_ast_switch_stmt(const void *expr) LUISA_NOEXCEPT {
    return FunctionBuilder::current()->switch_(static_cast<const Expression *>(expr));
}

void *luisa_compute_ast_case_stmt(const void *expr) LUISA_NOEXCEPT {
    return FunctionBuilder::current()->case_(static_cast<const Expression *>(expr));
}

void *luisa_compute_ast_default_stmt() LUISA_NOEXCEPT {
    return FunctionBuilder::current()->default_();
}

void *luisa_compute_ast_for_stmt(const void *var, const void *cond, const void *update) LUISA_NOEXCEPT {
    return FunctionBuilder::current()->for_(
        static_cast<const Expression *>(var),
        static_cast<const Expression *>(cond),
        static_cast<const Expression *>(update));
}

void luisa_compute_ast_assign_stmt(uint32_t op, const void *lhs, const void *rhs) LUISA_NOEXCEPT {
    FunctionBuilder::current()->assign(
        static_cast<AssignOp>(op),
        static_cast<const Expression *>(lhs),
        static_cast<const Expression *>(rhs));
}

void luisa_compute_ast_comment(const char *comment) LUISA_NOEXCEPT {
    FunctionBuilder::current()->comment_(comment);
}

void luisa_compute_ast_push_scope(void *scope) LUISA_NOEXCEPT {
    FunctionBuilder::current()->push_scope(static_cast<ScopeStmt *>(scope));
}

void luisa_compute_ast_pop_scope(void *scope) LUISA_NOEXCEPT {
    FunctionBuilder::current()->pop_scope(static_cast<ScopeStmt *>(scope));
}
