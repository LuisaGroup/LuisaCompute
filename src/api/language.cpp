#include <ast/function_builder.h>
#include <api/language.h>

using namespace luisa;
using namespace luisa::compute;

using luisa::compute::detail::FunctionBuilder;

LCKernel luisa_compute_ast_begin_kernel() LUISA_NOEXCEPT {
    auto shared_f = luisa::make_shared<FunctionBuilder>(Function::Tag::KERNEL);
    auto f = shared_f.get();
    FunctionBuilder::push(f);
    f->push_scope(f->body());
    // f->push_meta(f->body());
    auto gid = f->dispatch_id();
    auto gs = f->dispatch_size();
    auto less = f->binary(Type::of<bool3>(), BinaryOp::LESS, gid, gs);
    auto cond = f->call(Type::of<bool>(), CallOp::ALL, {less});
    auto if_stmt = f->if_(cond);
    f->push_scope(if_stmt->true_branch());
    return (LCKernel)new_with_allocator<luisa::shared_ptr<FunctionBuilder>>(std::move(shared_f));
}

void luisa_compute_ast_end_kernel(LCKernel kernel) LUISA_NOEXCEPT {
    auto pf = reinterpret_cast<luisa::shared_ptr<FunctionBuilder> *>(kernel);
    auto f = pf->get();
    f->pop_scope(nullptr);
    f->pop_scope(nullptr);
    // f->pop_meta(f->body());
    FunctionBuilder::pop(f);
}

void luisa_compute_ast_destroy_function(LCFunction function) LUISA_NOEXCEPT {
    delete_with_allocator(reinterpret_cast<luisa::shared_ptr<FunctionBuilder> *>(function));
}

LCCallable luisa_compute_ast_begin_callable() LUISA_NOEXCEPT {
    auto shared_f = luisa::make_shared<FunctionBuilder>(Function::Tag::CALLABLE);
    auto f = shared_f.get();
    FunctionBuilder::push(f);
    f->push_scope(f->body());
    return (LCCallable)new_with_allocator<luisa::shared_ptr<FunctionBuilder>>(std::move(shared_f));
}

void luisa_compute_ast_end_callable(LCCallable callable) LUISA_NOEXCEPT {
    auto pf = reinterpret_cast<luisa::shared_ptr<FunctionBuilder> *>(callable);
    auto f = pf->get();
    f->pop_scope(f->body());
    FunctionBuilder::pop(f);
}


LUISA_EXPORT_API LCConstantData luisa_compute_ast_create_constant_data(LCType t, void * data, size_t n) LUISA_NOEXCEPT {
    auto view = [type = reinterpret_cast<const Type *>(t), data, n]() noexcept -> ConstantData::View {
        switch (type->tag()) {
            case Type::Tag::BOOL:
                return luisa::span{reinterpret_cast<const bool *>(data), n};
            case Type::Tag::FLOAT:
                return luisa::span{reinterpret_cast<const float *>(data), n};
            case Type::Tag::INT:
                return luisa::span{reinterpret_cast<const int *>(data), n};
            case Type::Tag::UINT:
                return luisa::span{reinterpret_cast<const uint *>(data), n};
            case Type::Tag::VECTOR:
                return [dim = type->dimension(), elem = type->element(), data, n]() noexcept -> ConstantData::View {
                    auto vector_view = [dim, data, n]<typename T>() noexcept -> ConstantData::View {
                        if (dim == 2) {
                            return luisa::span{reinterpret_cast<const Vector<T, 2> *>(data), n};
                        } else if (dim == 3) {
                            return luisa::span{reinterpret_cast<const Vector<T, 3> *>(data), n};
                        } else if (dim == 4) {
                            return luisa::span{reinterpret_cast<const Vector<T, 4> *>(data), n};
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
                        return luisa::span{reinterpret_cast<const float2x2 *>(data), n};
                    } else if (dim == 3) {
                        return luisa::span{reinterpret_cast<const float3x3 *>(data), n};
                    } else if (dim == 4) {
                        return luisa::span{reinterpret_cast<const float4x4 *>(data), n};
                    }
                    LUISA_ERROR_WITH_LOCATION(
                        "Invalid matrix dimension: {}.", dim);
                }();
            default: [[unlikely]] LUISA_ERROR_WITH_LOCATION(
                "Invalid constant data type: {}",
                type->description());
        }
    }();
    return (LCConstantData)new_with_allocator<ConstantData>(ConstantData::create(view));
}
LUISA_EXPORT_API void luisa_compute_ast_destroy_constant_data(LCConstantData data) LUISA_NOEXCEPT {
    return delete_with_allocator(reinterpret_cast<ConstantData *>(data));
}
LUISA_EXPORT_API void luisa_compute_ast_set_block_size(uint32_t sx, uint32_t sy, uint32_t sz) LUISA_NOEXCEPT {
     auto f = FunctionBuilder::current();
    f->set_block_size(make_uint3(sx, sy, sz));
}

LCExpression luisa_compute_ast_thread_id() LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->thread_id();
}

LCExpression luisa_compute_ast_block_id() LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->block_id();
}

LCExpression luisa_compute_ast_dispatch_id() LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->dispatch_id();
}

LCExpression luisa_compute_ast_dispatch_size() LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->dispatch_size();
}



LCExpression luisa_compute_ast_local_variable(LCType *t) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->local(reinterpret_cast<const Type *>(t));
}

LCExpression luisa_compute_ast_shared_variable(LCType t) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->shared(reinterpret_cast<const Type *>(t));
}

LCExpression luisa_compute_ast_constant_variable(LCType t, const void *data) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->constant(reinterpret_cast<const Type *>(t), *reinterpret_cast<const ConstantData *>(data));
}


LCExpression luisa_compute_ast_buffer_binding(const void *elem_t, uint64_t buffer, size_t offset_bytes,size_t size_bytes) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->buffer_binding(reinterpret_cast<const Type *>(elem_t), buffer, offset_bytes, size_bytes);
}

LCExpression luisa_compute_ast_texture_binding(LCType t, uint64_t texture, uint32_t level) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->texture_binding(reinterpret_cast<const Type *>(t), texture, level);
}

LCExpression luisa_compute_ast_bindless_array_binding(uint64_t array) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->bindless_array_binding(array);
}

LCExpression luisa_compute_ast_accel_binding(uint64_t accel) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->accel_binding(accel);
}

LCExpression luisa_compute_ast_value_argument(LCType t) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->argument(reinterpret_cast<const Type *>(t));
}

LCExpression luisa_compute_ast_reference_argument(LCType t) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->reference(reinterpret_cast<const Type *>(t));
}

LCExpression luisa_compute_ast_buffer_argument(LCType t) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->buffer(reinterpret_cast<const Type *>(t));
}

LCExpression luisa_compute_ast_texture_argument(LCType t) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->texture(reinterpret_cast<const Type *>(t));
}

LCExpression luisa_compute_ast_bindless_array_argument() LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->bindless_array();
}

LCExpression luisa_compute_ast_accel_argument() LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->accel();
}

LCExpression luisa_compute_ast_literal_expr(LCType t, const void *value, const char *meta_value) LUISA_NOEXCEPT {
    auto v = [type = reinterpret_cast<const Type *>(t), value, meta_value]() noexcept -> LiteralExpr::Value {
        switch (type->tag()) {
            case Type::Tag::BOOL:
                return *reinterpret_cast<const bool *>(value);
            case Type::Tag::FLOAT:
                return *reinterpret_cast<const float *>(value);
            case Type::Tag::INT:
                return *reinterpret_cast<const int *>(value);
            case Type::Tag::UINT:
                return *reinterpret_cast<const uint *>(value);
            case Type::Tag::VECTOR:
                return [dim = type->dimension(), elem = type->element(), value]() noexcept -> LiteralExpr::Value {
                    auto vector_view = [dim, value]<typename T>() noexcept -> LiteralExpr::Value {
                        if (dim == 2) {
                            return *reinterpret_cast<const Vector<T, 2> *>(value);
                        } else if (dim == 3) {
                            return *reinterpret_cast<const Vector<T, 3> *>(value);
                        } else if (dim == 4) {
                            return *reinterpret_cast<const Vector<T, 4> *>(value);
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
                        return *reinterpret_cast<const float2x2 *>(value);
                    } else if (dim == 3) {
                        return *reinterpret_cast<const float3x3 *>(value);
                    } else if (dim == 4) {
                        return *reinterpret_cast<const float4x4 *>(value);
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
    return (LCExpression)f->literal(reinterpret_cast<const Type *>(t), v);
}

LCExpression luisa_compute_ast_unary_expr(LCType t, LCUnaryOp op, const LCExpression expr) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->unary(
        reinterpret_cast<const Type *>(t),
        static_cast<UnaryOp>(op),
        reinterpret_cast<const Expression *>(expr));
}

LCExpression luisa_compute_ast_binary_expr(LCType t, LCBinaryOp op, const void *lhs, const void *rhs) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->binary(
        reinterpret_cast<const Type *>(t),
        static_cast<BinaryOp>(op),
        reinterpret_cast<const Expression *>(lhs),
        reinterpret_cast<const Expression *>(rhs));
}

LCExpression luisa_compute_ast_member_expr(LCType t, LCExpression self, size_t member_id) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->member(
        reinterpret_cast<const Type *>(t),
        reinterpret_cast<const Expression *>(self),
        member_id);
}

LCExpression luisa_compute_ast_swizzle_expr(LCType t, LCExpression self, size_t swizzle_size, uint64_t swizzle_code) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->swizzle(
        reinterpret_cast<const Type *>(t),
        reinterpret_cast<const Expression *>(self),
        swizzle_size, swizzle_code);
}

LCExpression luisa_compute_ast_access_expr(LCType t, const void *range, const void *index) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->access(
        reinterpret_cast<const Type *>(t),
        reinterpret_cast<const Expression *>(range),
        reinterpret_cast<const Expression *>(index));
}

LCExpression luisa_compute_ast_cast_expr(LCType t, LCCastOp op, const LCExpression expr) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    return (LCExpression)f->cast(
        reinterpret_cast<const Type *>(t),
        static_cast<CastOp>(op),
        reinterpret_cast<const Expression *>(expr));
}

LCExpression luisa_compute_ast_call_expr(LCType t, LCCallOp call_op, const void *custom_callable, const void *args, size_t arg_count) LUISA_NOEXCEPT {
    auto f = FunctionBuilder::current();
    auto ret = reinterpret_cast<const Type *>(t);
    auto op = static_cast<CallOp>(call_op);
    luisa::span a{reinterpret_cast<Expression const *const *>(args), arg_count};
    if (op == CallOp::CUSTOM) {
        auto callable = Function{reinterpret_cast<const luisa::shared_ptr<FunctionBuilder> *>(custom_callable)->get()};
        if (ret != nullptr) {
            return (LCExpression)f->call(ret, callable, a);
        }
        f->call(callable, a);
        return nullptr;
    }
    if (ret != nullptr) { return (LCExpression)f->call(ret, op, a); }
    f->call(op, a);
    return nullptr;
}


void luisa_compute_ast_break_stmt() LUISA_NOEXCEPT {
    FunctionBuilder::current()->break_();
}

void luisa_compute_ast_continue_stmt() LUISA_NOEXCEPT {
    FunctionBuilder::current()->continue_();
}

void luisa_compute_ast_return_stmt(const LCExpression expr) LUISA_NOEXCEPT {
    FunctionBuilder::current()->return_(reinterpret_cast<const Expression *>(expr));
}

LCStmt luisa_compute_ast_if_stmt(const LCExpression cond) LUISA_NOEXCEPT {
    return (LCStmt)FunctionBuilder::current()->if_(reinterpret_cast<const Expression *>(cond));
}

LCStmt luisa_compute_ast_loop_stmt() LUISA_NOEXCEPT {
    return (LCStmt)FunctionBuilder::current()->loop_();
}

LCStmt luisa_compute_ast_switch_stmt(const LCExpression expr) LUISA_NOEXCEPT {
    return (LCStmt)FunctionBuilder::current()->switch_(reinterpret_cast<const Expression *>(expr));
}

LCStmt luisa_compute_ast_case_stmt(const LCExpression expr) LUISA_NOEXCEPT {
    return (LCStmt)FunctionBuilder::current()->case_(reinterpret_cast<const Expression *>(expr));
}

LCStmt luisa_compute_ast_default_stmt() LUISA_NOEXCEPT {
    return (LCStmt)FunctionBuilder::current()->default_();
}

LCStmt luisa_compute_ast_for_stmt(const LCExpression var, const LCExpression cond, const LCExpression update) LUISA_NOEXCEPT {
    return (LCStmt)FunctionBuilder::current()->for_(
        reinterpret_cast<const Expression *>(var),
        reinterpret_cast<const Expression *>(cond),
        reinterpret_cast<const Expression *>(update));
}

void luisa_compute_ast_assign_stmt(const LCExpression lhs, const LCExpression rhs) LUISA_NOEXCEPT {
    FunctionBuilder::current()->assign(
        reinterpret_cast<const Expression *>(lhs),
        reinterpret_cast<const Expression *>(rhs));
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

// void *luisa_compute_ast_meta_stmt(const char *meta_expr) LUISA_NOEXCEPT {
//     return FunctionBuilder::current()->meta(meta_expr);
// }

LCStmt luisa_compute_ast_if_stmt_true_scope(void *stmt) LUISA_NOEXCEPT {
    return (LCStmt)static_cast<IfStmt *>(stmt)->true_branch();
}

LCStmt luisa_compute_ast_if_stmt_false_scope(void *stmt) LUISA_NOEXCEPT {
    return (LCStmt)static_cast<IfStmt *>(stmt)->false_branch();
}

LCStmt luisa_compute_ast_loop_stmt_scope(void *stmt) LUISA_NOEXCEPT {
    return (LCStmt)static_cast<LoopStmt *>(stmt)->body();
}

LCStmt luisa_compute_ast_switch_stmt_scope(void *stmt) LUISA_NOEXCEPT {
    return (LCStmt)static_cast<SwitchStmt *>(stmt)->body();
}

LCStmt luisa_compute_ast_switch_case_stmt_scope(void *stmt) LUISA_NOEXCEPT {
    return (LCStmt)static_cast<SwitchCaseStmt *>(stmt)->body();
}

LCStmt luisa_compute_ast_switch_default_stmt_scope(void *stmt) LUISA_NOEXCEPT {
    return (LCStmt)static_cast<SwitchDefaultStmt *>(stmt)->body();
}

LCStmt luisa_compute_ast_for_stmt_scope(void *stmt) LUISA_NOEXCEPT {
    return (LCStmt)static_cast<ForStmt *>(stmt)->body();
}

// void *luisa_compute_ast_meta_stmt_scope(void *stmt) LUISA_NOEXCEPT {
//     return static_cast<MetaStmt *>(stmt)->scope();
// }

// void luisa_compute_ast_push_meta(void *meta) LUISA_NOEXCEPT {
//     FunctionBuilder::current()->push_meta(static_cast<MetaStmt *>(meta));
// }

// void luisa_compute_ast_pop_meta(void *meta) LUISA_NOEXCEPT {
//     FunctionBuilder::current()->pop_meta(static_cast<MetaStmt *>(meta));
// }
