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

LCType luisa_compute_type_from_description(const char* desc) LUISA_NOEXCEPT {
    return LCType(Type::from(desc));
}

LUISA_EXPORT_API LCConstantData luisa_compute_ast_create_constant_data(LCType t, void * data, size_t n) LUISA_NOEXCEPT {
    auto view = [type = reinterpret_cast<const Type *>(t), data, n]() noexcept -> ConstantData::View {
        switch (type->tag()) {
            case Type::Tag::BOOL:
                return luisa::span{static_cast<const bool *>(data), n};
            case Type::Tag::FLOAT:
                return luisa::span{static_cast<const float *>(data), n};
            case Type::Tag::INT:
                return luisa::span{static_cast<const int *>(data), n};
            case Type::Tag::UINT:
                return luisa::span{static_cast<const uint *>(data), n};
            case Type::Tag::VECTOR:
                return [dim = type->dimension(), elem = type->element(), data, n]() noexcept -> ConstantData::View {
                    auto vector_view = [dim, data, n]<typename T>() noexcept -> ConstantData::View {
                        if (dim == 2) {
                            return luisa::span{static_cast<const Vector<T, 2> *>(data), n};
                        } else if (dim == 3) {
                            return luisa::span{static_cast<const Vector<T, 3> *>(data), n};
                        } else if (dim == 4) {
                            return luisa::span{static_cast<const Vector<T, 4> *>(data), n};
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
                        return luisa::span{static_cast<const float2x2 *>(data), n};
                    } else if (dim == 3) {
                        return luisa::span{static_cast<const float3x3 *>(data), n};
                    } else if (dim == 4) {
                        return luisa::span{static_cast<const float4x4 *>(data), n};
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
    return (LCExpression)f->constant(reinterpret_cast<const Type *>(t), *static_cast<const ConstantData *>(data));
}