#include <luisa/core/logging.h>
#include <luisa/core/magic_enum.h>
#include <luisa/ir/ast2ir.h>
#include <luisa/ast/function_builder.h>
#include <luisa/ast/ast2json.h>

namespace luisa::compute {

[[nodiscard]] luisa::shared_ptr<ir::CArc<ir::KernelModule>> AST2IR::build_kernel(Function function) noexcept {
    auto j = to_json(function);
    auto slice = ir::CBoxedSlice<uint8_t>{
        .ptr = reinterpret_cast<uint8_t *>(j.data()),
        .len = j.size(),
        .destructor = nullptr,
    };
    auto f = ir::luisa_compute_ir_ast_json_to_ir_kernel(slice);
    return {luisa::new_with_allocator<ir::CArc<ir::KernelModule>>(f),
            [](ir::CArc<ir::KernelModule> *p) noexcept {
                p->release();
                luisa::delete_with_allocator(p);
            }};
}

[[nodiscard]] luisa::shared_ptr<ir::CArc<ir::CallableModule>> AST2IR::build_callable(Function function) noexcept {
    auto j = to_json(function);
    auto slice = ir::CBoxedSlice<uint8_t>{
        .ptr = reinterpret_cast<uint8_t *>(j.data()),
        .len = j.size(),
        .destructor = nullptr,
    };
    auto f = ir::luisa_compute_ir_ast_json_to_ir_callable(slice);
    return {luisa::new_with_allocator<ir::CArc<ir::CallableModule>>(f),
            [](ir::CArc<ir::CallableModule> *p) noexcept {
                p->release();
                luisa::delete_with_allocator(p);
            }};
}

ir::CArc<ir::Type> AST2IR::build_type(const Type *type) noexcept {
    static luisa::spin_mutex mutex;
    static luisa::unordered_map<const Type *, ir::CArc<ir::Type>> cache;

    std::scoped_lock lock{mutex};
    if (auto iter = cache.find(type); iter != cache.end()) { return iter->second.clone(); }

    auto j = to_json(type);
    auto slice = ir::CBoxedSlice<uint8_t>{
        .ptr = reinterpret_cast<uint8_t *>(j.data()),
        .len = j.size(),
        .destructor = nullptr,
    };
    auto t = ir::luisa_compute_ir_ast_json_to_ir_type(slice);
    return cache.emplace(type, t).first->second.clone();
}

}// namespace luisa::compute
