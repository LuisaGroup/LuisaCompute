#include <luisa/ir/transform.h>

namespace luisa::compute {

namespace detail {

template<typename M>
void perform_ir_transform(M *m, luisa::span<const luisa::string> transforms) noexcept {
    auto pipeline = ir::luisa_compute_ir_transform_pipeline_new();
    for (auto &&transform : transforms) {
        ir::luisa_compute_ir_transform_pipeline_add_transform(pipeline, transform.c_str());
    }
    auto converted_module = ir::luisa_compute_ir_transform_pipeline_transform(pipeline, m->module);
    ir::luisa_compute_ir_transform_pipeline_destroy(pipeline);
    m->module = converted_module;
}

}// namespace detail

void transform_ir_callable_module(ir::CallableModule *m, luisa::span<const luisa::string> transforms) noexcept {
    compute::detail::perform_ir_transform(m, transforms);
}

void transform_ir_kernel_module(ir::KernelModule *m, luisa::span<const luisa::string> transforms) noexcept {
    compute::detail::perform_ir_transform(m, transforms);
}

void transform_ir_kernel_module_auto(ir::KernelModule *m) noexcept {
    ir::luisa_compute_ir_transform_auto(m->module);
}

}// namespace luisa::compute
