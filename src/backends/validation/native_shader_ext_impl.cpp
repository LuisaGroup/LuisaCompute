#include "native_shader_ext_impl.h"
#include "rw_resource.h"

namespace lc::validation {

// Opaque bookkeeping entry for a native shader instance: it owns no upload
// memory, but it must be registered so that a dispatch referencing it (and its
// destruction) can be tracked like any other resource handle.
class NativeShaderHandle final : public RWResource {
public:
    static constexpr luisa::string_view validation_res_name{"NativeShader"};
    explicit NativeShaderHandle(uint64_t handle) noexcept
        : RWResource{handle, Tag::SHADER, false} {}
};

NativeShaderMetadata NativeShaderExtImpl::load(
    const NativeShaderCompileResult &result,
    luisa::span<const Usage> usage_override) noexcept {
    auto metadata = _native->load(result, usage_override);
    if (metadata.valid()) {
        new NativeShaderHandle{metadata.handle};
    }
    return metadata;
}

void NativeShaderExtImpl::destroy_shader(uint64_t handle) noexcept {
    RWResource::dispose(handle);
    _native->destroy_shader(handle);
}

}// namespace lc::validation
