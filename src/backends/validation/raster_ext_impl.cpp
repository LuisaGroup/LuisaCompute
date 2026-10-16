#include "raster_ext_impl.h"
#include "raster_shader.h"
#include "rw_resource.h"
#include "depth_buffer.h"
#include <luisa/core/logging.h>
namespace lc::validation {

namespace {

void append_root_arguments(
    Function stage,
    luisa::vector<RasterShader::RootArgument> &arguments) noexcept {
    auto variables = stage.arguments();
    auto bindings = stage.bound_arguments();
    LUISA_ASSERT(!variables.empty() && bindings.size() == variables.size(),
                 "Invalid raster-stage argument metadata.");
    arguments.reserve(arguments.size() + variables.size() - 1u);
    for (auto i = 1u; i < variables.size(); i++) {
        auto is_bound = !luisa::holds_alternative<luisa::monostate>(bindings[i]);
        arguments.emplace_back(RasterShader::RootArgument{
            .binding = bindings[i],
            .usage = stage.variable_usage(variables[i].uid()),
            .is_bound = is_bound});
    }
}

}// namespace

ResourceCreationInfo RasterExtImpl::create_raster_shader(
    const MeshFormat &mesh_format,
    Function vert,
    Function pixel,
    const ShaderOption &shader_option) noexcept {
    luisa::vector<RasterShader::RootArgument> root_arguments;
    append_root_arguments(vert, root_arguments);
    append_root_arguments(pixel, root_arguments);
    auto res = _impl->create_raster_shader(mesh_format, vert, pixel, shader_option);
    if (res.valid()) {
        new RasterShader{res.handle, std::move(root_arguments), false};
    }
    return res;
}

ResourceCreationInfo RasterExtImpl::load_raster_shader(
    luisa::span<Type const *const> types,
    luisa::string_view ser_path) noexcept {
    auto res = _impl->load_raster_shader(types, ser_path);
    if (res.valid()) {
        luisa::vector<RasterShader::RootArgument> root_arguments;
        root_arguments.resize(types.size());
        for (auto &argument : root_arguments) {
            argument.usage = Usage::READ_WRITE;
        }
        new RasterShader{res.handle, std::move(root_arguments), true};
    }
    return res;
}

void RasterExtImpl::destroy_raster_shader(uint64_t handle) noexcept {
    _impl->destroy_raster_shader(handle);
    RWResource::dispose(handle);
}

// depth buffer
ResourceCreationInfo RasterExtImpl::create_depth_buffer(DepthFormat format, uint width, uint height) noexcept {
    auto buffer = _impl->create_depth_buffer(format, width, height);
    new DepthBuffer(buffer.handle);
    return buffer;
}
void RasterExtImpl::destroy_depth_buffer(uint64_t handle) noexcept {
    _impl->destroy_depth_buffer(handle);
    RWResource::dispose(handle);

}
}// namespace lc::validation
