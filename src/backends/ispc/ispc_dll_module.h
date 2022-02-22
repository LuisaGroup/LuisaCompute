//
// Created by Mike on 2021/11/16.
//

#pragma once

#include <filesystem>

#include <core/dynamic_module.h>
#include <backends/ispc/ispc_module.h>

namespace luisa::compute {
class Context;
}

namespace luisa::compute::ispc {

class ISPCDLLModule final : public ISPCModule {

private:
    DynamicModule _module;

private:
    explicit ISPCDLLModule(DynamicModule m) noexcept
        : _module{std::move(m)} {
        _f_ptr = _module.function<ISPCModule::function_type>("kernel_main");
    }

public:
    [[nodiscard]] static luisa::shared_ptr<ISPCModule> load(
        const Context &ctx, const std::filesystem::path &obj_path) noexcept;
};

}// namespace luisa::compute::ispc
