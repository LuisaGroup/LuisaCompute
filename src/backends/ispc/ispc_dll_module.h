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

/**
 * @brief DLL moudle of ispc
 * 
 */
class ISPCDLLModule final : public ISPCModule {

private:
    DynamicModule _module;

private:
    explicit ISPCDLLModule(DynamicModule m) noexcept
        : _module{std::move(m)} {
        _f_ptr = _module.function<ISPCModule::function_type>("kernel_main");
    }

public:
    /**
     * @brief load object
     * 
     * @param ctx context
     * @param obj_path object path
     * @return luisa::unique_ptr<ISPCModule> 
     */
    [[nodiscard]] static luisa::unique_ptr<ISPCModule> load(
        const Context &ctx, const std::filesystem::path &obj_path) noexcept;
};

}// namespace luisa::compute::ispc
