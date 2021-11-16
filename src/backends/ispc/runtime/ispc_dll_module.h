//
// Created by Mike on 2021/11/16.
//

#pragma once

#include <filesystem>

#include <core/dynamic_module.h>
#include <backends/ispc/runtime/ispc_module.h>

namespace lc::ispc {

class DLLModule final : public Module {

private:
    DynamicModule _module;

private:
    explicit DLLModule(DynamicModule m) noexcept
        : _module{std::move(m)} {
        _f_ptr = _module.function<Module::function_type>("run");
    }

public:
    [[nodiscard]] static luisa::unique_ptr<Module> load(
        const Context &ctx, const std::filesystem::path &obj_path) noexcept;
};

}
