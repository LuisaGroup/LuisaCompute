//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <ast/function.h>
#include <backends/ispc/ispc_module.h>

namespace luisa::compute {
class Context;
}

namespace luisa::compute::ispc {

class ISPCShader {

private:
    luisa::shared_ptr<ISPCModule> _module;
    luisa::unordered_map<uint, size_t> _argument_offsets;
    size_t _argument_buffer_size{};

public:
    ISPCShader(const Context &ctx, Function func) noexcept;
    [[nodiscard]] auto module() const noexcept { return _module.get(); }
    [[nodiscard]] auto shared_module() const noexcept { return _module; }
    [[nodiscard]] auto argument_buffer_size() const noexcept { return _argument_buffer_size; }
    [[nodiscard]] size_t argument_offset(uint uid) const noexcept;
};

}
