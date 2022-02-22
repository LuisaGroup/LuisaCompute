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

/**
 * @brief Shader of ISPC
 * 
 */
class ISPCShader {

private:
    luisa::shared_ptr<ISPCModule> _module;
    luisa::unordered_map<uint, size_t> _argument_offsets;
    size_t _argument_buffer_size{};

public:
    /**
     * @brief Construct a new ISPCShader object
     * 
     * @param ctx context
     * @param func kernel of shader
     */
    ISPCShader(const Context &ctx, Function func) noexcept;
    /**
     * @brief Return module
     * 
     * @return const ISPCModule *
     */
    [[nodiscard]] auto module() const noexcept { return _module.get(); }

    /**
     * @brief Get module with shared ownership
     *
     * @return shared_ptr<ISPCModule>
     */
    [[nodiscard]] auto shared_module() const noexcept { return _module; }
    /**
     * @brief Return size of argument buffer
     * 
     * @return size of argument buffer
     */
    [[nodiscard]] auto argument_buffer_size() const noexcept { return _argument_buffer_size; }
    /**
     * @brief Return offset of argument
     * 
     * @param uid argument uid
     * @return offset of argument
     */
    [[nodiscard]] size_t argument_offset(uint uid) const noexcept;
};

}
