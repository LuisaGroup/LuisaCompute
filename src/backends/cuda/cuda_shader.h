//
// Created by Mike on 2021/12/4.
//

#pragma once

#include <span>
#include <memory>

#include <core/basic_types.h>

namespace luisa::compute {
class ShaderDispatchCommand;
}

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAStream;
class CUDACommandEncoder;

/**
 * @brief Shader on CUDA
 * 
 */
struct CUDAShader {
    CUDAShader() noexcept = default;
    CUDAShader(CUDAShader &&) noexcept = delete;
    CUDAShader(const CUDAShader &) noexcept = delete;
    CUDAShader &operator=(CUDAShader &&) noexcept = delete;
    CUDAShader &operator=(const CUDAShader &) noexcept = delete;
    virtual ~CUDAShader() noexcept = default;
    /**
     * @brief Create a shader object from code
     * 
     * @param device CUDADevice
     * @param ptx kernel code
     * @param ptx_size code size
     * @param entry name of function
     * @param is_raytracing is raytracing
     * @return CUDAShader* 
     */
    [[nodiscard]] static CUDAShader *create(CUDADevice *device, const char *ptx, size_t ptx_size, const char *entry, bool is_raytracing) noexcept;
    /**
     * @brief Destroy a CUDAShader
     * 
     * @param shader 
     */
    static void destroy(CUDAShader *shader) noexcept;
    /**
     * @brief Launch command to stream
     * 
     * @param stream 
     * @param command 
     */
    virtual void launch(CUDACommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept = 0;
};

}
