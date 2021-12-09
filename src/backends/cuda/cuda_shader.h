//
// Created by Mike on 2021/12/4.
//

#include <span>
#include <memory>

#include <core/basic_types.h>

namespace luisa::compute {
class ShaderDispatchCommand;
}

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAStream;

struct CUDAShader {
    CUDAShader() noexcept = default;
    CUDAShader(CUDAShader &&) noexcept = delete;
    CUDAShader(const CUDAShader &) noexcept = delete;
    CUDAShader &operator=(CUDAShader &&) noexcept = delete;
    CUDAShader &operator=(const CUDAShader &) noexcept = delete;
    virtual ~CUDAShader() noexcept = default;
    [[nodiscard]] static CUDAShader *create(CUDADevice *device, const char *ptx, size_t ptx_size, const char *entry, bool is_raytracing) noexcept;
    static void destroy(CUDAShader *shader) noexcept;
    virtual void launch(CUDAStream *stream, const ShaderDispatchCommand *command) const noexcept = 0;
};

}
