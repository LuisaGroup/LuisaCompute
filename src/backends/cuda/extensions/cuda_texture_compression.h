#pragma once

#include <cuda.h>
#include <luisa/backends/ext/tex_compress_ext.h>

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAEvent;

class CUDATexCompressExt final : public TexCompressExt {

private:
    std::mutex _mutex;
    CUDADevice *_device;
    CUevent _event;

private:
    void _compress(Stream &stream,
                   const ImageView<float> &src,
                   const BufferView<uint> &result,
                   float alpha_importance,
                   PixelFormat target_format) noexcept;

public:
    explicit CUDATexCompressExt(CUDADevice *device) noexcept;
    ~CUDATexCompressExt() noexcept;
    Result compress_bc6h(Stream &stream,
                         const ImageView<float> &src,
                         const BufferView<uint> &result) noexcept override;
    Result compress_bc7(Stream &stream,
                        const ImageView<float> &src,
                        const BufferView<uint> &result,
                        float alpha_importance) noexcept override;
    Result check_builtin_shader() noexcept override;
};

}// namespace luisa::compute::cuda
