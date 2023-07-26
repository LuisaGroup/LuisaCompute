#include "../cuda_command_encoder.h"
#include <luisa/backends/ext/cuda/lcub/cuda_lcub_command.h>

namespace luisa::compute::cuda {
void CudaLCubCommand::accept(luisa::compute::MutableCommandVisitor &visitor) noexcept {
    auto encoder = dynamic_cast<luisa::compute::cuda::CUDACommandEncoder *>(&visitor);
    LUISA_ASSERT(encoder, "cuda command is only allowed in cuda backend - your visitor = {}", typeid(visitor).name());
    auto stream = encoder->stream();
    func(stream->handle());
}
}