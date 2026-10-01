#pragma once

namespace luisa::compute::tile::bridge::tirx::detail {

// Supply only absent TVM CUDA callbacks for native C++ clients. Existing Python
// or application registrations remain authoritative and are never replaced.
void initialize_native_cuda_codegen();

}// namespace luisa::compute::tile::bridge::tirx::detail
