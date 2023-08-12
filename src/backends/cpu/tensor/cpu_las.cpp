#include "cpu_las.h"
#include <luisa/tensor/tensor.h>
#include "cpu_tensor_res.h"

namespace luisa::compute::cpu::tensor {
using namespace luisa::compute::tensor;

CpuLAS::CpuLAS(DeviceInterface &device, uint64_t stream_handle) noexcept
    : _device{device},
      _stream_handle{stream_handle} {}

CpuLAS::~CpuLAS() noexcept {}

CpuLAS::S<BackendTensorRes> CpuLAS::alloc_backend_tensor_res(const DTensor & d) noexcept 
{
    if (d.is_batched()) return make_shared<CblasDenseRes>(d);
    if (d.is_sparse() && d.is_matrix()) return make_shared<CblasSparseMatrixRes>(d);
    else return nullptr; 
}
}// namespace luisa::compute::cpu::tensor