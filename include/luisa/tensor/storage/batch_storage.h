#pragma once

#include <luisa/tensor/storage/dense_storage.h>
#include <luisa/tensor/view/batch_storage_view.h>
namespace luisa::compute::tensor {
using BatchStorage = DenseStorage<uint64_t>;
}// namespace luisa::compute::tensor