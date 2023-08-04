#pragma once

#include "batch_storage_view.h"

namespace luisa::compute::tensor {
class BatchDesc {
public:
    int _batch_count;
    int _batch_stride = 0; // only for strided batch
};

class BatchView {
public:
    BatchStorageView storage;
    BatchDesc desc;
};

}// namespace luisa::compute::tensor