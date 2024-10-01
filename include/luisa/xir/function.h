#pragma once

#include <luisa/xir/pool.h>

namespace luisa::compute::xir {

class LC_XIR_API Function : public PooledObject {

public:
    explicit Function(Pool *pool) noexcept : PooledObject{pool} {}

};

}// namespace luisa::compute::xir
