#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/xir/pool.h>

namespace luisa::compute::xir {

class Name : public PooledObject {

private:
    luisa::string _s;

public:
    [[nodiscard]] const auto &operator*() const noexcept { return _s; }
    [[nodiscard]] const auto *operator->() const noexcept { return &_s; }
};

}// namespace luisa::compute::xir
