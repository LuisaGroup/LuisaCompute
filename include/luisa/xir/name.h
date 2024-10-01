#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/xir/pool.h>

namespace luisa::compute::xir {

class LC_XIR_API Name : public PooledObject {

private:
    luisa::string _s;

public:
    explicit Name(Pool *pool, luisa::string s = {}) noexcept;
    [[nodiscard]] auto string() const noexcept { return _s; }
    [[nodiscard]] const auto &operator*() const noexcept { return _s; }
    [[nodiscard]] const auto *operator->() const noexcept { return &_s; }
};

}// namespace luisa::compute::xir
