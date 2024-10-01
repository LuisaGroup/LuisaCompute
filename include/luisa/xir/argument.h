#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class LC_XIR_API Argument : public Value {

private:
    bool _by_ref = false;

public:
    explicit Argument(Pool *pool,
                      const Type *type = nullptr,
                      bool by_ref = false,
                      const Name *name = nullptr) noexcept;
    void set_by_ref(bool by_ref) noexcept { _by_ref = by_ref; }
    [[nodiscard]] auto by_ref() const noexcept { return _by_ref; }
};

}// namespace luisa::compute::xir
