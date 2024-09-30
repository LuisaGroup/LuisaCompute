#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class LC_XIR_API User : public Value {

private:
    luisa::vector<Use *> _operands;

public:
    [[nodiscard]] auto operands() noexcept { return luisa::span{_operands}; }
    [[nodiscard]] auto operands() const noexcept { return luisa::span{_operands}; }
};

}// namespace luisa::compute::xir
