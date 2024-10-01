#pragma once

#include <luisa/ast/constant_data.h>
#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class LC_XIR_API Constant : public Value {

private:
    ConstantData _data;

public:
    Constant() noexcept = default;
    explicit Constant(ConstantData data, const Name *name = nullptr) noexcept;
    void set_data(ConstantData data) noexcept;
    [[nodiscard]] auto data() const noexcept { return _data; }
};

}// namespace luisa::compute::xir
