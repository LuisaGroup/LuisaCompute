#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class LC_XIR_API Shared : public Value {
public:
    using Value::Value;
};

}// namespace luisa::compute::xir
