#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class Shared : public Value {
public:
    using Value::Value;
};

}// namespace luisa::compute::xir
