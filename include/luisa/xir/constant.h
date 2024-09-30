#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class LC_XIR_API Constant : public Value {

private:
    luisa::vector<std::byte> _data;

public:

};

}// namespace luisa::compute::xir
