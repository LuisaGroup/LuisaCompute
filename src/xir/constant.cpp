#include <luisa/core/logging.h>
#include <luisa/xir/constant.h>

namespace luisa::compute::xir {

Constant::Constant(Pool *pool, ConstantData data, const Name *name) noexcept
    : Value{pool, data.type(), name} { set_data(data); }

void Constant::set_data(ConstantData data) noexcept {
    LUISA_DEBUG_ASSERT(!data || data.type() == type(),
                       "Constant data type mismatch: {} vs {}",
                       data.type()->description(),
                       type()->description());
    _data = data;
}

}// namespace luisa::compute::xir
