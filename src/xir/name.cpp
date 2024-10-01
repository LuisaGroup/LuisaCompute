#include <luisa/xir/name.h>

namespace luisa::compute::xir {

Name::Name(Pool *pool, luisa::string s) noexcept
    : PooledObject{pool}, _s{std::move(s)} {}

}// namespace luisa::compute::xir
