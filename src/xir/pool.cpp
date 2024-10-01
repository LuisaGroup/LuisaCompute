#include <luisa/xir/pool.h>

namespace luisa::compute::xir {

Pool::Pool(size_t init_cap) noexcept {
    if (init_cap != 0u) {
        _objects.reserve(init_cap);
    }
}

Pool::~Pool() noexcept {
    for (auto object : _objects) {
        luisa::delete_with_allocator(object);
    }
}

}// namespace luisa::compute::xir
