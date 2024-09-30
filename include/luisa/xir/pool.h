#pragma once

#include <luisa/core/dll_export.h>

#include <luisa/core/concepts.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

struct LC_XIR_API PooledObject {

    PooledObject() noexcept = default;
    virtual ~PooledObject() noexcept = default;

    // make the object pinned to its memory location
    PooledObject(PooledObject &&) noexcept = delete;
    PooledObject(const PooledObject &) noexcept = delete;
    PooledObject &operator=(PooledObject &&) noexcept = delete;
    PooledObject &operator=(const PooledObject &) noexcept = delete;
};

class LC_XIR_API Pool : public concepts::Noncopyable {

private:
    luisa::vector<PooledObject *> _objects;

public:
    explicit Pool(size_t init_cap = 0u) noexcept;
    ~Pool() noexcept;

public:
    template<typename T, typename... Args>
        requires std::derived_from<T, PooledObject>
    [[nodiscard]] T *create(Args &&...args) {
        auto object = luisa::new_with_allocator<T>(std::forward<Args>(args)...);
        _objects.emplace_back(object);
        return object;
    }
};

}// namespace luisa::compute::xir
