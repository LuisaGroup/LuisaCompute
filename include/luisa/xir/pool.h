#pragma once

#include <luisa/core/dll_export.h>

#include <luisa/core/concepts.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class Pool;

class LC_XIR_API PooledObject {

private:
    Pool *_pool;

protected:
    explicit PooledObject(Pool *pool) noexcept : _pool{pool} {}

public:
    virtual ~PooledObject() noexcept = default;
    [[nodiscard]] auto pool() const noexcept { return _pool; }

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
        auto object = luisa::new_with_allocator<T>(this, std::forward<Args>(args)...);
        assert(object->pool() == this && "PooledObject must be created with the correct pool.");
        _objects.emplace_back(object);
        return object;
    }
};

}// namespace luisa::compute::xir
