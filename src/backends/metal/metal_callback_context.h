#pragma once

#include <luisa/core/pool.h>
#include <luisa/core/stl/functional.h>

namespace luisa::compute::metal {

struct MetalCallbackContext {
    virtual void recycle() noexcept = 0;
    virtual ~MetalCallbackContext() noexcept = default;
};

class FunctionCallbackContext : public MetalCallbackContext {

private:
    luisa::move_only_function<void()> _function;

private:
    using Self = FunctionCallbackContext;
    [[nodiscard]] static Pool<Self, true, true> &_object_pool() noexcept;

public:
    template<typename F>
    explicit FunctionCallbackContext(F &&f) noexcept
        : _function{std::forward<F>(f)} {}
    ~FunctionCallbackContext() noexcept override = default;

    FunctionCallbackContext(FunctionCallbackContext &&) = default;
    FunctionCallbackContext(const FunctionCallbackContext &) = delete;

    template<typename F>
    [[nodiscard]] static auto create(F &&f) noexcept {
        return _object_pool().create(std::forward<F>(f));
    }

    void recycle() noexcept override;
};

}// namespace luisa::compute::metal

