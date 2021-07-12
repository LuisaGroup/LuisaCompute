//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <runtime/device.h>
#include <runtime/buffer.h>
#include <rtx/ray.h>
#include <rtx/hit.h>

namespace luisa::compute {

class Accel {

private:
    Device::Interface *_device;
    uint64_t _handle;

public:
    ~Accel() noexcept { _device->destroy_accel(_handle); }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] detail::Expr<Hit> trace_closest(detail::Expr<Ray> ray) const noexcept;
    [[nodiscard]] detail::Expr<bool> trace_any(detail::Expr<Ray> ray) const noexcept;
    [[nodiscard]] CommandHandle trace_closest(BufferView<Ray> rays, BufferView<Hit> hits) const noexcept;
    [[nodiscard]] CommandHandle trace_closest(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<Hit> hits) const noexcept;
    [[nodiscard]] CommandHandle trace_closest(BufferView<Ray> rays, BufferView<Hit> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] CommandHandle trace_closest(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<Hit> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] CommandHandle trace_any(BufferView<Ray> rays, BufferView<bool> hits) const noexcept;
    [[nodiscard]] CommandHandle trace_any(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<bool> hits) const noexcept;
    [[nodiscard]] CommandHandle trace_any(BufferView<Ray> rays, BufferView<bool> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] CommandHandle trace_any(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<bool> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] CommandHandle update() const noexcept;
};

}
