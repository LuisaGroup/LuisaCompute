//
// Created by Mike Smith on 2023/4/20.
//

#include <backends/metal/metal_device.h>
#include <backends/metal/metal_accel.h>

namespace luisa::compute::metal {

MetalAccel::MetalAccel(MetalDevice *device, const AccelOption &option) noexcept
    : _update{device->builtin_update_accel_instances()}, _option{option} {

}

MetalAccel::~MetalAccel() noexcept {
    if (_handle) { _handle->release(); }
    if (_instance_buffer) { _instance_buffer->release(); }
    if (_update_buffer) { _update_buffer->release(); }
}

}// namespace luisa::compute::metal
