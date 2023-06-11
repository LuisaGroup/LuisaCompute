#pragma once
#include "ref_counter.h"
#include <luisa/runtime/device.h>
#include <luisa/vstl/meta_lib.h>
using namespace luisa::compute;
class ManagedDevice {
public:
    Device device;
    bool valid;
    ManagedDevice(Device &&device) noexcept;
    ManagedDevice(ManagedDevice &&v) noexcept;
    ManagedDevice(ManagedDevice const &) = delete;
    ~ManagedDevice() noexcept;
};
