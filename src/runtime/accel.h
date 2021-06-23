//
// Created by Mike Smith on 2021/5/24.
//

#pragma once

#include <core/concepts.h>
#include <runtime/command.h>
#include <runtime/device.h>

namespace luisa::compute {

class Geometry {

private:
    Device::Interface *_device;
    uint64_t _handle;

};

class Accel {

private:
    Device::Interface *_device;
    uint64_t _handle;

public:


};

}
