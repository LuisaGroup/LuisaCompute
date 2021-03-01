//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <runtime/buffer.h>
#include <runtime/texture.h>

namespace luisa::compute {

class KernelArgumentBinding {

};

class KernelLaunchCommand {

public:


private:
    uint3 _block_size;
    uint3 _grid_size;
    uint32_t _kernel_uid;

public:

};

}// namespace luisa::compute
