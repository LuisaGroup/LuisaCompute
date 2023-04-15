//
// Created by Mike Smith on 2023/4/15.
//

#pragma once

#include <ast/function.h>

namespace luisa::compute::metal {

class MetalDevice;

class MetalCompiler {

public:
    explicit MetalCompiler(const MetalDevice *device) noexcept;
};

}// namespace luisa::compute::metal
