//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <core/memory.h>

namespace luisa::compute {

struct Statement;
struct Expression;

class Function {

public:
    enum struct Tag {
        KERNEL,
        DEVICE
    };

private:
    Arena _arena;

public:

    
};

}// namespace luisa
