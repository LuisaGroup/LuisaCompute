//
// Created by Mike Smith on 2021/3/13.
//

#pragma once

namespace luisa::compute {

enum Usage : uint16_t {
    NONE = 0,
    READ = 0x01,
    WRITE = 0x02,
    READ_WRITE = READ | WRITE
};

}
