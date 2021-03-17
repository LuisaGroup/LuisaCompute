//
// Created by Mike Smith on 2020/12/2.
//

#include <runtime/device.h>
#include <runtime/stream.h>

namespace luisa::compute {

Stream Device::create_stream() noexcept {
    return Stream{this, _create_stream()};
}

}
