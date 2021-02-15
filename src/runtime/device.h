//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <core/memory.h>

namespace luisa::compute {

class Device {

protected:
    // for buffer
    template<typename T> friend class Buffer;
    virtual void _dispose_buffer(uint64_t handle) noexcept = 0;
    [[nodiscard]] virtual uint64_t _create_buffer(size_t size_bytes) noexcept = 0;
    [[nodiscard]] virtual uint64_t _create_buffer_with_data(size_t size_bytes, const void *data) noexcept = 0;

public:

};

}
