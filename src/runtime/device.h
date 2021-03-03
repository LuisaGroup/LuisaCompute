//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <core/memory.h>

namespace luisa::compute {

struct Device {
    [[nodiscard]] virtual uint64_t create_buffer(size_t size_bytes) noexcept = 0;
    [[nodiscard]] virtual uint64_t create_buffer_with_data(size_t size_bytes, const void *data) noexcept = 0;
    virtual void dispose_buffer(uint64_t handle) noexcept = 0;
};

}
