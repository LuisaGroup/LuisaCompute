#pragma once

#include <cstdint>

namespace luisa::compute {

struct Argument {

    enum struct Tag {
        BUFFER,
        TEXTURE,
        UNIFORM,
        BINDLESS_ARRAY,
        ACCEL
    };

    struct Buffer {
        uint64_t handle;
        size_t offset;
        size_t size;
    };

    struct Texture {
        uint64_t handle;
        uint32_t level;
    };

    struct Uniform {
        size_t offset;
        size_t size;
    };

    struct BindlessArray {
        uint64_t handle;
    };

    struct Accel {
        uint64_t handle;
    };

    Tag tag;
    union {
        Buffer buffer;
        Texture texture;
        Uniform uniform;
        BindlessArray bindless_array;
        Accel accel;
    };
};

}// namespace luisa::compute

