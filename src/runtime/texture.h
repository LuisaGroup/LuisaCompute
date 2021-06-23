//
// Created by Mike Smith on 2021/4/7.
//

#pragma once

#include <core/mathematics.h>

namespace luisa::compute {

class TextureView;

namespace detail {

[[nodiscard]] auto valid_mipmap_levels(uint width, uint height, uint requested_levels) noexcept {
    auto rounded_size = next_pow2(std::min(width, height));
    auto max_levels = static_cast<uint>(std::log2(rounded_size));
    return requested_levels == 0u
               ? max_levels
               : std::min(requested_levels, max_levels);
}

}// namespace detail

class Texture {

};

class TextureView {

};

class TextureHeap {
};

}// namespace luisa::compute
