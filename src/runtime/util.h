#pragma once

#include <cstdint>
#include <core/concepts.h>
#include <runtime/image.h>

namespace luisa::compute {

class Stream;
class IUtil {

protected:
    ~IUtil() = default;

public:
    enum class Result : int8_t {
        NotImplemented = -1,
        Success = 0,
        Failed = 1
    };
    static size_t bc_byte_size(Image<float> const &tex) {
        uint2 size = tex.size();
        uint xBlocks = std::max<uint>(1, (size.x + 3) >> 2);
        uint yBlocks = std::max<uint>(1, (size.y + 3) >> 2);
        uint numBlocks = xBlocks * yBlocks;
        static constexpr size_t BLOCK_SIZE = 16;
        return numBlocks * BLOCK_SIZE;
    }
    virtual Result compress_bc6h(Stream &stream, Image<float> const &src, luisa::vector<std::byte> &result) noexcept { return Result::NotImplemented; }
    virtual Result compress_bc7(Stream &stream, Image<float> const &src, luisa::vector<std::byte> &result, float alpha_importance) noexcept { return Result::NotImplemented; }
    virtual Result check_builtin_shader() noexcept { return Result::NotImplemented; }
};

}// namespace luisa::compute
