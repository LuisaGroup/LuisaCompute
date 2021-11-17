//
// Created by Mike on 2021/11/16.
//

#pragma once

#include <core/basic_types.h>

namespace lc::ispc {

using luisa::uint3;

class Module {

public:
    using function_type = void(
        uint32_t,// blk_cX
        uint32_t,// blk_cY
        uint32_t,// blk_cZ
        uint32_t,// blk_idX
        uint32_t,// blk_idY
        uint32_t,// blk_idZ
        uint32_t,// dsp_cX
        uint32_t,// dsp_cY
        uint32_t,// dsp_cZ
        uint64_t// args
    );

protected:
    function_type *_f_ptr{nullptr};

public:
    explicit Module(function_type *f_ptr = nullptr) noexcept
        : _f_ptr{f_ptr} {}
    void invoke(uint3 blockCount, uint3 blockIdx, uint3 dispatchSize, const void *args) noexcept {
        _f_ptr(
            blockCount.x, blockCount.y, blockCount.z,
            blockIdx.x, blockIdx.y, blockIdx.z,
            dispatchSize.x, dispatchSize.y, dispatchSize.z,
            reinterpret_cast<uint64_t>(args));
    }
    virtual ~Module() noexcept = default;
};

}