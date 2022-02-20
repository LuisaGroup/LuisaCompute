//
// Created by Mike on 2021/11/16.
//

#pragma once

#include <core/basic_types.h>

namespace luisa::compute::ispc {

using luisa::uint3;

class ISPCModule {

public:
    using function_type = void(
        const void * /* args */,
        const uint /* block_index.x */,
        const uint /* block_index.y */,
        const uint /* block_index.z */,
        const uint /* dispatch_size.x */,
        const uint /* dispatch_size.y */,
        const uint /* dispatch_size.z */);

protected:
    function_type *_f_ptr{nullptr};

public:
    explicit ISPCModule(function_type *f_ptr = nullptr) noexcept : _f_ptr{f_ptr} {}
    void invoke(const void *args, uint3 block_index, uint3 dispatch_size) noexcept {
        _f_ptr(
            args, block_index.x, block_index.y, block_index.z,
            dispatch_size.x, dispatch_size.y, dispatch_size.z);
    }
    virtual ~ISPCModule() noexcept = default;
};

}// namespace luisa::compute::ispc
