//
// Created by Mike Smith on 2021/6/30.
//

#pragma once

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <core/first_fit.h>

namespace luisa::compute::metal {

class MetalBufferView {

private:
    id<MTLBuffer> _handle;
    FirstFit::Node *_node;

public:
    MetalBufferView(id<MTLBuffer> handle, FirstFit::Node *node) noexcept
        : _handle{handle}, _node{node} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _node == nullptr ? 0u : _node->offset(); }
    [[nodiscard]] auto size() const noexcept { return _node == nullptr ? _handle.length : _node->size(); }
    [[nodiscard]] auto is_pooled() const noexcept { return _node != nullptr; }
    [[nodiscard]] auto node() const noexcept { return _node; }
};

}// namespace luisa::compute::metal
