#pragma once
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
namespace toolhub::directx {
class TopAccel;
using BindProperty = vstd::variant<
    BufferView,
    DescriptorHeapView,
    TopAccel const *,
    std::pair<uint, uint4>>;
}// namespace toolhub::directx