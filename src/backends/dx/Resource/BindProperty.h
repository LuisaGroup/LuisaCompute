#pragma once
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
namespace lc::dx {
class TopAccel;
using BindProperty = vstd::variant<
    BufferView,
    DescriptorHeapView,
    TopAccel const *,
    std::pair<uint, uint4>>;
}// namespace lc::dx
