#pragma once
#include <Resource/Resource.h>
namespace lc::dx {
template<typename T>
    requires(std::is_base_of_v<Resource, T>)
uint64 resource_to_handle(T const *ptr) {
    auto handle = reinterpret_cast<uint64>(ptr);
    assert(handle == reinterpret_cast<uint64>(static_cast<Resource const*>(ptr)));
    return handle;
}
}// namespace lc::dx
