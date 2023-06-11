#pragma once
#include <luisa/vstl/common.h>
namespace lc::vk {
class Device;
class Resource : public vstd::IOperatorNewBase {
    Device *_device;

public:
    Resource(Resource const &) = delete;
    Resource(Resource &&) = default;
    Resource(Device *device) : _device{device} {}
    virtual ~Resource() = default;
    auto device() const { return _device; }
};
}// namespace lc::vk
