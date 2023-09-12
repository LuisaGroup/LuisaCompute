#pragma once
namespace luisa::compute::graph {
enum class MemoryNodeDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
};
}// namespace luisa::compute::graph