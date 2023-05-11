#pragma once
#include <core/stl/memory.h>
#include <core/stl/string.h>
class ID3D12Device;
namespace luisa {
namespace compute {
class Context;
}// namespace compute
class BinaryStream;
class DStorageInterface {
public:
    virtual ~DStorageInterface() noexcept = default;
    virtual bool dstorage_supported() const noexcept = 0;
    virtual luisa::unique_ptr<BinaryStream> create_stream(luisa::string_view path) noexcept = 0;
};
// extern "C" DStorageInterface *create(compute::Context const &runtime_dir, ID3D12Device *device_ptr) noexcept;
}// namespace luisa