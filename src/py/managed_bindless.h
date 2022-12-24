#include <runtime/device.h>
#include <py/managed_collector.h>
namespace luisa::compute {
class ManagedBindless : public vstd::IOperatorNewBase {
    uint64 handle;
    ManagedCollector collector;
    DeviceInterface *device;

public:
    uint64 GetHandle() const noexcept{return handle;}
    ManagedBindless(DeviceInterface *device,uint64 handle) noexcept;
    ManagedBindless(ManagedBindless &&) noexcept = delete;
    ~ManagedBindless() noexcept;
    void emplace_buffer( size_t index, uint64 handle, size_t offset) noexcept;
    void emplace_tex2d( size_t index, uint64 handle, Sampler sampler) noexcept;
    void emplace_tex3d( size_t index, uint64 handle, Sampler sampler) noexcept;
    void remove_buffer( size_t index) noexcept;
    void remove_tex2d( size_t index) noexcept;
    void remove_tex3d( size_t index) noexcept;
    void Update(PyStream& stream) noexcept;
};
}// namespace luisa::compute