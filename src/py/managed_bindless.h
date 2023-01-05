#include <runtime/device.h>
#include <runtime/bindless_array.h>
#include <py/managed_collector.h>

namespace luisa::compute {

class ManagedBindless : public vstd::IOperatorNewBase {

private:
    ManagedCollector collector;
    BindlessArray array;

public:
    [[nodiscard]] uint64 GetHandle() const noexcept { return array.handle(); }
    ManagedBindless(DeviceInterface *device, size_t slots) noexcept;
    ManagedBindless(ManagedBindless &&) noexcept = delete;
    ~ManagedBindless() noexcept = default;
    void emplace_buffer(size_t index, uint64 handle, size_t offset) noexcept;
    void emplace_tex2d(size_t index, uint64 handle, Sampler sampler) noexcept;
    void emplace_tex3d(size_t index, uint64 handle, Sampler sampler) noexcept;
    void remove_buffer(size_t index) noexcept;
    void remove_tex2d(size_t index) noexcept;
    void remove_tex3d(size_t index) noexcept;
    void Update(PyStream &stream) noexcept;
};

}// namespace luisa::compute
