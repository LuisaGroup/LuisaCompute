#pragma once
#include <vstl/common.h>
#include <rtx/accel.h>
#include <rtx/mesh.h>
namespace luisa::compute {
class DeviceInterface;
class RefCounter : public vstd::IOperatorNewBase {
public:
    using Handle = uint64;
    using Disposer = std::pair<vstd::funcPtr_t<void(DeviceInterface *, Handle)>, DeviceInterface *>;
    vstd::spin_mutex mtx;

private:
    vstd::unordered_map<Handle, std::pair<int64, Disposer>> refCounts;

public:
    RefCounter() noexcept;
    static vstd::unique_ptr<RefCounter> current;
    ~RefCounter() noexcept;
    void AddObject(Handle handle, Disposer disposer) noexcept;
    void InRef(Handle handle) noexcept;
    void DeRef(Handle handle) noexcept;
};
}// namespace luisa::compute