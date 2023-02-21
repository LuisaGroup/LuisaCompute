#pragma once

#include <vstl/common.h>
#include <runtime/rtx/accel.h>
#include <runtime/rtx/mesh.h>

namespace luisa::compute {

class DeviceInterface;

class RefCounter : public vstd::IOperatorNewBase {
public:
    using Handle = uint64;
    using Disposer = std::pair<vstd::func_ptr_t<void(DeviceInterface *, Handle)>, DeviceInterface *>;
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
