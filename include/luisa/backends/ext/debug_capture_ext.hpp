#pragma once

#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/stream.h>

namespace luisa::compute {

struct DebugCaptureOption {
    enum struct Output {
        DEVELOPER_TOOLS,
        GPU_TRACE_DOCUMENT,
    };
    Output output{Output::DEVELOPER_TOOLS};
    luisa::string file_name;
};

class DebugCaptureExt;

class DebugCaptureScope final : public concepts::Noncopyable {

private:
    const DebugCaptureExt *_ext;
    uint64_t _handle;

private:
    friend class DebugCaptureExt;
    DebugCaptureScope(const DebugCaptureExt *ext, uint64_t handle) noexcept
        : _ext{ext}, _handle{handle} {}

public:
    ~DebugCaptureScope() noexcept;
    DebugCaptureScope(DebugCaptureScope &&scope) noexcept
        : _ext{scope._ext}, _handle{scope._handle} {
        scope._ext = nullptr;
        scope._handle = 0u;
    }
    DebugCaptureScope &operator=(DebugCaptureScope &&rhs) noexcept {
        if (this != &rhs) {
            this->~DebugCaptureScope();
            new (this) DebugCaptureScope{std::move(rhs)};
        }
        return *this;
    }

public:
    void start() const noexcept;
    void stop() const noexcept;
};

class DebugCaptureExt : public DeviceExtension {

public:
    using Option = DebugCaptureOption;
    static constexpr luisa::string_view name = "DebugCaptureExt";

private:
    friend class DebugCaptureScope;
    [[nodiscard]] virtual uint64_t create_device_capture_scope(luisa::string_view label, const Option &option) const noexcept = 0;
    [[nodiscard]] virtual uint64_t create_stream_capture_scope(uint64_t stream_handle, luisa::string_view label, const Option &option) const noexcept = 0;
    virtual void destroy_capture_scope(uint64_t handle) const noexcept = 0;
    virtual void start_capture(uint64_t handle) const noexcept = 0;
    virtual void stop_capture(uint64_t handle) const noexcept = 0;

public:
    [[nodiscard]] auto create_scope(luisa::string_view label, const Option &option = {}) const noexcept {
        return DebugCaptureScope{this, create_device_capture_scope(label, option)};
    }
    [[nodiscard]] auto create_scope(const Stream &stream, luisa::string_view label, const Option &option = {}) const noexcept {
        return DebugCaptureScope{this, create_stream_capture_scope(stream.handle(), label, option)};
    }
};

inline DebugCaptureScope::~DebugCaptureScope() noexcept {
    if (_ext) { _ext->destroy_capture_scope(_handle); }
}

inline void DebugCaptureScope::start() const noexcept {
    _ext->start_capture(_handle);
}

inline void DebugCaptureScope::stop() const noexcept {
    _ext->stop_capture(_handle);
}

}// namespace luisa::compute
