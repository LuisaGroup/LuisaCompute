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
    void mark_begin() const noexcept;
    void mark_end() const noexcept;

    template<typename F>
    void with(F &&f) const noexcept {
        mark_begin();
        std::invoke(std::forward<F>(f));
        mark_end();
    }
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
    virtual void start_debug_capture(uint64_t handle) const noexcept = 0;
    virtual void stop_debug_capture() const noexcept = 0;
    virtual void mark_scope_begin(uint64_t handle) const noexcept = 0;
    virtual void mark_scope_end(uint64_t handle) const noexcept = 0;

public:
    [[nodiscard]] auto create_scope(luisa::string_view label, const Option &option = {}) const noexcept {
        return DebugCaptureScope{this, create_device_capture_scope(label, option)};
    }
    [[nodiscard]] auto create_scope(luisa::string_view label, const Stream &stream, const Option &option = {}) const noexcept {
        return DebugCaptureScope{this, create_stream_capture_scope(stream.handle(), label, option)};
    }
    void start_capture(const DebugCaptureScope &scope) const noexcept { start_debug_capture(scope._handle); }
    void stop_capture() const noexcept { stop_debug_capture(); }
};

inline DebugCaptureScope::~DebugCaptureScope() noexcept {
    if (_ext) { _ext->destroy_capture_scope(_handle); }
}

inline void DebugCaptureScope::mark_begin() const noexcept {
    _ext->mark_scope_begin(_handle);
}

inline void DebugCaptureScope::mark_end() const noexcept {
    _ext->mark_scope_end(_handle);
}

}// namespace luisa::compute
