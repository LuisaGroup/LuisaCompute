#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/runtime/rhi/device_interface.h>

#include "remote_protocol.h"

namespace luisa::compute::remote {

struct ServerOptions {
    luisa::string listen_address{"127.0.0.1"};
    uint16_t port{18080u};
    luisa::string token;
    uint64_t max_resource_size{16ull * 1024ull * 1024ull * 1024ull};
    uint64_t max_resources{1ull << 20u};
    uint64_t max_pending_submissions{4096u};
    uint64_t max_blob_cache_bytes{512ull * 1024ull * 1024ull};
    uint64_t max_blob_entry_size{64ull * 1024ull * 1024ull};
    uint64_t blob_cache_min_size{64ull * 1024ull};
    uint64_t max_blobs_per_batch{4096u};
    uint64_t max_prepared_blob_batches{64u};
    uint64_t max_concurrent_sessions{64u};
    ProtocolLimits protocol_limits;
};

struct DeviceRequest {
    luisa::string backend;
    size_t device_index{std::numeric_limits<size_t>::max()};
    bool enable_validation{false};
};

using DeviceFactory = std::function<luisa::shared_ptr<DeviceInterface>(
    const DeviceRequest &request, luisa::string &error)>;

class Server final {

private:
    class Impl;
    std::unique_ptr<Impl> _impl;

public:
    Server(luisa::shared_ptr<DeviceInterface> native_device,
           ServerOptions options = {});
    Server(DeviceFactory device_factory, ServerOptions options = {});
    ~Server() noexcept;
    Server(Server const &) = delete;
    Server(Server &&) = delete;
    Server &operator=(Server const &) = delete;
    Server &operator=(Server &&) = delete;

    void run();
    void stop() noexcept;
    [[nodiscard]] uint16_t port() const noexcept;
};

}// namespace luisa::compute::remote
