#pragma once

#include <cstdint>
#include <limits>

#include <luisa/core/stl/string.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

class RemoteDeviceConfigExt final : public DeviceConfigExt {

private:
    luisa::string _host;
    uint16_t _port;
    luisa::string _token;
    uint64_t _connect_timeout_ms;
    uint64_t _request_timeout_ms;
    uint64_t _max_in_flight_bytes;
    bool _enable_blob_cache;
    uint64_t _blob_cache_min_size;
    luisa::string _local_present_backend;
    luisa::string _server_backend;
    size_t _server_device_index;
    bool _server_enable_validation;

public:
    explicit RemoteDeviceConfigExt(
        luisa::string host = "127.0.0.1",
        uint16_t port = 18080u,
        luisa::string token = {},
        uint64_t connect_timeout_ms = 10'000u,
        uint64_t request_timeout_ms = 60'000u,
        uint64_t max_in_flight_bytes = 256ull * 1024ull * 1024ull,
        bool enable_blob_cache = true,
        uint64_t blob_cache_min_size = 64ull * 1024ull,
        luisa::string local_present_backend = {},
        luisa::string server_backend = {},
        size_t server_device_index = std::numeric_limits<size_t>::max(),
        bool server_enable_validation = false) noexcept
        : _host{std::move(host)},
          _port{port},
          _token{std::move(token)},
          _connect_timeout_ms{connect_timeout_ms},
          _request_timeout_ms{request_timeout_ms},
          _max_in_flight_bytes{max_in_flight_bytes},
          _enable_blob_cache{enable_blob_cache},
          _blob_cache_min_size{blob_cache_min_size},
          _local_present_backend{std::move(local_present_backend)},
          _server_backend{std::move(server_backend)},
          _server_device_index{server_device_index},
          _server_enable_validation{server_enable_validation} {}

    [[nodiscard]] auto host() const noexcept { return luisa::string_view{_host}; }
    [[nodiscard]] auto port() const noexcept { return _port; }
    [[nodiscard]] auto token() const noexcept { return luisa::string_view{_token}; }
    [[nodiscard]] auto connect_timeout_ms() const noexcept { return _connect_timeout_ms; }
    [[nodiscard]] auto request_timeout_ms() const noexcept { return _request_timeout_ms; }
    [[nodiscard]] auto max_in_flight_bytes() const noexcept { return _max_in_flight_bytes; }
    [[nodiscard]] auto enable_blob_cache() const noexcept { return _enable_blob_cache; }
    [[nodiscard]] auto blob_cache_min_size() const noexcept { return _blob_cache_min_size; }
    [[nodiscard]] auto local_present_backend() const noexcept { return luisa::string_view{_local_present_backend}; }
    [[nodiscard]] auto server_backend() const noexcept { return luisa::string_view{_server_backend}; }
    [[nodiscard]] auto server_device_index() const noexcept { return _server_device_index; }
    [[nodiscard]] auto server_enable_validation() const noexcept { return _server_enable_validation; }
};

}// namespace luisa::compute
