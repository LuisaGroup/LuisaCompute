// Newline-delimited JSON-RPC 2.0 subset and TCP transport for embedding
// retrieval (retrieval plan phase 3). The protocol entry point is separately
// testable without sockets; the server owns all socket/thread lifetimes.
#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>

#include "retrieval_service.h"

namespace luisa::embed {

class RetrievalRpcServer {
public:
    static constexpr size_t kDefaultMaxRequestLineBytes = 1024u * 1024u;

    explicit RetrievalRpcServer(RetrievalService &service) noexcept;
    ~RetrievalRpcServer() noexcept;

    RetrievalRpcServer(const RetrievalRpcServer &) = delete;
    RetrievalRpcServer(RetrievalRpcServer &&) = delete;
    RetrievalRpcServer &operator=(const RetrievalRpcServer &) = delete;
    RetrievalRpcServer &operator=(RetrievalRpcServer &&) = delete;

    // Executes one request. nullopt denotes a notification (including a failed
    // notification); protocol errors on non-notifications always have a response.
    [[nodiscard]] static luisa::optional<luisa::string> process_json_line(
        RetrievalService &service,
        luisa::string_view json_line);

    // Starts an internal accept thread. Port zero requests an ephemeral port.
    // A server instance is single-start; stop() is idempotent and joins every
    // accept/client thread before returning.
    [[nodiscard]] bool start(
        luisa::string_view listen_address,
        uint16_t port,
        luisa::string &error,
        size_t max_request_line_bytes = kDefaultMaxRequestLineBytes) noexcept;
    void stop() noexcept;

    [[nodiscard]] bool running() const noexcept;
    [[nodiscard]] uint16_t port() const noexcept;
    [[nodiscard]] luisa::string runtime_error() const;

private:
    struct Impl;
    luisa::unique_ptr<Impl> _impl;
};

}// namespace luisa::embed
