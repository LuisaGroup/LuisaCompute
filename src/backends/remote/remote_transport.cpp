#include "remote_transport.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <asio.hpp>

#include <luisa/core/stl/format.h>

namespace luisa::compute::remote {

using Tcp = asio::ip::tcp;

class Connection::Impl {

private:
    struct Pending {
        MessageKind expected_kind{};
        std::condition_variable cv;
        Response response;
        bool done{false};
    };

    ProtocolLimits _limits;
    asio::io_context _io;
    Tcp::socket _socket{_io};
    std::thread _reader;
    std::thread::id _reader_thread_id{};
    mutable std::mutex _state_mutex;
    std::mutex _write_mutex;
    std::mutex _close_mutex;
    std::mutex _join_mutex;
    std::unordered_map<uint64_t, std::shared_ptr<Pending>> _pending;
    NotificationHandler _notification_handler;
    CloseHandler _close_handler;
    std::atomic_uint64_t _next_request_id{1u};
    bool _connected{false};
    bool _closing{false};
    luisa::string _error;

private:
    void _set_closed(luisa::string error) noexcept {
        CloseHandler close_handler;
        luisa::string callback_error;
        {
            std::scoped_lock lock{_state_mutex};
            if (!_connected && !_error.empty()) { return; }
            _connected = false;
            if (_error.empty()) { _error = std::move(error); }
            callback_error = _error;
            close_handler = _close_handler;
            for (auto &[id, pending] : _pending) {
                static_cast<void>(id);
                if (!pending->done) {
                    pending->response.status = Status::CONNECTION_CLOSED;
                    pending->response.message = _error;
                    pending->done = true;
                    pending->cv.notify_one();
                }
            }
        }
        if (close_handler) { close_handler(callback_error); }
    }

    [[nodiscard]] bool _write_frame(
        MessageKind kind,
        uint64_t request_id,
        luisa::span<const std::byte> payload,
        luisa::string &error) noexcept {
        if (payload.size() > _limits.max_frame_payload) {
            error = "Remote protocol payload exceeds the configured frame limit.";
            return false;
        }
        auto header = encode_frame_header(FrameHeader{
            .kind = kind,
            .request_id = request_id,
            .payload_size = payload.size(),
            .payload_checksum = payload_checksum(payload)});
        asio::error_code ec;
        {
            std::scoped_lock lock{_write_mutex};
            std::array<asio::const_buffer, 2u> buffers{
                asio::buffer(static_cast<const void *>(header.data()), header.size()),
                asio::buffer(payload.data(), payload.size())};
            asio::write(_socket, buffers, ec);
        }
        if (ec) {
            error = luisa::format("Remote socket write failed: {}", ec.message());
            _set_closed(error);
            return false;
        }
        return true;
    }

    [[nodiscard]] bool _read_exact(
        void *data, size_t size, asio::error_code &error) noexcept {
        if (size == 0u) {
            error.clear();
            return true;
        }
        auto completed = false;
        asio::async_read(
            _socket, asio::buffer(data, size),
            [&](const asio::error_code &read_error, size_t) noexcept {
                error = read_error;
                completed = true;
            });
        _io.run();
        _io.restart();
        return completed && !error;
    }

    void _reader_loop() noexcept {
        {
            std::scoped_lock lock{_state_mutex};
            _reader_thread_id = std::this_thread::get_id();
        }
        while (true) {
            std::array<std::byte, frame_header_size> header_bytes{};
            asio::error_code ec;
            if (!_read_exact(header_bytes.data(), header_bytes.size(), ec)) {
                bool closing;
                {
                    std::scoped_lock lock{_state_mutex};
                    closing = _closing;
                }
                _set_closed(closing ? "Remote connection closed." :
                                      luisa::format("Remote socket read failed: {}", ec.message()));
                return;
            }
            FrameHeader header;
            luisa::string decode_error;
            if (!decode_frame_header(header_bytes, header, decode_error, _limits)) {
                _set_closed(std::move(decode_error));
                asio::error_code ignored;
                _socket.close(ignored);
                return;
            }
            luisa::vector<std::byte> payload;
            payload.resize(static_cast<size_t>(header.payload_size));
            if (!payload.empty()) {
                if (!_read_exact(payload.data(), payload.size(), ec)) {
                    _set_closed(luisa::format(
                        "Remote socket payload read failed: {}", ec.message()));
                    return;
                }
            }
            if (payload_checksum(payload) != header.payload_checksum) {
                _set_closed("Remote protocol payload checksum mismatch.");
                asio::error_code ignored;
                _socket.close(ignored);
                return;
            }
            if (header.kind == MessageKind::RESPONSE ||
                header.kind == MessageKind::ERROR) {
                std::shared_ptr<Pending> pending;
                {
                    std::scoped_lock lock{_state_mutex};
                    if (auto iter = _pending.find(header.request_id);
                        iter != _pending.end()) {
                        pending = iter->second;
                    }
                }
                if (!pending) { continue; }
                ResponseView view;
                luisa::string response_error;
                Response response;
                if (!decode_response_payload(
                        payload, view, response_error, _limits)) {
                    response.status = Status::INVALID_REQUEST;
                    response.message = std::move(response_error);
                } else if (view.request_kind != pending->expected_kind) {
                    response.status = Status::INVALID_REQUEST;
                    response.message = "Remote response kind does not match its request.";
                } else {
                    response.request_kind = view.request_kind;
                    response.status = view.status;
                    response.message = std::move(view.message);
                    response.body.assign(view.body.begin(), view.body.end());
                }
                {
                    std::scoped_lock lock{_state_mutex};
                    if (!pending->done) {
                        pending->response = std::move(response);
                        pending->done = true;
                        pending->cv.notify_one();
                    }
                }
                continue;
            }
            NotificationHandler handler;
            {
                std::scoped_lock lock{_state_mutex};
                handler = _notification_handler;
            }
            if (handler) { handler(header.kind, header.request_id, payload); }
        }
    }

public:
    explicit Impl(ProtocolLimits limits) noexcept : _limits{limits} {}

    ~Impl() noexcept { close(); }

    [[nodiscard]] bool connect(luisa::string_view host,
                               uint16_t port,
                               std::chrono::milliseconds timeout,
                               luisa::string &error) noexcept {
        std::thread::id reader_thread_id;
        {
            std::scoped_lock lock{_state_mutex};
            if (_connected) {
                error = "Remote connection is already open.";
                return false;
            }
            reader_thread_id = _reader_thread_id;
        }
        if (reader_thread_id == std::this_thread::get_id()) {
            error = "Remote connection cannot reconnect from its reader callback.";
            return false;
        }
        {
            std::scoped_lock close_lock{_close_mutex};
            {
                std::scoped_lock lock{_state_mutex};
                if (_connected) {
                    error = "Remote connection is already open.";
                    return false;
                }
            }
            asio::error_code ignored;
            _socket.cancel(ignored);
            _socket.shutdown(Tcp::socket::shutdown_both, ignored);
            _socket.close(ignored);
        }
        {
            std::scoped_lock join_lock{_join_mutex};
            if (_reader.joinable()) { _reader.join(); }
        }
        std::scoped_lock close_lock{_close_mutex};
        {
            std::scoped_lock lock{_state_mutex};
            if (_connected) {
                error = "Remote connection is already open.";
                return false;
            }
            _error.clear();
            _closing = false;
            _reader_thread_id = {};
        }
        _io.restart();
        Tcp::resolver resolver{_io};
        asio::error_code ec;
        auto endpoints = resolver.resolve(
            luisa::string{host}, std::to_string(port), ec);
        if (ec) {
            error = luisa::format("Remote endpoint resolution failed: {}", ec.message());
            return false;
        }
        asio::steady_timer timer{_io};
        bool connect_done = false;
        bool timed_out = false;
        timer.expires_after(timeout);
        timer.async_wait([&](const asio::error_code &timer_error) noexcept {
            if (!timer_error && !connect_done) {
                timed_out = true;
                asio::error_code ignored;
                _socket.close(ignored);
            }
        });
        asio::async_connect(
            _socket, endpoints,
            [&](const asio::error_code &connect_error, const Tcp::endpoint &) noexcept {
                ec = connect_error;
                connect_done = true;
                asio::error_code ignored;
                static_cast<void>(timer.cancel());
            });
        _io.run();
        _io.restart();
        if (timed_out) {
            error = "Remote connection timed out.";
            return false;
        }
        if (ec) {
            error = luisa::format("Remote connection failed: {}", ec.message());
            return false;
        }
        {
            std::scoped_lock lock{_state_mutex};
            _connected = true;
        }
        _reader = std::thread{[this]() noexcept { _reader_loop(); }};
        return true;
    }

    [[nodiscard]] Response request(
        MessageKind kind,
        luisa::span<const std::byte> payload,
        std::chrono::milliseconds timeout) noexcept {
        auto request_id = _next_request_id.fetch_add(1u, std::memory_order_relaxed);
        auto pending = std::make_shared<Pending>();
        pending->expected_kind = kind;
        {
            std::scoped_lock lock{_state_mutex};
            if (!_connected) {
                return Response{
                    .request_kind = kind,
                    .status = Status::CONNECTION_CLOSED,
                    .message = _error.empty() ? "Remote connection is closed." : _error};
            }
            _pending.emplace(request_id, pending);
        }
        luisa::string write_error;
        if (!_write_frame(kind, request_id, payload, write_error)) {
            std::scoped_lock lock{_state_mutex};
            _pending.erase(request_id);
            return Response{
                .request_kind = kind,
                .status = Status::CONNECTION_CLOSED,
                .message = std::move(write_error)};
        }
        std::unique_lock lock{_state_mutex};
        auto completed = pending->cv.wait_for(
            lock, timeout, [&]() noexcept { return pending->done || !_connected; });
        if (!completed) {
            _pending.erase(request_id);
            return Response{
                .request_kind = kind,
                .status = Status::TIMEOUT,
                .message = "Remote request timed out."};
        }
        auto response = std::move(pending->response);
        _pending.erase(request_id);
        return response;
    }

    [[nodiscard]] bool notify(MessageKind kind,
                              uint64_t request_id,
                              luisa::span<const std::byte> payload,
                              luisa::string &error) noexcept {
        {
            std::scoped_lock lock{_state_mutex};
            if (!_connected) {
                error = _error.empty() ? "Remote connection is closed." : _error;
                return false;
            }
        }
        return _write_frame(kind, request_id, payload, error);
    }

    void set_notification_handler(NotificationHandler handler) noexcept {
        std::scoped_lock lock{_state_mutex};
        _notification_handler = std::move(handler);
    }

    void set_close_handler(CloseHandler handler) noexcept {
        std::scoped_lock lock{_state_mutex};
        _close_handler = std::move(handler);
    }

    void close() noexcept {
        auto initiate_close = false;
        std::thread::id reader_thread_id;
        {
            std::scoped_lock close_lock{_close_mutex};
            {
                std::scoped_lock lock{_state_mutex};
                initiate_close = !_closing;
                _closing = true;
                reader_thread_id = _reader_thread_id;
            }
            if (initiate_close) {
                asio::error_code ignored;
                _socket.cancel(ignored);
                _socket.shutdown(Tcp::socket::shutdown_both, ignored);
                _socket.close(ignored);
            }
        }
        if (reader_thread_id != std::this_thread::get_id()) {
            std::scoped_lock join_lock{_join_mutex};
            if (_reader.joinable()) { _reader.join(); }
        }
        _set_closed("Remote connection closed.");
    }

    [[nodiscard]] bool connected() const noexcept {
        std::scoped_lock lock{_state_mutex};
        return _connected;
    }

    [[nodiscard]] luisa::string error() const noexcept {
        std::scoped_lock lock{_state_mutex};
        return _error;
    }
};

Connection::Connection(ProtocolLimits limits) noexcept
    : _impl{std::make_unique<Impl>(limits)} {}

Connection::~Connection() noexcept = default;

bool Connection::connect(luisa::string_view host,
                         uint16_t port,
                         std::chrono::milliseconds timeout,
                         luisa::string &error) noexcept {
    return _impl->connect(host, port, timeout, error);
}

Response Connection::request(MessageKind kind,
                             luisa::span<const std::byte> payload,
                             std::chrono::milliseconds timeout) noexcept {
    return _impl->request(kind, payload, timeout);
}

bool Connection::notify(MessageKind kind,
                        uint64_t request_id,
                        luisa::span<const std::byte> payload,
                        luisa::string &error) noexcept {
    return _impl->notify(kind, request_id, payload, error);
}

void Connection::set_notification_handler(NotificationHandler handler) noexcept {
    _impl->set_notification_handler(std::move(handler));
}

void Connection::set_close_handler(CloseHandler handler) noexcept {
    _impl->set_close_handler(std::move(handler));
}

void Connection::close() noexcept { _impl->close(); }

bool Connection::connected() const noexcept { return _impl->connected(); }

luisa::string Connection::error() const noexcept { return _impl->error(); }

}// namespace luisa::compute::remote
