// Retrieval JSON-RPC protocol and small socket transport (retrieval plan phase
// 3). yyjson documents/writer buffers are scoped per call, while accepted client
// handlers share only the mutex-serialized RetrievalService.

#include "retrieval_rpc.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/vector.h>
#include <yyjson.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace luisa::embed {
namespace {

struct ReadDocumentGuard {
    yyjson_doc *document{};
    ~ReadDocumentGuard() noexcept { yyjson_doc_free(document); }
};

struct WriteDocumentGuard {
    yyjson_mut_doc *document{};
    ~WriteDocumentGuard() noexcept { yyjson_mut_doc_free(document); }
};

struct WriterBufferGuard {
    char *data{};
    ~WriterBufferGuard() noexcept { std::free(data); }
};

class JsonResponse {
public:
    explicit JsonResponse(yyjson_val *id) noexcept {
        _document.document = yyjson_mut_doc_new(nullptr);
        if (_document.document == nullptr) {
            return;
        }
        _root = yyjson_mut_obj(_document.document);
        if (_root == nullptr) {
            return;
        }
        yyjson_mut_doc_set_root(_document.document, _root);
        _valid = yyjson_mut_obj_add_str(_document.document, _root, "jsonrpc", "2.0");
        if (id == nullptr) {
            _valid = yyjson_mut_obj_add_null(_document.document, _root, "id") && _valid;
        } else {
            auto *copied_id = yyjson_val_mut_copy(_document.document, id);
            _valid = yyjson_mut_obj_add_val(_document.document, _root, "id", copied_id) && copied_id != nullptr && _valid;
        }
    }

    [[nodiscard]] bool valid() const noexcept { return _valid; }
    [[nodiscard]] yyjson_mut_doc *document() const noexcept { return _document.document; }

    [[nodiscard]] yyjson_mut_val *add_object(const char *key, yyjson_mut_val *object = nullptr) {
        auto *created = yyjson_mut_obj(_document.document);
        auto *parent = object == nullptr ? _root : object;
        if (created == nullptr ||
            !yyjson_mut_obj_add_val(_document.document, parent, key, created)) {
            _valid = false;
            return nullptr;
        }
        return created;
    }

    [[nodiscard]] yyjson_mut_val *add_array(const char *key, yyjson_mut_val *object = nullptr) {
        auto *created = yyjson_mut_arr(_document.document);
        auto *parent = object == nullptr ? _root : object;
        if (created == nullptr ||
            !yyjson_mut_obj_add_val(_document.document, parent, key, created)) {
            _valid = false;
            return nullptr;
        }
        return created;
    }

    void add_string(yyjson_mut_val *object, const char *key, luisa::string_view value) {
        auto *copied = yyjson_mut_strncpy(_document.document, value.data(), value.size());
        if (copied == nullptr || !yyjson_mut_obj_add_val(_document.document, object, key, copied)) {
            _valid = false;
        }
    }

    void append_string(yyjson_mut_val *array, luisa::string_view value) {
        auto *copied = yyjson_mut_strncpy(_document.document, value.data(), value.size());
        if (copied == nullptr || !yyjson_mut_arr_append(array, copied)) {
            _valid = false;
        }
    }

    void add_uint(yyjson_mut_val *object, const char *key, uint64_t value) {
        if (!yyjson_mut_obj_add_val(_document.document, object, key,
                                    yyjson_mut_uint(_document.document, value))) {
            _valid = false;
        }
    }

    void add_sint(yyjson_mut_val *object, const char *key, int64_t value) {
        if (!yyjson_mut_obj_add_val(_document.document, object, key,
                                    yyjson_mut_sint(_document.document, value))) {
            _valid = false;
        }
    }

    void add_real(yyjson_mut_val *object, const char *key, double value) {
        if (!yyjson_mut_obj_add_val(_document.document, object, key,
                                    yyjson_mut_real(_document.document, value))) {
            _valid = false;
        }
    }

    [[nodiscard]] luisa::string finish() const {
        if (!_valid) {
            return {};
        }
        size_t length = 0u;
        WriterBufferGuard output{yyjson_mut_write(_document.document, YYJSON_WRITE_NOFLAG, &length)};
        if (output.data == nullptr) {
            return {};
        }
        return {output.data, length};
    }

private:
    WriteDocumentGuard _document;
    yyjson_mut_val *_root{};
    bool _valid{false};
};

[[nodiscard]] luisa::string error_response(
    yyjson_val *id,
    int32_t code,
    luisa::string_view message) {
    JsonResponse response{id};
    if (auto *error = response.add_object("error"); error != nullptr) {
        response.add_sint(error, "code", code);
        response.add_string(error, "message", message);
    }
    auto json = response.finish();
    if (json.empty()) {
        return R"({"jsonrpc":"2.0","id":null,"error":{"code":-32603,"message":"response allocation failed"}})";
    }
    return json;
}

[[nodiscard]] int32_t operational_error_code(RetrievalErrorCode code) noexcept {
    return code == RetrievalErrorCode::invalid_argument ? -32602 : -32000;
}

[[nodiscard]] bool copy_required_string(
    yyjson_val *params,
    const char *key,
    luisa::string &output,
    bool reject_nul = false) {
    auto *value = yyjson_obj_get(params, key);
    if (!yyjson_is_str(value)) {
        return false;
    }
    const auto *data = yyjson_get_str(value);
    const auto length = yyjson_get_len(value);
    if (reject_nul && std::memchr(data, '\0', length) != nullptr) {
        return false;
    }
    output.assign(data, length);
    return true;
}

[[nodiscard]] bool required_params(yyjson_val *root, yyjson_val *&params) {
    params = yyjson_obj_get(root, "params");
    return yyjson_is_obj(params);
}

[[nodiscard]] luisa::optional<luisa::string> dispatch_error(
    yyjson_val *id,
    bool notification,
    int32_t code,
    luisa::string_view message) {
    if (notification) {
        LUISA_WARNING("retrieval RPC notification failed: {}", message);
        return std::nullopt;
    }
    return error_response(id, code, message);
}

[[nodiscard]] luisa::string load_response(yyjson_val *id, const RetrievalLoadResult &result) {
    JsonResponse response{id};
    if (auto *object = response.add_object("result"); object != nullptr) {
        response.add_string(object, "path", result.root);
        if (auto *extensions = response.add_array("extensions", object); extensions != nullptr) {
            for (const auto &extension : result.extensions) {
                response.append_string(extensions, extension);
            }
        }
        response.add_uint(object, "added", result.added);
        response.add_uint(object, "skipped", result.skipped);
        response.add_uint(object, "failed", result.failed);
        response.add_uint(object, "total_documents", result.total_documents);
    }
    return response.finish();
}

[[nodiscard]] luisa::string unload_response(yyjson_val *id, const RetrievalUnloadResult &result) {
    JsonResponse response{id};
    if (auto *object = response.add_object("result"); object != nullptr) {
        response.add_string(object, "path", result.root);
        response.add_uint(object, "removed_documents", result.removed_documents);
        response.add_uint(object, "total_documents", result.total_documents);
    }
    return response.finish();
}

[[nodiscard]] luisa::string update_response(yyjson_val *id, const RetrievalUpdateResult &result) {
    JsonResponse response{id};
    if (auto *object = response.add_object("result"); object != nullptr) {
        response.add_string(object, "path", result.path);
        response.add_uint(object, "added", result.added);
        response.add_uint(object, "updated", result.updated);
        response.add_uint(object, "removed", result.removed);
        response.add_uint(object, "unchanged", result.unchanged);
        response.add_uint(object, "failed", result.failed);
        response.add_uint(object, "total_documents", result.total_documents);
    }
    return response.finish();
}

[[nodiscard]] luisa::string query_response(yyjson_val *id, const RetrievalQueryResult &result) {
    JsonResponse response{id};
    if (auto *object = response.add_object("result"); object != nullptr) {
        if (auto *hits = response.add_array("results", object); hits != nullptr) {
            for (const auto &hit : result.hits) {
                auto *entry = yyjson_mut_obj(response.document());
                if (entry == nullptr || !yyjson_mut_arr_append(hits, entry)) {
                    return error_response(id, -32603, "response allocation failed");
                }
                response.add_string(entry, "path", hit.path);
                response.add_real(entry, "score", hit.score);
            }
        }
    }
    return response.finish();
}

[[nodiscard]] bool json_string_equals(yyjson_val *value, const char *expected) noexcept {
    return value != nullptr && yyjson_equals_str(value, expected);
}

#ifdef _WIN32
using SocketHandle = SOCKET;
constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
constexpr int kSocketError = SOCKET_ERROR;
#else
using SocketHandle = int;
constexpr SocketHandle kInvalidSocket = -1;
constexpr int kSocketError = -1;
#endif

class Socket {
public:
    Socket() noexcept = default;
    explicit Socket(SocketHandle handle) noexcept : _handle{handle} {}
    ~Socket() noexcept { close(); }
    Socket(const Socket &) = delete;
    Socket &operator=(const Socket &) = delete;
    Socket(Socket &&other) noexcept : _handle{other.release()} {}
    Socket &operator=(Socket &&other) noexcept {
        if (this != &other) {
            close();
            _handle = other.release();
        }
        return *this;
    }

    [[nodiscard]] SocketHandle get() const noexcept { return _handle; }
    [[nodiscard]] bool valid() const noexcept { return _handle != kInvalidSocket; }
    [[nodiscard]] SocketHandle release() noexcept {
        const auto handle = _handle;
        _handle = kInvalidSocket;
        return handle;
    }
    void close() noexcept {
        if (!valid()) {
            return;
        }
#ifdef _WIN32
        closesocket(_handle);
#else
        ::close(_handle);
#endif
        _handle = kInvalidSocket;
    }

private:
    SocketHandle _handle{kInvalidSocket};
};

struct AddrInfoGuard {
    addrinfo *value{};
    ~AddrInfoGuard() noexcept {
        if (value != nullptr) {
            freeaddrinfo(value);
        }
    }
};

class SocketRuntime {
public:
    ~SocketRuntime() noexcept {
#ifdef _WIN32
        if (_active) {
            WSACleanup();
        }
#endif
    }

    [[nodiscard]] bool initialize(luisa::string &error) noexcept {
#ifdef _WIN32
        WSADATA data{};
        const auto result = WSAStartup(MAKEWORD(2, 2), &data);
        if (result != 0) {
            error = luisa::format("WSAStartup failed with error {}", result);
            return false;
        }
        _active = true;
#else
        (void)error;
#endif
        return true;
    }

private:
    bool _active{false};
};

[[nodiscard]] int last_socket_error() noexcept {
#ifdef _WIN32
    return WSAGetLastError();
#else
    return errno;
#endif
}

void shutdown_socket(SocketHandle socket) noexcept {
#ifdef _WIN32
    shutdown(socket, SD_BOTH);
#else
    shutdown(socket, SHUT_RDWR);
#endif
}

[[nodiscard]] bool send_all(SocketHandle socket, luisa::string_view data) noexcept {
    size_t offset = 0u;
    while (offset < data.size()) {
        const auto remaining = data.size() - offset;
        const auto chunk = std::min(remaining, static_cast<size_t>(std::numeric_limits<int>::max()));
#ifdef MSG_NOSIGNAL
        constexpr int flags = MSG_NOSIGNAL;
#else
        constexpr int flags = 0;
#endif
        const auto sent = send(socket, data.data() + offset, static_cast<int>(chunk), flags);
        if (sent <= 0) {
            return false;
        }
        offset += static_cast<size_t>(sent);
    }
    return true;
}

}// namespace

luisa::optional<luisa::string> RetrievalRpcServer::process_json_line(
    RetrievalService &service,
    luisa::string_view json_line) {
    ReadDocumentGuard request{yyjson_read(json_line.data(), json_line.size(), YYJSON_READ_NOFLAG)};
    if (request.document == nullptr) {
        return error_response(nullptr, -32700, "parse error: expected one JSON-RPC request object");
    }
    auto *root = yyjson_doc_get_root(request.document);
    if (!yyjson_is_obj(root)) {
        return error_response(nullptr, -32600, "invalid request: batch arrays and non-object requests are unsupported");
    }
    auto *id = yyjson_obj_get(root, "id");
    if (id != nullptr && !yyjson_is_str(id) && !yyjson_is_num(id)) {
        return error_response(nullptr, -32600, "invalid request: id must be a string or number");
    }
    const bool notification = id == nullptr;
    if (!json_string_equals(yyjson_obj_get(root, "jsonrpc"), "2.0")) {
        return error_response(id, -32600, "invalid request: jsonrpc must be exactly '2.0'");
    }
    auto *method_value = yyjson_obj_get(root, "method");
    if (!yyjson_is_str(method_value)) {
        return error_response(id, -32600, "invalid request: method must be a string");
    }

    const auto invalid_params = [&](luisa::string_view message) {
        return dispatch_error(id, notification, -32602, message);
    };
    yyjson_val *params = nullptr;

    if (json_string_equals(method_value, "load")) {
        if (!required_params(root, params)) {
            return invalid_params("invalid params: load requires an object");
        }
        luisa::string path;
        if (!copy_required_string(params, "path", path, true) || path.empty()) {
            return invalid_params("invalid params: load.path must be a nonempty UTF-8 string");
        }
        std::filesystem::path filesystem_path;
        auto path_status = RetrievalIndex::path_from_utf8(path, filesystem_path);
        if (!path_status.ok()) {
            return invalid_params(path_status.message);
        }
        auto *extensions_value = yyjson_obj_get(params, "extensions");
        RetrievalLoadResult result;
        if (extensions_value == nullptr) {
            result = service.load(filesystem_path);
        } else {
            if (!yyjson_is_arr(extensions_value)) {
                return invalid_params("invalid params: load.extensions must be an array of strings");
            }
            luisa::vector<luisa::string> extensions;
            const auto count = yyjson_arr_size(extensions_value);
            extensions.reserve(count);
            for (size_t i = 0u; i < count; i++) {
                auto *entry = yyjson_arr_get(extensions_value, i);
                if (!yyjson_is_str(entry)) {
                    return invalid_params("invalid params: every extension must be a string");
                }
                extensions.emplace_back(yyjson_get_str(entry), yyjson_get_len(entry));
            }
            result = service.load(
                filesystem_path,
                luisa::span<const luisa::string>{extensions.data(), extensions.size()});
        }
        if (!result.status.ok()) {
            return dispatch_error(id, notification, operational_error_code(result.status.code), result.status.message);
        }
        if (notification) {
            return std::nullopt;
        }
        auto response = load_response(id, result);
        return response.empty() ? error_response(id, -32603, "response allocation failed") : std::move(response);
    }

    if (json_string_equals(method_value, "unload")) {
        if (!required_params(root, params)) {
            return invalid_params("invalid params: unload requires an object");
        }
        luisa::string path;
        if (!copy_required_string(params, "path", path, true) || path.empty()) {
            return invalid_params("invalid params: unload.path must be a nonempty UTF-8 string");
        }
        std::filesystem::path filesystem_path;
        auto path_status = RetrievalIndex::path_from_utf8(path, filesystem_path);
        if (!path_status.ok()) {
            return invalid_params(path_status.message);
        }
        auto result = service.unload(filesystem_path);
        if (!result.status.ok()) {
            return dispatch_error(id, notification, operational_error_code(result.status.code), result.status.message);
        }
        if (notification) {
            return std::nullopt;
        }
        auto response = unload_response(id, result);
        return response.empty() ? error_response(id, -32603, "response allocation failed") : std::move(response);
    }

    if (json_string_equals(method_value, "update")) {
        if (!required_params(root, params)) {
            return invalid_params("invalid params: update requires an object");
        }
        luisa::string path;
        if (!copy_required_string(params, "path", path, true) || path.empty()) {
            return invalid_params("invalid params: update.path must be a nonempty UTF-8 string");
        }
        std::filesystem::path filesystem_path;
        auto path_status = RetrievalIndex::path_from_utf8(path, filesystem_path);
        if (!path_status.ok()) {
            return invalid_params(path_status.message);
        }
        auto result = service.update(filesystem_path);
        if (!result.status.ok()) {
            return dispatch_error(id, notification, operational_error_code(result.status.code), result.status.message);
        }
        if (notification) {
            return std::nullopt;
        }
        auto response = update_response(id, result);
        return response.empty() ? error_response(id, -32603, "response allocation failed") : std::move(response);
    }

    if (json_string_equals(method_value, "query")) {
        if (!required_params(root, params)) {
            return invalid_params("invalid params: query requires an object");
        }
        luisa::string text;
        if (!copy_required_string(params, "text", text)) {
            return invalid_params("invalid params: query.text must be a string");
        }
        uint32_t top_k = kRetrievalDefaultTopK;
        if (auto *top_value = yyjson_obj_get(params, "top_k"); top_value != nullptr) {
            if (!yyjson_is_uint(top_value)) {
                return invalid_params("invalid params: query.top_k must be an integer");
            }
            const auto value = yyjson_get_uint(top_value);
            if (value == 0u || value > kRetrievalMaxTopK) {
                return invalid_params("invalid params: query.top_k is outside [1, 100]");
            }
            top_k = static_cast<uint32_t>(value);
        }
        auto result = service.query(text, top_k);
        if (!result.status.ok()) {
            return dispatch_error(id, notification, operational_error_code(result.status.code), result.status.message);
        }
        if (notification) {
            return std::nullopt;
        }
        auto response = query_response(id, result);
        return response.empty() ? error_response(id, -32603, "response allocation failed") : std::move(response);
    }

    return dispatch_error(id, notification, -32601, "method not found");
}

struct RetrievalRpcServer::Impl {
    explicit Impl(RetrievalService &service) noexcept : service{service} {}

    RetrievalService &service;
    SocketRuntime runtime;
    Socket listener;
    std::thread accept_thread;
    luisa::vector<std::thread> client_threads;
    luisa::vector<SocketHandle> clients;
    mutable std::mutex mutex;
    std::mutex stop_mutex;
    std::atomic<bool> stopping{false};
    std::atomic<bool> active{false};
    uint16_t bound_port{};
    size_t max_request_line_bytes{kDefaultMaxRequestLineBytes};
    luisa::string error;

    void set_error(luisa::string message) {
        std::lock_guard lock{mutex};
        error = std::move(message);
    }

    void remove_and_close_client(SocketHandle handle) noexcept {
        Socket socket{handle};
        std::lock_guard lock{mutex};
        const auto it = std::find(clients.begin(), clients.end(), handle);
        if (it != clients.end()) {
            clients.erase(it);
        }
        socket.close();
    }

    void handle_client(SocketHandle handle) noexcept {
        luisa::string pending;
        char buffer[16384];
        bool close_after_error = false;
        while (!stopping.load(std::memory_order_acquire) && !close_after_error) {
            const auto received = recv(handle, buffer, static_cast<int>(sizeof(buffer)), 0);
            if (received == 0) {
                break;
            }
            if (received < 0) {
                if (!stopping.load(std::memory_order_acquire)) {
                    LUISA_WARNING("retrieval RPC client receive failed with error {}", last_socket_error());
                }
                break;
            }
            pending.append(buffer, static_cast<size_t>(received));
            size_t consumed = 0u;
            while (true) {
                const auto newline = pending.find('\n', consumed);
                if (newline == luisa::string::npos) {
                    if (pending.size() - consumed > max_request_line_bytes) {
                        auto response = error_response(nullptr, -32700, "request line exceeds the configured size limit");
                        response.push_back('\n');
                        (void)send_all(handle, response);
                        close_after_error = true;
                    }
                    break;
                }
                const auto line_length = newline - consumed;
                if (line_length > max_request_line_bytes) {
                    auto response = error_response(nullptr, -32700, "request line exceeds the configured size limit");
                    response.push_back('\n');
                    (void)send_all(handle, response);
                    close_after_error = true;
                    break;
                }
                const luisa::string_view line{pending.data() + consumed, line_length};
                auto response = RetrievalRpcServer::process_json_line(service, line);
                if (response.has_value()) {
                    response->push_back('\n');
                    if (!send_all(handle, *response)) {
                        close_after_error = true;
                        break;
                    }
                }
                consumed = newline + 1u;
            }
            if (consumed > 0u) {
                pending.erase(0u, consumed);
            }
        }
        remove_and_close_client(handle);
    }

    void accept_loop() noexcept {
        while (!stopping.load(std::memory_order_acquire)) {
            fd_set read_set;
            FD_ZERO(&read_set);
            FD_SET(listener.get(), &read_set);
            timeval timeout{};
            timeout.tv_sec = 0;
            timeout.tv_usec = 100000;
#ifdef _WIN32
            const auto selected = select(0, &read_set, nullptr, nullptr, &timeout);
#else
            const auto selected = select(listener.get() + 1, &read_set, nullptr, nullptr, &timeout);
#endif
            if (selected == 0) {
                continue;
            }
            if (selected == kSocketError) {
                if (!stopping.load(std::memory_order_acquire)) {
                    set_error(luisa::format("listener select failed with error {}", last_socket_error()));
                }
                break;
            }
            Socket client{accept(listener.get(), nullptr, nullptr)};
            if (!client.valid()) {
                if (!stopping.load(std::memory_order_acquire)) {
                    LUISA_WARNING("retrieval RPC accept failed with error {}", last_socket_error());
                    std::this_thread::sleep_for(std::chrono::milliseconds{10});
                }
                continue;
            }
            const auto handle = client.release();
#if defined(__APPLE__) && defined(SO_NOSIGPIPE)
            const int no_sigpipe = 1;
            setsockopt(handle, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe));
#endif
            {
                std::lock_guard lock{mutex};
                clients.push_back(handle);
            }
            client_threads.emplace_back([this, handle] { handle_client(handle); });
        }
        active.store(false, std::memory_order_release);
    }

    [[nodiscard]] bool bind_listener(luisa::string_view address, uint16_t port) {
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;
        hints.ai_flags = AI_PASSIVE;
        const luisa::string host{address};
        const auto service_name = luisa::format("{}", port);
        addrinfo *raw_info = nullptr;
        const auto gai_result = getaddrinfo(host.c_str(), service_name.c_str(), &hints, &raw_info);
        if (gai_result != 0) {
            set_error(luisa::format("cannot resolve listen address '{}': {}", host, gai_result));
            return false;
        }
        AddrInfoGuard info{raw_info};
        int last_error = 0;
        for (auto *candidate = info.value; candidate != nullptr; candidate = candidate->ai_next) {
            Socket socket{::socket(candidate->ai_family, candidate->ai_socktype, candidate->ai_protocol)};
            if (!socket.valid()) {
                last_error = last_socket_error();
                continue;
            }
            if (bind(socket.get(), candidate->ai_addr, static_cast<int>(candidate->ai_addrlen)) == kSocketError) {
                last_error = last_socket_error();
                continue;
            }
            if (listen(socket.get(), 16) == kSocketError) {
                last_error = last_socket_error();
                continue;
            }
            sockaddr_storage bound_address{};
#ifdef _WIN32
            int bound_length = static_cast<int>(sizeof(bound_address));
#else
            socklen_t bound_length = static_cast<socklen_t>(sizeof(bound_address));
#endif
            if (getsockname(socket.get(), reinterpret_cast<sockaddr *>(&bound_address), &bound_length) == kSocketError) {
                last_error = last_socket_error();
                continue;
            }
            if (bound_address.ss_family == AF_INET) {
                bound_port = ntohs(reinterpret_cast<sockaddr_in *>(&bound_address)->sin_port);
            } else if (bound_address.ss_family == AF_INET6) {
                bound_port = ntohs(reinterpret_cast<sockaddr_in6 *>(&bound_address)->sin6_port);
            } else {
                last_error = 0;
                continue;
            }
            listener = std::move(socket);
            return true;
        }
        set_error(luisa::format("cannot bind/listen on '{}:{}' (socket error {})", host, port, last_error));
        return false;
    }

    void stop() noexcept {
        std::lock_guard stop_lock{stop_mutex};
        if (stopping.exchange(true, std::memory_order_acq_rel) && !active.load(std::memory_order_acquire)) {
            return;
        }
        if (accept_thread.joinable()) {
            accept_thread.join();
        }
        listener.close();
        {
            std::lock_guard lock{mutex};
            for (const auto client : clients) {
                shutdown_socket(client);
            }
        }
        for (auto &thread : client_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        client_threads.clear();
        active.store(false, std::memory_order_release);
    }
};

RetrievalRpcServer::RetrievalRpcServer(RetrievalService &service) noexcept
    : _impl{luisa::make_unique<Impl>(service)} {}

RetrievalRpcServer::~RetrievalRpcServer() noexcept {
    stop();
}

bool RetrievalRpcServer::start(
    luisa::string_view listen_address,
    uint16_t port,
    luisa::string &error_message,
    size_t max_request_line_bytes) noexcept {
    if (_impl->active.load(std::memory_order_acquire) || _impl->listener.valid() || _impl->stopping.load(std::memory_order_acquire)) {
        error_message = "server instance has already been started";
        return false;
    }
    if (listen_address.empty() || max_request_line_bytes == 0u) {
        error_message = "listen address and request-line limit must be nonempty";
        return false;
    }
    if (!_impl->runtime.initialize(error_message)) {
        return false;
    }
    _impl->max_request_line_bytes = max_request_line_bytes;
    if (!_impl->bind_listener(listen_address, port)) {
        error_message = _impl->error.empty() ? "listener setup failed" : _impl->error;
        return false;
    }
    _impl->stopping.store(false, std::memory_order_release);
    _impl->active.store(true, std::memory_order_release);
    _impl->accept_thread = std::thread{[impl = _impl.get()] { impl->accept_loop(); }};
    LUISA_INFO("retrieval RPC listening on {}:{} (request-line limit {} bytes)",
               listen_address, _impl->bound_port, max_request_line_bytes);
    return true;
}

void RetrievalRpcServer::stop() noexcept {
    _impl->stop();
}

bool RetrievalRpcServer::running() const noexcept {
    return _impl->active.load(std::memory_order_acquire) &&
           !_impl->stopping.load(std::memory_order_acquire);
}

uint16_t RetrievalRpcServer::port() const noexcept {
    return _impl->bound_port;
}

luisa::string RetrievalRpcServer::runtime_error() const {
    std::lock_guard lock{_impl->mutex};
    return _impl->error;
}

}// namespace luisa::embed
