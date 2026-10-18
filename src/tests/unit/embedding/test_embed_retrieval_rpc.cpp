// Test for embedding/embed/retrieval_rpc.h — host-only protocol and TCP transport.
// This test covers all RPC methods, JSON-RPC errors and IDs, notifications,
// yyjson string lifetimes/escaping, fragmented/coalesced NDJSON frames, oversized
// lines, disconnects, concurrent clients, ephemeral ports, and clean teardown.
// Run: xmake run test_embed_retrieval_rpc
#include "ut/ut.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/mathematics.h>
#include <luisa/core/stl/filesystem.h>
#include <yyjson.h>

#include "embedding/embed/retrieval_rpc.h"

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
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#endif

using namespace luisa;
using namespace luisa::embed;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<uint64_t> sequence{0u};
        const auto ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::error_code ec;
        const auto base = luisa::filesystem::temp_directory_path(ec);
        if (ec) {
            LUISA_ERROR("cannot locate temporary directory");
        }
        _path = base / luisa::format("lc_retrieval_rpc_{}_{}", ticks, sequence.fetch_add(1u));
        luisa::filesystem::create_directories(_path, ec);
        if (ec) {
            LUISA_ERROR("cannot create temporary test directory");
        }
    }
    ~TempDirectory() noexcept {
        std::error_code ec;
        luisa::filesystem::remove_all(_path, ec);
    }
    [[nodiscard]] const luisa::filesystem::path &path() const noexcept { return _path; }

private:
    luisa::filesystem::path _path;
};

void write_file(const luisa::filesystem::path &path, luisa::string_view contents) {
    std::error_code ec;
    luisa::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        LUISA_ERROR("cannot create test directory");
    }
    std::ofstream stream{path, std::ios::binary | std::ios::trunc};
    if (!stream.is_open()) {
        LUISA_ERROR("cannot create test file '{}'", path.string());
    }
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
}

[[nodiscard]] luisa::string utf8_path(const luisa::filesystem::path &path) {
    std::error_code ec;
    const auto canonical = luisa::filesystem::canonical(path, ec);
    const auto &selected = ec ? path : canonical;
    const auto utf8 = selected.generic_u8string();
    return {reinterpret_cast<const char *>(utf8.data()), utf8.size()};
}

[[nodiscard]] RetrievalEncodeCallback fake_encoder() {
    return [](luisa::string_view text) -> luisa::vector<float> {
        if (text.find("beta") != luisa::string_view::npos) {
            return {0.0f, 1.0f};
        }
        return {1.0f, 0.0f};
    };
}

[[nodiscard]] luisa::string json_quote(luisa::string_view value) {
    luisa::string output{"\""};
    constexpr char hex[] = "0123456789abcdef";
    for (const char c : value) {
        const auto byte = static_cast<uint8_t>(c);
        switch (c) {
            case '\"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (byte < 0x20u) {
                    output += "\\u00";
                    output.push_back(hex[byte >> 4u]);
                    output.push_back(hex[byte & 0x0fu]);
                } else {
                    output.push_back(c);
                }
                break;
        }
    }
    output.push_back('\"');
    return output;
}

class JsonDocument {
public:
    explicit JsonDocument(luisa::string_view text)
        : _document{yyjson_read(text.data(), text.size(), YYJSON_READ_NOFLAG)} {}
    ~JsonDocument() noexcept { yyjson_doc_free(_document); }
    [[nodiscard]] yyjson_val *root() const noexcept {
        return _document == nullptr ? nullptr : yyjson_doc_get_root(_document);
    }
    [[nodiscard]] bool valid() const noexcept { return _document != nullptr; }

private:
    yyjson_doc *_document{};
};

void expect_error(luisa::string_view json, int32_t code) {
    JsonDocument document{json};
    expect(document.valid());
    auto *root = document.root();
    auto *error = yyjson_obj_get(root, "error");
    expect(yyjson_is_obj(error));
    expect(yyjson_get_sint(yyjson_obj_get(error, "code")) == code);
    expect(yyjson_is_str(yyjson_obj_get(error, "message")));
}

void expect_numeric_id(luisa::string_view json, double expected) {
    JsonDocument document{json};
    auto *id = yyjson_obj_get(document.root(), "id");
    expect(yyjson_is_num(id));
    expect(luisa::abs(yyjson_get_num(id) - expected) < 1e-12);
}

[[nodiscard]] luisa::string request_with_path(
    luisa::string_view method,
    luisa::string_view path,
    int32_t id = 1) {
    return luisa::format(
        "{{\"jsonrpc\":\"2.0\",\"id\":{},\"method\":\"{}\",\"params\":{{\"path\":{}}}}}",
        id, method, json_quote(path));
}

[[nodiscard]] luisa::string query_request(int32_t id, luisa::string_view text = "alpha") {
    return luisa::format(
        "{{\"jsonrpc\":\"2.0\",\"id\":{},\"method\":\"query\",\"params\":{{\"text\":{}}}}}",
        id, json_quote(text));
}

void reg_rpc_methods_and_ids() {
    "retrieval_rpc_methods_and_ids"_test = [] {
        TempDirectory temporary;
        const auto root = temporary.path() / "root with spaces";
        const auto file = root / "a.txt";
        write_file(file, "alpha");
        RetrievalService service{fake_encoder()};

        auto response = RetrievalRpcServer::process_json_line(
            service, request_with_path("load", utf8_path(root), 1));
        expect(response.has_value());
        if (response.has_value()) {
            expect_numeric_id(*response, 1.0);
            JsonDocument document{*response};
            auto *result = yyjson_obj_get(document.root(), "result");
            expect(yyjson_is_obj(result));
            expect(yyjson_get_uint(yyjson_obj_get(result, "added")) == 1u);
            expect(yyjson_get_uint(yyjson_obj_get(result, "total_documents")) == 1u);
            expect(yyjson_is_arr(yyjson_obj_get(result, "extensions")));
            auto *path = yyjson_obj_get(result, "path");
            expect(yyjson_is_str(path));
            expect(utf8_path(root) == luisa::string{yyjson_get_str(path), yyjson_get_len(path)});
        }

        write_file(file, "beta");
        response = RetrievalRpcServer::process_json_line(
            service, request_with_path("update", utf8_path(file), 2));
        expect(response.has_value());
        if (response.has_value()) {
            JsonDocument document{*response};
            auto *result = yyjson_obj_get(document.root(), "result");
            expect(yyjson_get_uint(yyjson_obj_get(result, "updated")) == 1u);
            expect(yyjson_is_str(yyjson_obj_get(result, "path")));
        }

        response = RetrievalRpcServer::process_json_line(
            service,
            R"({"jsonrpc":"2.0","id":"query-id","method":"query","params":{"text":"beta","top_k":1}})");
        expect(response.has_value());
        if (response.has_value()) {
            JsonDocument document{*response};
            expect(yyjson_equals_str(yyjson_obj_get(document.root(), "id"), "query-id"));
            auto *result = yyjson_obj_get(document.root(), "result");
            auto *hits = yyjson_obj_get(result, "results");
            expect(yyjson_arr_size(hits) == 1u);
            auto *hit = yyjson_arr_get(hits, 0u);
            expect(yyjson_is_str(yyjson_obj_get(hit, "path")));
            expect(luisa::abs(yyjson_get_num(yyjson_obj_get(hit, "score")) - 1.0) < 1e-12);
        }

        response = RetrievalRpcServer::process_json_line(
            service, request_with_path("unload", utf8_path(root), 9));
        expect(response.has_value());
        if (response.has_value()) {
            JsonDocument document{*response};
            auto *result = yyjson_obj_get(document.root(), "result");
            expect(yyjson_get_uint(yyjson_obj_get(result, "removed_documents")) == 1u);
            expect(yyjson_get_uint(yyjson_obj_get(result, "total_documents")) == 0u);
        }

        response = RetrievalRpcServer::process_json_line(
            service, R"({"jsonrpc":"2.0","id":1.25,"method":"query","params":{"text":"alpha"}})");
        expect(response.has_value());
        if (response.has_value()) {
            expect_numeric_id(*response, 1.25);
            expect_error(*response, -32000);
        }
    };
}

void reg_rpc_errors_and_notifications() {
    "retrieval_rpc_errors_and_notifications"_test = [] {
        TempDirectory temporary;
        RetrievalService service{fake_encoder()};
        const auto parse_error = RetrievalRpcServer::process_json_line(service, "{");
        expect(parse_error.has_value());
        if (parse_error) expect_error(*parse_error, -32700);
        const auto batch = RetrievalRpcServer::process_json_line(service, "[]");
        expect(batch.has_value());
        if (batch) expect_error(*batch, -32600);
        const auto version = RetrievalRpcServer::process_json_line(
            service, R"({"jsonrpc":"1.0","id":7,"method":"query","params":{"text":"x"}})");
        expect(version.has_value());
        if (version) {
            expect_error(*version, -32600);
            expect_numeric_id(*version, 7.0);
        }
        const auto bad_id = RetrievalRpcServer::process_json_line(
            service, R"({"jsonrpc":"2.0","id":true,"method":"query","params":{}})");
        expect(bad_id.has_value());
        if (bad_id) expect_error(*bad_id, -32600);
        const auto missing_params = RetrievalRpcServer::process_json_line(
            service, R"({"jsonrpc":"2.0","id":1,"method":"query"})");
        expect(missing_params.has_value());
        if (missing_params) expect_error(*missing_params, -32602);
        const auto wrong_params = RetrievalRpcServer::process_json_line(
            service, R"({"jsonrpc":"2.0","id":1,"method":"query","params":[]})");
        expect(wrong_params.has_value());
        if (wrong_params) expect_error(*wrong_params, -32602);
        const auto unknown = RetrievalRpcServer::process_json_line(
            service, R"({"jsonrpc":"2.0","id":2,"method":"help","params":{}})");
        expect(unknown.has_value());
        if (unknown) expect_error(*unknown, -32601);
        const luisa::vector<luisa::string> invalid_top{
            R"({"jsonrpc":"2.0","id":1,"method":"query","params":{"text":"x","top_k":0}})",
            R"({"jsonrpc":"2.0","id":1,"method":"query","params":{"text":"x","top_k":101}})",
            R"({"jsonrpc":"2.0","id":1,"method":"query","params":{"text":"x","top_k":1.5}})",
            R"({"jsonrpc":"2.0","id":1,"method":"query","params":{"text":"x","top_k":-1}})",
            R"({"jsonrpc":"2.0","id":1,"method":"query","params":{"text":"x","top_k":"5"}})"};
        for (const auto &request : invalid_top) {
            const auto result = RetrievalRpcServer::process_json_line(service, request);
            expect(result.has_value());
            if (result) expect_error(*result, -32602);
        }
        const auto empty_text = RetrievalRpcServer::process_json_line(
            service, R"({"jsonrpc":"2.0","id":3,"method":"query","params":{"text":"  "}})");
        expect(empty_text.has_value());
        if (empty_text) expect_error(*empty_text, -32602);
        const auto empty_corpus = RetrievalRpcServer::process_json_line(
            service, R"({"jsonrpc":"2.0","id":4,"method":"query","params":{"text":"x"}})");
        expect(empty_corpus.has_value());
        if (empty_corpus) expect_error(*empty_corpus, -32000);

        const auto bad_extensions = RetrievalRpcServer::process_json_line(
            service, R"({"jsonrpc":"2.0","id":5,"method":"load","params":{"path":".","extensions":[]}})");
        expect(bad_extensions.has_value());
        if (bad_extensions) expect_error(*bad_extensions, -32602);
        expect(!RetrievalRpcServer::process_json_line(
                    service, R"({"jsonrpc":"2.0","method":"unknown","params":{}})")
                    .has_value());
        expect(!RetrievalRpcServer::process_json_line(
                    service, R"({"jsonrpc":"2.0","method":"query","params":{}})")
                    .has_value());

        write_file(temporary.path() / "note.txt", "alpha");
        const auto notification = luisa::format(
            "{{\"jsonrpc\":\"2.0\",\"method\":\"load\",\"params\":{{\"path\":{}}}}}",
            json_quote(utf8_path(temporary.path())));
        expect(!RetrievalRpcServer::process_json_line(service, notification).has_value());
        const auto query = RetrievalRpcServer::process_json_line(service, query_request(12));
        expect(query.has_value());
        if (query) {
            JsonDocument document{*query};
            auto *result = yyjson_obj_get(document.root(), "result");
            expect(yyjson_arr_size(yyjson_obj_get(result, "results")) == 1u);
        }
    };
}

void reg_rpc_json_escaping() {
    "retrieval_rpc_json_escaping"_test = [] {
        RetrievalService service{fake_encoder()};
        const luisa::string id = "\"quoted\"\\line\n雪";
        const auto request = luisa::format(
            "{{\"jsonrpc\":\"2.0\",\"id\":{},\"method\":\"unknown\",\"params\":{{}}}}",
            json_quote(id));
        const auto response = RetrievalRpcServer::process_json_line(service, request);
        expect(response.has_value());
        if (!response) return;
        expect(response->find("\\\"") != luisa::string::npos);
        expect(response->find("\\\\") != luisa::string::npos);
        expect(response->find("\\n") != luisa::string::npos);
        expect(response->find("雪") != luisa::string::npos);
        JsonDocument document{*response};
        expect(document.valid());
        auto *response_id = yyjson_obj_get(document.root(), "id");
        expect(yyjson_is_str(response_id));
        if (yyjson_is_str(response_id)) {
            const luisa::string_view decoded{yyjson_get_str(response_id), yyjson_get_len(response_id)};
            expect(decoded == id);
            expect(decoded.find("\"quoted\"") != luisa::string_view::npos);
            expect(decoded.find('\n') != luisa::string_view::npos);
            expect(decoded.find("雪") != luisa::string_view::npos);
        }
        expect_error(*response, -32601);
    };
}

#ifdef _WIN32
using ClientSocketHandle = SOCKET;
constexpr ClientSocketHandle kInvalidClientSocket = INVALID_SOCKET;
#else
using ClientSocketHandle = int;
constexpr ClientSocketHandle kInvalidClientSocket = -1;
#endif

class ClientSocket {
public:
    ClientSocket() noexcept = default;
    explicit ClientSocket(ClientSocketHandle handle) noexcept : _handle{handle} {}
    ~ClientSocket() noexcept { close(); }
    ClientSocket(const ClientSocket &) = delete;
    ClientSocket &operator=(const ClientSocket &) = delete;

    [[nodiscard]] bool valid() const noexcept { return _handle != kInvalidClientSocket; }

    [[nodiscard]] bool connect(const char *host, uint16_t port) {
        close();
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;
        const auto service = luisa::format("{}", port);
        addrinfo *info = nullptr;
        if (getaddrinfo(host, service.c_str(), &hints, &info) != 0) {
            return false;
        }
        for (auto *candidate = info; candidate != nullptr; candidate = candidate->ai_next) {
            ClientSocketHandle handle = ::socket(candidate->ai_family, candidate->ai_socktype, candidate->ai_protocol);
            if (handle == kInvalidClientSocket) {
                continue;
            }
            if (::connect(handle, candidate->ai_addr, static_cast<int>(candidate->ai_addrlen)) == 0) {
                _handle = handle;
                break;
            }
#ifdef _WIN32
            closesocket(handle);
#else
            ::close(handle);
#endif
        }
        freeaddrinfo(info);
        if (!valid()) {
            return false;
        }
#ifdef _WIN32
        const DWORD timeout_ms = 3000u;
        setsockopt(_handle, SOL_SOCKET, SO_RCVTIMEO,
                   reinterpret_cast<const char *>(&timeout_ms), sizeof(timeout_ms));
#else
        timeval timeout{};
        timeout.tv_sec = 3;
        setsockopt(_handle, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#endif
        return true;
    }

    [[nodiscard]] bool send(luisa::string_view data) const {
        size_t offset = 0u;
        while (offset < data.size()) {
#ifdef MSG_NOSIGNAL
            constexpr int flags = MSG_NOSIGNAL;
#else
            constexpr int flags = 0;
#endif
            const auto sent = ::send(_handle, data.data() + offset,
                                     static_cast<int>(data.size() - offset), flags);
            if (sent <= 0) {
                return false;
            }
            offset += static_cast<size_t>(sent);
        }
        return true;
    }

    [[nodiscard]] bool send_line(luisa::string_view line) {
        luisa::string framed{line};
        framed.push_back('\n');
        return send(framed);
    }

    [[nodiscard]] bool read_line(luisa::string &line) {
        while (true) {
            const auto newline = _buffer.find('\n');
            if (newline != luisa::string::npos) {
                line.assign(_buffer.data(), newline);
                _buffer.erase(0u, newline + 1u);
                return true;
            }
            char bytes[4096];
            const auto received = recv(_handle, bytes, static_cast<int>(sizeof(bytes)), 0);
            if (received <= 0) {
                return false;
            }
            _buffer.append(bytes, static_cast<size_t>(received));
        }
    }

    [[nodiscard]] bool receive_eof() const {
        char byte{};
        return recv(_handle, &byte, 1, 0) == 0;
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
        _handle = kInvalidClientSocket;
    }

private:
    ClientSocketHandle _handle{kInvalidClientSocket};
    luisa::string _buffer;
};

void reg_rpc_tcp_transport() {
    "retrieval_rpc_tcp_transport"_test = [] {
        TempDirectory temporary;
        write_file(temporary.path() / "a.txt", "alpha");
        std::atomic<int32_t> active_encodes{0};
        std::atomic<bool> overlapping_encodes{false};
        RetrievalService service{[&](luisa::string_view text) -> luisa::vector<float> {
            if (active_encodes.fetch_add(1) != 0) {
                overlapping_encodes.store(true);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
            active_encodes.fetch_sub(1);
            if (text.find("beta") != luisa::string_view::npos) {
                return {0.0f, 1.0f};
            }
            return {1.0f, 0.0f};
        }};
        auto loaded = service.load(temporary.path());
        expect(loaded.status.ok());
        RetrievalRpcServer server{service};
        luisa::string error;
        const bool started = server.start("127.0.0.1", 0u, error, 256u);
        expect(started) << error;
        if (!started) return;
        expect(server.running() && server.port() != 0u);

        ClientSocket client;
        expect(client.connect("127.0.0.1", server.port()));
        const auto request10 = query_request(10);
        const auto request11 = query_request(11, "beta");
        expect(client.send(request10 + "\n" + request11 + "\n"));
        luisa::string response;
        expect(client.read_line(response));
        expect_numeric_id(response, 10.0);
        expect(client.read_line(response));
        expect_numeric_id(response, 11.0);

        const auto fragmented = query_request(12);
        expect(client.send(luisa::string_view{fragmented.data(), 5u}));
        expect(client.send(luisa::string_view{fragmented.data() + 5u, 17u}));
        expect(client.send(luisa::string_view{fragmented.data() + 22u, fragmented.size() - 22u}));
        expect(client.send("\n"));
        expect(client.read_line(response));
        expect_numeric_id(response, 12.0);
        expect(client.send_line(query_request(13)));
        expect(client.read_line(response));
        expect_numeric_id(response, 13.0);

        const auto notification_request = luisa::format(
            "{{\"jsonrpc\":\"2.0\",\"method\":\"update\",\"params\":{{\"path\":{}}}}}",
            json_quote(utf8_path(temporary.path())));
        expect(client.send(notification_request + "\n" + query_request(14) + "\n"));
        expect(client.read_line(response));
        expect_numeric_id(response, 14.0);

        ClientSocket disconnected;
        expect(disconnected.connect("127.0.0.1", server.port()));
        disconnected.close();
        expect(client.send_line(query_request(15)));
        expect(client.read_line(response));
        expect_numeric_id(response, 15.0);
        expect(client.send_line("{"));
        expect(client.read_line(response));
        expect_error(response, -32700);
        expect(client.send_line(query_request(16)));
        expect(client.read_line(response));
        expect_numeric_id(response, 16.0);

        ClientSocket oversized;
        expect(oversized.connect("127.0.0.1", server.port()));
        luisa::string oversized_line(300u, 'x');
        oversized_line.push_back('\n');
        expect(oversized.send(oversized_line));
        expect(oversized.read_line(response));
        expect_error(response, -32700);
        expect(oversized.receive_eof());
        oversized.close();

        std::atomic<bool> concurrent_ok{true};
        const auto worker = [&](int32_t base) {
            ClientSocket concurrent;
            if (!concurrent.connect("127.0.0.1", server.port())) {
                concurrent_ok.store(false);
                return;
            }
            for (int32_t i = 0; i < 6; i++) {
                if (!concurrent.send_line(query_request(base + i))) {
                    concurrent_ok.store(false);
                    return;
                }
                luisa::string concurrent_response;
                if (!concurrent.read_line(concurrent_response)) {
                    concurrent_ok.store(false);
                    return;
                }
                JsonDocument document{concurrent_response};
                auto *id = yyjson_obj_get(document.root(), "id");
                if (!yyjson_is_num(id) ||
                    luisa::abs(yyjson_get_num(id) - static_cast<double>(base + i)) > 1e-12 ||
                    yyjson_obj_get(document.root(), "error") != nullptr) {
                    concurrent_ok.store(false);
                    return;
                }
            }
        };
        std::thread first{worker, 100};
        std::thread second{worker, 200};
        first.join();
        second.join();
        expect(concurrent_ok.load());
        expect(!overlapping_encodes.load()) << "service calls must serialize encoder access";

        client.close();
        server.stop();
        expect(!server.running());
        server.stop();
    };
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    reg_rpc_methods_and_ids();
    reg_rpc_errors_and_notifications();
    reg_rpc_json_escaping();
    reg_rpc_tcp_transport();
    return 0;
}
