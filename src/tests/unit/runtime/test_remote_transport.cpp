// Test for the C++ remote backend TCP transport.
// This test covers:
// - request/response correlation over a localhost socket
// - asynchronous notification delivery
// - checksum failure closing the connection and waking waiters

#include "ut/ut.hpp"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

#include <asio.hpp>

#include "remote_protocol.h"
#include "remote_transport.h"

using namespace luisa;
using namespace luisa::compute::remote;
using namespace boost::ut;
using namespace boost::ut::literals;
using namespace std::chrono_literals;

namespace {

using Tcp = asio::ip::tcp;

[[nodiscard]] vector<std::byte> bytes_of(string_view text) {
    vector<std::byte> bytes;
    bytes.resize(text.size());
    std::memcpy(bytes.data(), text.data(), text.size());
    return bytes;
}

[[nodiscard]] bool equal_bytes(span<const std::byte> lhs,
                               span<const std::byte> rhs) noexcept {
    return lhs.size() == rhs.size() &&
           std::memcmp(lhs.data(), rhs.data(), lhs.size()) == 0;
}

struct ReceivedFrame {
    FrameHeader header;
    vector<std::byte> payload;
};

[[nodiscard]] bool read_frame(Tcp::socket &socket, ReceivedFrame &frame) {
    std::array<std::byte, frame_header_size> header_bytes{};
    asio::error_code error;
    asio::read(socket, asio::buffer(header_bytes), error);
    if (error) { return false; }
    string decode_error;
    if (!decode_frame_header(
            header_bytes, frame.header, decode_error)) {
        return false;
    }
    frame.payload.resize(static_cast<size_t>(frame.header.payload_size));
    asio::read(socket, asio::buffer(frame.payload), error);
    return !error &&
           payload_checksum(frame.payload) == frame.header.payload_checksum;
}

void write_frame(Tcp::socket &socket, MessageKind kind, uint64_t request_id,
                 span<const std::byte> payload, bool valid_checksum = true) {
    auto header = encode_frame_header(FrameHeader{
        .kind = kind,
        .request_id = request_id,
        .payload_size = payload.size(),
        .payload_checksum = valid_checksum ? payload_checksum(payload) : 1u});
    std::array<asio::const_buffer, 2u> buffers{
        asio::buffer(static_cast<const void *>(header.data()), header.size()),
        asio::buffer(payload.data(), payload.size())};
    asio::error_code error;
    asio::write(socket, buffers, error);
    expect(!error);
}

void test_request_and_notification() {
    asio::io_context io;
    Tcp::acceptor acceptor{io, {Tcp::v4(), 0u}};
    auto port = acceptor.local_endpoint().port();
    std::thread server{[&] {
        Tcp::socket socket{io};
        asio::error_code error;
        acceptor.accept(socket, error);
        expect(!error);
        ReceivedFrame request;
        expect(read_frame(socket, request));
        expect(request.header.kind == MessageKind::QUERY);
        expect(request.payload == bytes_of("backend"));
        auto response_body = bytes_of("metal");
        auto response = make_response_payload(
            MessageKind::QUERY, Status::OK, {}, response_body);
        write_frame(socket, MessageKind::RESPONSE,
                    request.header.request_id, response);
        auto notification = bytes_of("done");
        write_frame(socket, MessageKind::STREAM_LOG, 91u, notification);
        std::this_thread::sleep_for(50ms);
    }};

    Connection connection;
    std::mutex notification_mutex;
    std::condition_variable notification_cv;
    bool notified = false;
    connection.set_notification_handler(
        [&](MessageKind kind, uint64_t id, span<const std::byte> payload) {
            std::scoped_lock lock{notification_mutex};
            expect(kind == MessageKind::STREAM_LOG);
            expect(id == 91u);
            auto expected = bytes_of("done");
            expect(equal_bytes(payload, expected));
            notified = true;
            notification_cv.notify_one();
        });
    string error;
    expect(connection.connect("127.0.0.1", port, 2s, error));
    auto query = bytes_of("backend");
    auto response = connection.request(MessageKind::QUERY, query, 2s);
    expect(static_cast<bool>(response));
    expect(response.request_kind == MessageKind::QUERY);
    expect(response.body == bytes_of("metal"));
    {
        std::unique_lock lock{notification_mutex};
        expect(notification_cv.wait_for(lock, 2s, [&] { return notified; }));
    }
    connection.close();
    server.join();
}

void test_bad_checksum_closes_connection() {
    asio::io_context io;
    Tcp::acceptor acceptor{io, {Tcp::v4(), 0u}};
    auto port = acceptor.local_endpoint().port();
    std::thread server{[&] {
        Tcp::socket socket{io};
        asio::error_code error;
        acceptor.accept(socket, error);
        expect(!error);
        ReceivedFrame request;
        expect(read_frame(socket, request));
        auto response = make_response_payload(
            MessageKind::QUERY, Status::OK, {});
        write_frame(socket, MessageKind::RESPONSE,
                    request.header.request_id, response, false);
        std::this_thread::sleep_for(20ms);
    }};

    Connection connection;
    string error;
    expect(connection.connect("127.0.0.1", port, 2s, error));
    auto response = connection.request(MessageKind::QUERY, {}, 2s);
    expect(!static_cast<bool>(response));
    expect(response.status == Status::CONNECTION_CLOSED);
    expect(response.message ==
           "Remote protocol payload checksum mismatch.");
    expect(!connection.connected());
    connection.close();
    server.join();
}

void test_reconnect_same_connection() {
    asio::io_context io;
    Tcp::acceptor acceptor{io, {Tcp::v4(), 0u}};
    auto port = acceptor.local_endpoint().port();
    std::thread server{[&] {
        for (auto attempt = 0u; attempt < 2u; attempt++) {
            Tcp::socket socket{io};
            asio::error_code error;
            acceptor.accept(socket, error);
            expect(!error);
            ReceivedFrame request;
            expect(read_frame(socket, request));
            expect(request.header.kind == MessageKind::QUERY);
            Writer body;
            body.write_u32(attempt);
            auto response = make_response_payload(
                MessageKind::QUERY, Status::OK, {}, body.bytes());
            write_frame(socket, MessageKind::RESPONSE,
                        request.header.request_id, response);
            socket.shutdown(Tcp::socket::shutdown_both, error);
            socket.close(error);
        }
    }};

    Connection connection;
    string error;
    expect(connection.connect("127.0.0.1", port, 2s, error));
    auto first = connection.request(MessageKind::QUERY, {}, 2s);
    expect(static_cast<bool>(first));
    Reader first_reader{first.body};
    expect(first_reader.read_u32() == 0u);
    expect(first_reader.finish());
    auto deadline = std::chrono::steady_clock::now() + 2s;
    while (connection.connected() &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    expect(!connection.connected());

    error.clear();
    expect(connection.connect("127.0.0.1", port, 2s, error)) << error;
    auto second = connection.request(MessageKind::QUERY, {}, 2s);
    expect(static_cast<bool>(second));
    Reader second_reader{second.body};
    expect(second_reader.read_u32() == 1u);
    expect(second_reader.finish());
    connection.close();
    server.join();
}

}// namespace

static auto test_remote_transport_registration = [] {
    "remote_transport_request_and_notification"_test =
        test_request_and_notification;
    "remote_transport_bad_checksum"_test =
        test_bad_checksum_closes_connection;
    "remote_transport_reconnect_same_connection"_test =
        test_reconnect_same_connection;
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}
