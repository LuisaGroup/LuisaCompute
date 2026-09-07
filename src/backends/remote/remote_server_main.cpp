#include <algorithm>
#include <charconv>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <string_view>
#include <thread>

#include <asio.hpp>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>

#include "remote_server.h"

namespace {

struct CommandLine {
    luisa::string backend;
    luisa::string listen{"127.0.0.1"};
    luisa::string token;
    luisa::vector<luisa::string> allowed_backends;
    uint16_t port{18080u};
    size_t device_index{std::numeric_limits<size_t>::max()};
    uint64_t max_sessions{64u};
    uint64_t blob_cache_bytes{512ull * 1024ull * 1024ull};
    uint64_t blob_cache_entry_bytes{64ull * 1024ull * 1024ull};
    uint64_t blob_cache_min_bytes{64ull * 1024ull};
    bool validation{false};
    bool allow_client_validation{false};
    bool print_ready{false};
    bool help{false};
};

[[nodiscard]] bool parse_unsigned(
    std::string_view text, uint64_t &value) noexcept {
    auto result = std::from_chars(
        text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} &&
           result.ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_command_line(
    int argc, char *argv[], CommandLine &command,
    luisa::string &error) noexcept {
    if (auto token = std::getenv("LUISA_REMOTE_TOKEN")) {
        command.token = token;
    }
    for (auto i = 1; i < argc; i++) {
        auto argument = std::string_view{argv[i]};
        if (argument == "--help" || argument == "-h") {
            command.help = true;
            continue;
        }
        if (argument == "--validation") {
            command.validation = true;
            continue;
        }
        if (argument == "--print-ready") {
            command.print_ready = true;
            continue;
        }
        if (argument == "--allow-client-validation") {
            command.allow_client_validation = true;
            continue;
        }
        if (i + 1 >= argc) {
            error = luisa::format("Missing value after '{}'.", argument);
            return false;
        }
        auto value = std::string_view{argv[++i]};
        if (argument == "--backend") {
            command.backend = value;
        } else if (argument == "--allow-backend") {
            command.allowed_backends.emplace_back(value);
        } else if (argument == "--listen") {
            command.listen = value;
        } else if (argument == "--token") {
            command.token = value;
        } else if (argument == "--port") {
            uint64_t parsed{};
            if (!parse_unsigned(value, parsed) ||
                parsed > std::numeric_limits<uint16_t>::max()) {
                error = "Invalid remote server port.";
                return false;
            }
            command.port = static_cast<uint16_t>(parsed);
        } else if (argument == "--device-index") {
            uint64_t parsed{};
            if (!parse_unsigned(value, parsed) ||
                parsed > std::numeric_limits<size_t>::max()) {
                error = "Invalid native device index.";
                return false;
            }
            command.device_index = static_cast<size_t>(parsed);
        } else if (argument == "--blob-cache-bytes") {
            if (!parse_unsigned(value, command.blob_cache_bytes)) {
                error = "Invalid remote blob-cache byte capacity.";
                return false;
            }
        } else if (argument == "--blob-cache-entry-bytes") {
            if (!parse_unsigned(value, command.blob_cache_entry_bytes)) {
                error = "Invalid remote blob-cache entry byte limit.";
                return false;
            }
        } else if (argument == "--blob-cache-min-bytes") {
            if (!parse_unsigned(value, command.blob_cache_min_bytes)) {
                error = "Invalid remote blob-cache minimum byte size.";
                return false;
            }
        } else if (argument == "--max-sessions") {
            if (!parse_unsigned(value, command.max_sessions) ||
                command.max_sessions == 0u ||
                command.max_sessions > 4096u) {
                error = "Invalid remote concurrent-session limit.";
                return false;
            }
        } else {
            error = luisa::format("Unknown option '{}'.", argument);
            return false;
        }
    }
    return true;
}

void print_usage(std::ostream &stream) {
    stream << "Usage: luisa-remote-server [options]\n"
              "  --backend <name>       Default backend (default: first non-remote backend)\n"
              "  --allow-backend <name> Allow a client-selectable backend; repeatable\n"
              "  --listen <address>     Listen address (default: 127.0.0.1)\n"
              "  --port <port>          TCP port (default: 18080)\n"
              "  --token <token>        Shared token (or LUISA_REMOTE_TOKEN)\n"
              "  --device-index <index> Native device index\n"
              "  --max-sessions <n>     Concurrent client limit (default: 64)\n"
              "  --blob-cache-bytes <n> Shared blob-cache capacity; 0 disables it\n"
              "  --blob-cache-entry-bytes <n> Maximum cached upload size\n"
              "  --blob-cache-min-bytes <n> Recommended minimum cached upload size\n"
              "  --validation           Enable native validation layer\n"
              "  --allow-client-validation Allow clients to request native validation\n"
              "  --print-ready          Print 'LCRP_READY_V1 <port>' after binding\n"
              "  --help                  Show this help\n";
}

class SignalStopper final {

private:
    asio::io_context _io;
    asio::signal_set _signals;
    std::thread _thread;

public:
    explicit SignalStopper(luisa::compute::remote::Server &server)
        : _signals{_io, SIGINT, SIGTERM} {
        _signals.async_wait(
            [&server](const asio::error_code &error, int signal) {
                if (!error) {
                    LUISA_INFO(
                        "Stopping Luisa remote server after signal {}.",
                        signal);
                    server.stop();
                }
            });
        _thread = std::thread{[this] { _io.run(); }};
    }

    ~SignalStopper() noexcept {
        _io.stop();
        if (_thread.joinable()) { _thread.join(); }
    }

    SignalStopper(SignalStopper const &) = delete;
    SignalStopper(SignalStopper &&) = delete;
    SignalStopper &operator=(SignalStopper const &) = delete;
    SignalStopper &operator=(SignalStopper &&) = delete;
};

}// namespace

int main(int argc, char *argv[]) {
    CommandLine command;
    luisa::string error;
    if (!parse_command_line(argc, argv, command, error)) {
        std::cerr << error << '\n';
        print_usage(std::cerr);
        return 2;
    }
    if (command.help) {
        print_usage(std::cout);
        return 0;
    }
    try {
        luisa::compute::Context context{argv[0]};
        if (command.backend.empty()) {
            for (auto &&backend : context.installed_backends()) {
                if (backend != "remote") {
                    command.backend = backend;
                    break;
                }
            }
        }
        if (command.backend.empty() || command.backend == "remote") {
            std::cerr << "A non-remote native backend is required.\n";
            return 2;
        }
        auto installed = context.installed_backends();
        auto is_installed = [installed](luisa::string_view backend) noexcept {
            return backend != "remote" &&
                   std::find(installed.begin(), installed.end(), backend) !=
                       installed.end();
        };
        if (!is_installed(command.backend)) {
            std::cerr << "Default backend '" << command.backend
                      << "' is not installed or cannot be remote.\n";
            return 2;
        }
        if (command.allowed_backends.empty()) {
            command.allowed_backends.emplace_back(command.backend);
        } else if (std::find(
                       command.allowed_backends.begin(),
                       command.allowed_backends.end(), command.backend) ==
                   command.allowed_backends.end()) {
            command.allowed_backends.emplace_back(command.backend);
        }
        for (auto &&backend : command.allowed_backends) {
            if (!is_installed(backend)) {
                std::cerr << "Allowed backend '" << backend
                          << "' is not installed or cannot be remote.\n";
                return 2;
            }
        }
        std::sort(
            command.allowed_backends.begin(),
            command.allowed_backends.end());
        command.allowed_backends.erase(
            std::unique(
                command.allowed_backends.begin(),
                command.allowed_backends.end()),
            command.allowed_backends.end());
        auto factory_mutex = luisa::make_shared<std::mutex>();
        auto device_factory =
            [&context, factory_mutex,
             default_backend = command.backend,
             default_device_index = command.device_index,
             allowed_backends = command.allowed_backends,
             validation = command.validation,
             allow_client_validation = command.allow_client_validation](
                const luisa::compute::remote::DeviceRequest &request,
                luisa::string &factory_error)
            -> luisa::shared_ptr<luisa::compute::DeviceInterface> {
            auto backend = request.backend.empty() ?
                               default_backend :
                               request.backend;
            if (!std::binary_search(
                    allowed_backends.begin(), allowed_backends.end(), backend)) {
                factory_error = luisa::format(
                    "Backend '{}' is not allowed by this remote service.",
                    backend);
                return nullptr;
            }
            if (request.enable_validation && !allow_client_validation) {
                factory_error = "Client-requested native validation is disabled by this remote service.";
                return nullptr;
            }
            auto device_index =
                request.device_index == std::numeric_limits<size_t>::max() ?
                    default_device_index :
                    request.device_index;
            std::scoped_lock lock{*factory_mutex};
            if (device_index != std::numeric_limits<size_t>::max()) {
                auto names = context.backend_device_names(backend);
                if (device_index >= names.size()) {
                    factory_error = luisa::format(
                        "Device index {} is out of range for backend '{}'.",
                        device_index, backend);
                    return nullptr;
                }
            }
            luisa::compute::DeviceConfig config;
            config.device_index = device_index;
            // AST compilation must remain available on the execution host.
            config.headless = false;
            auto device = context.create_device(
                backend, &config,
                validation || request.enable_validation);
            auto native = device.impl_shared();
            if (native == nullptr) {
                factory_error = luisa::format(
                    "Backend '{}' failed to create a device.", backend);
                return nullptr;
            }
            LUISA_INFO(
                "Created remote service device using backend '{}'.", backend);
            return native;
        };
        luisa::compute::remote::ServerOptions options;
        options.listen_address = command.listen;
        options.port = command.port;
        options.token = command.token;
        options.max_blob_cache_bytes = command.blob_cache_bytes;
        options.max_blob_entry_size = command.blob_cache_entry_bytes;
        options.blob_cache_min_size = command.blob_cache_min_bytes;
        options.max_concurrent_sessions = command.max_sessions;
        luisa::compute::remote::Server server{
            std::move(device_factory), std::move(options)};
        LUISA_INFO(
            "Luisa remote service is listening on {}:{} with default backend '{}' and {} allowed backend(s).",
            command.listen, server.port(), command.backend,
            command.allowed_backends.size());
        if (command.print_ready) {
            std::cout << "LCRP_READY_V1 " << server.port() << '\n'
                      << std::flush;
            if (!std::cout) {
                std::cerr << "Failed to publish remote server readiness.\n";
                return 1;
            }
        }
        SignalStopper signal_stopper{server};
        server.run();
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "Remote server failed: " << exception.what() << '\n';
        return 1;
    }
}
