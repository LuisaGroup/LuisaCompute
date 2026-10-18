// Interactive whole-file embedding retrieval console and TCP JSON-RPC server
// (retrieval plan phases 2-4; target example_embed_retrieval). The existing
// example_embed_text executable and its command-line contract are independent.

#include <charconv>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <thread>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>

#include "embedding/embed/retrieval_rpc.h"
#include "embedding/embed/retrieval_service.h"
#include "embedding/embed/session.h"

using namespace luisa;

namespace {

volatile std::sig_atomic_t interrupted = 0;

void handle_interrupt(int) noexcept {
    interrupted = 1;
}

struct ProgramOptions {
    luisa::filesystem::path model;
    luisa::filesystem::path tokenizer;
    luisa::string backend{"vk"};
    luisa::string listen_address{"127.0.0.1"};
    uint16_t port{8765u};
    bool server{};
    bool help{};
    bool ok{true};
    luisa::string error;
};

void print_usage() {
    std::cout
        << "usage: example_embed_retrieval --model <model.bgm> --tokenizer <tokenizer.json>\n"
        << "                              [--backend <name>] [--server]\n"
        << "                              [--listen <address>] [--port <port>]\n\n"
        << "  --backend defaults to vk; --server selects TCP JSON-RPC instead of the REPL.\n"
        << "  Server defaults: --listen 127.0.0.1 --port 8765 (port 0 selects an ephemeral port).\n";
}

void print_command_help() {
    std::cout
        << "commands:\n"
        << "  load <directory> [<extension-list>]   recursively index; default .txt,.md\n"
        << "  unload <loaded-root-directory>       unload exactly one loaded root\n"
        << "  query [--top <N>] <query text>       rank files; default top 5, N in [1,100]\n"
        << "  update <file-or-subdirectory>        explicitly re-index changed content\n"
        << "  help                                 show this grammar\n"
        << "  quit | exit                          leave the console\n"
        << "Use quotes for spaces and \\\" for a quoted quote. Backslashes otherwise remain literal.\n"
        << "One file has one embedding; default encoding truncates at 512 tokens.\n";
}

[[nodiscard]] bool parse_uint32(luisa::string_view text, uint32_t &value) noexcept {
    if (text.empty()) {
        return false;
    }
    uint32_t parsed = 0u;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return false;
    }
    value = parsed;
    return true;
}

[[nodiscard]] ProgramOptions parse_program_options(int argc, char *argv[]) {
    ProgramOptions options;
    for (int i = 1; i < argc; i++) {
        const luisa::string_view argument{argv[i]};
        const auto require_value = [&](luisa::string_view &value) {
            if (i + 1 >= argc) {
                options.ok = false;
                options.error = luisa::format("option '{}' requires a value", argument);
                return false;
            }
            value = luisa::string_view{argv[++i]};
            return true;
        };
        if (argument == "--help") {
            options.help = true;
        } else if (argument == "--server") {
            options.server = true;
        } else if (argument == "--model" || argument == "--tokenizer" ||
                   argument == "--backend" || argument == "--listen" || argument == "--port") {
            luisa::string_view value;
            if (!require_value(value)) {
                return options;
            }
            if (argument == "--model") {
                auto status = embed::RetrievalIndex::path_from_utf8(value, options.model);
                if (!status.ok()) {
                    options.ok = false;
                    options.error = std::move(status.message);
                    return options;
                }
            } else if (argument == "--tokenizer") {
                auto status = embed::RetrievalIndex::path_from_utf8(value, options.tokenizer);
                if (!status.ok()) {
                    options.ok = false;
                    options.error = std::move(status.message);
                    return options;
                }
            } else if (argument == "--backend") {
                options.backend.assign(value.data(), value.size());
                if (options.backend.empty()) {
                    options.ok = false;
                    options.error = "backend name must not be empty";
                    return options;
                }
            } else if (argument == "--listen") {
                options.listen_address.assign(value.data(), value.size());
                if (options.listen_address.empty()) {
                    options.ok = false;
                    options.error = "listen address must not be empty";
                    return options;
                }
            } else {
                uint32_t port = 0u;
                if (!parse_uint32(value, port) || port > 65535u) {
                    options.ok = false;
                    options.error = "port must be an integer in [0, 65535]";
                    return options;
                }
                options.port = static_cast<uint16_t>(port);
            }
        } else {
            options.ok = false;
            options.error = luisa::format("unknown option '{}'", argument);
            return options;
        }
    }
    return options;
}

[[nodiscard]] bool validate_startup_files(const ProgramOptions &options) {
    if (options.model.empty() || options.tokenizer.empty()) {
        std::cerr << "error: --model and --tokenizer are required\n";
        print_usage();
        return false;
    }
    std::error_code ec;
    if (!luisa::filesystem::is_regular_file(options.model, ec) || ec) {
        std::cerr << "error: packed model does not exist or is not a regular file: "
                  << options.model << '\n';
        return false;
    }
    ec.clear();
    if (!luisa::filesystem::is_regular_file(options.tokenizer, ec) || ec) {
        std::cerr << "error: tokenizer does not exist or is not a regular file: "
                  << options.tokenizer << '\n';
        return false;
    }
    return true;
}

[[nodiscard]] bool tokenize_command(
    luisa::string_view line,
    luisa::vector<luisa::string> &tokens,
    luisa::string &error) {
    luisa::string current;
    char quote = '\0';
    bool token_started = false;
    for (size_t i = 0u; i < line.size(); i++) {
        const char c = line[i];
        if (quote == '\0' && std::isspace(static_cast<unsigned char>(c)) != 0) {
            if (token_started) {
                tokens.push_back(std::move(current));
                current.clear();
                token_started = false;
            }
            continue;
        }
        if (c == '\\' && i + 1u < line.size()) {
            const auto next = line[i + 1u];
            const bool escaped_quote = next != '\0' &&
                                       (next == quote || (quote == '\0' && (next == '\"' || next == '\'')));
            if (escaped_quote) {
                current.push_back(line[++i]);
                token_started = true;
                continue;
            }
        }
        if (c == '\"' || c == '\'') {
            if (quote == '\0') {
                quote = c;
                token_started = true;
                continue;
            }
            if (quote == c) {
                quote = '\0';
                continue;
            }
        }
        current.push_back(c);
        token_started = true;
    }
    if (quote != '\0') {
        error = "unterminated quoted argument";
        return false;
    }
    if (token_started) {
        tokens.push_back(std::move(current));
    }
    return true;
}

void print_status_error(const embed::RetrievalStatus &status) {
    std::cerr << "error: " << status.message << '\n';
}

[[nodiscard]] luisa::string join_extensions(luisa::span<const luisa::string> extensions) {
    luisa::string result;
    for (size_t i = 0u; i < extensions.size(); i++) {
        if (i != 0u) {
            result.push_back(',');
        }
        result += extensions[i];
    }
    return result;
}

void print_load(const embed::RetrievalLoadResult &result) {
    std::cout << "loaded " << result.root
              << " extensions=" << join_extensions(result.extensions)
              << " added=" << result.added
              << " skipped=" << result.skipped
              << " failed=" << result.failed
              << " total_documents=" << result.total_documents << '\n';
}

void print_unload(const embed::RetrievalUnloadResult &result) {
    std::cout << "unloaded " << result.root
              << " removed_documents=" << result.removed_documents
              << " total_documents=" << result.total_documents << '\n';
}

void print_update(const embed::RetrievalUpdateResult &result) {
    std::cout << "updated " << result.path
              << " added=" << result.added
              << " updated=" << result.updated
              << " removed=" << result.removed
              << " unchanged=" << result.unchanged
              << " failed=" << result.failed
              << " total_documents=" << result.total_documents << '\n';
}

void print_query(const embed::RetrievalQueryResult &result) {
    if (result.hits.empty()) {
        std::cout << "no results\n";
        return;
    }
    for (size_t i = 0u; i < result.hits.size(); i++) {
        const auto &hit = result.hits[i];
        std::cout << (i + 1u) << " " << luisa::format("{:.6f}", hit.score)
                  << " " << hit.path << '\n';
    }
}

void execute_command(embed::RetrievalService &service, const luisa::vector<luisa::string> &arguments) {
    const auto &command = arguments.front();
    if (command == "help") {
        if (arguments.size() != 1u) {
            std::cerr << "error: help takes no arguments\n";
        } else {
            print_command_help();
        }
        return;
    }
    if (command == "load") {
        if (arguments.size() < 2u || arguments.size() > 3u) {
            std::cerr << "error: usage: load <directory> [<extension-list>]\n";
            return;
        }
        luisa::filesystem::path root;
        auto path_status = embed::RetrievalIndex::path_from_utf8(arguments[1], root);
        if (!path_status.ok()) {
            print_status_error(path_status);
            return;
        }
        embed::RetrievalLoadResult result;
        if (arguments.size() == 2u) {
            result = service.load(root);
        } else {
            luisa::vector<luisa::string> extensions;
            auto status = embed::RetrievalIndex::parse_extension_list(arguments[2], extensions);
            if (!status.ok()) {
                print_status_error(status);
                return;
            }
            result = service.load(
                root,
                luisa::span<const luisa::string>{extensions.data(), extensions.size()});
        }
        if (!result.status.ok()) {
            print_status_error(result.status);
        } else {
            print_load(result);
        }
        return;
    }
    if (command == "unload") {
        if (arguments.size() != 2u) {
            std::cerr << "error: usage: unload <loaded-root-directory>\n";
            return;
        }
        luisa::filesystem::path root;
        auto path_status = embed::RetrievalIndex::path_from_utf8(arguments[1], root);
        if (!path_status.ok()) {
            print_status_error(path_status);
            return;
        }
        auto result = service.unload(root);
        if (!result.status.ok()) {
            print_status_error(result.status);
        } else {
            print_unload(result);
        }
        return;
    }
    if (command == "update") {
        if (arguments.size() != 2u) {
            std::cerr << "error: usage: update <file-or-subdirectory>\n";
            return;
        }
        luisa::filesystem::path path;
        auto path_status = embed::RetrievalIndex::path_from_utf8(arguments[1], path);
        if (!path_status.ok()) {
            print_status_error(path_status);
            return;
        }
        auto result = service.update(path);
        if (!result.status.ok()) {
            print_status_error(result.status);
        } else {
            print_update(result);
        }
        return;
    }
    if (command == "query") {
        size_t next = 1u;
        uint32_t top_k = embed::kRetrievalDefaultTopK;
        if (arguments.size() >= 2u && arguments[1] == "--top") {
            if (arguments.size() < 3u || !parse_uint32(arguments[2], top_k) ||
                top_k == 0u || top_k > embed::kRetrievalMaxTopK) {
                std::cerr << "error: --top requires an integer in [1,100]\n";
                return;
            }
            next = 3u;
        }
        if (next >= arguments.size()) {
            std::cerr << "error: query text must not be empty\n";
            return;
        }
        luisa::string text;
        for (size_t i = next; i < arguments.size(); i++) {
            if (i != next) {
                text.push_back(' ');
            }
            text += arguments[i];
        }
        auto result = service.query(text, top_k);
        if (!result.status.ok()) {
            print_status_error(result.status);
        } else {
            print_query(result);
        }
        return;
    }
    std::cerr << "error: unknown command '" << command << "'; type help for usage\n";
}

void run_repl(embed::RetrievalService &service) {
    std::cout << "Embedding retrieval console; type help for commands.\n";
    print_command_help();
      while (true) {
          std::cout << "embed> " << std::flush;
          std::string line;
          if (!std::getline(std::cin, line)) {
              break;
          }
          luisa::string input{line.data(), line.size()};
        luisa::vector<luisa::string> arguments;
        luisa::string error;
        if (!tokenize_command(input, arguments, error)) {
            std::cerr << "error: " << error << '\n';
            continue;
        }
        if (arguments.empty()) {
            continue;
        }
        if (arguments.front() == "quit" || arguments.front() == "exit") {
            if (arguments.size() != 1u) {
                std::cerr << "error: quit/exit takes no arguments\n";
                continue;
            }
            break;
        }
        execute_command(service, arguments);
    }
}

int run_server(embed::RetrievalService &service, const ProgramOptions &options) {
    embed::RetrievalRpcServer server{service};
    luisa::string error;
    if (!server.start(options.listen_address, options.port, error)) {
        std::cerr << "error: " << error << '\n';
        return 1;
    }
    interrupted = 0;
    std::signal(SIGINT, handle_interrupt);
    std::signal(SIGTERM, handle_interrupt);
#ifdef _WIN32
    std::signal(SIGBREAK, handle_interrupt);
#endif
    while (interrupted == 0 && server.running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    error = server.runtime_error();
    server.stop();
    if (interrupted == 0 && !error.empty()) {
        std::cerr << "error: " << error << '\n';
        return 1;
    }
    return 0;
}

}// namespace

int main(int argc, char *argv[]) {
    const auto options = parse_program_options(argc, argv);
    if (!options.ok) {
        std::cerr << "error: " << options.error << '\n';
        print_usage();
        return 1;
    }
    if (options.help) {
        print_usage();
        return 0;
    }
    if (!validate_startup_files(options)) {
        return 1;
    }

    compute::Context context{argv[0]};
    LUISA_INFO("example_embed_retrieval: selecting backend '{}'", options.backend);
    auto device = context.create_device(options.backend);
    if (!device) {
        luisa::string installed;
        for (const auto &backend : context.installed_backends()) {
            if (!installed.empty()) {
                installed += ", ";
            }
            installed += backend;
        }
        std::cerr << "error: backend '" << options.backend
                  << "' is unavailable; installed backends: " << installed << '\n';
        return 1;
    }
    auto session = embed::Session::create(device, options.model, options.tokenizer);
    if (session == nullptr) {
        std::cerr << "error: Session::create failed\n";
        return 1;
    }
    // Fully initialize and drain the backend execution path once at startup.
    // This also makes an encode-free REPL session follow the same orderly
    // scheduler/backend teardown path as a session that served retrieval calls.
    if (session->encode("retrieval session warmup").dense.empty()) {
        std::cerr << "error: Session warm-up returned an empty dense embedding\n";
        return 1;
    }
    embed::RetrievalService service{[session = session.get()](luisa::string_view text) {
        return session->encode(text).dense;
    }};
    if (options.server) {
        return run_server(service, options);
    }
    run_repl(service);
    return 0;
}
