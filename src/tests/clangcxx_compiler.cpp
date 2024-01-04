#include <iostream>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include <luisa/clangcxx/compiler.h>
#include <luisa/core/thread_pool.h>
#include <luisa/runtime/context.h>
using namespace luisa;
using namespace luisa::compute;

static bool kTestRuntime = false;
string to_lower(string_view value) {
    string value_str{value};
    for (auto &i : value_str) {
        if (i >= 'A' && i <= 'Z') {
            i += 'a' - 'A';
        }
    }
    return value_str;
}

int main(int argc, char *argv[]) {
    log_level_error();
    //////// Properties
    if (argc <= 1) {
        LUISA_ERROR("Empty argument not allowed.");
    }
    std::filesystem::path src_path;
    std::filesystem::path dst_path;
    std::filesystem::path inc_path;
    luisa::unordered_set<luisa::string> defines;
    luisa::string backend = "dx";
    bool use_optimize = true;
    bool enable_help = false;
    bool enable_lsp = false;
    vstd::HashMap<vstd::string, vstd::function<void(vstd::string_view)>> cmds(16);
    auto invalid_arg = []() {
        LUISA_ERROR("Invalid argument, use --help please.");
    };
    cmds.emplace(
        "opt"sv,
        [&](string_view name) {
            auto lower_name = to_lower(name);
            if (lower_name == "on"sv) {
                use_optimize = true;
            } else if (lower_name == "off"sv) {
                use_optimize = false;
            } else {
                invalid_arg();
            }
        });
    cmds.emplace(
        "help"sv,
        [&](string_view name) {
            enable_help = true;
        });
    cmds.emplace(
        "backend"sv,
        [&](string_view name) {
            if (name.empty()) {
                invalid_arg();
            }
            auto lower_name = to_lower(name);
            backend = lower_name;
        });
    cmds.emplace(
        "in"sv,
        [&](string_view name) {
            if (name.empty()) {
                invalid_arg();
            }
            if (!src_path.empty()) {
                LUISA_ERROR("Source path set multiple times.");
            }
            src_path = name;
        });
    cmds.emplace(
        "out"sv,
        [&](string_view name) {
            if (name.empty()) {
                invalid_arg();
            }
            if (!dst_path.empty()) {
                LUISA_ERROR("Dest path set multiple times.");
            }
            dst_path = name;
        });
    cmds.emplace(
        "include"sv,
        [&](string_view name) {
            if (name.empty()) {
                invalid_arg();
            }
            if (!inc_path.empty()) {
                LUISA_ERROR("Include path set multiple times.");
            }
            inc_path = name;
        });
    cmds.emplace(
        "define"sv,
        [&](string_view name) {
            if (name.empty()) {
                invalid_arg();
            }
            defines.emplace(name);
        });
    cmds.emplace(
        "lsp"sv,
        [&](string_view name) {
            enable_lsp = true;
        });
    // TODO: define
    for (auto i : vstd::ptr_range(argv + 1, argc - 1)) {
        string arg = i;
        string_view kv_pair = arg;
        for (auto i : vstd::range(arg.size())) {
            if (arg[i] == '-')
                continue;
            else {
                kv_pair = string_view(arg.data() + i, arg.size() - i);
                break;
            }
        }
        if (kv_pair.empty() || kv_pair.size() == arg.size()) {
            invalid_arg();
        }
        string_view key = kv_pair;
        string_view value;
        for (auto i : vstd::range(kv_pair.size())) {
            if (kv_pair[i] == '=') {
                key = string_view(kv_pair.data(), i);
                value = string_view(kv_pair.data() + i + 1, kv_pair.size() - i - 1);
                break;
            }
        }
        auto iter = cmds.find(key);
        if (!iter) {
            invalid_arg();
        }
        iter.value()(value);
    }

    if (enable_help) {
        // print help list
        string_view helplist = R"(
Argument format:
    argument should be -ARG=VALUE or --ARG=VALUE, invalid argument will cause fatal error and never been ignored.

Argument list:
    --opt: enable or disable optimize, E.g --opt=on, --opt=off, case ignored.
    --backend: backend name, currently support "dx", "cuda", "metal", case ignored.
    --in: input file or dir, E.g --in=./my_dir/my_shader.cpp
    --out: output file or dir, E.g --out=./my_dir/my_shader.bin
    --include: include file directory, E.g --include=./shader_dir/
    --define: shader predefines, this can be set multiple times, E.g --define=MY_MACRO
    --lsp: enable compile_commands.json generation, E.g --lsp
)"sv;
        std::cout << helplist << '\n';
        return 0;
    }
    if (src_path.empty()) {
        LUISA_ERROR("Input file path not defined.");
    }
    Context context{argv[0]};
    if (src_path.is_relative()) {
        src_path = std::filesystem::current_path() / src_path;
    }
    std::error_code code;
    src_path = std::filesystem::canonical(src_path, code);
    if (code.value() != 0) {
        LUISA_ERROR("Invalid source file path");
    }
    auto format_path = [&]() {
        if (inc_path.is_relative()) {
            inc_path = std::filesystem::current_path() / inc_path;
        }
        if (dst_path.is_relative()) {
            dst_path = std::filesystem::current_path() / dst_path;
        }
        if (src_path == dst_path) {
            LUISA_ERROR("Source file and dest file path can not be the same.");
        }
        dst_path = std::filesystem::weakly_canonical(dst_path);
        inc_path = std::filesystem::weakly_canonical(inc_path, code);
        if (code.value() != 0) {
            LUISA_ERROR("Invalid include file path");
        }
    };
    auto ite_dir = [&](auto &&ite_dir, auto const &path, auto &&func) -> void {
        for (auto &i : std::filesystem::directory_iterator(path)) {
            if (i.is_directory()) {
                auto path_str = luisa::to_string(i.path().filename());
                if (path_str[0] == '.') {
                    continue;
                }
                ite_dir(ite_dir, i.path(), func);
                continue;
            }
            func(i.path());
        }
    };
    //////// LSP print
    if (enable_lsp) {
        if (!std::filesystem::is_directory(src_path)) {
            LUISA_ERROR("Source path must be a directory.");
        }
        if (dst_path.empty()) {
            dst_path = "compile_commands.json";
        } else if (std::filesystem::exists(dst_path) && !std::filesystem::is_regular_file(dst_path)) {
            LUISA_ERROR("Dest path must be a file.");
        }
        if (inc_path.empty()) {
            inc_path = src_path;
        }
        format_path();

        luisa::vector<char> result;
        result.reserve(16384);
        result.emplace_back('[');
        luisa::vector<std::filesystem::path> paths;
        auto func = [&](auto const &file_path_ref) {
            // auto file_path = file_path_ref;
            auto const &ext = file_path_ref.extension();
            if (ext != ".cpp" && ext != ".h" && ext != ".hpp") return;
            paths.emplace_back(file_path_ref);
        };
        ite_dir(ite_dir, src_path, func);
        if (!paths.empty()) {
            luisa::ThreadPool thread_pool(std::min<uint>(std::thread::hardware_concurrency(), paths.size()));
            std::mutex mtx;
            thread_pool.parallel(paths.size(), [&](size_t i) {
                auto &file_path = paths[i];
                if (file_path.is_absolute()) {
                    file_path = std::filesystem::relative(file_path, src_path);
                }
                luisa::vector<char> local_result;
                auto iter = vstd::range_linker{
                    vstd::make_ite_range(defines),
                    vstd::transform_range{[&](auto &&v) { return luisa::string_view{v}; }}}
                                .i_range();
                luisa::clangcxx::Compiler::lsp_compile_commands(
                    iter,
                    src_path,
                    file_path,
                    inc_path,
                    local_result);
                local_result.emplace_back(',');
                size_t idx = [&]() {
                    std::lock_guard lck{mtx};
                    auto sz = result.size();
                    result.push_back_uninitialized(local_result.size());
                    return sz;
                }();
                memcpy(result.data() + idx, local_result.data(), local_result.size());
            });
            thread_pool.synchronize();
        }
        if (result.size() > 1) {
            result.pop_back();
        }
        result.emplace_back(']');
        auto dst_path_str = luisa::to_string(dst_path);
        auto f = fopen(dst_path_str.c_str(), "wb");
        fwrite(result.data(), result.size(), 1, f);
        fclose(f);
        return 0;
    }
    //////// Compile all
    if (std::filesystem::is_directory(src_path)) {
        if (dst_path.empty()) {
            dst_path = src_path / "out";
        } else if (std::filesystem::exists(dst_path) && !std::filesystem::is_directory(dst_path)) {
            LUISA_ERROR("Dest path must be a directory.");
        }
        luisa::vector<std::filesystem::path> paths;
        auto create_dir = [&](auto &&path) {
            auto const &parent_path = path.parent_path();
            if (!std::filesystem::exists(parent_path))
                std::filesystem::create_directories(parent_path);
        };
        auto func = [&](auto const &file_path_ref) {
            if (file_path_ref.extension() != ".cpp") return;
            paths.emplace_back(file_path_ref);
        };
        ite_dir(ite_dir, src_path, func);
        if (paths.empty()) return 0;
        luisa::ThreadPool thread_pool(std::min<uint>(std::thread::hardware_concurrency(), paths.size()));
        auto add = [&]<typename T>(auto &&result, T c) {
            if constexpr (std::is_same_v<T, char const *> || std::is_same_v<T, char *>) {
                vstd::push_back_all(result, span<char const>(c, strlen(c)));
            } else if constexpr (std::is_same_v<T, char>) {
                result.emplace_back(c);
            } else {
                vstd::push_back_all(result, span<char const>(c.data(), c.size()));
            }
        };
        if (inc_path.empty()) {
            inc_path = src_path.parent_path();
        }
        format_path();
        auto inc_path_str = luisa::to_string(inc_path);
        log_level_info();
        thread_pool.parallel(
            paths.size(),
            [&](size_t i) {
                auto const &src_file_path = paths[i];
                auto file_path = src_file_path;
                if (file_path.is_absolute()) {
                    file_path = std::filesystem::relative(file_path, src_path);
                }
                auto out_path = dst_path / file_path;
                create_dir(out_path);
                out_path.replace_extension("bin");
                luisa::vector<char> vec;
                add(vec, argv[0]);
                add(vec, ' ');
                add(vec, "--opt="sv);
                add(vec, use_optimize ? "on"sv : "off"sv);
                add(vec, ' ');
                add(vec, "--backend="sv);
                add(vec, backend);
                add(vec, ' ');
                add(vec, "--in="sv);
                add(vec, luisa::to_string(src_file_path));
                add(vec, ' ');
                add(vec, "--out="sv);
                add(vec, luisa::to_string(out_path));
                add(vec, ' ');
                add(vec, "--include="sv);
                add(vec, inc_path_str);
                for (auto &i : defines) {
                    add(vec, ' ');
                    add(vec, "--define="sv);
                    add(vec, i);
                }
                vec.emplace_back(0);
                LUISA_INFO("{}", string_view(vec.data(), vec.size() - 1));
                system(vec.data());
            });
        thread_pool.synchronize();
        return 0;
    }
    //////// Compile
    if (dst_path.empty()) {
        dst_path = src_path.filename();
        if (dst_path.has_extension() && dst_path.extension() == "bin") {
            dst_path.replace_extension();
            dst_path.replace_extension(luisa::to_string(dst_path.filename()) + string("_out.bin"sv));
        } else {
            dst_path.replace_extension("bin");
        }
        // src_path.replace_extension()
    }
    if (inc_path.empty()) {
        inc_path = src_path.parent_path();
    }
    format_path();
    DeviceConfig config{
        .headless = true};
    Device device = context.create_device(backend, &config);
    auto iter = vstd::range_linker{
        vstd::make_ite_range(defines),
        vstd::transform_range{[&](auto &&v) { return luisa::string_view{v}; }}}
                    .i_range();
    luisa::clangcxx::Compiler::create_shader(
        ShaderOption{
            .enable_fast_math = use_optimize,
            .enable_debug_info = !use_optimize,
            .compile_only = true,
            .name = luisa::to_string(dst_path)},
        device, iter, src_path, inc_path);
    return 0;
}