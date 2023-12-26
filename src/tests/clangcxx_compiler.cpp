#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <iostream>
#include <luisa/core/stl/filesystem.h>
#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include <luisa/clangcxx/compiler.h>
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
    log_level_warning();
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
    vstd::HashMap<vstd::string, vstd::function<void(vstd::string_view)>> cmds;
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
    --in: input file, E.g --in=./my_dir/my_shader.cpp
    --out: output file, E.g --out=./my_dir/my_shader.bin
    --include: include file directory, E.g --include=./shader_dir/
    --define: shader predefines, this can be set multiple times, E.g --define=MY_MACRO
)"sv;
        std::cout << helplist << '\n';
        return 0;
    }
    //////// Compile
    if (src_path.empty()) {
        LUISA_ERROR("Input file path not defined.");
    }
    if (!std::filesystem::is_regular_file(src_path)) {
        LUISA_ERROR("Source path must be a file.");
    }
    if (dst_path.empty()) {
        dst_path = src_path;
        if (dst_path.has_extension() && dst_path.extension() == "bin") {
            dst_path.replace_extension();
            dst_path.replace_extension(luisa::to_string(dst_path.filename()) + string("_out.bin"sv));
        } else {
            dst_path.replace_extension("bin");
        }
        // src_path.replace_extension()
    } else if (!std::filesystem::is_regular_file(dst_path)) {
        LUISA_ERROR("Dest path must be a file.");
    }

    Context context{argv[0]};
    if (inc_path.empty()) {
        inc_path = src_path.parent_path();
    } else if (inc_path.is_relative()) {
        inc_path = context.runtime_directory() / inc_path;
    }
    if (src_path.is_relative()) {
        src_path = context.runtime_directory() / src_path;
    }
    if (dst_path.is_relative()) {
        dst_path = context.runtime_directory() / dst_path;
    }
    if (src_path == dst_path) {
        LUISA_ERROR("Source file and dest file path can not be the same.");
    }

    DeviceConfig config{
        .headless = true};
    Device device = context.create_device(backend, &config);
    auto compiler = luisa::clangcxx::Compiler(
        ShaderOption{
            .compile_only = true,
            .enable_fast_math = use_optimize,
            .enable_debug_info = !use_optimize,
            .name = luisa::to_string(dst_path)});
    compiler.create_shader(context, device, src_path, inc_path);
    return 0;
}