#include <iostream>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include <luisa/vstl/lmdb.hpp>
#include <luisa/vstl/md5.h>
#include <luisa/clangcxx/compiler.h>
#include <luisa/core/fiber.h>
#include <luisa/runtime/context.h>
#include <luisa/core/binary_file_stream.h>
#include <luisa/vstl/v_guid.h>
#include <luisa/vstl/spin_mutex.h>
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
template<typename Vec, typename T>
void add(Vec &&result, T c) {
    if constexpr (std::is_same_v<T, char const *> || std::is_same_v<T, char *>) {
        vstd::push_back_all(result, span<char const>(c, strlen(c)));
    } else if constexpr (std::is_same_v<T, char>) {
        result.emplace_back(c);
    } else {
        vstd::push_back_all(result, luisa::span<char const>(c.data(), c.size()));
    }
};

class Preprocessor {
    vstd::LMDB db;
    luisa::string _clang_path;
    std::filesystem::path _cache_path;
    luisa::vector<luisa::string_view> _defines;
    luisa::vector<luisa::string> _inc_paths;
    vstd::spin_mutex _time_mtx;
    using DBValue = luisa::vector<std::byte>;
    vstd::HashMap<luisa::string, DBValue> _last_write_times;
    vstd::spin_mutex _remove_mtx;
    luisa::vector<luisa::vector<std::byte>> _remove_list;
    void update_file(luisa::string_view name, std::filesystem::file_time_type time, luisa::span<const std::byte> data) {
        DBValue vec;
        vec.reserve(data.size() + sizeof(time));
        vstd::push_back_all(vec, reinterpret_cast<std::byte const *>(&time), sizeof(time));
        if (!data.empty()) {
            vstd::push_back_all(vec, reinterpret_cast<std::byte const *>(data.data()), data.size());
        }
        std::lock_guard lck{_time_mtx};
        auto iter = _last_write_times.try_emplace(name);
        if (iter.second || (iter.first.value().size() < data.size() + sizeof(time))) {
            iter.first.value() = std::move(vec);
        }
    }
    bool file_is_new(luisa::string_view name) {
        luisa::span<const std::byte> db_value;
        return file_is_new(name, db_value);
    }
    bool file_is_new(luisa::string_view name, luisa::span<const std::byte> &db_value) {
        db_value = db.read(name);
        auto cur_time = std::filesystem::last_write_time(std::filesystem::path{name});
        if (db_value.size_bytes() >= sizeof(std::filesystem::file_time_type)) {
            std::filesystem::file_time_type old_time;
            memcpy(&old_time, db_value.data(), sizeof(old_time));
            if (cur_time <= old_time) {
                return false;
            }
        }
        update_file(name, cur_time, {});
        return true;
    }

public:
    Preprocessor(
        std::filesystem::path const &clang_path,
        std::filesystem::path const &db_path,
        std::filesystem::path &&cache_path,
        vstd::IRange<luisa::string_view> &defines,
        vstd::IRange<luisa::string> &inc_paths)
        : db(db_path, std::max<size_t>(126ull, std::thread::hardware_concurrency() * 2)),
          _clang_path(luisa::to_string(clang_path)),
          _cache_path(std::move(cache_path)) {
        if (!std::filesystem::exists(_cache_path)) {
            std::error_code ec;
            std::filesystem::create_directories(_cache_path, ec);
            if (ec) [[unlikely]] {
                LUISA_ERROR("Create cache path '{}' failed, message: {}", luisa::to_string(_cache_path), ec.message());
            }
        }
        for (auto &&i : defines) {
            _defines.emplace_back(i);
        }
        for (auto &&i : inc_paths) {
            _inc_paths.emplace_back(std::move(i));
        }
    }
    void remove_file(luisa::string_view name) {
        std::lock_guard lck{_remove_mtx};
        auto &v = _remove_list.emplace_back();
        v.push_back_uninitialized(name.size());
        memcpy(v.data(), name.data(), name.size());
    }
    void post_process() {
        luisa::vector<vstd::LMDBWriteCommand> write_cmds;
        write_cmds.reserve(_last_write_times.size());
        for (auto &&i : _last_write_times) {
            auto &&kv = write_cmds.emplace_back();
            vstd::push_back_all(kv.key, reinterpret_cast<std::byte const *>(i.first.data()), i.first.size());
            kv.value = std::move(i.second);
        }
        _last_write_times.clear();
        db.write_all(std::move(write_cmds));
        db.remove_all(std::move(_remove_list));
        _remove_list.clear();
        if (std::filesystem::exists(_cache_path)) {
            std::error_code ec;
            std::filesystem::remove_all(_cache_path, ec);
            if (ec) [[unlikely]] {
                LUISA_WARNING("Delete temp dir: {} failed: {}", luisa::to_string(_cache_path), ec.message());
            }
        }
    }
    static luisa::unordered_set<luisa::string> include_dirs(luisa::string_view text) {
        auto ptr = text.data();
        auto end = text.data() + text.size();
        luisa::unordered_set<luisa::string> r;
#define NEXT_PTR                     \
    do {                             \
        ++ptr;                       \
        if (ptr == end) [[unlikely]] \
            return r;                \
    } while (false)
        for (; ptr != end; ++ptr) {
            if (*ptr != '#') [[likely]] {
                continue;
            }
            NEXT_PTR;
            while (*ptr != '"') {
                NEXT_PTR;
            }
            NEXT_PTR;
            auto path_begin = ptr;
            while (*ptr != '"') {
                if (*ptr == '<') [[unlikely]] {
                    ++ptr;
                    goto CONTINUE;
                }
                NEXT_PTR;
            }
            r.emplace(luisa::string{path_begin, ptr});
        CONTINUE:
            continue;
        }
        return r;
#undef NEXT_PTR
    }
    bool require_recompile(
        std::filesystem::path const &src_dir,
        std::filesystem::path const &file_dir) {
        std::error_code ec;
        auto file_abs_dir = std::filesystem::canonical(src_dir / file_dir, ec);
        if (ec) [[unlikely]] {
            LUISA_ERROR("Invalid canonical file path '{}' failed, message: {}", luisa::to_string(file_abs_dir), ec.message());
        }
        if (ec) [[unlikely]] {
            LUISA_ERROR("Get file last write time '{}' failed, message: {}", luisa::to_string(_cache_path), ec.message());
        }
        auto file_abs_dir_str = luisa::to_string(file_abs_dir);
        if (ec) [[unlikely]] {
            LUISA_ERROR("Invalid canonical file path '{}' failed, message: {}", luisa::to_string(file_dir), ec.message());
        }
        vstd::Guid guid{true};
        if (ec) [[unlikely]] {
            LUISA_ERROR("Invalid canonical file path '{}' failed, message: {}", luisa::to_string(_cache_path / file_dir), ec.message());
        }
        luisa::span<const std::byte> db_value;
        auto preprocess = [&]<bool check_preprocess = true>() {
            auto preprocess_path = std::filesystem::weakly_canonical(_cache_path / guid.to_string(false), ec);
            vstd::MD5 old_md5;
            if constexpr (check_preprocess) {
                int64_t lefted_size = db_value.size_bytes() - sizeof(std::filesystem::file_time_type);
                if (lefted_size >= int64_t(sizeof(vstd::MD5))) {
                    memcpy(&old_md5, db_value.data() + sizeof(std::filesystem::file_time_type), sizeof(vstd::MD5));
                }
            }
            {
                luisa::vector<char> vec;
                vec.reserve(1024);
                add(vec, luisa::to_string(_clang_path));
                add(vec, " --preprocess");
                for (auto &&i : _inc_paths) {
                    add(vec, " -I"sv);
                    add(vec, i);
                }
                for (auto &&i : _defines) {
                    add(vec, " -D"sv);
                    add(vec, i);
                }

                add(vec, " -o ");
                add(vec, luisa::to_string(preprocess_path));
                add(vec, ' ');
                add(vec, file_abs_dir_str);
                vec.push_back(0);
                auto r = system(vec.data());
                if (r != 0) {
                    remove_file(file_abs_dir_str);
                }
            }
            {
                luisa::vector<std::byte> vec;
                {
                    BinaryFileStream file_stream{luisa::to_string(preprocess_path)};
                    vec.push_back_uninitialized(file_stream.length());
                    file_stream.read(vec);
                }
                vstd::MD5 md5{{reinterpret_cast<uint8_t const *>(vec.data()), vec.size()}};
                if constexpr (check_preprocess) {
                    if (old_md5 == md5) {
                        return false;
                    }
                }
                auto include_set = include_dirs(luisa::string_view{
                    reinterpret_cast<char const *>(vec.data()),
                    vec.size()});
                vec.clear();
                auto push = [&]<typename T>(T const &a) {
                    auto last_size = vec.size();
                    if constexpr (std::is_trivial_v<T>) {
                        vec.push_back_uninitialized(sizeof(T));
                        memcpy(vec.data() + last_size, &a, sizeof(T));
                    } else if constexpr (std::is_same_v<T, luisa::string_view> || std::is_same_v<T, luisa::string>) {
                        vec.push_back_uninitialized(a.size());
                        memcpy(vec.data() + last_size, a.data(), a.size());
                    } else {
                        vec.push_back_uninitialized(a.size_bytes());
                        memcpy(vec.data() + last_size, a.data(), a.size_bytes());
                    }
                };
                push(md5.to_binary());
                // nlohmann::json js_arr;
                for (auto &&i : include_set) {
                    auto path = luisa::to_string(std::filesystem::weakly_canonical(i, ec));
                    update_file(path, std::filesystem::last_write_time(i), {});
                    push(path.size());
                    push(path);
                }
                // auto js_str = js_arr.dump();
                update_file(file_abs_dir_str, std::filesystem::last_write_time(file_abs_dir), vec);
            }
            return true;
        };
        if (file_is_new(file_abs_dir_str, db_value)) {
            return preprocess();
        }
        // check include files
        auto header_size = sizeof(std::filesystem::file_time_type) + sizeof(vstd::MD5);
        int64_t data_size = db_value.size() - header_size;
        if (data_size > 0) {
            auto ptr = db_value.data() + header_size;
            auto end_ptr = db_value.data() + db_value.size();
            while (ptr < end_ptr) {
                size_t str_size;
                memcpy(&str_size, ptr, sizeof(size_t));
                ptr += sizeof(size_t);
                if (ptr >= end_ptr) [[unlikely]]
                    break;
                luisa::string_view name = {reinterpret_cast<char const *>(ptr), reinterpret_cast<char const *>(ptr + str_size)};
                ptr += str_size;
                if (ptr > end_ptr) [[unlikely]] {
                    LUISA_WARNING("Invalid cache data.");
                    return preprocess();
                }
                if (file_is_new(name)) {
                    return preprocess();
                }
            }
            return preprocess();
        }
        return false;
    }
};

int main(int argc, char *argv[]) {
    log_level_error();
    //////// Properties
    if (argc <= 1) {
        LUISA_ERROR("Empty argument not allowed.");
    }
    std::filesystem::path src_path;
    std::filesystem::path dst_path;
    luisa::vector<std::filesystem::path> inc_paths;
    luisa::unordered_set<luisa::string> defines;
    luisa::string backend = "dx";
    bool use_optimize = true;
    bool enable_help = false;
    bool enable_lsp = false;
    bool rebuild = false;
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
            inc_paths.emplace_back(name);
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
    cmds.emplace(
        "rebuild"sv,
        [&](string_view name) {
            rebuild = true;
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
        for (auto &&i : inc_paths) {
            if (i.is_relative()) {
                i = std::filesystem::current_path() / i;
            }
            i = std::filesystem::weakly_canonical(i, code);
            if (code.value() != 0) {
                LUISA_ERROR("Invalid include file path");
            }
        }

        if (dst_path.is_relative()) {
            dst_path = std::filesystem::current_path() / dst_path;
        }
        if (src_path == dst_path) {
            LUISA_ERROR("Source file and dest file path can not be the same.");
        }
        dst_path = std::filesystem::weakly_canonical(dst_path, code);
        if (code.value() != 0) {
            LUISA_ERROR("Invalid dest file path");
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
        if (inc_paths.empty()) {
            inc_paths.emplace_back(src_path);
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
            luisa::fiber::scheduler thread_pool(std::min<uint>(std::thread::hardware_concurrency(), paths.size()));
            std::mutex mtx;
            luisa::fiber::parallel(paths.size(), [&](size_t i) {
                auto &file_path = paths[i];
                if (file_path.is_absolute()) {
                    file_path = std::filesystem::relative(file_path, src_path);
                }
                luisa::vector<char> local_result;
                auto iter = vstd::range_linker{
                    vstd::make_ite_range(defines),
                    vstd::transform_range{[&](auto &&v) { return luisa::string_view{v}; }}}
                                .i_range();
                auto inc_iter = vstd::range_linker{
                    vstd::make_ite_range(inc_paths),
                    vstd::transform_range{
                        [&](auto &&path) { return luisa::to_string(path); }}}
                                    .i_range();
                luisa::clangcxx::Compiler::lsp_compile_commands(
                    iter,
                    src_path,
                    file_path,
                    inc_iter,
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
        luisa::fiber::scheduler thread_pool(std::min<uint>(std::thread::hardware_concurrency(), paths.size()));

        format_path();
        log_level_info();
        auto clang_path = std::filesystem::path{argv[0]}.parent_path() / "clang.exe";
        auto iter = vstd::range_linker{
            vstd::make_ite_range(defines),
            vstd::transform_range{[&](auto &&v) { return luisa::string_view{v}; }}}
                        .i_range();
        auto inc_iter = vstd::range_linker{
            vstd::make_ite_range(inc_paths),
            vstd::transform_range{
                [&](auto &&path) { return luisa::to_string(path); }}}
                            .i_range();
        if (rebuild) {
            auto cache_path = std::filesystem::current_path() / ".cache";
            if (std::filesystem::exists(cache_path)) {
                std::error_code ec;
                std::filesystem::remove_all(cache_path, ec);
                if (ec) [[unlikely]] {
                    LUISA_ERROR("Try clear cache dir {} failed {}.", luisa::to_string(cache_path), ec.message());
                }
            }
        }
        Preprocessor processor{
            clang_path,
            std::filesystem::path{"./.cache/.lmdb"},
            std::filesystem::path{"./.cache/.obj"},
            iter,
            inc_iter};
        luisa::fiber::parallel(
            paths.size(),
            [&](size_t i) {
                auto const &src_file_path = paths[i];
                auto file_path = src_file_path;
                if (file_path.is_absolute()) {
                    file_path = std::filesystem::relative(file_path, src_path);
                }
                if (!processor.require_recompile(src_path, file_path)) return;
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
                for (auto &i : inc_paths) {
                    add(vec, ' ');
                    add(vec, "--include="sv);
                    add(vec, luisa::to_string(i));
                }
                for (auto &i : defines) {
                    add(vec, ' ');
                    add(vec, "--define="sv);
                    add(vec, i);
                }
                vec.emplace_back(0);
                LUISA_INFO("compiling {}", luisa::to_string(file_path.filename()));
                auto result = system(vec.data());
                if (result != 0) {
                    processor.remove_file(luisa::to_string(std::filesystem::weakly_canonical(src_path / file_path)));
                }
            });
        processor.post_process();
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
    }
    format_path();
    DeviceConfig config{
        .headless = true};
    Device device = context.create_device(backend, &config);
    auto iter = vstd::range_linker{
        vstd::make_ite_range(defines),
        vstd::transform_range{[&](auto &&v) { return luisa::string_view{v}; }}}
                    .i_range();
    auto inc_iter = vstd::range_linker{
        vstd::make_ite_range(inc_paths),
        vstd::transform_range{
            [&](auto &&path) { return luisa::to_string(path); }}}
                        .i_range();
    if (!luisa::clangcxx::Compiler::create_shader(
            ShaderOption{
                .enable_fast_math = use_optimize,
                .enable_debug_info = !use_optimize,
                .compile_only = true,
                .name = luisa::to_string(dst_path)},
            device, iter, src_path, inc_iter)) {
        return 1;
    }
    return 0;
}