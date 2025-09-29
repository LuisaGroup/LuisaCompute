#include <cstring>
#include <vector>
#include <type_traits>
#include <filesystem>
#include <unordered_map>

template<typename T>
struct always_false {
    static constexpr bool value = false;
};

template<typename T>
constexpr bool always_false_v = always_false<T>::value;

static char const *bytes_str[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "173", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", "195", "196", "197", "198", "199", "200", "201", "202", "203", "204", "205", "206", "207", "208", "209", "210", "211", "212", "213", "214", "215", "216", "217", "218", "219", "220", "221", "222", "223", "224", "225", "226", "227", "228", "229", "230", "231", "232", "233", "234", "235", "236", "237", "238", "239", "240", "241", "242", "243", "244", "245", "246", "247", "248", "249", "250", "251", "252", "253", "254", "255"};
static constexpr uint64_t magic_number = 1145141919810ull;

struct FileMeta {
    std::string file_name;
    alignas(std::filesystem::file_time_type) char src_file_time[sizeof(std::filesystem::file_time_type)];
};
bool check_file(std::byte const *&ptr, std::byte const *end);
bool deser_meta(std::byte const *&ptr, std::byte const *end, FileMeta &meta);
void ser_meta(std::vector<char> &data, FileMeta const &meta);

int main(int argc, char *argv[]) {
    const uint32_t arg_start = 7;
    if (argc < arg_start + 1) {
        printf("Usage <soruce dir> <dest dir> <meta dir> <var_name_prefix> <remove_ext(y/n)> <remove_/r(y/n)> <file_list> ... \n");
        return 1;
    }
    auto src_dir = std::filesystem::path{argv[1]};
    auto dst_dir = std::filesystem::path{argv[2]};
    auto meta_dir = std::filesystem::path{argv[3]};
    std::vector<char> result;
    std::unordered_map<std::string, FileMeta> file_metas;
    auto meta_dir_str = meta_dir.string();
    bool meta_dirty = false;
    auto var_name_prefix = std::string(argv[4]);
    if (var_name_prefix == "_") {
        var_name_prefix.clear();
    }
    auto remove_ext_str = std::string_view(argv[5]);
    auto remove_slash_r_str = std::string_view(argv[6]);
    bool remove_ext = remove_ext = remove_ext_str == "y" || remove_ext_str == "Y";
    auto remove_slash_r = remove_slash_r_str == "y" || remove_slash_r_str == "Y";

    if (std::filesystem::exists(meta_dir) && std::filesystem::exists(dst_dir)) {
        auto f = fopen(meta_dir_str.c_str(), "rb");
        if (f) {
            std::vector<std::byte> meta_bytes;
            fseek(f, 0, SEEK_END);
            auto meta_length = ftell(f);
            fseek(f, 0, SEEK_SET);
            meta_bytes.resize(meta_length);
            fread(meta_bytes.data(), meta_length, 1, f);
            fclose(f);
            const std::byte *ptr = meta_bytes.data();
            const std::byte *end = ptr + meta_length;
            std::filesystem::file_time_type dst_time;
            if (check_file(ptr, end)) {
                std::memcpy(&dst_time, ptr, sizeof(dst_time));
                ptr += sizeof(dst_time);
                if (dst_time != std::filesystem::last_write_time(dst_dir)) {
                    meta_dirty = true;
                } else {
                    while (ptr < end) {
                        FileMeta meta;
                        if (!deser_meta(ptr, end, meta)) {
                            meta_dirty = true;
                            file_metas.clear();
                            break;
                        }
                        file_metas.try_emplace(meta.file_name, std::move(meta));
                    }
                }
            } else {
                meta_dirty = true;
            }
        } else {
            meta_dirty = true;
        }
    } else {
        meta_dirty = true;
    }

    auto push = [&]<typename T>(T const &t) {
        if constexpr (requires {t.size(); t.data(); }) {
            auto end = t.data() + t.size();
            auto size_bytes = reinterpret_cast<size_t>(end) - reinterpret_cast<size_t>(t.data());
            auto last_size = result.size();
            result.resize(last_size + size_bytes);
            std::memcpy(result.data() + last_size, t.data(), size_bytes);
        } else if constexpr (std::is_same_v<std::decay_t<T>, const char *> || std::is_same_v<std::decay_t<T>, char *>) {
            auto size_bytes = strlen(t);
            auto last_size = result.size();
            result.resize(last_size + size_bytes);
            std::memcpy(result.data() + last_size, t, size_bytes);
        } else if constexpr (std::is_same_v<T, char>) {
            result.push_back(t);
        } else if constexpr (std::is_trivially_destructible_v<T>) {
            auto last_size = result.size();
            result.resize(last_size + sizeof(T));
            std::memcpy(result.data() + last_size, &t, sizeof(T));
        } else {
            static_assert(always_false_v<T>, "Bad type.");
        }
    };
    std::vector<uint8_t> src_data;
    result.clear();
    auto get_var_name = [&](auto &&file_name) {
        auto filename = std::filesystem::path{file_name}.filename();
        if (remove_ext) {
            filename = filename.replace_extension();
        }
        auto var_name = filename.string();
        for (auto &i : var_name) {
            if (i == '.') {
                i = '_';
            }
        }
        if (!var_name_prefix.empty())
            return var_name_prefix + var_name;
        else
            return var_name;
    };
    if (!meta_dirty) {
        for (int i = arg_start; i < argc; ++i) {
            std::string file_name = argv[i];
            auto file_meta_iter = file_metas.find(file_name);
            auto src_file_path = (src_dir / file_name).string();
            auto var_name = get_var_name(file_name);
            auto src_last_time = std::filesystem::last_write_time(src_file_path);
            if (file_meta_iter != file_metas.end()) {
                auto &&file_meta = file_meta_iter->second;
                if (std::memcmp(&file_meta.src_file_time, &src_last_time, sizeof(src_last_time)) != 0) {
                    meta_dirty = true;
                    break;
                }
            } else {
                meta_dirty = true;
                break;
            }
        }
    }
    if (meta_dirty) {
        for (int i = arg_start; i < argc; ++i) {
            std::string file_name = argv[i];
            auto src_file_path = (src_dir / file_name).string();
            auto f = fopen(src_file_path.c_str(), "rb");
            if (!f) {
                printf("Source file not exists.\n");
                return 1;
            }
            auto src_last_time = std::filesystem::last_write_time(src_file_path);
            auto var_name = get_var_name(file_name);
            fseek(f, 0, SEEK_END);
            auto src_length = ftell(f);
            fseek(f, 0, SEEK_SET);
            src_data.clear();
            src_data.resize(src_length);
            fread(src_data.data(), src_data.size(), 1, f);
            fclose(f);
            if (remove_slash_r) {
                std::vector<uint8_t> new_src_data;
                new_src_data.reserve(src_data.size());
                for (auto i : src_data) {
                    if (i != '\r') {
                        new_src_data.emplace_back(i);
                    }
                }
                src_data = std::move(new_src_data);
            }
            push("extern \"C\" const unsigned char ");
            push(var_name);
            push('[');
            push(std::to_string(src_data.size()));
            push(']');
            push("={");
            if (src_data.size() > 0) [[likely]] {
                push(bytes_str[src_data[0]]);
                for (size_t i = 1; i < src_data.size(); ++i) {
                    push(',');
                    push(bytes_str[src_data[i]]);
                }
            }
            push("};\n");
            push("extern \"C\" const unsigned long long ");
            push(var_name);
            push("_size=");
            push(std::to_string(src_data.size()));
            push(";\n");
            FileMeta file_meta{.file_name = file_name};
            std::memcpy(file_meta.src_file_time, &src_last_time, sizeof(src_last_time));
            file_metas[file_name] = std::move(file_meta);
        }
        auto dst_dir_str = dst_dir.string();
        auto f = fopen(dst_dir_str.c_str(), "wb");
        if (!f) {
            printf("Dest file write error.\n");
            return 1;
        }
        fwrite(result.data(), result.size(), 1, f);
        fclose(f);

        auto dst_last_time = std::filesystem::last_write_time(dst_dir);
        result.clear();
        push(magic_number);
        size_t file_size_index = result.size();
        push(size_t(0));// file_size
        push(dst_last_time);
        for (auto &kv : file_metas) {
            ser_meta(result, kv.second);
        }
        size_t size = result.size();
        std::memcpy(result.data() + file_size_index, &size, sizeof(size_t));
        f = fopen(meta_dir_str.c_str(), "wb");
        if (f) {
            fwrite(result.data(), result.size(), 1, f);
            fclose(f);
        }
    }
    return 0;
}

bool check_file(std::byte const *&ptr, std::byte const *end) {
    size_t file_size = end - ptr;
    size_t sizes[2];
    if (file_size < sizeof(size_t) * 2) return false;
    auto copy = [&](void *dst_ptr, size_t size) {
        std::memcpy(dst_ptr, ptr, size);
        ptr += size;
    };
    copy(sizes, sizeof(size_t) * 2);
    return sizes[0] = magic_number && sizes[1] == file_size;
}
bool deser_meta(std::byte const *&ptr, std::byte const *end, FileMeta &meta) {
    uint32_t sizes[2];
    if ((end - ptr) < 2 * sizeof(uint32_t)) {
        return false;
    }
    auto copy = [&](void *dst_ptr, size_t size) {
        std::memcpy(dst_ptr, ptr, size);
        ptr += size;
    };
    copy(sizes, 2 * sizeof(uint32_t));
    if (sizes[0] != sizes[1] + sizeof(std::filesystem::file_time_type)) {
        return false;
    }
    if ((end - ptr) < sizes[0]) {
        return false;
    }
    meta.file_name.clear();
    meta.file_name.resize(sizes[1]);
    copy(meta.file_name.data(), sizes[1]);
    copy(&meta.src_file_time, sizeof(std::filesystem::file_time_type));
    return true;
}
void ser_meta(std::vector<char> &data, FileMeta const &meta) {
    auto push = [&](void const *ptr, size_t size) {
        auto last_size = data.size();
        data.resize(last_size + size);
        std::memcpy(data.data() + last_size, ptr, size);
    };
    uint32_t sizes[2];
    sizes[0] = meta.file_name.size() + sizeof(meta.src_file_time);
    sizes[1] = meta.file_name.size();
    push(sizes, sizeof(uint32_t) * 2);
    push(meta.file_name.data(), meta.file_name.size());
    push(&meta.src_file_time, sizeof(meta.src_file_time));
}
