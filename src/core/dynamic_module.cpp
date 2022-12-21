#include <core/dynamic_module.h>
#include <core/clock.h>
#include <core/logging.h>

namespace luisa {

DynamicModule &DynamicModule::operator=(DynamicModule &&rhs) noexcept {
    if (&rhs != this) [[likely]] {
        _handle = rhs._handle;
        rhs._handle = nullptr;
    }
    return *this;
}

DynamicModule::DynamicModule(DynamicModule &&another) noexcept
    : _handle{another._handle} { another._handle = nullptr; }

DynamicModule::~DynamicModule() noexcept { dynamic_module_destroy(_handle); }

inline std::mutex &DynamicModule::_search_path_mutex() noexcept {
    static std::mutex mutex;
    return mutex;
}

inline luisa::vector<std::pair<std::filesystem::path, size_t>> &DynamicModule::_search_paths() noexcept {
    static luisa::vector<std::pair<std::filesystem::path, size_t>> paths;
    return paths;
}

void DynamicModule::add_search_path(const std::filesystem::path &path) noexcept {
    std::scoped_lock lock{_search_path_mutex()};
    auto canonical_path = std::filesystem::canonical(path);
    auto &&paths = _search_paths();
    if (auto iter = std::find_if(paths.begin(), paths.end(), [&canonical_path](auto &&p) noexcept {
            return p.first == canonical_path;
        });
        iter != paths.end()) {
        iter->second++;
    } else {
        paths.emplace_back(std::move(canonical_path), 0u);
    }
}

void DynamicModule::remove_search_path(const std::filesystem::path &path) noexcept {
    std::scoped_lock lock{_search_path_mutex()};
    auto canonical_path = std::filesystem::canonical(path);
    auto &&paths = _search_paths();
    if (auto iter = std::find_if(paths.begin(), paths.end(), [&canonical_path](auto &&p) noexcept {
            return p.first == canonical_path;
        });
        iter != paths.end()) {
        if (--iter->second == 0u) {
            paths.erase(iter);
        }
    }
}

luisa::optional<DynamicModule> DynamicModule::load(std::string_view name) noexcept {
    std::scoped_lock lock{_search_path_mutex()};
    Clock clock;
    auto &&paths = _search_paths();
    for (auto iter = paths.crbegin(); iter != paths.crend(); iter++) {
        auto p = iter->first / dynamic_module_name(name);
        if (auto handle = dynamic_module_load(p)) {
            LUISA_INFO(
                "Loaded dynamic module '{}' in {} ms.",
                p.string(), clock.toc());
            return DynamicModule{handle};
        }
    }
    auto module_name = dynamic_module_name(name);
    if (auto handle = dynamic_module_load(module_name)) {
        LUISA_INFO(
            "Loaded dynamic module '{}' in {} ms.",
            module_name, clock.toc());
        return DynamicModule{handle};
    }
    return luisa::nullopt;
}

luisa::optional<DynamicModule> DynamicModule::load(const std::filesystem::path &folder,
                                                   std::string_view name) noexcept {
    Clock clock;
    auto p = folder / dynamic_module_name(name);
    if (auto handle = dynamic_module_load(p)) {
        LUISA_INFO(
            "Loaded dynamic module '{}' in {} ms.",
            p.string(), clock.toc());
        return DynamicModule{handle};
    }
    return luisa::nullopt;
}

}// namespace luisa
