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

DynamicModule::DynamicModule(const std::filesystem::path &folder, std::string_view name) noexcept {
    Clock clock;
    auto p = folder / dynamic_module_name(name);
    if ((_handle = dynamic_module_load(p)) == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load dynamic module: {}.",
            p.string());
    }
    LUISA_INFO(
        "Loaded dynamic module '{}' in {} ms.",
        p.string(), clock.toc());
}

std::mutex &DynamicModule::_search_path_mutex() noexcept {
    static std::mutex mutex;
    return mutex;
}

luisa::vector<std::pair<std::filesystem::path, size_t>> &DynamicModule::_search_paths() noexcept {
    static luisa::vector<std::pair<std::filesystem::path, size_t>> paths;
    return paths;
}

DynamicModule::DynamicModule(std::string_view name) noexcept {
    std::scoped_lock lock{_search_path_mutex()};
    Clock clock;
    auto &&paths = _search_paths();
    // TODO: use ranges...
    for (auto iter = paths.crbegin(); iter != paths.crend(); iter++) {
        auto p = iter->first / dynamic_module_name(name);
        if ((_handle = dynamic_module_load(p)) != nullptr) {
            LUISA_INFO(
                "Loaded dynamic module '{}' in {} ms.",
                p.string(), clock.toc());
            return;
        }
    }
    if ((_handle = dynamic_module_load(dynamic_module_name(name))) == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load dynamic module '{}' from all search paths.",
            name);
    }
    LUISA_INFO(
        "Loaded dynamic module '{}' in {} ms.",
        name, clock.toc());
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

}// namespace luisa
