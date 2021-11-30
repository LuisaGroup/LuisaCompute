//
// Created by Mike Smith on 2021/2/2.
//

#include <core/logging.h>
#include <runtime/context.h>

namespace luisa::compute {

namespace detail {
[[nodiscard]] auto runtime_directory(const std::filesystem::path &p) noexcept {
    auto cp = std::filesystem::canonical(p);
    if (std::filesystem::is_directory(cp)) { return cp; }
    return std::filesystem::canonical(cp.parent_path());
}
}// namespace detail

Context::Context(const std::filesystem::path &program) noexcept
    : _runtime_directory{detail::runtime_directory(program)} {
    LUISA_INFO("Created context for program: {}.", program.filename().string<char>());
    LUISA_INFO("Runtime directory: {}.", _runtime_directory.string<char>());
    _cache_directory = _runtime_directory / ".cache";
    LUISA_INFO("Cache directory: {}.", _cache_directory.string<char>());
    if (!std::filesystem::exists(_cache_directory)) {
        LUISA_INFO("Created cache directory.");
        std::filesystem::create_directories(_cache_directory);
    }
    DynamicModule::add_search_path(_runtime_directory);
}

const std::filesystem::path &Context::runtime_directory() const noexcept {
    return _runtime_directory;
}

const std::filesystem::path &Context::cache_directory() const noexcept {
    return _cache_directory;
}

Device Context::create_device(std::string_view backend_name, const nlohmann::json &properties) noexcept {
    if (!properties.is_object()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid device properties: {}.",
            properties.dump());
    }
    auto [create, destroy] = [backend_name, this] {
        if (auto iter = std::find(_device_identifiers.cbegin(),
                                  _device_identifiers.cend(),
                                  backend_name);
            iter != _device_identifiers.cend()) {
            auto i = iter - _device_identifiers.cbegin();
            auto c = _device_creators[i];
            auto d = _device_deleters[i];
            return std::make_pair(c, d);
        }
        auto &&m = _loaded_modules.emplace_back(
            _runtime_directory / "backends",
            fmt::format("luisa-compute-backend-{}", backend_name));
        auto c = m.function<Device::Creator>("create");
        auto d = m.function<Device::Deleter>("destroy");
        _device_identifiers.emplace_back(backend_name);
        _device_creators.emplace_back(c);
        _device_deleters.emplace_back(d);
        return std::make_pair(c, d);
    }();
    return Device{Device::Handle{create(*this, properties.dump()), destroy}};
}

Context::~Context() noexcept {
    DynamicModule::remove_search_path(_runtime_directory);
}

}// namespace luisa::compute
