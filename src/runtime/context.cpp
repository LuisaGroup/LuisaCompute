//
// Created by Mike Smith on 2021/2/2.
//

#include <core/logging.h>
#include <runtime/context.h>

namespace luisa::compute {

Context::Context(const std::filesystem::path &program) noexcept
    : _runtime_directory{std::filesystem::canonical(program).parent_path()} {
    LUISA_INFO("Created context for program: {}.", program.filename().string<char>());
    LUISA_INFO("Runtime directory: {}.", _runtime_directory.string<char>());
    _cache_directory = _runtime_directory / ".cache";
    if (!std::filesystem::exists(_cache_directory)) {
        std::filesystem::create_directories(_cache_directory);
    }
}

const std::filesystem::path &Context::runtime_directory() const noexcept {
    return _runtime_directory;
}

DeviceHandle Context::create_device(std::string_view backend_name, uint32_t index) noexcept {
    auto [create, destroy] = [backend_name, this] {
        if (auto iter = std::find(_device_identifiers.cbegin(),
                                  _device_identifiers.cend(),
                                  backend_name);
            iter != _device_identifiers.cend()) {
            auto i = iter - _device_identifiers.cbegin();
            auto create = _device_creators[i];
            auto destroy = _device_deleters[i];
            return std::make_pair(create, destroy);
        }
        auto &&m = _loaded_modules.emplace_back(
            _runtime_directory / "backends",
            fmt::format("luisa-compute-backend-{}", backend_name));
        auto create = m.function<DeviceCreator>("create");
        auto destroy = m.function<DeviceDeleter>("destroy");
        _device_identifiers.emplace_back(backend_name);
        _device_creators.emplace_back(create);
        _device_deleters.emplace_back(destroy);
        return std::make_pair(create, destroy);
    }();
    return {create(*this, index), destroy};
}

const std::filesystem::path &Context::cache_directory() const noexcept {
    return _cache_directory;
}

}// namespace luisa::compute
