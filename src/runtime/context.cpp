//
// Created by Mike Smith on 2021/2/2.
//

#include <core/logging.h>
#include <runtime/context.h>

namespace luisa::compute {

Context::Context(const std::filesystem::path &rt_dir,
                 const std::filesystem::path &work_dir) noexcept
    : _runtime_directory{std::filesystem::canonical(rt_dir)} {

    if (!std::filesystem::exists(work_dir)) {
        LUISA_INFO("Creating working directory: {}.", work_dir.string());
        std::filesystem::create_directories(work_dir);
    }
    _working_directory = std::filesystem::canonical(work_dir);
}

const std::filesystem::path &Context::runtime_directory() const noexcept {
    return _runtime_directory;
}

const std::filesystem::path &Context::working_directory() const noexcept {
    return _working_directory;
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
        auto &&m = _loaded_modules.emplace_back(_runtime_directory / "backends", backend_name);
        auto create = m.function<DeviceCreator>("create");
        auto destroy = m.function<DeviceDeleter>("destroy");
        _device_identifiers.emplace_back(backend_name);
        _device_creators.emplace_back(create);
        _device_deleters.emplace_back(destroy);
        return std::make_pair(create, destroy);
    }();
    return {create(index), destroy};
}

}// namespace luisa::compute
