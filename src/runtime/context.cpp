#include <luisa/core/dynamic_module.h>
#include <luisa/core/logging.h>
#include <luisa/core/platform.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/core/binary_io.h>
#include <luisa/vstl/pdqsort.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute {

struct BackendModule {
    using BackendDeviceNames = void(luisa::vector<luisa::string> &);
    DynamicModule module;
    Device::Creator *creator;
    Device::Deleter *deleter;
    BackendDeviceNames *backend_device_names;
};

struct ValidationLayer {
    using Creator = DeviceInterface *(Context &&ctx, luisa::shared_ptr<DeviceInterface> &&native);
    DynamicModule module;
    Creator *creator{};
    Device::Deleter *deleter{};
};

// Make context global, so dynamic modules cannot be redundantly loaded
namespace detail {

class ContextImpl {

public:
    std::filesystem::path runtime_directory;
    luisa::unordered_map<luisa::string, luisa::unique_ptr<BackendModule>> loaded_backends;
    luisa::vector<luisa::string> installed_backends;
    ValidationLayer validation_layer;
    luisa::unordered_map<luisa::string, luisa::unique_ptr<std::filesystem::path>> runtime_subdir_paths;
    std::mutex runtime_subdir_mutex;
    std::mutex module_mutex;

    [[nodiscard]] const BackendModule &load_backend(const luisa::string &backend_name) noexcept {
        if (std::find(installed_backends.cbegin(),
                      installed_backends.cend(),
                      backend_name) == installed_backends.cend()) {
            LUISA_ERROR_WITH_LOCATION("Backend '{}' is not installed.", backend_name);
        }
        std::scoped_lock lock{module_mutex};
        if (auto iter = loaded_backends.find(backend_name);
            iter != loaded_backends.cend()) { return *iter->second; }
        BackendModule m{.module = DynamicModule::load(
                            runtime_directory,
                            luisa::format("lc-backend-{}", backend_name))};
        LUISA_ASSERT(m.module, "Failed to load backend '{}'.", backend_name);
        m.creator = m.module.function<Device::Creator>("create");
        m.deleter = m.module.function<Device::Deleter>("destroy");
        m.backend_device_names = m.module.function<BackendModule::BackendDeviceNames>("backend_device_names");
        auto pm = loaded_backends
                      .emplace(backend_name, luisa::make_unique<BackendModule>(std::move(m)))
                      .first->second.get();
        return *pm;
    }

    [[nodiscard]] const ValidationLayer &load_validation_layer() noexcept {
        std::scoped_lock lock{module_mutex};
        if (!validation_layer.module) {
            validation_layer.module = DynamicModule::load(runtime_directory, "lc-validation-layer");
            validation_layer.creator = validation_layer.module.function<ValidationLayer::Creator>("create");
            validation_layer.deleter = validation_layer.module.function<Device::Deleter>("destroy");
        }
        return validation_layer;
    }

    explicit ContextImpl(luisa::string_view program_path) noexcept {
        std::filesystem::path program{program_path};
        {
            auto cp = std::filesystem::canonical(program);
            if (std::filesystem::is_directory(cp)) {
                runtime_directory = std::move(cp);
            } else {
                runtime_directory = std::filesystem::canonical(cp.parent_path());
            }
        }
        LUISA_INFO("Created context for program '{}'.", to_string(program.filename()));
        LUISA_INFO("Runtime directory: {}.", to_string(runtime_directory));
        DynamicModule::add_search_path(runtime_directory);
        for (auto &&p : std::filesystem::directory_iterator{runtime_directory}) {
            if (auto &&path = p.path();
                p.is_regular_file() &&
                (path.extension() == ".so" ||
                 path.extension() == ".dll" ||
                 path.extension() == ".dylib")) {
                using namespace std::string_view_literals;
                constexpr std::array possible_prefixes{
                    "lc-backend-"sv,
                    // Make Mingw happy
                    "liblc-backend-"sv};
                auto filename = to_string(path.stem());
                for (auto prefix : possible_prefixes) {
                    if (filename.starts_with(prefix)) {
                        auto name = filename.substr(prefix.size());
                        for (auto &c : name) { c = static_cast<char>(std::tolower(c)); }
                        LUISA_VERBOSE_WITH_LOCATION("Found backend: {}.", name);
                        installed_backends.emplace_back(std::move(name));
                        break;
                    }
                }
            }
        }
        pdqsort(installed_backends.begin(), installed_backends.end());
        installed_backends.erase(
            std::unique(installed_backends.begin(), installed_backends.end()),
            installed_backends.end());
    }
    ~ContextImpl() noexcept {
        DynamicModule::remove_search_path(runtime_directory);
    }
};

}// namespace detail

Context::Context(string_view program_path) noexcept
    : _impl{luisa::make_shared<detail::ContextImpl>(program_path)} {}

Device Context::create_device(luisa::string_view backend_name_in, const DeviceConfig *settings, bool enable_validation) noexcept {
    luisa::string backend_name{backend_name_in};
    for (auto &c : backend_name) { c = static_cast<char>(std::tolower(c)); }
    auto &&m = _impl->load_backend(backend_name);
    auto interface = m.creator(Context{_impl}, settings);
    interface->_backend_name = std::move(backend_name);
    auto handle = Device::Handle{
        interface,
        [impl = _impl, deleter = m.deleter](auto p) noexcept {
            deleter(p);
        }};
    if (enable_validation) {
        auto &validation_layer = _impl->load_validation_layer();
        auto layer_handle = Device::Handle{
            validation_layer.creator(Context{_impl}, std::move(handle)),
            [impl = _impl](auto layer) noexcept {
                impl->validation_layer.deleter(layer);
            }};
        return Device{std::move(layer_handle)};
    } else {
        return Device{std::move(handle)};
    }
}

Context::Context(luisa::shared_ptr<detail::ContextImpl> impl) noexcept
    : _impl{std::move(impl)} {}

Context::~Context() noexcept = default;

luisa::span<const luisa::string> Context::installed_backends() const noexcept {
    return _impl->installed_backends;
}

Device Context::create_default_device() noexcept {
    LUISA_ASSERT(!installed_backends().empty(), "No backends installed.");
    return create_device(installed_backends().front());
}

luisa::vector<luisa::string> Context::backend_device_names(luisa::string_view backend_name_in) const noexcept {
    luisa::string backend_name{backend_name_in};
    for (auto &c : backend_name) { c = static_cast<char>(std::tolower(c)); }
    auto &&m = _impl->load_backend(backend_name);
    luisa::vector<luisa::string> names;
    m.backend_device_names(names);
    return names;
}

const luisa::filesystem::path &Context::runtime_directory() const noexcept {
    return _impl->runtime_directory;
}

const luisa::filesystem::path &Context::create_runtime_subdir(luisa::string_view folder_name) const noexcept {
    std::lock_guard lock{_impl->runtime_subdir_mutex};
    auto iter = _impl->runtime_subdir_paths.try_emplace(
        folder_name,
        luisa::lazy_construct([&]() {
            auto dir = runtime_directory() / folder_name;
            std::error_code ec;
            luisa::filesystem::create_directories(dir, ec);
            if (ec) {
                LUISA_WARNING_WITH_LOCATION(
                    "Failed to create runtime sub-directory '{}': {}.",
                    to_string(dir), ec.message());
            }
            return luisa::make_unique<std::filesystem::path>(std::move(dir));
        }));
    return *iter.first->second;
}

}// namespace luisa::compute
