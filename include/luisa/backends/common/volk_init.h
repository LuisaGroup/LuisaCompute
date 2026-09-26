#pragma once

#include <volk.h>

#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>
#include <luisa/core/magic_enum.h>
#include <luisa/core/dynamic_module.h>
#include <luisa/backends/common/vulkan_check_error.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>

namespace luisa::compute {

class VolkInitializer {

private:
    [[nodiscard]] static DynamicModule _try_load_vulkan_dylib(const luisa::filesystem::path &search_path, luisa::string_view specified_name) noexcept {
        if (!specified_name.empty()) {
            return DynamicModule::load(search_path, specified_name);
        }
        using namespace std::string_view_literals;
        constexpr std::array candidate_names = {
#ifdef LUISA_PLATFORM_WINDOWS
            "vulkan.dll"sv,
            "vulkan-1.dll"sv,
#else
            "libvulkan.so"sv,
            "libvulakn.so.1"sv,
#ifdef LUISA_PLATFORM_APPLE
            "libvulkan.dylib"sv,
            "libvulakn.dylib.1"sv,
            "libMoltenVK.so"sv,
            "libMoltenVK.dylib"sv,
            "MoltenVK.so"sv,
            "MoltenVK.dylib"sv,
#endif
#endif
        };
        for (auto &&name : candidate_names) {
            if (auto m = DynamicModule::load(search_path, name)) {
                return m;
            }
        }
        return {};
    }

public:
    DynamicModule vk_module;

    void init(const luisa::filesystem::path &custom_path = {}, luisa::string_view lib_name = {}) {
        if (custom_path.empty() && lib_name.empty()) {
            LUISA_CHECK_VULKAN(volkInitialize());
        } else {
            auto search_path = custom_path.empty() ?
                                   luisa::filesystem::canonical(current_executable_path()).parent_path() :
                                   custom_path;
            vk_module = _try_load_vulkan_dylib(search_path, lib_name);
            LUISA_ASSERT(vk_module, "Failed to load vulkan module from {}/{}", search_path.string(), lib_name);
            auto ptr = vk_module.address("vkGetInstanceProcAddr");
            LUISA_ASSERT(ptr != nullptr, "vkGetInstanceProcAddr symbol not found.");
            volkInitializeCustom(reinterpret_cast<PFN_vkGetInstanceProcAddr>(ptr));
        }
    }
};

}// namespace luisa::compute