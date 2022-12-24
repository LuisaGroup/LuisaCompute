#pragma once
#include <vstl/common.h>
#include <filesystem>
namespace toolhub::directx {
struct ShaderPaths {
    std::filesystem::path const& shaderCacheFolder;
    std::filesystem::path dataFolder;
    std::filesystem::path const& runtimeFolder;
};
}// namespace toolhub::directx