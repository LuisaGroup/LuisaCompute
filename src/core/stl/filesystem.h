#pragma once

#include <filesystem>
#include <core/stl/string.h>

namespace luisa {

namespace filesystem = std::filesystem;
[[nodiscard]] LC_CORE_API luisa::string to_string(const luisa::filesystem::path &path);

}// namespace luisa
