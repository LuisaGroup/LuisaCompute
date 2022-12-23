#include <core/stl/filesystem.h>

namespace luisa {

LC_CORE_API luisa::string to_string(const luisa::filesystem::path &path) {
    return path.string<char, std::char_traits<char>, luisa::allocator<char>>();
}

}// namespace luisa
