#include <luisa/core/binary_io.h>

namespace luisa {

luisa::unique_ptr<BinaryStream> BinaryIO::read_shader_source(luisa::string_view name) const noexcept {
    return this->read_shader_cache(name);
}

luisa::filesystem::path BinaryIO::write_shader_source(luisa::string_view name, luisa::span<const std::byte> data) const noexcept {
    return this->write_shader_cache(name, data);
}

}
