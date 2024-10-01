#include <luisa/xir/metadata/location.h>

namespace luisa::compute::xir {

LocationMD::LocationMD(Pool *pool,
                       luisa::filesystem::path file,
                       int line, int column) noexcept
    : Metadata{pool},
      _file{std::move(file)},
      _line{line},
      _column{column} {}

}// namespace luisa::compute::xir
