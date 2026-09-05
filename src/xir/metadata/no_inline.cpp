#include <luisa/xir/metadata/no_inline.h>

namespace luisa::compute::xir {

ManagedPtr<Metadata> NoInlineMD::clone() const noexcept {
    return luisa::make_managed<NoInlineMD>();
}

}// namespace luisa::compute::xir
