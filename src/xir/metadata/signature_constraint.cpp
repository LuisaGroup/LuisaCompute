#include <luisa/xir/metadata/signature_constraint.h>

namespace luisa::compute::xir {

ManagedPtr<Metadata> SignatureConstraintMD::clone() const noexcept {
    return luisa::make_managed<SignatureConstraintMD>();
}

}// namespace luisa::compute::xir
