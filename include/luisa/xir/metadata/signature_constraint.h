#pragma once

#include <luisa/xir/metadata.h>

namespace luisa::compute::xir {

class LUISA_XIR_API SignatureConstraintMD final : public DerivedMetadata<SignatureConstraintMD, DerivedMetadataTag::SIGNATURE_CONSTRAINT> {

public:
    SignatureConstraintMD() noexcept = default;
    [[nodiscard]] ManagedPtr<Metadata> clone() const noexcept override;
};

}// namespace luisa::compute::xir
