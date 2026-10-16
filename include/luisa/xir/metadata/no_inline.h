#pragma once

#include <luisa/xir/metadata.h>

namespace luisa::compute::xir {

// Requires a backend-visible call boundary without constraining the callable
// signature or keeping an otherwise unused definition alive.
class LUISA_XIR_API NoInlineMD final
    : public DerivedMetadata<NoInlineMD, DerivedMetadataTag::NO_INLINE> {

public:
    NoInlineMD() noexcept = default;
    [[nodiscard]] ManagedPtr<Metadata> clone() const noexcept override;
};

}// namespace luisa::compute::xir
