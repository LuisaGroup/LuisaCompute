#pragma once

#include <cstdint>
#include <limits>

#include <luisa/xir/metadata.h>

namespace luisa::compute::xir {

struct ContiguousCopyDescriptor {
    uint64_t element_count{0u};
    uint32_t vector_width{1u};
};

[[nodiscard]] inline bool is_valid_contiguous_copy_descriptor(
    const ContiguousCopyDescriptor &descriptor) noexcept {
    return descriptor.element_count != 0u && descriptor.vector_width != 0u &&
           descriptor.element_count <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 4u;
}

// Required executable semantics, not a symbol-name convention. An external
// void(buffer<float> resource, uint64 element_offset, array<float> reference)
// copies element_count exact FP32 bit patterns at the call site. The caller
// supplies a valid contiguous source range and a complete private destination
// disjoint from that resource, with sufficient capacity. This does not make
// the source globally readonly or disjoint from other user resources. Targets
// must validate the signature, storage provenance and supported vector width;
// a target without this capability must reject the call, not ignore this tag.
class LUISA_XIR_API ContiguousCopyMD final
    : public DerivedMetadata<ContiguousCopyMD, DerivedMetadataTag::CONTIGUOUS_COPY> {
public:
    ContiguousCopyDescriptor descriptor;
    ContiguousCopyMD() noexcept = default;
    explicit ContiguousCopyMD(ContiguousCopyDescriptor value) noexcept;
    [[nodiscard]] ManagedPtr<Metadata> clone() const noexcept override;
};

}// namespace luisa::compute::xir
