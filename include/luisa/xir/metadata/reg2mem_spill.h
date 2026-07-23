#pragma once

#include <cstdint>

#include <luisa/xir/metadata.h>

namespace luisa::compute::xir {

enum struct Reg2MemSpillKind : uint8_t {
    PHI,
    CROSS_BLOCK,
};

[[nodiscard]] constexpr luisa::string_view
to_string(Reg2MemSpillKind kind) noexcept {
    switch (kind) {
        case Reg2MemSpillKind::PHI: return "phi";
        case Reg2MemSpillKind::CROSS_BLOCK: return "cross_block";
    }
    return "unknown";
}

class LUISA_XIR_API Reg2MemSpillMD final
    : public DerivedMetadata<Reg2MemSpillMD, DerivedMetadataTag::REG2MEM_SPILL> {

private:
    Reg2MemSpillKind _kind;

public:
    explicit Reg2MemSpillMD(Reg2MemSpillKind kind = Reg2MemSpillKind::PHI) noexcept;
    [[nodiscard]] auto kind() const noexcept { return _kind; }
    void set_kind(Reg2MemSpillKind kind) noexcept { _kind = kind; }
    [[nodiscard]] ManagedPtr<Metadata> clone() const noexcept override;
};

}// namespace luisa::compute::xir
