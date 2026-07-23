#include <luisa/xir/metadata/reg2mem_spill.h>

namespace luisa::compute::xir {

Reg2MemSpillMD::Reg2MemSpillMD(Reg2MemSpillKind kind) noexcept
    : _kind{kind} {}

ManagedPtr<Metadata> Reg2MemSpillMD::clone() const noexcept {
    return luisa::make_managed<Reg2MemSpillMD>(this->kind());
}

}// namespace luisa::compute::xir
