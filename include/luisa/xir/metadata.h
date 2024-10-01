#pragma once

#include <luisa/xir/ilist.h>

namespace luisa::compute::xir {

class LC_XIR_API Metadata : public IntrusiveForwardNode<Metadata> {

public:
    explicit Metadata(Pool *pool) noexcept;
};

using MetadataList = IntrusiveForwardList<Metadata>;

}// namespace luisa::compute::xir
