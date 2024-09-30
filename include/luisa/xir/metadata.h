#pragma once

#include <luisa/xir/ilist.h>

namespace luisa::compute::xir {

struct LC_XIR_API Metadata : IntrusiveForwardNode<Metadata> {};

using MetadataList = IntrusiveForwardList<Metadata>;

}// namespace luisa::compute::xir
