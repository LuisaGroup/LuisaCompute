#pragma once

#include <luisa/xir/ilist.h>

namespace luisa::compute::xir {

struct LC_XIR_API Metadata : IntrusiveSDNode<Metadata> {};

using MetadataList = IntrusiveSDList<Metadata>;

}// namespace luisa::compute::xir
