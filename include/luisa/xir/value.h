#pragma once

#include <luisa/ast/type.h>
#include <luisa/xir/use.h>
#include <luisa/xir/metadata.h>

namespace luisa::compute::xir {

class LC_XIR_API Value : public PooledObject {

private:
    const Type *_type = nullptr;
    UseList _use_list;
    MetadataList _metadata_list;

public:
};

}// namespace luisa::compute::xir
