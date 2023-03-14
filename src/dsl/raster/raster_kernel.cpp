#include <dsl/raster/raster_kernel.h>
#include <core/logging.h>
#include <ast/type.h>
namespace luisa::compute {
LC_DSL_API void check_vert_ret_type(Type const *type) {
    LUISA_ASSERT((type->is_vector() && type->element()->tag() == Type::Tag::FLOAT32 && type->dimension() == 4) || (type->is_structure() && type->members().size() >= 1 && type->members()[0]->is_vector() && type->members()[0]->element()->tag() == Type::Tag::FLOAT32 && type->members()[0]->dimension() == 4), "invalid vertex return type.");
}
}// namespace luisa::compute