#include <luisa/core/pool.h>
#include "metal_callback_context.h"

namespace luisa::compute::metal {

Pool<FunctionCallbackContext, true, true> &FunctionCallbackContext::_object_pool() noexcept {
    static Pool<FunctionCallbackContext, true, true> pool;
    return pool;
}

void FunctionCallbackContext::recycle() noexcept {
    _function();
    _object_pool().destroy(this);
}

}// namespace luisa::compute::metal

