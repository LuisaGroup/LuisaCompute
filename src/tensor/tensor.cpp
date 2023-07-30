#include <luisa/tensor/tensor.h>
#include <luisa/core/logging.h>
namespace luisa::compute::tensor {
class JitSession::Impl {
};
thread_local JitSession *_current = nullptr;
JitSession &JitSession::get() noexcept {
    if (_current == nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "No evaluation scope found. "
            "Please make sure you are calling this function inside a kernel.");
    }
    return *_current;
}
}// namespace luisa::compute::tensor