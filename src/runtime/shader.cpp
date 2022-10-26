//
// Created by Mike Smith on 2022/10/24.
//

#include <runtime/shader.h>
#include <runtime/bindless_array.h>

namespace luisa::compute::detail {

ShaderInvokeBase &ShaderInvokeBase::operator<<(const BindlessArray &array) noexcept {
    _shader_command()->encode_bindless_array(array.handle());
    return *this;
}

}// namespace luisa::compute::detail
