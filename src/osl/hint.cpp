#include <luisa/core/stl/format.h>
#include <luisa/osl/hint.h>

namespace luisa::compute::osl {

luisa::string Hint::dump() const noexcept {
    auto s = luisa::format("%{}", identifier());
    if (!args().empty()) {
        s.append("{");
        for (auto &&a : _args) {
            s.append(a).append(",");
        }
        s.pop_back();
        s.append("}");
    }
    return s;
}

}// namespace luisa::compute::osl
