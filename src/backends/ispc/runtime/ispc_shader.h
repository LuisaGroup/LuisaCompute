#pragma once
#include <vstl/Common.h>
#include <core/dynamic_module.h>
namespace lc::ispc {
using namespace luisa;
class Shader : public vstd::IOperatorNewBase {
private:
    DynamicModule dllModule;
    vstd::funcPtr_t<void(uint, uint, uint, uint64)> exportFunc;

public:
    Shader(std::string const &str);

    template<typename... Args>
    static void PackArgs(
        vstd::vector<void const *> &vec,
        Args const &...args) {
        vec.clear();
        auto getPointer = [&](auto const &v) -> uint8_t {
            vec.emplace_back(&v);
            return 0;
        };
        auto lst = {getPointer(args)...};
    }
    void dispatch(
        uint x,
        uint y,
        uint z,
        vstd::vector<void const *> const &vec) const;
};
}// namespace lc::ispc