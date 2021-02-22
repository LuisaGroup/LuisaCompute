#pragma once
#include "interface_common.h"
namespace luisa::compute {
class IFunction {
public:
    enum struct Tag {
        KERNEL,
        DEVICE,
        // TODO: Ray-tracing functions...
    };
};
}// namespace luisa::compute