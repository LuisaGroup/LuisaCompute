#pragma once

#include <core/stl.h>

namespace lc::ispc {

class ISPCBindlessArray {
private:
    size_t size;
public:
    explicit ISPCBindlessArray(size_t size) noexcept;

};

}