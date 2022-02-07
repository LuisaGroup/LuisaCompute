//
// Created by Mike Smith on 2022/2/7.
//

#pragma once

#include <compile/codegen.h>

namespace luisa::compute::ispc {

class ISPCCodegen final : public Codegen {

public:
    void emit(Function f) override;
};

}// namespace luisa::compute::ispc
