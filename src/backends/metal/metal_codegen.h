//
// Created by Mike Smith on 2021/3/25.
//

#pragma once

#import <compile/codegen.h>

namespace luisa::compute::metal {

class MetalCodegen : public compile::Codegen {
public:
    explicit MetalCodegen(Scratch &scratch) noexcept;
    void emit(Function f) override;
};

}
