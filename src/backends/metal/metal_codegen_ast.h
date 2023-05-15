//
// Created by Mike Smith on 2023/4/15.
//

#pragma once

#include <ast/function.h>
#include <backends/common/string_scratch.h>

namespace luisa::compute::metal {

class MetalCodegenAST {

private:
    StringScratch &_scratch;
    const Type *_ray_type;
    const Type *_triangle_hit_type;
    const Type *_procedural_hit_type;
    const Type *_committed_hit_type;
    const Type *_ray_query_all_type;
    const Type *_ray_query_any_type;

private:
    void _emit_type_decls(Function kernel) noexcept;
    void _emit_type_name(const Type *type) noexcept;

public:
    explicit MetalCodegenAST(StringScratch &scratch) noexcept;
    void emit(Function kernel) noexcept;
    [[nodiscard]] static size_t type_size_bytes(const Type *type) noexcept;
};

}// namespace luisa::compute::metal
