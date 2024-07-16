#include <spdlog/fmt/fmt.h>
#include <string>
#include <iostream>
#include "../backends/common/c_codegen/codegen_utils.h"
using namespace luisa;
using namespace luisa::compute;
int main(int argc, char *argv[]) {
    Clanguage_CodegenUtils codegen;
    vstd::StringBuilder decl_sb;
    auto bool4_type = Type::vector(Type::of<bool>(), 4);
    auto float4_type = Type::vector(Type::of<float>(), 4);
    {
        auto args = {
            bool4_type,
            bool4_type,
            bool4_type};
        codegen.gen_callop(decl_sb, CallOp::SELECT, bool4_type, args);
    }
    {
        auto args = {
            bool4_type};
        codegen.gen_callop(decl_sb, CallOp::ALL, bool4_type, args);
        codegen.gen_callop(decl_sb, CallOp::ANY, bool4_type, args);
    }
    {
        auto args = {
            float4_type,
            float4_type};
        codegen.gen_callop(decl_sb, CallOp::MIN, bool4_type, args);
        codegen.gen_callop(decl_sb, CallOp::MAX, bool4_type, args);
    }
    {
        auto args = {
            float4_type,
            float4_type,
            float4_type};
        codegen.gen_callop(decl_sb, CallOp::CLAMP, bool4_type, args);
    }
    {
        auto args = {
            float4_type};
        codegen.gen_callop(decl_sb, CallOp::SATURATE, bool4_type, args);
    }
    std::cout << decl_sb.view() << "\n";
}
