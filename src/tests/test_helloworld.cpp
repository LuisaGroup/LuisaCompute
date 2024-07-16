#include <spdlog/fmt/fmt.h>
#include <string>
#include <iostream>
#include "../backends/common/c_codegen/codegen_utils.h"
using namespace luisa;
using namespace luisa::compute;
struct TT{
    float a, b;
};
int main(int argc, char *argv[]) {  
    Clanguage_CodegenUtils codegen;
    auto bool4_type = Type::vector(Type::of<bool>(), 4);
    auto float4_type = Type::vector(Type::of<float>(), 4);
    {
        auto args = {
            bool4_type,
            bool4_type,
            bool4_type};
        codegen.gen_callop(CallOp::SELECT, bool4_type, args);
    }
    {
        auto args = {
            bool4_type};
        codegen.gen_callop(CallOp::ALL, bool4_type, args);
        codegen.gen_callop(CallOp::ANY, bool4_type, args);
    }
    {
        auto args = {
            float4_type,
            float4_type};
        codegen.gen_callop(CallOp::MIN, bool4_type, args);
        codegen.gen_callop(CallOp::MAX, bool4_type, args);
    }
    {
        auto args = {
            float4_type,
            float4_type,
            float4_type};
        codegen.gen_callop(CallOp::CLAMP, bool4_type, args);
    }
    {
        auto args = {
            float4_type};
        codegen.gen_callop(CallOp::SATURATE, bool4_type, args);
    }
    std::cout << codegen.decl_sb.view() << "\n";
}
