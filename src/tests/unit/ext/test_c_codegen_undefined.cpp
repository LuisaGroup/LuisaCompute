// Regression for direct C AST refinement of CallOp::UNDEFINED.

#include "ut/ut.hpp"

#include <array>
#include <string_view>

#include <luisa/dsl/sugar.h>

#include "codegen_visitor.h"

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "direct_c_undefined_is_a_complete_expression"_test = [] {
        using Bank = std::array<float4, 3u>;
        Callable<Bank()> callable{[]() noexcept {
            return undefined<Bank>();
        }};

        Clanguage_CodegenUtils utilities;
        vstd::StringBuilder source;
        CodegenVisitor visitor{
            source, "undefined_seed", utilities, callable.function()};
        auto text = source.view();
        expect(text.find("{0})") != luisa::string_view::npos)
            << "direct C must refine undefined to a typed zero compound literal";
        expect(text.find("{0})()") == luisa::string_view::npos)
            << "the generic call suffix must not be appended to a value expression";
    };
}
