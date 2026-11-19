#include <luisa/dsl/sugar.h>
#include <type_traits>
#include <tuple>
#include <utility>

using namespace luisa::compute;
using luisa::uint;
using luisa::compute::detail::dsl_function_t;

static_assert(std::is_same_v<dsl_function_t<void(UInt)>, void(uint)>);
static_assert(std::is_same_v<dsl_function_t<Float(UInt)>, float(uint)>);
static_assert(std::is_same_v<dsl_function_t<std::tuple<Float, UInt>(UInt)>, std::tuple<float, uint>(uint)>);
static_assert(std::is_same_v<dsl_function_t<std::pair<Float, UInt>(UInt)>, std::tuple<float, uint>(uint)>);
static_assert(std::is_same_v<dsl_function_t<decltype([](UInt) noexcept {})>, void(uint)>);
static_assert(std::is_same_v<dsl_function_t<decltype([](UInt x) noexcept { return x; })>, uint(uint)>);
static_assert(std::is_same_v<dsl_function_t<decltype([](UInt x) noexcept { return std::pair{x, x}; })>, std::tuple<uint, uint>(uint)>);
static_assert(std::is_same_v<dsl_function_t<decltype([](UInt x) noexcept { return std::tuple{x, x}; })>, std::tuple<uint, uint>(uint)>);

int main() {}
