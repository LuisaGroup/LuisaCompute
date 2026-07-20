// Test for AST (Abstract Syntax Tree) type system and attribute handling.
// This test verifies the creation of buffer types with custom attributes.

#include <luisa/luisa-compute.h>
#include <exception>
#include "ut/ut.hpp"
#include "test_device.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int test_ast(Device &device) {
    static_cast<void>(device);
    // Enable verbose logging for debugging
    luisa::log_level_verbose();

    // Create a list of custom attributes for the buffer type
    luisa::vector<Attribute> attris;
    attris.emplace_back("attr0", "attr1");

    // Create a buffer type with float elements and custom attributes
    auto t = Type::buffer(Type::of<float>(), attris);

    // Print the type description to verify attribute handling
    LUISA_INFO("{}", t->description());

    expect(t != nullptr);
    expect(t->is_buffer());
    expect(t->element() == Type::of<float>());
    expect(eq(luisa::string_view{t->description()}, luisa::string_view{"buffer[attr0(attr1)]<float>"}));
    return 0;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_ast(device);
}
