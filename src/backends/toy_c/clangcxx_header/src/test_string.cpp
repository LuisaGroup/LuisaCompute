#include <luisa/std.hpp>
#include <luisa/printer.hpp>
using namespace luisa::shader;

StringView string_add(
	StringView left,
	StringView right) {
    auto bf = temp_new_buffer<int8>(left._len + right._len);
    memcpy(bf._ptr, left._ptr, left._len);
    memcpy(bf._ptr + left._len, right._ptr, right._len);
    return StringView(bf._ptr, bf._len);
}

[[kernel_1d(1)]] int kernel() {
	int a = 66;
	float b = 77;
	device_log("test: {}, {}", a, b);
	auto str = to_strview("hello world, {}, {}");
	device_log(str, b, a);
    auto add_str = string_add(to_strview("first "), to_strview("second"));
    device_log(add_str);
	return 0;
}
