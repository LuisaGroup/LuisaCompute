#include <luisa/std.hpp>
#include <luisa/printer.hpp>
using namespace luisa::shader;

template <typename T>
int device_func(T& a, T b) {
	a = a + b;
	device_log("result: {}", a);
	return a;
}

[[kernel_1d(1)]] int kernel(
	FunctionRef<int(int, int)> host_func) {
	// invoke host function
    host_func(3, 6);
	// Bind device function
	FunctionRef<int(int&, int)> device_func_ref = BIND_FUNCTION(int(int&, int), device_func<int>);
	// invoke device function
	int input = 6;
    device_func_ref(input, 9);
	device_log("ref: {}", input);
	return 0;
}