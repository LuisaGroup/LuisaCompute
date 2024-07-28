#include <luisa/std.hpp>
#include <luisa/printer.hpp>
using namespace luisa::shader;

int fib(int layer) {
	if (layer == 0)
		return 0;
	else if (layer == 1)
		return 1;
	else {
		return fib(layer - 1) + fib(layer - 2);
	}
}

[[kernel_1d(1)]] int kernel(
	int layer) {
    device_log("Fibo: {}", fib(layer));
	return 0;
}