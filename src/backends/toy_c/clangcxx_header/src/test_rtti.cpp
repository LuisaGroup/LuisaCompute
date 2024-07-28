#include <luisa/std.hpp>
#include <luisa/printer.hpp>
using namespace luisa::shader;

template<typename Int>
void test_callback(Int usr_data, StringView type_desc, Int value_ref) {
	device_log(type_desc);
}
struct MyTest {
	[[clang::annotate("luisa-shader", "my_attr")]] int x;
	[[clang::annotate("luisa-shader", "my_attr_key", "my_attr_value")]] double y;
	StringView str;
};
struct DestructableClass;
void my_func(DestructableClass&);
struct DestructableClass {
	int x;
	Finalizer finalizer;
};
void my_func(DestructableClass& d) {
	device_log("finalizer, x: {}", d.x);
}
[[kernel_1d(1)]] int kernel(
	FunctorRef<void(StringView, uint64)> type_to_json) {

	FunctorRef<void(StringView, uint64)> callback;

	callback = BIND_FUNCTOR(void(StringView, uint64), 0, test_callback<uint64>);
	MyTest mt;
	mt.x = 66;
	mt.y = 77.0f;
	mt.str = to_strview("hello world");
	rtti_call(callback, mt);
	// call host function
	rtti_call(type_to_json, mt);
	Array<DestructableClass, 3> d;
	for (int i = 0; i < 3; ++i) {
		d[i].x = i;
		if (i == 1) {
			d[i].finalizer = 0;
			
		} else {
			d[i].finalizer = BIND_FINALIZER(DestructableClass, my_func);
		}
	}
	// Call member "finalizer" a
	dispose(d);
	return 0;
}