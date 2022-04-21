#pragma once
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
namespace toolhub::directx {
class TopAccel;
struct BindProperty {
	vstd::string name;
	vstd::variant<
		BufferView,
		DescriptorHeapView,
		TopAccel const*>
		prop;
    BindProperty() {}
	template<typename A, typename B>
	requires(
		std::is_constructible_v<decltype(name), A&&> || std::is_constructible_v<decltype(prop), B&&>)
		BindProperty(
			A&& a,
			B&& b)
		: name(std::forward<A>(a)),
		  prop(std::forward<B>(b)) {}
};
}// namespace toolhub::directx