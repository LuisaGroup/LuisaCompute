#pragma once
#include <vstl/MetaLib.h>
#include <vstl/Memory.h>
namespace vstd {
struct unique_ptr_deleter {
    template<typename T>
    void operator()(T *ptr) const noexcept {
		if constexpr (std::is_base_of_v<IDisposable, T>) {
			ptr->Dispose();
		} else {
			delete ptr;
		}
	}
};
template<typename T>
using unique_ptr = std::unique_ptr<T, unique_ptr_deleter>;
template<typename T>
unique_ptr<T> make_unique(T* ptr) {
	return unique_ptr<T>(ptr);
}
}// namespace vstd