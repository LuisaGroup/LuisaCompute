#pragma once
#include <type_traits>
#include <stdint.h>

template<typename F>
constexpr void QuicksortStack(std::remove_cvref_t<F>* a, int32_t p, int32_t q) noexcept {
	using T = std::remove_cvref_t<F>;
	int32_t i = p;
	int32_t j = q;
	T temp = a[p];

	while (i < j) {
		while (a[j] >= temp && j > i) j--;

		if (j > i) {
			a[i] = a[j];
			i++;
			while (a[i] <= temp && i < j) i++;
			if (i < j) {
				a[j] = a[i];
				j--;
			}
		}
	}
	a[i] = temp;
	if (p < (i - 1)) QuicksortStack<F>(a, p, i - 1);
	if ((j + 1) < q) QuicksortStack<F>(a, j + 1, q);
}

template<typename F, typename CompareFunc>
constexpr void QuicksortStackCustomCompare(std::remove_cvref_t<F>* a, const CompareFunc& compareFunc, int32_t p, int32_t q) noexcept {
	using T = std::remove_cvref_t<F>;
	int32_t i = p;
	int32_t j = q;
	T temp = a[p];

	while (i < j) {
		while (compareFunc(a[j], temp) >= 0 && j > i) j--;

		if (j > i) {
			a[i] = a[j];
			i++;
			while (compareFunc(a[i], temp) <= 0 && i < j) i++;
			if (i < j) {
				a[j] = a[i];
				j--;
			}
		}
	}
	a[i] = temp;
	if (p < (i - 1)) QuicksortStackCustomCompare<F, CompareFunc>(a, compareFunc, p, i - 1);
	if ((j + 1) < q) QuicksortStackCustomCompare<F, CompareFunc>(a, compareFunc, j + 1, q);
}