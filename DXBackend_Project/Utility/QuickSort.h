#pragma once
#include <type_traits>
#include <stdint.h>
#include "../Common/vector.h"
enum class SortOption {
	SmallToLarge,
	LargeToSmall
};
template<typename F, SortOption opt>
void QuicksortStack(std::remove_cvref_t<F>* a, int64_t count) noexcept {
	using T = std::remove_cvref_t<F>;
	struct StackData {
		int64_t p;
		int64_t q;
		bool state;
		int64_t i;
		int64_t j;
	};
	ArrayList<StackData> stackDataVector;
	stackDataVector.reserve(count);
	stackDataVector.push_back({0, count - 1, false});

	while (!stackDataVector.empty()) {
		auto&& last = *(stackDataVector.end() - 1);
		if (!last.state) {
			int64_t& i = last.i;
			int64_t& j = last.j;
			int64_t& p = last.p;
			int64_t& q = last.q;
			i = p;
			j = q;
			T temp = a[p];
			if constexpr (opt == SortOption::SmallToLarge) {
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
			} else {
				while (i < j) {
					while (a[j] <= temp && j > i) j--;

					if (j > i) {
						a[i] = a[j];
						i++;
						while (a[i] >= temp && i < j) i++;
						if (i < j) {
							a[j] = a[i];
							j--;
						}
					}
				}
			}
			a[i] = temp;
			if (p < (i - 1)) {
				stackDataVector.push_back({p, i - 1, false});
				last.state = true;
			} else if ((j + 1) < q) {
				last = {j + 1, q, false};
			} else {
				stackDataVector.erase_last();
			}
		} else {
			int64_t& i = last.i;
			int64_t& j = last.j;
			int64_t& p = last.p;
			int64_t& q = last.q;
			if ((j + 1) < q) {
				last = {j + 1, q, false};
			} else {
				stackDataVector.erase_last();
			}
		}
	}
}