#pragma once
#include <Common/Runnable.h>
#include <Common/Common.h>

namespace vengine::linq {
template<typename T>
class IEnumeratorBase {
public:
	template<typename Y>
	decltype(auto) make_select(
		Y&& ff) && {
		return SelectIEnumerator<T, Y>(std::forward<T>(*static_cast<T*>(this)), std::forward<Y>(ff));
	}
	template<typename Y>
	decltype(auto) make_transform(
		Y&& ff) && {
		return TransformIEnumerator<T, Y>(std::forward<T>(*static_cast<T*>(this)), std::forward<Y>(ff));
	}

	template<typename Y>
	decltype(auto) make_select(
		Y&& ff) const& {
		return SelectIEnumerator<T const&, Y>(*static_cast<T const*>(this), std::forward<Y>(ff));
	}

	template<typename Y>
	decltype(auto) make_transform(
		Y&& ff) const& {
		return TransformIEnumerator<T const&, Y>(*static_cast<T const*>(this), std::forward<Y>(ff));
	}
	template<typename Y>
	decltype(auto) make_pick(
		Y&& ff) const& {
		return PickIEnumerator<T const&, Y>(*static_cast<T const*>(this), std::forward<Y>(ff));
	}

	decltype(auto) to_vector() const {
		vengine::vector<T::ElementType> vec;
		auto tt = static_cast<T const*>(this);
		for (auto i : (*tt)) {
			vec.emplace_back(std::move(i));
		}
		return vec;
	}
	template<typename Y>
	bool same_check(Y&& another_ite) const {
		auto selfPtr = static_cast<T const*>(this);
		auto selfIte = selfPtr->begin();
		auto selfEnd = selfPtr->end();

		auto ite = another_ite.begin();
		auto endIte = another_ite.end();
		while (true) {
			bool finish = ite == endIte;
			bool selfFinish = selfIte == selfEnd;
			bool bothFinish = finish && selfFinish;
			bool anyFinish = finish || selfFinish;
			if (bothFinish != anyFinish) return false;
			if (anyFinish) return true;
			if (*ite != *selfIte) return false;
			++ite;
			++selfIte;
		}
		return true;
	}
};

template<typename T>
class IEnumerator : public IEnumeratorBase<IEnumerator<T>> {
	T const* ptr;

public:
	decltype(auto) begin() const {
		return ptr->begin();
	}
	decltype(auto) end() const {
		return ptr->end();
	}
	IEnumerator(
		T const& collection) : ptr(&collection) {}
};

template<typename T, typename Func>
class SelectIEnumerator : public IEnumeratorBase<SelectIEnumerator<T, Func>> {
public:
	using Type = std::remove_cvref_t<T>;
	using ElementType = std::remove_cvref_t<decltype(*(std::declval<Type>().begin()))>;
	using IteratorType = std::remove_cvref_t<decltype(std::declval<Type>().begin())>;
	using Functor = const std::remove_reference_t<Func>;

private:
	Type ptr;
	Functor func;

public:
	struct Iterator {
		IteratorType curIte;
		IteratorType endIte;
		Functor* ff;
		Iterator(
			Functor* ff,
			IteratorType cc,
			IteratorType ee)
			: curIte(cc), endIte(ee), ff(ff) {
			while (!(curIte == endIte || (*ff)(*curIte)))
				++curIte;
		}
		void operator++() {
			while (true) {
				++curIte;
				if (curIte == endIte || (*ff)(*curIte)) break;
			}
		}
		bool operator==(Iterator const& ite) const {
			return ite.curIte == curIte;
		}
		bool operator!=(Iterator const& ite) const {
			return !operator==(ite);
		}
		ElementType operator*() {
			return *curIte;
		}
	};
	Iterator begin() const {
		return Iterator(&func, ptr.begin(), ptr.end());
	}
	Iterator end() const {
		return Iterator(nullptr, ptr.end(), ptr.end());
	}
	SelectIEnumerator(
		T&& collection,
		Func&& ff)
		: ptr(std::forward<T>(collection)),
		  func(std::forward<Func>(ff)) {}
};
template<typename T, typename Func>
class TransformIEnumerator : public IEnumeratorBase<TransformIEnumerator<T, Func>> {
public:
	using Type = std::remove_cvref_t<T>;
	using ElementType = std::remove_cvref_t<decltype(*(std::declval<Type>().begin()))>;
	using IteratorType = std::remove_cvref_t<decltype(std::declval<Type>().begin())>;
	using Functor = const std::remove_reference_t<Func>;

private:
	Type ptr;
	Functor func;

public:
	struct Iterator {
		IteratorType curIte;
		IteratorType endIte;
		Functor* ff;
		Iterator(
			Functor* ff,
			IteratorType cc,
			IteratorType ee)
			: curIte(cc), endIte(ee), ff(ff) {
		}
		void operator++() {
			++curIte;
		}
		bool operator==(Iterator const& ite) const {
			return ite.curIte == curIte;
		}
		bool operator!=(Iterator const& ite) const {
			return !operator==(ite);
		}
		ElementType operator*() {
			return (*ff)(*curIte);
		}
	};
	Iterator begin() const {
		return Iterator(&func, ptr.begin(), ptr.end());
	}
	Iterator end() const {
		return Iterator(nullptr, ptr.end(), ptr.end());
	}
	TransformIEnumerator(
		T&& collection,
		Func&& ff)
		: ptr(std::forward<T>(collection)),
		  func(std::forward<Func>(ff)) {}
};

template<typename T, typename Func>
class PickIEnumerator : public IEnumeratorBase<PickIEnumerator<T, Func>> {
public:
	using Type = std::remove_cvref_t<T>;
	using ElementType = std::remove_cvref_t<decltype(*(std::declval<Type>().begin()))>;
	using IteratorType = std::remove_cvref_t<decltype(std::declval<Type>().begin())>;
	using Functor = const std::remove_reference_t<Func>;

private:
	Type ptr;
	Functor func;

public:
	struct Iterator {
		IteratorType curIte;
		IteratorType endIte;
		Functor* ff;
		Iterator(
			Functor* ff,
			IteratorType cc,
			IteratorType ee)
			: curIte(cc), endIte(ee), ff(ff) {
			if (curIte != endIte && !(*ff)(*curIte)) {
				curIte = endIte;
			}
		}
		void operator++() {
			++curIte;
			if (curIte != endIte && !(*ff)(*curIte)) {
				curIte = endIte;
			}
		}
		bool operator==(Iterator const& ite) const {
			return ite.curIte == curIte;
		}
		bool operator!=(Iterator const& ite) const {
			return !operator==(ite);
		}
		ElementType operator*() {
			return *curIte;
		}
	};
	Iterator begin() const {
		return Iterator(&func, ptr.begin(), ptr.end());
	}
	Iterator end() const {
		return Iterator(nullptr, ptr.end(), ptr.end());
	}
	PickIEnumerator(
		T&& collection,
		Func&& ff)
		: ptr(std::forward<T>(collection)),
		  func(std::forward<Func>(ff)) {}
};

}// namespace vengine::linq
