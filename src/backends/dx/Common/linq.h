#pragma once
#include <Common/Runnable.h>
#include <Common/Common.h>

namespace vengine::linq {
template<typename T, typename Func>
class SelectIEnumerator;
template<typename T, typename Func>
class TransformIEnumerator;

template<typename T>
class IEnumeratorBase {
public:
	template<typename Y>
	SelectIEnumerator<T, Y> make_select(
		Y&& ff) && {
		return SelectIEnumerator<T, Y>(std::forward<T>(*static_cast<T*>(this)), std::forward<Y>(ff));
	}
	template<typename Y>
	TransformIEnumerator<T, Y> make_transform(
		Y&& ff) && {
		return TransformIEnumerator<T, Y>(std::forward<T>(*static_cast<T*>(this)), std::forward<Y>(ff));
	}

	template<typename Y>
	SelectIEnumerator<T const&, Y> make_select(
		Y&& ff) const& {
		return SelectIEnumerator<T const&, Y>(*static_cast<T const*>(this), std::forward<Y>(ff));
	}

	template<typename Y>
	TransformIEnumerator<T const&, Y> make_transform(
		Y&& ff) const& {
		return TransformIEnumerator<T const&, Y>(*static_cast<T const*>(this), std::forward<Y>(ff));
	}

	size_t count() const {
		auto self = static_cast<T const*>(this);
		size_t c = 0;
		for (auto&& i : *self) {
			c++;
		}
		return c;
	}
};

class Range : public IEnumeratorBase<Range> {
private:
	int64 begin;
	int64 end;
	int64 step;
	int64 value;

public:
	using ElementType = int64;
	using ObjType = StackObject<int64>;
	bool GetNext(ObjType& result) {
		*result = value++;
		return *result < end;
	}
	ObjType Reset() {
		value = begin;
		return ObjType();
	}
	Range(
		int64 begin,
		int64 end,
		int64 step = 1) : begin(begin), end(end), step(step) {}
};

template<typename T>
class IEnumerator : public IEnumeratorBase<IEnumerator<T>> {
public:
	using IteratorType = std::remove_cvref_t<decltype(std::declval<T>().begin())>;
	using ElementType = std::remove_cvref_t<decltype(*(std::declval<T>().begin()))>;
	using ObjType = StackObject<ElementType, true>;

private:
	T const* ptr;
	IteratorType ite;

public:
	bool GetNext(ObjType& result) {
		if (ite == ptr->end()) return false;
		auto&& v = *ite;
		if (!result.New(v)) {
			*result = v;
		}
		ite++;
		return true;
	}
	ObjType Reset() {
		ite = ptr->begin();
		return ObjType();
	}
	IEnumerator(
		T const& collection) : ptr(&collection), ite(collection.begin()) {
	}
};

template<typename T, typename Func>
class SelectIEnumerator : public IEnumeratorBase<SelectIEnumerator<T, Func>> {
public:
	using Type = std::remove_cvref_t<T>;
	using ElementType = typename Type::ElementType;
	using Functor = const std::remove_reference_t<Func>;
	using ObjType = typename Type::ObjType;

private:
	Type ptr;
	Functor func;

public:
	bool GetNext(ObjType& result) {
		while (ptr.GetNext(result)) {
			if (func(*result)) {
				return true;
			}
		}
		return false;
	}
	ObjType Reset() {
		return ptr.Reset();
	}

	SelectIEnumerator(
		T&& collection,
		Func&& ff)
		: ptr(std::forward<T>(collection)),
		  func(std::forward<Func>(ff)) { Reset(); }
};
template<typename T, typename Func>
class TransformIEnumerator : public IEnumeratorBase<TransformIEnumerator<T, Func>> {
public:
	using Type = std::remove_cvref_t<T>;
	using ElementType = typename Type::ElementType;
	using Functor = const std::remove_reference_t<Func>;
	using ObjType = typename Type::ObjType;

private:
	Type ptr;
	Functor func;

public:
	bool GetNext(ObjType& result) {
		if (ptr.GetNext(result)) {
			func(*result);
			return true;
		}
		return false;
	}
	ObjType Reset() {
		return ptr.Reset();
	}

	TransformIEnumerator(
		T&& collection,
		Func&& ff)
		: ptr(std::forward<T>(collection)),
		  func(std::forward<Func>(ff)) { Reset(); }
};

#define LINQ_LOOP(V, S) \
	for (auto&& V = (S).Reset(); (S).GetNext(V);)
}// namespace vengine::linq
