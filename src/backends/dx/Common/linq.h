#pragma once
#include <Common/Common.h>

namespace vengine::linq {

template<typename T>
class Iterator {
public:
	using SelfType = Iterator<T>;
	virtual T* Init() = 0;
	virtual T* Available() = 0;
	virtual void GetNext() = 0;
	virtual ~Iterator() {}
	virtual SelfType* CopyNew() const = 0;
	virtual SelfType* MoveNew() = 0;
	template<typename Func>
	decltype(auto) make_filter(Func&& f) const&;
	template<typename Func>
	decltype(auto) make_filter(Func&& f) &&;

	template<typename Func>
	decltype(auto) make_transformer(Func&& f) const&;
	template<typename Func>
	decltype(auto) make_transformer(Func&& f) &&;

	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
};
#define LINQ_DECLARE_COPY_MOVE                 \
	BaseType* CopyNew() const override {       \
		return new SelfType(*this);            \
	}                                          \
	BaseType* MoveNew() override {             \
		return new SelfType(std::move(*this)); \
	}
template<typename T>
using Linq_ElementType = std::remove_reference_t<decltype(*(std::declval<T>().begin()))>;
template<typename T>
class IEnumerator : public Iterator<Linq_ElementType<T>> {
public:
	using BeginIteratorType = std::remove_reference_t<decltype(std::declval<T>().begin())>;
	using ElementType = Linq_ElementType<T>;
	using SelfType = IEnumerator<T>;
	using BaseType = Iterator<Linq_ElementType<T>>;

private:
	StackObject<BeginIteratorType, true> curType;
	T* colPtr;
	ElementType* ptr;

public:
	LINQ_DECLARE_COPY_MOVE
	IEnumerator(
		IEnumerator const& ie)
		: colPtr(ie.colPtr),
		  ptr(ie.ptr) {
		if (ie.curType) {
			curType.New(*ie.curType);
		}
	}
	IEnumerator(
		IEnumerator&& ie)
		: colPtr(ie.colPtr),
		  ptr(ie.ptr) {
		if (ie.curType) {
			curType.New(std::move(*ie.curType));
		}
	}

	virtual ~IEnumerator() {}
	IEnumerator(
		T& collection) {
		colPtr = &collection;
	}
	IEnumerator(T&&) = delete;
	ElementType* Init() override {
		curType.Delete();
		curType.New(colPtr->begin());
		ptr = &(**curType);
		return ptr;
	}
	ElementType* Available() override {
		return (*curType == colPtr->end()) ? nullptr : ptr;
	}
	void GetNext() override {
		++(*curType);
		ptr = &(**curType);
	}
};

template<typename IteType, typename T>
class FilterIterator final : public Iterator<IteType> {
public:
	using ElementType = IteType;
	using BaseType = Iterator<ElementType>;
	using SelfType = FilterIterator<IteType, T>;

private:
	std::unique_ptr<BaseType> ite;
	std::remove_reference_t<T> func;
	ElementType* ptr = nullptr;

public:
	LINQ_DECLARE_COPY_MOVE
	FilterIterator(
		FilterIterator const& v)
		: ite(v.ite->CopyNew()),
		  func(v.func) {}

	FilterIterator(
		FilterIterator&& v)
		: ite(std::move(v.ite)),
		  func(std::move(v.func)) {}

	virtual ~FilterIterator() {}
	FilterIterator(
		BaseType&& lastIte,
		T&& func)
		: ite(lastIte.MoveNew()),
		  func(std::forward<T>(func)) {
	}
	FilterIterator(
		std::unique_ptr<BaseType>&& lastIte,
		T&& func)
		: ite(std::move(lastIte)),
		  func(std::forward<T>(func)) {
	}
	FilterIterator(
		BaseType const& lastIte,
		T&& func)
		: ite(lastIte.CopyNew()),
		  func(std::forward<T>(func)) {
	}
	ElementType* Init() override {
		for (ptr = ite->Init(); ptr = ite->Available(); ite->GetNext()) {
			if (func(static_cast<ElementType const&>(*ptr))) {
				return ptr;
			}
		}
		ptr = nullptr;
		return nullptr;
	}
	ElementType* Available() override {
		return ptr;
	}
	void GetNext() override {
		ite->GetNext();
		while (ptr = ite->Available()) {
			if (func(static_cast<ElementType const&>(*ptr))) {
				return;
			}
			ite->GetNext();
		}
	}
};

template<typename Func, typename Arg>
using TransformIteratorRetType = decltype(std::declval<Func>()(std::declval<Arg>()));

template<typename IteType, typename T>
using TransformIteratorElementType = std::remove_reference_t<TransformIteratorRetType<T, IteType>>;

template<typename IteType, typename T>
class TransformIterator final
	: public Iterator<TransformIteratorElementType<IteType, T>> {
public:
	using OriginType = IteType;
	using ElementType = TransformIteratorElementType<IteType, T>;
	using BaseType = Iterator<ElementType>;
	using SelfType = TransformIterator<IteType, T>;

private:
	std::unique_ptr<Iterator<IteType>> ite;
	std::remove_cvref_t<T> func;
	StackObject<ElementType, true> curEle;

public:
	LINQ_DECLARE_COPY_MOVE
	TransformIterator(
		TransformIterator const& v)
		: ite(v.ite->CopyNew()),
		  func(v.func) {
		if (v.curEle) {
			curEle.New(*v.curEle);
		}
	}
	TransformIterator(
		TransformIterator&& v)
		: ite(std::move(v.ite)),
		  func(std::move(v.func)) {
		if (v.curEle) {
			curEle.New(std::move(*v.curEle));
		}
	}
	TransformIterator(
		Iterator<IteType>&& lastIte,
		T&& func)
		: ite(lastIte.MoveNew()),
		  func(std::forward<T>(func)) {
	}
	TransformIterator(
		std::unique_ptr<Iterator<IteType>>&& lastIte,
		T&& func)
		: ite(std::move(lastIte)),
		  func(std::forward<T>(func)) {
	}
	TransformIterator(
		Iterator<IteType> const& lastIte,
		T&& func)
		: ite(lastIte.CopyNew()),
		  func(std::forward<T>(func)) {
	}
	ElementType* Init() override {
		curEle.Delete();
		OriginType* ptr = ite->Init();
		if (!ite->Available())
			return nullptr;
		curEle.New(std::move(func(std::move(*ptr))));
		ite->GetNext();
		return curEle;
	}
	ElementType* Available() override {
		return curEle ? curEle : nullptr;
	}
	void GetNext() override {
		OriginType* ptr = ite->Available();
		if (!ptr) {
			curEle.Delete();
			return;
		}
		*curEle = std::move(func(std::move(*ptr)));

		ite->GetNext();
	}
	virtual ~TransformIterator() {}
};

template<typename T>
class CombinedIterator final
	: public Iterator<T> {
	vengine::vector<Iterator<T>*> iterators;
	Iterator<T>** curIte = nullptr;
	T* curPtr;

public:
	using SelfType = CombinedIterator<T>;
	using BaseType = Iterator<T>;
	LINQ_DECLARE_COPY_MOVE
	CombinedIterator(
		vengine::vector<Iterator<T>*>&& iterators)
		: iterators(std::move(iterators)) {}
	CombinedIterator(
		vengine::vector<Iterator<T>*> const& iterators)
		: iterators(iterators) {}
	T* Init() override {
		curIte = &iterators[0];
		curPtr = (*curIte)->Init();
		return curPtr;
	}
	T* Available() override {
		return curPtr;
	}
	void GetNext() override {
		(*curIte)->GetNext();
		if (!(curPtr = (*curIte)->Available())) {
			curIte++;
			if (curIte == iterators.end()) {
				curPtr = nullptr;
			} else {
				curPtr = (*curIte)->Init();
			}
		}
	}
	~CombinedIterator() {}
};

template<typename T>
template<typename Func>
decltype(auto) Iterator<T>::make_filter(Func&& f) const& {
	return FilterIterator<T, Func>(*this, std::forward<Func>(f));
}

template<typename T>
template<typename Func>
decltype(auto) Iterator<T>::make_filter(Func&& f) && {
	return FilterIterator<T, Func>(std::move(*this), std::forward<Func>(f));
}

template<typename T>
template<typename Func>
decltype(auto) Iterator<T>::make_transformer(Func&& f) const& {
	return TransformIterator<T, Func>(*this, std::forward<Func>(f));
}

template<typename T>
template<typename Func>
decltype(auto) Iterator<T>::make_transformer(Func&& f) && {
	return TransformIterator<T, Func>(std::move(*this), std::forward<Func>(f));
}

#define LINQ_LOOP(value, iterator) for (auto value = (iterator).Init(); (value = (iterator).Available()); (iterator).GetNext())
#undef LINQ_DECLARE_COPY_MOVE
}// namespace vengine::linq