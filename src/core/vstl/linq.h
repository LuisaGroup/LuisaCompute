#pragma once
#include <core/vstl/MetaLib.h>
#include <core/vstl/vector.h>

namespace vstd::linq {

template<typename T>
class Iterator {
public:
	using SelfType = Iterator<T>;
	virtual T const* Init() = 0;
	virtual T const* Available() = 0;
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
#define VENGINE_LINQ_DECLARE_COPY_MOVE(BType, SType) \
	BType* CopyNew() const override {                \
		return new SType(*this);                     \
	}                                                \
	BType* MoveNew() override {                      \
		return new SType(std::move(*this));          \
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
	T const* colPtr;
	ElementType const* ptr;

public:
	VENGINE_LINQ_DECLARE_COPY_MOVE(BaseType, SelfType)
	IEnumerator(
		IEnumerator const& ie)
		: colPtr(ie.colPtr) {
	}
	IEnumerator(
		IEnumerator&& ie)
		: colPtr(ie.colPtr) {
	}

	virtual ~IEnumerator() {}
	IEnumerator(
		T const& collection) {
		colPtr = &collection;
	}
	IEnumerator(T&&) = delete;
	ElementType const* Init() override {
		curType.Delete();
		curType.New(colPtr->begin());
		ptr = &(**curType);
		return ptr;
	}
	ElementType const* Available() override {
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
	ElementType const* ptr = nullptr;

public:
	VENGINE_LINQ_DECLARE_COPY_MOVE(BaseType, SelfType)
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
	ElementType const* Init() override {
		for (ptr = ite->Init(); ptr = ite->Available(); ite->GetNext()) {
			if (func(*ptr)) {
				return ptr;
			}
		}
		ptr = nullptr;
		return nullptr;
	}
	ElementType const* Available() override {
		return ptr;
	}
	void GetNext() override {
		ite->GetNext();
		while (ptr = ite->Available()) {
			if (func(*ptr)) {
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
	VENGINE_LINQ_DECLARE_COPY_MOVE(BaseType, SelfType)
	TransformIterator(
		TransformIterator const& v)
		: ite(v.ite->CopyNew()),
		  func(v.func) {
	}
	TransformIterator(
		TransformIterator&& v)
		: ite(std::move(v.ite)),
		  func(std::move(v.func)) {
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
	ElementType const* Init() override {
		curEle.Delete();
		OriginType const* ptr = ite->Init();
		if (!ite->Available())
			return nullptr;
		curEle.New(std::move(func(*ptr)));
		ite->GetNext();
		return curEle;
	}
	ElementType const* Available() override {
		return curEle ? static_cast<ElementType const*>(curEle) : nullptr;
	}
	void GetNext() override {
		OriginType const* ptr = ite->Available();
		if (!ptr) {
			curEle.Delete();
			return;
		}
		*curEle = std::move(func(*ptr));

		ite->GetNext();
	}
	virtual ~TransformIterator() {}
};

template<typename T>
class CombinedIterator final
	: public Iterator<T> {
	vstd::vector<Iterator<T>*> iterators;
	Iterator<T>** curIte = nullptr;
	T const* curPtr = nullptr;

public:
	using SelfType = CombinedIterator<T>;
	using BaseType = Iterator<T>;
	VENGINE_LINQ_DECLARE_COPY_MOVE(BaseType, SelfType)
	CombinedIterator(
		SelfType const& t)
		: iterators (t.iterators){

	}
	CombinedIterator(
		SelfType&& t)
		: iterators(std::move(t.iterators)) {
	}

	CombinedIterator(
		vstd::vector<Iterator<T>*>&& iterators)
		: iterators(std::move(iterators)) {}
	CombinedIterator(
		vstd::vector<Iterator<T>*> const& iterators)
		: iterators(iterators) {}
	T const* Init() override {
		curIte = &iterators[0];
		curPtr = (*curIte)->Init();
		return curPtr;
	}
	T const* Available() override {
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
}// namespace vstd::linq