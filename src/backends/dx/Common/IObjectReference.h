#pragma once
#include <vstl/MetaLib.h>
class VObject;
class IObjectReference {
public:
	template<typename T>
	int64 GetVObjectPtrOffset() const {
		size_t thisPtr = reinterpret_cast<size_t>(static_cast<PureType_t<T> const*>(this));
		size_t vobjPtr = reinterpret_cast<size_t>(GetVObjectPtr());
		return static_cast<int64>(thisPtr - vobjPtr);
	}
	virtual VObject* GetVObjectPtr() = 0;
	virtual VObject const* GetVObjectPtr() const = 0;
};

#define VENGINE_IOBJREF_OVERRIDE          \
	VObject* GetVObjectPtr() override {             \
		return this;                                \
	}                                               \
	VObject const* GetVObjectPtr() const override { \
		return this;                                \
	}
