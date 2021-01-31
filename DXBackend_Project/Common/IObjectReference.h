#pragma once
class VObject;
class IObjectReference {
public:
	virtual VObject* GetVObjectPtr() = 0;
	virtual VObject const* GetVObjectPtr() const = 0;
};

#define IOBJECTREFERENCE_OVERRIDE_FUNCTION          \
	VObject* GetVObjectPtr() override {             \
		return this;                                \
	}                                               \
	VObject const* GetVObjectPtr() const override { \
		return this;                                \
	}