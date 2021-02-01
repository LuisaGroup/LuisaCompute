#include "Actor.h"
void* Actor::GetComponent(Type t) const {
	auto ite = hash.Find(t);
	if (ite) return ite.Value().ptr;
	return nullptr;
}
Actor::Pointer::~Pointer() {
	disposer(ptr);
}
void Actor::Pointer::operator=(Pointer const& p) {
	disposer(ptr);
	ptr = p.ptr;
	disposer = p.disposer;
}
void Actor::RemoveComponent(Type t) {
	hash.Remove(t);
}
void Actor::SetComponent(Type t, void* ptr, void (*disposer)(void*)) {
	auto&& ite = hash.Insert(t).Value();
	ite.ptr = ptr;
	ite.disposer = disposer;
}
Actor::Actor() {}
Actor::Actor(uint32_t initComponentCapacity) : hash(initComponentCapacity) {}
Actor::~Actor() {}
