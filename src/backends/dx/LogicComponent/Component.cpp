#include <LogicComponent/Component.h>
#include <LogicComponent/Transform.h>
#include <Common/RandomVector.h>
Component::Component(Transform* trans, bool onTransformUpdateEvent, bool onMoveTheWorldUpdateEvent)
	: transform(trans) {
	assert(trans);
	if (trans) {
		std::lock_guard lck(trans->compLock);
		trans->allComponents.Add(this, &componentIndex);
		if (onTransformUpdateEvent) {
			trans->allTransformUpdateComponents.Add(this, &onTransformUpdateIndex);
		}
		if (onMoveTheWorldUpdateEvent) {
			trans->allMoveTheWorldComponents.Add(this, &onMoveTheWorldUpdateIndex);
		}
	}
}
Component::~Component() {
	if (transform) {
		std::lock_guard lck(transform->compLock);
		transform->allComponents.Remove(componentIndex);
		if (onTransformUpdateIndex != -1) {
			transform->allTransformUpdateComponents.Remove(onTransformUpdateIndex);
		}
		if (onMoveTheWorldUpdateIndex != -1) {
			transform->allMoveTheWorldComponents.Remove(onMoveTheWorldUpdateIndex);
		}
	}
}
