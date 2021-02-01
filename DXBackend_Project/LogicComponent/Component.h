#pragma once
#include "../Common/VObject.h"
#include "../Common/Common.h"
class Transform;
class World;
class Component : public VObject
{
	friend class Transform;
	friend class World;
private:
	uint componentIndex;
	uint onTransformUpdateIndex = -1;
	uint onMoveTheWorldUpdateIndex = -1;
	Transform* transform;
protected:
	bool enabled = false;
	Component(Transform* trans, bool onTransformUpdateEvent = false, bool onMoveTheWorldUpdateEvent = false);
	virtual void OnTransformUpdated() {}
	virtual void OnMovingTheWorld(Math::Vector3 const& moveDir) {}
public:
	Component(Component const&) = delete;
	Component(Component&&) = delete;
	virtual ~Component();
	Transform* GetTransform() const { return transform; }
};