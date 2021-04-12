#include <LogicComponent/RendererBase.h>
#include <LogicComponent/Transform.h>
#include <RenderComponent/IMesh.h>
RendererBase::RendererBase(Transform* trans, bool transformUpdate, bool moveTheWorldUpdate) noexcept :
	Component(trans, transformUpdate, moveTheWorldUpdate)
{
	
}
RendererBase::~RendererBase() noexcept
{
	
}

void RendererBase::ReCalculateSphereBounding(IMesh const* mesh) {
	using namespace Math;
	static const Vector3 offsets[] =
		{
			{1, 1, 1},
			{1, 1, -1},
			{1, -1, 1},
			{1, -1, -1},
			{-1, 1, 1},
			{-1, 1, -1},
			{-1, -1, 1},
			{-1, -1, -1}};
	auto localToWorld = GetTransform()->GetLocalToWorldMatrixCPU();
	Vector3 worldCenter = (Vector3)mul(localToWorld, Vector4(mesh->GetBoundingCenter(), 1));
	Vector3 worldExtent = (Vector3)mul(localToWorld, Vector4(mesh->GetBoundingExtent(), 0));
	Vector3 minPos = worldCenter + worldExtent * offsets[0];
	Vector3 maxPos = minPos;
	for (uint i = 1; i < 8; ++i) {
		auto point = worldCenter + worldExtent * offsets[i];
		minPos = Min(minPos, point);
		maxPos = Max(maxPos, point);
	}
	sphereBounding = distance(maxPos, minPos) * 0.5f;
}
