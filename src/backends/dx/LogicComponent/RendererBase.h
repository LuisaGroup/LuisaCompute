#pragma once
#include <Common/GFXUtil.h>
#include <RenderComponent/Utility/CullingMask.h>
#include <LogicComponent/Component.h>
#include <Common/RandomVector.h>
class IMesh;
class RendererBase : public Component {
protected:
	float sphereBounding = 0;
	void ReCalculateSphereBounding(IMesh const* mesh);

public:
	RendererBase(Transform* trans, bool transformUpdate = false, bool moveTheWorldUpdate = false) noexcept;
	virtual void UpdateMeshLodToLevel(uint level) noexcept = 0;
	virtual void CullMesh() noexcept = 0;
	float GetBoundingRadius() const noexcept { return sphereBounding; }
	virtual CullingMask GetCullingMask() const noexcept = 0;
	virtual ~RendererBase() noexcept;
};