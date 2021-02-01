#pragma once
#include "RendererBase.h"
class IMesh;
class RayRendererData;
class RayRenderer final : public RendererBase {
public:
	RayRenderer(
		Transform* tr);
	RayRenderer(
		Transform* tr,
		GFXDevice* device,
		vengine::vector<vengine::string> const& meshGuids);
	RayRenderer(
		Transform* tr,
		GFXDevice* device,
		vengine::vector<ObjectPtr<IMesh>>&& meshGuids);
	void UpdateMeshLodToLevel(uint level) noexcept override;
	void CullMesh() noexcept override;
	CullingMask GetCullingMask() const noexcept override {
		//Currently Ray-Tracing need not support mask
		return CullingMask::ALL;
	}
	void OnTransformUpdated() override;
	~RayRenderer() noexcept;


	vengine::vector<ObjectPtr<IMesh>> loadMeshes;
private:
	vengine::vector<RayRendererData*> renderDatas;
};
