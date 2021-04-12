#include <LogicComponent/RayRenderer.h>
#include <PipelineComponent/RayTracing/RayTracingManager.h>
#include <RenderComponent/Mesh.h>

RayRenderer::RayRenderer(Transform* tr)
	: RendererBase(tr, true) {
}
RayRenderer::RayRenderer(
	Transform* tr,
	GFXDevice* device,
	vengine::vector<vengine::string> const& meshGuids)
	: RendererBase(tr, true, true) {
	for (auto& i : meshGuids) {
		/*loadMeshes.emplace_back(
			std::move(
				AssetDatabase::GetInstance()
					->SyncLoad(
						device,
						i,
						AssetLoadType::Mesh)
					.CastTo<Mesh>()
					.CastTo<IMesh>()));*/
	}
}

RayRenderer::RayRenderer(Transform* tr, GFXDevice* device, vengine::vector<ObjectPtr<IMesh>>&& meshGuids)
	: RendererBase(tr, true),
	  loadMeshes(std::move(meshGuids)) {
}

void RayRenderer::UpdateMeshLodToLevel(uint level) noexcept {
	enabled = true;
	///// Cull Old Datas
	CullMesh();
	ObjectPtr<IMesh> mesh = loadMeshes[level];
	IMesh* mm = mesh;
	uint subCount = mm->GetSubMeshCount();
	renderDatas.reserve(subCount);
/*	for (uint i = 0; i < subCount; ++i) {
		uint matIndex = mm->GetSubMesh(i).materialIndex;
		if (matIndex < materials.size()) {
			GpuRPMaterial* mat = materials[matIndex];
			renderDatas.push_back(RayTracingManager::GetInstance()->AddRenderer(
				std::move(mesh),
				mat->GetMaterialTypeIndex(), mat->GetMaterialIndex(),
				GetTransform(), -1));
		}
		
	}*/
}

void RayRenderer::CullMesh() noexcept {
	/*if (RayTracingManager::GetInstance()) {
		for (auto& i : renderDatas) {
			RayTracingManager::GetInstance()->RemoveRenderer(
				i);
		}
	}*/
	renderDatas.clear();
}
RayRenderer::~RayRenderer() noexcept {
	CullMesh();
}



void RayRenderer::OnTransformUpdated() {
	/*for (auto& i : renderDatas) {
		RayTracingManager::GetInstance()->UpdateRenderer(
			nullptr,
			-1, -1,
			GetTransform(),
			i, -1);
	}*/
}
