#pragma once
//#include "../RenderComponent/MeshRenderer.h"
#include "../Common/Common.h"
#include "../Common/VObject.h"
#include "../Common/BitArray.h"
#include "../Common/RandomVector.h"
#include "../Utility/Actor.h"
#include <mutex>
class GameTimer;
class Transform;
class JobBucket;
class Camera;
class JobSystem;
class JobHandle;
struct TransformMoveStruct;
struct TransformMoveToEndStruct;
//Only For Test!
class World final
{
	friend class Transform;
	friend struct TransformMoveToEndStruct;
	friend struct TransformMoveStruct;
private:
	World();
	static World* current;
//	std::mutex mtx;
	Actor actor;
	//GRPRenderManager* grpRenderer;
	//GpuRPMaterialManager* grpMaterialManager;
	//SurfelLightProbeGI* surfelGI;
	RandomVector<Transform*, true> allTransformsPtr;
	//std::unique_ptr<LodGroupController> lodController;

	int3 blockIndex = { 0,0,0 };
	vengine::vector<Runnable<void(JobBucket*, double3 const&)>> moveTheWorldEvents;
	vengine::vector<Runnable<void(JobBucket*)>> frameUpdateEvents;
	ObjectPtr<Camera> mainCam = nullptr;
public:
	void SetMainCamera(ObjectPtr<Camera> const& cam);
	uint windowWidth;
	uint windowHeight;
	vengine::vector<ObjectPtr<Camera>> allCameras;
	static Actor& GetWorldActor()
	{
		return current->actor;
	}
	int3 GetBlockIndex() const { return blockIndex; }
	void ClearAllEvents();
	void AddMoveTheWorldEvent(Runnable<void(JobBucket*, double3 const&)> const& evt);
	void AddFrameUpdateEvent(Runnable<void(JobBucket*)> const& evt);
	static constexpr int32_t BLOCK_SIZE = 128;
	/*Light* testLight;
	ObjectPtr<Skybox> currentSkybox;
	ObjectPtr<TerrainVirtualTexture> virtualTexture;
	ObjectPtr<TerrainDrawer> terrainDrawer;
	std::unique_ptr<TerrainMainLogic> terrainMainLogic;
	std::unique_ptr<TerrainHeightMap> terrainHeightMap;
	GRPRenderManager* GetGRPRenderManager() const
	{
		return grpRenderer;
	}
	LodGroupController* GetLodGroupController() const
	{
		return lodController.get();
	}
	GpuRPMaterialManager* GetGpuRPMaterialManager() const
	{
		return grpMaterialManager;
	}*/
	vengine::vector<ObjectPtr<Camera>>& GetCameras()
	{
		return allCameras;
	}
	~World();
	void DestroyAllCameras();
	
	static constexpr World* GetInstance() { return current; }
	static constexpr World* CreateInstance()
	{
		if (current)
			return current;
		new World();
		return current;
	}
	static inline void DestroyInstance()
	{
		auto a = current;
		current = nullptr;
		if (a) delete a;
	}
	void PrepareUpdateJob(
		JobSystem* bucket,
		uint64 currentFrameCount,
		int2 screenSize,
		Math::Vector3& moveDir,
		double3& moveDirDouble,
		bool& isMovingWorld,
		ArrayList<JobBucket*>& mainDependedTasks);
	float3 GetRelativeWorldPos(double3 absoluteWorldPos) const;
	double3 GetWorldBlockPos() const;
};