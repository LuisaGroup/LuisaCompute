#include <WorldManagement/World.h>
#include <Common/Camera.h>
#include <LogicComponent/Transform.h>
#include <CJsonObject/CJsonObject.hpp>
#include <JobSystem/JobInclude.h>
#include <Singleton/MathLib.h>
using namespace Math;
using namespace neb;
World* World::current = nullptr;
World::World() : allTransformsPtr(500) {
	current = this;
}
void World::DestroyAllCameras() {
	for (auto ite = allCameras.begin(); ite != allCameras.end(); ++ite) {
		ite->Destroy();
	}
	allCameras.clear();
}
World::~World() {
	DestroyAllCameras();
}
double3 World::GetWorldBlockPos() const {
	return double3(blockIndex.x, blockIndex.y, blockIndex.z) * BLOCK_SIZE;
}
float3 World::GetRelativeWorldPos(double3 absoluteWorldPos) const {
	double3 moveDirDouble = double3(blockIndex.x, blockIndex.y, blockIndex.z) * BLOCK_SIZE;
	return float3(
		absoluteWorldPos.x - moveDirDouble.x,
		absoluteWorldPos.y - moveDirDouble.y,
		absoluteWorldPos.z - moveDirDouble.z);
}
void World::SetMainCamera(ObjectPtr<Camera> const& cam) {
	mainCam = cam;
}
void World::PrepareUpdateJob(
	JobSystem* sys,
	uint64 currentFrameCount,
	int2 screenSize,
	Math::Vector3& moveDir,
	double3& moveDirDouble,
	bool& isMovingWorld,
	ArrayList<JobBucket*>& buckets) {
	if (!allCameras.empty()) {
		if (!mainCam) {
			mainCam = allCameras[0];
		}
		Camera* mainCamera = mainCam;
		int3 blockOffset = {0, 0, 0};
		isMovingWorld = false;
		if (mainCamera->GetPosition().GetX() > BLOCK_SIZE) {
			blockOffset.x += 1;
			isMovingWorld = true;
		} else if (mainCamera->GetPosition().GetX() < -BLOCK_SIZE) {
			blockOffset.x -= 1;
			isMovingWorld = true;
		}
		if (mainCamera->GetPosition().GetY() > BLOCK_SIZE) {
			blockOffset.y += 1;
			isMovingWorld = true;
		} else if (mainCamera->GetPosition().GetY() < -BLOCK_SIZE) {
			blockOffset.y -= 1;
			isMovingWorld = true;
		}
		if (mainCamera->GetPosition().GetZ() > BLOCK_SIZE) {
			blockOffset.z += 1;
			isMovingWorld = true;
		} else if (mainCamera->GetPosition().GetZ() < -BLOCK_SIZE) {
			blockOffset.z -= 1;
			isMovingWorld = true;
		}
		moveDirDouble = double3(blockOffset.x, blockOffset.y, blockOffset.z) * -BLOCK_SIZE;
		moveDir = Vector3(moveDirDouble.x, moveDirDouble.y, moveDirDouble.z);
		if (isMovingWorld) {
			JobBucket* moveTheWorldBucket = buckets.emplace_back(sys->GetJobBucket());
			int3* bIndex = &blockIndex;
			moveTheWorldBucket->GetTask({}, [=]() -> void {
				auto& cameras = GetCameras();
				for (auto ite = cameras.begin(); ite != cameras.end(); ++ite) {
					(*ite)->MoveTheWorld(moveDir);
				}
			});
			/*mainDependedTasks.push_back(
				bucket->GetTask({}, [=]()->void
					{
						if (grpRenderer)
						{
							grpRenderer->MoveTheWorld(
								device, resource, moveDir, bucket);
						}
					}));*/
			for (auto&& ite : moveTheWorldEvents) {
				ite(moveTheWorldBucket, moveDirDouble);
			}
			Transform::MoveTheWorld(bIndex, blockOffset, moveDirDouble, moveTheWorldBucket);
		}
	}
	JobBucket* bucket = buckets.emplace_back(sys->GetJobBucket());
	for (auto&& ite : frameUpdateEvents) {
		ite(bucket);
	}
}
void World::ClearAllEvents() {
	moveTheWorldEvents.clear();
	frameUpdateEvents.clear();
}
void World::AddMoveTheWorldEvent(Runnable<void(JobBucket*, double3 const&)> const& evt) {
	moveTheWorldEvents.push_back(evt);
}
void World::AddFrameUpdateEvent(Runnable<void(JobBucket*)> const& evt) {
	frameUpdateEvents.push_back(evt);
}
