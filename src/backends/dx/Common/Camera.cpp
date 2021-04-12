#include <Common/Camera.h>
#include <Utility/PassConstants.h>
#include <WorldManagement/World.h>
#include <PipelineComponent/IPipelineResource.h>
using namespace DirectX;
using namespace Math;
Camera::Camera()
	: perCameraResource(32) {
	int3 blockIndex = World::GetInstance()->GetBlockIndex();
	worldOffset = double3(
		(double)blockIndex.x * World::BLOCK_SIZE,
		(double)blockIndex.y * World::BLOCK_SIZE,
		(double)blockIndex.z * World::BLOCK_SIZE);
	SetLens(0.25f * MathHelper::Pi, 1.0f, 1.0f, 1000.0f);
	afterRenderFunction.reserve(16);
}
void Camera::AddAfterRenderFunction(const Runnable<void(Camera*)>& func) {
	afterRenderFunction.push_back(func);
}
void Camera::ExecuteAllCallBack() noexcept {
	for (auto ite = afterRenderFunction.begin(); ite != afterRenderFunction.end(); ++ite) {
		(*ite)(this);
	}
	afterRenderFunction.clear();
}
Camera::~Camera() {
	perCameraResource.IterateAll([](IPipelineResource* ptr) -> void {
		delete ptr;
	});
}
void Camera::UploadCameraBuffer(PassConstants& mMainPassCB) const {
	Matrix4 proj = GetProj();
	Matrix4 viewProj = mul(mView, proj);
	Matrix4 invView = inverse(mView);
	Matrix4 invProj = inverse(proj);
	Matrix4 invViewProj = inverse(viewProj);
	mMainPassCB.View = mView;
	mMainPassCB.InvView = invView;
	mMainPassCB.Proj = proj;
	mMainPassCB.InvProj = invProj;
	mMainPassCB.ViewProj = viewProj;
	mMainPassCB.InvViewProj = invViewProj;
	mMainPassCB.NearZ = GetNearZ();
	mMainPassCB.worldSpaceCameraPos = GetPosition();
	mMainPassCB.FarZ = GetFarZ();
	//Flip
	proj[1].m128_f32[1] *= -1;
	invProj = inverse(proj);
	viewProj = mul(mView, proj);
	invViewProj = inverse(viewProj);
	mMainPassCB._FlipProj = proj;
	mMainPassCB._FlipVP = viewProj;
	mMainPassCB._FlipInvProj = invProj;
	mMainPassCB._FlipInvVP = invViewProj;
	//auto& currPassCB = res->GetCameraCBMap()[GetInstanceID()];
	//currPassCB.buffer->CopyData(currPassCB.element, &mMainPassCB);
}
void Camera::SetPosition(double x, double y, double z) {
	mPosition = float3(x, y, z);
	mViewDirty = true;
}
void Camera::MoveTheWorld(Math::Vector3 const& blockSize) {
	mPosition = (Vector3)mPosition + blockSize;
	worldOffset -= double3(
		blockSize.GetX(),
		blockSize.GetY(),
		blockSize.GetZ());
	mViewDirty = true;
}
void Camera::SetPosition(const float3& v) {
	mPosition = v;
	mViewDirty = true;
}
double3 Camera::GetAbsolutePos() {
	return worldOffset + double3(mPosition.GetX(), mPosition.GetY(), mPosition.GetZ());
}
void Camera::SetLens(double fovY, double aspect, double zn, double zf) {
	// cache properties
	mFovY = fovY;
	mAspect = aspect;
	mNearZ = zn;
	mFarZ = zf;
	mNearWindowHeight = 2.0 * (double)mNearZ * tan(0.5 * mFovY);
	mFarWindowHeight = 2.0 * (double)mFarZ * tan(0.5 * mFovY);
}
void Camera::LookAt(const Math::Vector3& pos, const Math::Vector3& target, const Math::Vector3& worldUp) {
	mLook = normalize(target - pos);
	mRight = normalize(cross(worldUp, mLook));
	mUp = cross(mLook, mRight);
	mPosition = pos;
	mViewDirty = true;
}
void Camera::SetProj(const float4x4& data) {
	memcpy(&mProj, &data, sizeof(float4x4));
}
void Camera::SetView(const float4x4& data) {
	memcpy(&mView, &data, sizeof(float4x4));
}
void Camera::SetProj(const Matrix4& data) {
	memcpy(&mProj, &data, sizeof(float4x4));
}
void Camera::SetView(const Matrix4& data) {
	memcpy(&mView, &data, sizeof(float4x4));
}
void Camera::UpdateProjectionMatrix() {
	if (isOrtho) {
		mFarZ = Max(mFarZ, mNearZ + 0.1);
		Matrix4 P = XMMatrixOrthographicLH(orthoSize, orthoSize, mFarZ, mNearZ);
		mProj = P;
	} else {
		mNearZ = Max(mNearZ, 0.01);
		mFarZ = Max(mFarZ, mNearZ + 0.1);
		Matrix4 P = XMMatrixPerspectiveFovLH(mFovY, mAspect, mFarZ, mNearZ);
		mProj = P;
	}
}
void Camera::UpdateViewMatrix() {
	Vector3& R = mRight;
	Vector3& U = mUp;
	Vector3& L = mLook;
	Vector3& P = mPosition;
	mView = GetInverseTransformMatrix(R, U, L, P);
	cameraToWorld[0] = R;
	cameraToWorld[1] = U;
	cameraToWorld[2] = L;
	cameraToWorld[3] = P;
	cameraToWorld[3].m128_f32[3] = 1;
}
