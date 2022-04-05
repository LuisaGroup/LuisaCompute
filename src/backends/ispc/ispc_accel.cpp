//
// Created by Mike Smith on 2022/2/11.
//

#include <core/stl.h>

#include <backends/ispc/ispc_mesh.h>
#include <backends/ispc/ispc_accel.h>

namespace luisa::compute::ispc {

ISPCAccel::ISPCAccel(RTCDevice device, AccelBuildHint hint) noexcept
    : _handle{rtcNewScene(device)} {
    switch (hint) {
        case AccelBuildHint::FAST_TRACE:
            rtcSetSceneBuildQuality(_handle, RTC_BUILD_QUALITY_HIGH);
            rtcSetSceneFlags(_handle, RTC_SCENE_FLAG_COMPACT);
            break;
        case AccelBuildHint::FAST_UPDATE:
            rtcSetSceneBuildQuality(_handle, RTC_BUILD_QUALITY_MEDIUM);
            rtcSetSceneFlags(_handle, RTC_SCENE_FLAG_COMPACT | RTC_SCENE_FLAG_DYNAMIC);
            break;
        case AccelBuildHint::FAST_REBUILD:
            rtcSetSceneBuildQuality(_handle, RTC_BUILD_QUALITY_LOW);
            rtcSetSceneFlags(_handle, RTC_SCENE_FLAG_DYNAMIC);
            break;
    }
}

ISPCAccel::~ISPCAccel() noexcept { rtcReleaseScene(_handle); }

void ISPCAccel::build(ThreadPool &pool, luisa::span<const uint64_t> mesh_handles, luisa::span<const AccelUpdateRequest> requests) noexcept {
    pool.async([this, meshes = luisa::vector<uint64_t>{mesh_handles.cbegin(), mesh_handles.cend()},
                requests = luisa::vector<AccelUpdateRequest>{requests.cbegin(), requests.cend()}] {
        if (meshes.size() < _instances.size()) {// remove redundant geometries
            for (auto i = meshes.size(); i < _instances.size(); i++) { rtcDetachGeometry(_handle, i); }
            _instances.resize(meshes.size());
        } else {// create new geometries
            auto device = rtcGetSceneDevice(_handle);
            _instances.reserve(next_pow2(meshes.size()));
            for (auto i = _instances.size(); i < meshes.size(); i++) {
                auto geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
                rtcSetGeometryBuildQuality(geometry, RTC_BUILD_QUALITY_HIGH);
                rtcAttachGeometryByID(_handle, geometry, i);
                rtcReleaseGeometry(geometry);// already moved into the scene
                _instances.emplace_back().geometry = geometry;
            }
        }
        for (auto r : requests) {
            if (r.flags & AccelUpdateRequest::update_flag_transform) {
                std::memcpy(_instances[r.index].affine, r.affine, sizeof(r.affine));
            }
            if (r.flags & AccelUpdateRequest::update_flag_visibility) {
                _instances[r.index].visible = r.visible;
            }
        }
        for (auto i = 0u; i < _instances.size(); i++) {
            auto geometry = _instances[i].geometry;
            auto mesh = reinterpret_cast<const ISPCMesh *>(meshes[i]);
            rtcSetGeometryInstancedScene(geometry, mesh->handle());
            rtcSetGeometryTransform(geometry, 0u, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, _instances[i].affine);
            if (_instances[i].visible) {
                rtcEnableGeometry(geometry);
            } else {
                rtcDisableGeometry(geometry);
            }
            rtcCommitGeometry(geometry);
            _instances[i].dirty = false;
        }
        rtcCommitScene(_handle);
    });
}

void ISPCAccel::update(ThreadPool &pool, luisa::span<const AccelUpdateRequest> requests) noexcept {
    pool.async([this, requests = luisa::vector<AccelUpdateRequest>{requests.cbegin(), requests.cend()}] {
        for (auto r : requests) {
            if (r.flags & AccelUpdateRequest::update_flag_transform) {
                std::memcpy(_instances[r.index].affine, r.affine, sizeof(r.affine));
            }
            if (r.flags & AccelUpdateRequest::update_flag_visibility) {
                _instances[r.index].visible = r.visible;
            }
            _instances[r.index].dirty = true;
        }
        for (auto &&instance : _instances) {
            if (instance.dirty) {
                auto geometry = instance.geometry;
                rtcSetGeometryTransform(geometry, 0u, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, instance.affine);
                if (instance.visible) {
                    rtcEnableGeometry(geometry);
                } else {
                    rtcDisableGeometry(geometry);
                }
                rtcCommitGeometry(geometry);
                instance.dirty = false;
            }
        }
        rtcCommitScene(_handle);
    });
}

std::array<float, 12> ISPCAccel::_compress(float4x4 m) noexcept {
    return {m[0].x, m[1].x, m[2].x, m[3].x,
            m[0].y, m[1].y, m[2].y, m[3].y,
            m[0].z, m[1].z, m[2].z, m[3].z};
}

float4x4 ISPCAccel::_decompress(std::array<float, 12> m) noexcept {
    return luisa::make_float4x4(
        m[0], m[4], m[8], 0.f,
        m[1], m[5], m[9], 0.f,
        m[2], m[6], m[10], 0.f,
        m[3], m[7], m[11], 1.f);
}

}// namespace luisa::compute::ispc
