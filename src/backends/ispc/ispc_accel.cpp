//
// Created by Mike Smith on 2022/2/11.
//

#include <core/stl.h>

#include <backends/ispc/ispc_mesh.h>
#include <backends/ispc/ispc_accel.h>

namespace luisa::compute::ispc {

ISPCAccel::ISPCAccel(RTCDevice device, AccelUsageHint hint) noexcept
    : _handle{rtcNewScene(device)} {
    switch (hint) {
        case AccelUsageHint::FAST_TRACE:
            rtcSetSceneBuildQuality(_handle, RTC_BUILD_QUALITY_HIGH);
            rtcSetSceneFlags(_handle, RTC_SCENE_FLAG_COMPACT);
            break;
        case AccelUsageHint::FAST_UPDATE:
            rtcSetSceneBuildQuality(_handle, RTC_BUILD_QUALITY_MEDIUM);
            rtcSetSceneFlags(_handle, RTC_SCENE_FLAG_COMPACT | RTC_SCENE_FLAG_DYNAMIC);
            break;
        case AccelUsageHint::FAST_BUILD:
            rtcSetSceneBuildQuality(_handle, RTC_BUILD_QUALITY_LOW);
            rtcSetSceneFlags(_handle, RTC_SCENE_FLAG_DYNAMIC);
            break;
    }
}

ISPCAccel::~ISPCAccel() noexcept { rtcReleaseScene(_handle); }

void ISPCAccel::build(ThreadPool &pool, size_t instance_count,
                      luisa::span<const AccelBuildCommand::Modification> mods) noexcept {

    using Mod = AccelBuildCommand::Modification;
    pool.async([this, instance_count, mods = luisa::vector<Mod>{mods.cbegin(), mods.cend()}]() {
        if (instance_count < _instances.size()) {// remove redundant geometries
            for (auto i = instance_count; i < _instances.size(); i++) { rtcDetachGeometry(_handle, i); }
            _instances.resize(instance_count);
        } else {// create new geometries
            auto device = rtcGetSceneDevice(_handle);
            _instances.reserve(next_pow2(instance_count));
            for (auto i = _instances.size(); i < instance_count; i++) {
                auto geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
                rtcSetGeometryBuildQuality(geometry, RTC_BUILD_QUALITY_HIGH);
                rtcAttachGeometryByID(_handle, geometry, i);
                rtcReleaseGeometry(geometry);// already moved into the scene
                _instances.emplace_back().geometry = geometry;
            }
        }
        for (auto m : mods) {
            auto geometry = _instances[m.index].geometry;
            if (m.flags & Mod::flag_mesh) { rtcSetGeometryInstancedScene(
                geometry, reinterpret_cast<const ISPCMesh *>(m.mesh)->handle()); }
            if (m.flags & Mod::flag_transform) { std::memcpy(
                _instances[m.index].affine, m.affine, sizeof(m.affine)); }
            if (m.flags & Mod::flag_visibility_on) { _instances[m.index].visible = true; }
            if (m.flags & Mod::flag_visibility_off) { _instances[m.index].visible = false; }
            _instances[m.index].dirty = true;
        }
        for (auto &&instance : _instances) {
            if (instance.dirty) {
                auto geometry = instance.geometry;
                rtcSetGeometryTransform(geometry, 0u, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, instance.affine);
                instance.visible ? rtcEnableGeometry(geometry) : rtcDisableGeometry(geometry);
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
