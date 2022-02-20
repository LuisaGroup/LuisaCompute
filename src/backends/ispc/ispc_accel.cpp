//
// Created by Mike Smith on 2022/2/11.
//

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

ISPCAccel::~ISPCAccel() noexcept {
    rtcReleaseScene(_handle);
    for (auto geometry : _committed_geometries) {
        rtcReleaseGeometry(geometry);
    }
}

void ISPCAccel::build(ThreadPool &pool) noexcept {
    _dirty.clear();
    _resources.clear();
    for (auto &_instance : _instances) {
        auto mesh = _instance.mesh;
        _resources.emplace(mesh->vertex_buffer());
        _resources.emplace(mesh->triangle_buffer());
        _resources.emplace(reinterpret_cast<uint64_t>(mesh));
    }
    pool.async([this, instances = _instances] {
        for (auto i = 0u; i < _committed_geometries.size(); i++) {
            rtcDetachGeometry(_handle, i);
            rtcReleaseGeometry(_committed_geometries[i]);
        }
        auto device = rtcGetSceneDevice(_handle);
        _committed_transforms.resize(instances.size());
        _committed_geometries.resize(instances.size());
        for (auto i = 0u; i < instances.size(); i++) {
            auto transform = instances[i].transform;
            _committed_transforms[i] = _decompress(transform);
            auto geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
            _committed_geometries[i] = geometry;
            rtcSetGeometryInstancedScene(geometry, instances[i].mesh->handle());
            rtcSetGeometryBuildQuality(geometry, RTC_BUILD_QUALITY_HIGH);
            rtcSetGeometryTransform(
                geometry, 0.f,
                RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR,
                &_committed_transforms[i]);
            if (!instances[i].visible) { rtcDisableGeometry(geometry); }
            rtcCommitGeometry(geometry);
            rtcAttachGeometryByID(_handle, geometry, i);
        }
        rtcCommitScene(_handle);
    });
}

void ISPCAccel::update(ThreadPool &pool) noexcept {
    auto s = luisa::span{_instances}.subspan(_dirty.offset(), _dirty.size());
    pool.async([this, offset = _dirty.offset(),
                instances = luisa::vector<Instance>{s.cbegin(), s.cend()}] {
        for (auto i = 0u; i < instances.size(); i++) {
            auto geometry = _committed_geometries[i + offset];
            if (instances[i].visible) {
                rtcEnableGeometry(geometry);
            } else {
                rtcDisableGeometry(geometry);
            }
            auto transform = instances[i].transform;
            _committed_transforms[i + offset] = _decompress(transform);
            rtcSetGeometryTransform(
                geometry, 0.f, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR,
                &_committed_transforms[i + offset]);
            rtcCommitGeometry(geometry);
        }
        rtcCommitScene(_handle);
    });
    _dirty.clear();
}

void ISPCAccel::push_mesh(const ISPCMesh *mesh, float4x4 transform, bool visible) noexcept {
    Instance instance{
        .mesh = mesh,
        .transform = _compress(transform),
        .visible = visible};
    _instances.emplace_back(instance);
    _resources.emplace(mesh->vertex_buffer());
    _resources.emplace(mesh->triangle_buffer());
    _resources.emplace(reinterpret_cast<uint64_t>(mesh));
}

void ISPCAccel::pop_mesh() noexcept { _instances.pop_back(); }

void ISPCAccel::set_mesh(size_t index, const ISPCMesh *mesh, float4x4 transform, bool visible) noexcept {
    _instances[index] = {
        .mesh = mesh,
        .transform = _compress(transform),
        .visible = visible};
    _resources.emplace(mesh->vertex_buffer());
    _resources.emplace(mesh->triangle_buffer());
    _resources.emplace(reinterpret_cast<uint64_t>(mesh));
}

void ISPCAccel::set_visibility(size_t index, bool visible) noexcept {
    _instances[index].visible = visible;
    _dirty.mark(index);
}

void ISPCAccel::set_transform(size_t index, float4x4 transform) noexcept {
    _instances[index].transform = _compress(transform);
    _dirty.mark(index);
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
