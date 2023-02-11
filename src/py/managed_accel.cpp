#include <py/managed_accel.h>
#include <py/ref_counter.h>

namespace luisa::compute {

ManagedAccel::Data::Data(Accel &&accel)
    : collector(2), accel(std::move(accel)) {
}

ManagedAccel::ManagedAccel(Accel &&accel) noexcept
    : data(vstd::create_unique(new Data(std::move(accel)))) {
}

ManagedAccel::~ManagedAccel() noexcept {
    if (!data) return;
    auto device = data->accel.device();
    for (auto &&i : data->meshes) {
        if (i.first != 0) {
            device->destroy_mesh(i.first);
        }
    }
}

void ManagedAccel::emplace(MeshUpdateCmd const &mesh, float4x4 const &transform, bool visible, bool opaque) noexcept {
    auto device = data->accel.device();
    auto newMesh = device->create_mesh(mesh.option).handle;
    auto lastSize = data->meshes.size();
    data->meshes.emplace_back(newMesh, mesh);
    data->requireUpdateMesh.emplace(newMesh, mesh);
    data->accel.emplace_back_handle(newMesh, transform, visible, opaque);
    uint64 handles[2] = {mesh.vertex_buffer, mesh.triangle_buffer};
    data->collector.InRef(lastSize, handles);
}

void ManagedAccel::pop_back() noexcept {
    data->accel.pop_back();
    auto lastMesh = std::move(data->meshes.back());
    data->meshes.pop_back();
    data->requireUpdateMesh.erase(lastMesh.first);
    data->meshDisposeList.emplace_back(lastMesh.first);
    data->collector.DeRef(data->meshes.size());
}

void ManagedAccel::set(size_t idx, MeshUpdateCmd const &mesh, float4x4 const &transform, bool visible, bool opaque) noexcept {
    auto &lastMesh = data->meshes[idx];
    data->requireUpdateMesh.erase(lastMesh.first);
    data->meshDisposeList.emplace_back(lastMesh.first);
    auto device = data->accel.device();
    lastMesh.first = device->create_mesh(mesh.option).handle;
    lastMesh.second = mesh;
    data->accel.set_handle(idx, lastMesh.first, transform, visible, opaque);
    data->requireUpdateMesh.emplace(lastMesh.first, lastMesh.second);
    uint64 handles[2] = {mesh.vertex_buffer, mesh.triangle_buffer};
    data->collector.InRef(idx, handles);
}

void ManagedAccel::update(PyStream &stream) noexcept {
    for (auto &&i : data->requireUpdateMesh) {
        auto &m = i.second;
        stream.add(MeshBuildCommand::create(
            i.first,
            AccelBuildRequest::FORCE_BUILD,
            m.vertex_buffer,
            m.vertex_buffer_offset,
            m.vertex_buffer_size,
            m.vertex_stride,
            m.triangle_buffer,
            m.triangle_buffer_offset,
            m.triangle_buffer_size));
    }
    data->requireUpdateMesh.clear();
    stream.add(data->accel.build(AccelBuildRequest::FORCE_BUILD));
    stream.delegates.emplace_back([lst = std::move(data->meshDisposeList), device = data->accel.device()] {
        for (auto &&i : lst) {
            device->destroy_mesh(i);
        }
    });
    data->collector.AfterExecuteStream(stream);
    // TODO: delete data->meshes, update mesh, deref
}

}// namespace luisa::compute