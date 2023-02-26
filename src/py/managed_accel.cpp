#include <py/managed_accel.h>
#include <py/ref_counter.h>

namespace luisa::compute {

ManagedAccel::Data::Data(Accel &&accel) noexcept
    : collector(2), accel(std::move(accel)) {
}

ManagedAccel::ManagedAccel(Accel &&accel) noexcept
    : data(vstd::create_unique(new Data(std::move(accel)))) {
}

ManagedAccel::~ManagedAccel() noexcept {
    if (!data) return;
    auto device = data->accel.device();
    for (auto &&i : data->created_mesh) {
        device->destroy_mesh(i.second.mesh);
    }
}
uint64 ManagedAccel::set_mesh(size_t index, MeshUpdateCmd const &mesh) noexcept {
    auto device = data->accel.device();
    MeshUpdateCmd temp;
    memcpy(&temp, &mesh, sizeof(temp));
    memset(&temp.option, 0, sizeof(temp.option));
    temp.option.allow_compaction = mesh.option.allow_compaction;
    temp.option.allow_update = mesh.option.allow_update;
    temp.option.hint = mesh.option.hint;
    auto &last_mesh = data->meshes[index];
    vstd::MD5 md5({reinterpret_cast<uint8_t *>(&md5), sizeof(MeshUpdateCmd)});
    if (last_mesh.first != invalid_resource_handle) {
        if (last_mesh.second == md5) {
            return last_mesh.first;
        } else {
            remove_mesh(index);
        }
    }
    auto iter = data->created_mesh.try_emplace(md5);
    uint64 new_mesh;
    if (iter.second) {
        new_mesh = device->create_mesh(mesh.option).handle;
        iter.first->second.mesh = new_mesh;
        iter.first->second.ref_count = 1;
        data->requireUpdateMesh.try_emplace(new_mesh, temp);
    } else {
        new_mesh = iter.first->second.mesh;
        iter.first->second.ref_count++;
    }
    last_mesh = {new_mesh, md5};
    uint64 handles[2] = {mesh.vertex_buffer, mesh.triangle_buffer};
    data->collector.InRef(index, handles);
    return new_mesh;
}
void ManagedAccel::remove_mesh(size_t index) noexcept {
    auto &&mesh = data->meshes[index];
    auto iter = data->created_mesh.find(mesh.second);
    if (iter == data->created_mesh.end()) [[unlikely]]
        return;
    iter->second.ref_count--;
    if (iter->second.ref_count == 0) {
        data->requireUpdateMesh.erase(mesh.first);
        data->meshDisposeList.emplace_back(mesh.first);
        data->collector.DeRef(index);
        data->created_mesh.erase(iter);
    }
}
void ManagedAccel::emplace(MeshUpdateCmd const &mesh, float4x4 const &transform, bool visible, bool opaque) noexcept {
    auto sz = data->meshes.size();
    data->meshes.emplace_back(invalid_resource_handle, vstd::MD5{});
    auto new_mesh = set_mesh(sz, mesh);
    data->accel.emplace_back_handle(new_mesh, transform, visible, opaque);
}

void ManagedAccel::pop_back() noexcept {
    remove_mesh(data->meshes.size() - 1);
    data->accel.pop_back();
    data->meshes.pop_back();
}

void ManagedAccel::set(size_t idx, MeshUpdateCmd const &mesh, float4x4 const &transform, bool visible, bool opaque) noexcept {
    auto &last_mesh = data->meshes[idx];
    auto new_mesh = set_mesh(idx, mesh);
    data->accel.set_handle(idx, new_mesh, transform, visible, opaque);
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