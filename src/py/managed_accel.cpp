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
        if (i.second.type == MeshRefType::Mesh) {
            device->destroy_mesh(i.second.mesh);
        } else {
            device->destroy_procedural_primitive(i.second.mesh);
        }
    }
}
template<typename T>
uint64 ManagedAccel::set_mesh(size_t index, T const &mesh) noexcept {
    constexpr MeshRefType _mesh_ref_type = std::is_same_v<T, MeshUpdateCmd> ? MeshRefType::Mesh : MeshRefType::Procedural;
    auto device = data->accel.device();
    T temp;
    memcpy(&temp, &mesh, sizeof(temp));
    // do this in case aliasing problem
    memset(&temp.option, 0, sizeof(temp.option));
    temp.option.allow_compaction = mesh.option.allow_compaction;
    temp.option.allow_update = mesh.option.allow_update;
    temp.option.hint = mesh.option.hint;
    auto &last_mesh = data->meshes[index];
    vstd::MD5 md5({reinterpret_cast<uint8_t *>(&temp), sizeof(T)});
    if (last_mesh.mesh != invalid_resource_handle) {
        if (last_mesh.md5 == md5) {
            return last_mesh.mesh;
        } else {
            remove_mesh(index);
        }
    }
    auto iter = data->created_mesh.try_emplace(md5);
    uint64 new_mesh;
    auto &v = iter.first->second;
    if (iter.second) {
        if constexpr (_mesh_ref_type == MeshRefType::Mesh)
            new_mesh = device->create_mesh(mesh.option).handle;
        else
            new_mesh = device->create_procedural_primitive(mesh.option).handle;
        v.mesh = new_mesh;
        v.ref_count = 1;
        v.type = _mesh_ref_type;
        data->require_update_mesh.try_emplace(new_mesh, temp);
    } else {
        new_mesh = v.mesh;
        v.ref_count++;
    }
    last_mesh = {new_mesh, md5};
    if constexpr (_mesh_ref_type == MeshRefType::Mesh) {
        uint64 handles[2] = {mesh.vertex_buffer, mesh.triangle_buffer};
        data->collector.InRef(index, handles);
    } else {
        uint64 handles[2] = {mesh.aabb_buffer, invalid_resource_handle};
        data->collector.InRef(index, handles);
    }
    return new_mesh;
}
void ManagedAccel::remove_mesh(size_t index) noexcept {
    auto &&mesh = data->meshes[index];
    auto iter = data->created_mesh.find(mesh.md5);
    if (iter == data->created_mesh.end()) [[unlikely]]
        return;
    iter->second.ref_count--;
    if (iter->second.ref_count == 0) {
        data->require_update_mesh.erase(mesh.mesh);
        data->mesh_dispose_list.emplace_back(mesh.mesh, mesh.ref_type);
        data->collector.DeRef(index);
        data->created_mesh.erase(iter);
    }
}
void ManagedAccel::emplace(MeshUpdateCmd const &mesh, float4x4 const &transform, uint visibility_mask, bool opaque) noexcept {
    auto sz = data->meshes.size();
    data->meshes.emplace_back(MeshValue{invalid_resource_handle, vstd::MD5{}, MeshRefType::Mesh});
    auto new_mesh = set_mesh<MeshUpdateCmd>(sz, mesh);
    data->accel.emplace_back_handle(new_mesh, transform, visibility_mask, opaque);
}
void ManagedAccel::emplace(ProceduralUpdateCmd const &procedural, float4x4 const &transform, uint visibility_mask, bool opaque) noexcept {
    auto sz = data->meshes.size();
    data->meshes.emplace_back(MeshValue{invalid_resource_handle, vstd::MD5{}, MeshRefType::Procedural});
    auto new_mesh = set_mesh<ProceduralUpdateCmd>(sz, procedural);
    data->accel.emplace_back_handle(new_mesh, transform, visibility_mask, opaque);
}

void ManagedAccel::pop_back() noexcept {
    remove_mesh(data->meshes.size() - 1);
    data->accel.pop_back();
    data->meshes.pop_back();
}

void ManagedAccel::set(size_t idx, MeshUpdateCmd const &mesh, float4x4 const &transform, uint visibility_mask, bool opaque) noexcept {
    auto &last_mesh = data->meshes[idx];
    auto new_mesh = set_mesh<MeshUpdateCmd>(idx, mesh);
    data->accel.set_handle(idx, new_mesh, transform, visibility_mask, opaque);
}
void ManagedAccel::set(size_t idx, ProceduralUpdateCmd const &procedural, float4x4 const &transform, uint visibility_mask, bool opaque) noexcept {
    auto &last_mesh = data->meshes[idx];
    auto new_mesh = set_mesh<ProceduralUpdateCmd>(idx, procedural);
    data->accel.set_handle(idx, new_mesh, transform, visibility_mask, opaque);
}

void ManagedAccel::update(PyStream &stream) noexcept {
    for (auto &&i : data->require_update_mesh) {
        auto &m = i.second;
        m.multi_visit(
            [&](MeshUpdateCmd const &cmd) {
                stream.add(MeshBuildCommand::create(
                    i.first,
                    AccelBuildRequest::PREFER_UPDATE,
                    cmd.vertex_buffer,
                    cmd.vertex_buffer_offset,
                    cmd.vertex_buffer_size,
                    cmd.vertex_stride,
                    cmd.triangle_buffer,
                    cmd.triangle_buffer_offset,
                    cmd.triangle_buffer_size));
            },
            [&](ProceduralUpdateCmd const &cmd) {
                stream.add(ProceduralPrimitiveBuildCommand::create(
                    i.first,
                    AccelBuildRequest::PREFER_UPDATE,
                    cmd.aabb_buffer,
                    cmd.aabb_offset,
                    cmd.aabb_count));
            });
    }
    data->require_update_mesh.clear();
    stream.add(data->accel.build(AccelBuildRequest::PREFER_UPDATE));
    stream.delegates.emplace_back([lst = std::move(data->mesh_dispose_list), device = data->accel.device()] {
        for (auto &&i : lst) {
            if (i.second == MeshRefType::Mesh) {
                device->destroy_mesh(i.first);
            } else {
                device->destroy_procedural_primitive(i.first);
            }
        }
    });
    data->collector.AfterExecuteStream(stream);
    // TODO: delete data->meshes, update mesh, deref
}

}// namespace luisa::compute