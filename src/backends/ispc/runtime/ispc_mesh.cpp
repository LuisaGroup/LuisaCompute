#include <backends/ispc/runtime/ispc_mesh.h>

namespace lc::ispc {

    ISPCMesh::ISPCMesh(uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
        uint64_t t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept :
        _v_buffer(v_buffer), _v_offset(v_offset), _v_stride(v_stride), _v_count(v_count),
        _t_buffer(t_buffer), _t_offset(t_offset), _t_count(t_count), _hint(hint) {}

}