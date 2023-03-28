#include "rw_resource.h"
namespace lc::validation {
class Shader : public RWResource {
public:
    Shader(uint64_t handle) : RWResource{handle, Tag::SHADER, false} {}
};
}// namespace lc::validation