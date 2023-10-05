#pragma once
#include "resource.h"
#include <luisa/ast/usage.h>
#include "range.h"
namespace lc::validation {
class Stream;
using namespace luisa::compute;
struct RWInfo {
    Usage usage{Usage::NONE};
    uint64_t last_frame{0};
};
class RWResource : public Resource {
    friend struct ResMap;
    vstd::unordered_map<uint64_t , RWInfo> _info;
    bool _non_simultaneous;
    uint64_t _handle;

protected:
    virtual ~RWResource();
    static RWResource *_get(uint64_t handle);

public:
    auto handle() const { return _handle; }
    static void set_usage(Stream *stream, RWResource *res, Usage usage, Range range);
    virtual void set(Stream *stream, Usage usage, Range range) {
        set_usage(stream, this, usage, range);
    }
    auto non_simultaneous() const { return _non_simultaneous; }
    auto const &info() const { return _info; }
    RWResource(RWResource &&) = delete;
    RWResource(RWResource const &) = delete;
    RWResource(uint64_t handle, Tag tag, bool non_simultaneous);
    static void dispose(uint64_t handle);
    template<typename T>
        requires(std::is_same_v<T, RWResource> || std::is_base_of_v<RWResource, T>)
    static T *get(uint64_t handle) {
        auto ptr = static_cast<T *>(_get(handle));
        if(ptr == nullptr) [[unlikely]]{
            LUISA_ERROR("Type {} instance not found.", typeid(T).name());
        }
        return ptr;
    }
    template<typename T>
        requires(std::is_same_v<T, RWResource> || std::is_base_of_v<RWResource, T>)
    static T *try_get(uint64_t handle) {
        return static_cast<T *>(_get(handle));
    }
};
}// namespace lc::validation
