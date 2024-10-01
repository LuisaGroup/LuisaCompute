#pragma once

#include <luisa/xir/use.h>
#include <luisa/xir/metadata.h>
#include <luisa/xir/name.h>

namespace luisa::compute {
class Type;
}// namespace luisa::compute

namespace luisa::compute::xir {

class LC_XIR_API Value : public PooledObject {

private:
    const Type *_type = nullptr;
    const Name *_name = nullptr;
    UseList _use_list;
    MetadataList _metadata_list;

public:
    explicit Value(Pool *pool, const Type *type = nullptr, const Name *name = nullptr) noexcept;
    void set_type(const Type *type) noexcept { _type = type; }
    void set_name(const Name *name) noexcept { _name = name; }

public:
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto name() const noexcept { return _name; }
    [[nodiscard]] auto &use_list() noexcept { return _use_list; }
    [[nodiscard]] auto &use_list() const noexcept { return _use_list; }
    [[nodiscard]] auto &metadata_list() noexcept { return _metadata_list; }
    [[nodiscard]] auto &metadata_list() const noexcept { return _metadata_list; }
};

}// namespace luisa::compute::xir
