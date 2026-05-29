#pragma once

#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/managed_ilist.h>
#include <luisa/xir/traits.h>

namespace luisa::compute::xir {

enum struct DerivedMetadataTag {
    NAME,
    LOCATION,
    COMMENT,
    CURVE_BASIS,
};

class LUISA_XIR_API Metadata : public ManagedIntrusiveForwardNode<Metadata> {

public:
    [[nodiscard]] virtual DerivedMetadataTag derived_metadata_tag() const noexcept = 0;
    [[nodiscard]] virtual ManagedPtr<Metadata> clone() const noexcept = 0;
    LUISA_XIR_DEFINED_ISA_METHOD(Metadata, metadata)
};

template<typename Derived, DerivedMetadataTag Tag, typename Base = Metadata>
    requires std::derived_from<Base, Metadata>
class LUISA_XIR_API DerivedMetadata : public Base {
public:
    using derived_metadata_type = Derived;
    using Super = DerivedMetadata;
    using Base::Base;

    [[nodiscard]] static constexpr auto
    static_derived_metadata_tag() noexcept { return Tag; }

    [[nodiscard]] DerivedMetadataTag
    derived_metadata_tag() const noexcept final { return static_derived_metadata_tag(); }

    [[nodiscard]] ManagedPtr<Derived> clone_into() const noexcept {
        return Metadata::clone().template into<Derived>();
    }
};

using MetadataList = ManagedIntrusiveForwardList<Metadata>;

namespace detail {
[[nodiscard]] LUISA_XIR_API Metadata *luisa_xir_metadata_list_mixin_find_metadata(MetadataList &list, DerivedMetadataTag tag) noexcept;
[[nodiscard]] LUISA_XIR_API Metadata *luisa_xir_metadata_list_mixin_create_metadata(MetadataList &list, DerivedMetadataTag tag) noexcept;
[[nodiscard]] LUISA_XIR_API Metadata *luisa_xir_metadata_list_mixin_find_or_create_metadata(MetadataList &list, DerivedMetadataTag tag) noexcept;
[[nodiscard]] LUISA_XIR_API luisa::optional<luisa::string_view> luisa_xir_metadata_list_mixin_get_name(const MetadataList &list) noexcept;
LUISA_XIR_API void luisa_xir_metadata_list_mixin_set_name(MetadataList &list, std::string_view name) noexcept;
LUISA_XIR_API void luisa_xir_metadata_list_mixin_set_location(MetadataList &list, const std::filesystem::path &file, int line) noexcept;
LUISA_XIR_API void luisa_xir_metadata_list_mixin_add_comment(MetadataList &list, std::string_view comment) noexcept;
}// namespace detail

class MetadataListMixin {

private:
    MetadataList _metadata_list;

protected:
    MetadataListMixin() noexcept = default;
    ~MetadataListMixin() noexcept = default;

public:
    [[nodiscard]] auto &metadata_list() noexcept { return _metadata_list; }
    [[nodiscard]] auto &metadata_list() const noexcept { return _metadata_list; }

    [[nodiscard]] Metadata *find_metadata(DerivedMetadataTag tag) noexcept {
        return detail::luisa_xir_metadata_list_mixin_find_metadata(_metadata_list, tag);
    }
    [[nodiscard]] const Metadata *find_metadata(DerivedMetadataTag tag) const noexcept {
        return const_cast<MetadataListMixin *>(this)->find_metadata(tag);
    }
    [[nodiscard]] Metadata *create_metadata(DerivedMetadataTag tag) noexcept {
        return detail::luisa_xir_metadata_list_mixin_create_metadata(_metadata_list, tag);
    }
    [[nodiscard]] Metadata *find_or_create_metadata(DerivedMetadataTag tag) noexcept {
        return detail::luisa_xir_metadata_list_mixin_find_or_create_metadata(_metadata_list, tag);
    }

    template<typename T>
    [[nodiscard]] auto find_metadata() noexcept {
        return static_cast<T *>(find_metadata(T::static_derived_metadata_tag()));
    }
    template<typename T>
    [[nodiscard]] auto find_metadata() const noexcept {
        return static_cast<const T *>(find_metadata(T::static_derived_metadata_tag()));
    }
    template<typename T>
    [[nodiscard]] auto create_metadata() noexcept {
        return static_cast<T *>(create_metadata(T::static_derived_metadata_tag()));
    }
    template<typename T>
    [[nodiscard]] auto find_or_create_metadata() noexcept {
        return static_cast<T *>(find_or_create_metadata(T::static_derived_metadata_tag()));
    }

    void set_name(std::string_view name) noexcept {
        detail::luisa_xir_metadata_list_mixin_set_name(_metadata_list, name);
    }
    void set_location(const std::filesystem::path &file, int line = -1) noexcept {
        detail::luisa_xir_metadata_list_mixin_set_location(_metadata_list, file, line);
    }
    void add_comment(std::string_view comment) noexcept {
        detail::luisa_xir_metadata_list_mixin_add_comment(_metadata_list, comment);
    }

    [[nodiscard]] luisa::optional<luisa::string_view> name() const noexcept {
        return detail::luisa_xir_metadata_list_mixin_get_name(_metadata_list);
    }
};

}// namespace luisa::compute::xir
