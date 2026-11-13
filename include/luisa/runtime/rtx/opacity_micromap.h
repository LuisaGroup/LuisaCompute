#pragma once

// X11 exposes None as a macro. Hide it before including dependencies too:
// platform headers such as Windows SDK headers can also use None as a name.
// Restore the caller's definition after this header.
#pragma push_macro("None")
#undef None

#include <luisa/core/logging.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rtx/triangle.h>

namespace luisa::compute {

class Device;

// Hardware opacity micromap (VK_KHR_opacity_micromap / the VK_EXT_opacity_micromap
// alias, or DXR 1.2 opacity micromaps on the DX backend).
//
// An opacity micromap ("cut-out mask") tells the ray traversal which micro
// triangles of a base triangle are fully transparent and which are fully opaque,
// so an alpha-tested mesh can be traced without running an any-hit shader for
// every candidate hit. It is *not* a texture:
//
//   * it has no image, no layout, no sampler and no mip chain; its whole payload is
//     the packed micro-triangle opacity bits,
//   * the bits are undefined until a build (`bake()` / `upload()`) produced them,
//     which is why the resource carries a `ready()` state,
//   * it cannot be a shader argument: Vulkan has no descriptor type for
//     `VkMicromapEXT`, and SPIR-V exposes a micromap only as the
//     `OpacityMicromapIdKHR` execution mode plus the `ForceOpacityMicromap2State*`
//     ray flag. What *is* bindable is `data()`, a plain `BufferView<uint>`, which is
//     the documented inspection surface (see also `Mesh::cutout_mask()`).
//
// The resource is a move-only `Resource`: a `Mesh` binds its *handle*
// (`Mesh::cutout_mask()`), so the micromap must outlive every traced acceleration
// structure built from that mesh unless the driver reports the build sizes as
// discardable.
class LUISA_RUNTIME_API OpacityMicromap : public Resource {

public:
    // VkOpacityMicromapFormatEXT. Two-state stores one bit per micro triangle
    // (0 = transparent, 1 = opaque); four-state stores two bits
    // (0 = fully transparent, 1 = fully opaque, 2 = unknown transparent,
    // 3 = unknown opaque), where the two "unknown" states ask the shader to
    // resolve the opacity itself (e.g. by sampling the mask texture in an any-hit
    // shader).
    enum class Format : uint32_t {
        TwoState = 0u,
        FourState = 1u,
    };

    // Byte size of one per-triangle micromap index, i.e. the VkIndexType subset
    // `VkAccelerationStructureTrianglesOpacityMicromapEXT::indexType` allows
    // (UINT8 additionally needs VK_KHR_index_type_uint8). `NoIndex` means the mesh
    // triangles map one-to-one onto the micromap triangles, which is the common
    // case and needs no index buffer at all.

    // The zero enumerator is not called `None`: X11 defines `None` as a macro and
    // depth_format.h restores that definition after its own enum (see the comment
    // there), so an enumeration constant spelled `None` only compiles on a
    // platform that never includes the X11 headers.
    enum class IndexFormat : uint32_t {
        NoIndex = 0u,
        UInt8 = 1u,
        UInt16 = 2u,
        UInt32 = 4u,
    };

    struct Option {
        // Format of every triangle of this micromap (uniform, like the byte size
        // Vulkan derives the allocation from).
        Format format{Format::TwoState};
        // 0..15; each level quadruples the micro-triangle count. Level 5 (1024
        // micro triangles, 128 bytes per triangle for the two-state format) is
        // enough to keep a 10-stripe cut-out.
        uint32_t subdivision_level{2u};
        // Four-state only: report a micro triangle whose three corners disagree as
        // "unknown" (2) instead of resolving it from the mask's own samples, so an
        // any-hit shader can sample the original texture and decide. Ignored by the
        // two-state format.
        bool resolve_boundary_in_any_hit{false};
        // Baked cut-out threshold: a mask sample below this value is cut out.
        float alpha_cutoff{0.5f};
        // How the three corner samples of a micro triangle become one bit:
        //   true: the micro triangle is cut out only when all three of its corners
        //         are transparent, so the cut-out can never grow past the mask (a
        //         mixed micro triangle stays opaque - the safe, hole-free rule);
        //   false: one extra sample at the centroid decides, which costs less than
        //         three samples and reproduces a per-hit-point alpha test, but may
        //         cut slightly more than the mask's own transparent area.
        // Either way `bake()` samples the mask at three (or four) points per micro
        // triangle, never per pixel.
        bool conservative{true};
        // Optional shared/per-triangle micromap index. `None` means the mesh's
        // triangles index the micromap triangles one-to-one.
        IndexFormat index_format{IndexFormat::NoIndex};
    };

    // The three corners of one micro triangle, as barycentric coordinates in the
    // *base* triangle: `u` is the weight of vertex 1 and `v` the weight of vertex 2
    // (vertex 0 has weight `1 - u - v`), i.e. exactly `TriangleHit::bary`.
    struct MicroTriangle {
        float2 c0;
        float2 c1;
        float2 c2;
    };

public:
    // 15 is the largest subdivision level whose micro-triangle count (4^15 =
    // 2^30) still fits a 32-bit byte count for the four-state format.
    static constexpr uint32_t max_representable_subdivision_level = 15u;

    // 4^level, clamped to the largest representable level.
    [[nodiscard]] static constexpr uint64_t micro_triangle_count(uint32_t level) noexcept {
        return uint64_t{1u} << (2u * (level > max_representable_subdivision_level ?
                                          max_representable_subdivision_level :
                                          level));
    }

    // Bytes one triangle consumes: 1 bit per micro triangle for the two-state
    // format, 2 bits for the four-state one, rounded up to whole bytes
    // (VUID-VkMicromapTriangleEXT-...: "calculated using its format and
    // subdivisionLevel rounded up to the nearest byte").
    [[nodiscard]] static constexpr uint64_t triangle_data_byte_size(
        Format format, uint32_t level) noexcept {
        auto bits = micro_triangle_count(level) *
                    (format == Format::FourState ? 2u : 1u);
        return (bits + 7u) / 8u;
    }

    // Bytes the packed bits of `triangle_count` triangles occupy.
    [[nodiscard]] static constexpr uint64_t data_byte_size(
        Format format, uint32_t level, uint64_t triangle_count) noexcept {
        return triangle_data_byte_size(format, level) * triangle_count;
    }

    // The three corners of micro triangle `index` at `level`, in the Vulkan
    // appendix's space-filling-curve order: the curve descends into the
    // sub-triangle nearest vertex 0, then the middle triangle, then the one nearest
    // vertex 1, then the one nearest vertex 2, i.e. the base-4 index digits are
    // read most-significant first. Only the *set* of corners is significant for
    // baking (every cut-out predicate below is order independent), which is why the
    // per-child vertex rotation the spec also fixes is not reproduced here.
    [[nodiscard]] static MicroTriangle micro_triangle(uint32_t level, uint64_t index) noexcept;

    // Inverse of `micro_triangle`: the index of the micro triangle containing the
    // barycentric point `bary` (out-of-range coordinates are clamped by the caller,
    // exact edge ties resolve to the lower-indexed sub-triangle). Mostly useful for
    // host-side inspection and tests.
    [[nodiscard]] static uint64_t micro_triangle_index(uint32_t level, float2 bary) noexcept;

    // Whether this device can build and attach an opacity micromap, i.e. whether
    // `Device::create_opacity_micromap` would produce a usable resource.
    [[nodiscard]] static bool is_supported(const Device &device) noexcept;

    // The implementation's largest legal subdivision level for `format`
    // (VkPhysicalDeviceOpacityMicromapPropertiesEXT::maxOpacity{2,4}StateSubdivision
    // Level), or 0 when the device has no micromap support.
    [[nodiscard]] static uint32_t max_subdivision_level(const Device &device, Format format) noexcept;

public:
    // Both are defined in the .cpp so the pimpl (`Baker`) may stay incomplete here.
    OpacityMicromap() noexcept;
    ~OpacityMicromap() noexcept override;
    OpacityMicromap(OpacityMicromap &&) noexcept;
    OpacityMicromap &operator=(OpacityMicromap &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    OpacityMicromap(OpacityMicromap const &) noexcept = delete;
    OpacityMicromap &operator=(OpacityMicromap const &) noexcept = delete;
    using Resource::operator bool;
    using Resource::release;

    [[nodiscard]] Format format() const noexcept;
    [[nodiscard]] uint32_t subdivision_level() const noexcept;
    [[nodiscard]] uint32_t triangle_count() const noexcept;
    [[nodiscard]] IndexFormat index_format() const noexcept;
    [[nodiscard]] const Option &option() const noexcept;

    // False until the packed bits are final, i.e. until a `bake()` was recorded or
    // an `upload()` was issued. `Mesh::build()` refuses to attach a micromap that is
    // not ready: in the two-state format a zero bit means *transparent*, so a
    // never-written micromap would make the whole mesh silently disappear.
    [[nodiscard]] bool ready() const noexcept;

    // The packed micro-triangle opacity bits, LSB-first, one block of
    // `4^level` (two-state) or `4^level * 2` (four-state) bits per triangle, in the
    // space-filling-curve order of `micro_triangle()`. This view is the shader-side
    // and inspection surface: writable by a user kernel (a custom baker) and
    // readable for debugging.
    [[nodiscard]] BufferView<uint> data() const noexcept;
    // The per-triangle micromap index, only when `option().index_format != None`.
    [[nodiscard]] BufferView<uint> index() const noexcept;

    // Packed bits straight from the host (`BufferUploadCommand`), or from another
    // buffer (`BufferCopyCommand`). The byte count must match
    // `data_byte_size(format(), subdivision_level(), triangle_count())`; both mark
    // the micromap ready.
    [[nodiscard]] luisa::unique_ptr<Command> upload(luisa::span<const uint> bits) noexcept;
    [[nodiscard]] luisa::unique_ptr<Command> upload(BufferView<uint> bits) noexcept;
    // Read the packed bits back (`BufferDownloadCommand` / `BufferCopyCommand`), for
    // tests and debugging.
    [[nodiscard]] luisa::unique_ptr<Command> readback(luisa::span<uint> bits) const noexcept;
    [[nodiscard]] luisa::unique_ptr<Command> readback(BufferView<uint> bits) const noexcept;

    // Built-in baker: sample `mask` at every micro triangle of every triangle of
    // `triangles` and pack the resulting opacity bits into this micromap's
    // `data()`.
    //
    // The mask is read with an *integer* texel fetch at mip level 0 and without a
    // sampler object: `texel = clamp(int(uv * size), 0, size - 1)`, so neither the
    // texture's filter/address state nor its mip chain can change a bake.
    // `uv` holds one UV per *vertex* of the mesh, so a per-triangle
    // parameterization needs a de-indexed mesh; `triangles` is the mesh's index
    // buffer. `mask_channel` selects the mask component (3 = alpha).
    [[nodiscard]] CommandList bake(ImageView<float> mask,
                                   BufferView<float2> uv,
                                   BufferView<Triangle> triangles,
                                   uint32_t mask_channel = 3u) const;
    // Same baker for a flat `width * height` mask buffer (row-major, x fastest).
    [[nodiscard]] CommandList bake(BufferView<float> mask, uint2 mask_size,
                                   BufferView<float2> uv,
                                   BufferView<Triangle> triangles,
                                   uint32_t mask_channel = 3u) const;

private:
    friend class Device;

    struct Baker;

    // Device-side construction: `Device::create_opacity_micromap` creates the
    // payload buffers (they are ordinary `Buffer<uint>`s, so the payload commands
    // stay ordinary buffer commands) and the backend object, then hands both over.
    OpacityMicromap(DeviceInterface *device, const ResourceCreationInfo &info,
                    const Option &option, uint32_t triangle_count,
                    Buffer<uint> data, Buffer<uint> index) noexcept;

    // Texture-like plumbing: the payload lives in these frontend buffers, so the
    // upload/readback commands are the ordinary buffer commands (no new
    // Command::Tag, no backend work).
    Buffer<uint> _data;
    Buffer<uint> _index;
    Option _option{};
    uint32_t _triangle_count{0u};
    // `bake()` is const (it only records a command) but does finalize the bits, so
    // both the ready state and the lazily compiled bake shader are mutable.
    mutable bool _ready{false};
    // Lazily compiled bake shader; the pimpl keeps the AST/shader headers out of
    // this public header and lets the shader die with the resource.
    mutable luisa::unique_ptr<Baker> _baker;

private:
    // `false` for a default-constructed (invalid) resource; every accessor below
    // checks it through `_check_is_valid()`.
    [[nodiscard]] uint64_t data_byte_size() const noexcept {
        return data_byte_size(_option.format, _option.subdivision_level, _triangle_count);
    }
    [[nodiscard]] uint64_t index_byte_size() const noexcept {
        return static_cast<uint64_t>(_option.index_format == IndexFormat::NoIndex ?
                                         0u :
                                         static_cast<uint32_t>(_option.index_format)) *
               _triangle_count;
    }
    // Validates a host/device bit span against this micromap's shape, then returns
    // the byte count. Raises (LUISA_ERROR) before any command is created.
    [[nodiscard]] size_t _check_bits_size(size_t size_bytes, const char *what) const noexcept;
    // Shared body of both bake() overloads: records one dispatch of the lazily
    // compiled bake kernel into a fresh command list.
    [[nodiscard]] CommandList _bake(bool mask_is_image,
                                    const ImageView<float> *image_mask,
                                    BufferView<float> flat_mask,
                                    uint2 mask_size,
                                    BufferView<float2> uv,
                                    BufferView<Triangle> triangles,
                                    uint32_t mask_channel) const;
};

}// namespace luisa::compute

#pragma pop_macro("None")
