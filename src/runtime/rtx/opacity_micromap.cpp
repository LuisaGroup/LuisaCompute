#include <luisa/runtime/rtx/opacity_micromap.h>

#include <luisa/ast/function_builder.h>
#include <luisa/ast/op.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

#include "opacity_micromap_detail.h"

namespace luisa::compute {

namespace opacity_micromap_detail {

using Builder = detail::FunctionBuilder;

inline uint32_t parse_uint(luisa::string_view s) noexcept {
    auto value = 0u;
    for (auto c : s) {
        if (c < '0' || c > '9') { break; }
        value = value * 10u + static_cast<uint32_t>(c - '0');
    }
    return value;
}

// One 32-bit output word of the packed opacity bits.
//
// The grid is `word_count()` threads wide and each thread owns exactly one output
// word, which is what makes the packing race-free without atomics: the bits of a
// word may span two triangles (a level-1 two-state triangle is a single byte), so
// the loop recomputes `(triangle, micro triangle)` per unit instead of iterating a
// triangle.
//
// `mask_is_image` selects the mask binding and the fetch: a `Texture2D` read at an
// explicit integer coordinate (no sampler, level 0) or a flat RGBA `Buffer<float>`.
[[nodiscard]] luisa::shared_ptr<const Builder> make_bake_kernel(bool mask_is_image) noexcept {
    return Builder::define_kernel([mask_is_image] {
        auto &cur = *Builder::current();

        // ---- bindings ----
        auto mask_texture = mask_is_image ? cur.texture(Type::of<Image<float>>()) : nullptr;
        auto mask_buffer = mask_is_image ? nullptr : cur.buffer(Type::of<Buffer<float>>());
        auto uv_buffer = cur.buffer(Type::of<Buffer<float2>>());
        // The triangle buffer is read as three uint indices per triangle: the
        // framead's `Triangle` is exactly `{uint, uint, uint}`, and it has no
        // registered AST struct type, so the bake reads the raw indices.
        auto triangle_buffer = cur.buffer(Type::of<Buffer<uint>>());
        auto out_buffer = cur.buffer(Type::of<Buffer<uint>>());

        // ---- uniforms (declaration order matches the host-side invocation) ----
        auto mask_width = cur.argument(Type::of<uint>());
        auto mask_height = cur.argument(Type::of<uint>());
        auto level = cur.argument(Type::of<uint>());
        auto micro_count = cur.argument(Type::of<uint>());
        auto channel = cur.argument(Type::of<uint>());
        auto unit_count = cur.argument(Type::of<uint>());
        auto units_per_word = cur.argument(Type::of<uint>());
        auto bits_per_unit = cur.argument(Type::of<uint>());
        auto four_state = cur.argument(Type::of<uint>());
        auto conservative = cur.argument(Type::of<uint>());
        auto resolve_boundary = cur.argument(Type::of<uint>());
        auto cutoff = cur.argument(Type::of<float>());

        // ---- small builders ----
        auto u32 = [&](uint32_t v) noexcept {
            return static_cast<const Expression *>(cur.literal(Type::of<uint>(), v));
        };
        auto f32 = [&](float v) noexcept {
            return static_cast<const Expression *>(cur.literal(Type::of<float>(), v));
        };
        auto u = [&](BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept {
            return cur.binary(Type::of<uint>(), op, lhs, rhs);
        };
        auto b = [&](BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept {
            return cur.binary(Type::of<bool>(), op, lhs, rhs);
        };
        auto v2 = [&](BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept {
            return cur.binary(Type::of<float2>(), op, lhs, rhs);
        };
        auto splat2 = [&](const Expression *scalar) noexcept {
            return cur.call(Type::of<float2>(), CallOp::MAKE_FLOAT2, {scalar, scalar});
        };
        auto component = [&](const Expression *vec, uint32_t i) noexcept {
            return cur.swizzle(Type::of<float>(), vec, 1u, static_cast<uint64_t>(i));
        };

        // `if (cond) { body }`, appended to the current scope.
        auto when = [&](const Expression *cond, const std::function<void()> &body) {
            auto if_stmt = cur.if_(cond);
            cur.with(if_stmt->true_branch(), body);
        };

        // ---- dispatch index ----
        auto word = cur.swizzle(Type::of<uint>(), cur.dispatch_id(), 1u, 0ull);

        auto acc = cur.local(Type::of<uint>());
        cur.assign(acc, u32(0u));
        auto i = cur.local(Type::of<uint>());
        cur.assign(i, u32(0u));

        auto half2 = cur.call(Type::of<float2>(), CallOp::MAKE_FLOAT2,
                              {f32(0.5f), f32(0.5f)});

        // ---- per-word packing loop ----
        auto word_loop = cur.loop_();
        cur.with(word_loop->body(), [&] {
            // if (i * bits_per_unit >= 32) break;
            when(b(BinaryOp::GREATER_EQUAL,
                   u(BinaryOp::MUL, i, bits_per_unit), u32(32u)),
                 [&] { cur.break_(); });
            // if (unit >= unit_count) break;
            auto unit = u(BinaryOp::ADD, u(BinaryOp::MUL, word, units_per_word), i);
            when(b(BinaryOp::GREATER_EQUAL, unit, unit_count),
                 [&] { cur.break_(); });
            auto triangle = u(BinaryOp::DIV, unit, micro_count);
            auto k = u(BinaryOp::MOD, unit, micro_count);

            // ---- micro triangle k -> its three corner barycentrics ----
            // The Vulkan appendix's space-filling curve: digit 0 = nearest vertex 0,
            // 1 = middle triangle, 2 = nearest vertex 1, 3 = nearest vertex 2, with
            // the index digits read most significant first.
            auto ca = cur.local(Type::of<float2>());
            auto cb = cur.local(Type::of<float2>());
            auto cc = cur.local(Type::of<float2>());
            cur.assign(ca, cur.call(Type::of<float2>(), CallOp::MAKE_FLOAT2, {f32(0.f), f32(0.f)}));
            cur.assign(cb, cur.call(Type::of<float2>(), CallOp::MAKE_FLOAT2, {f32(1.f), f32(0.f)}));
            cur.assign(cc, cur.call(Type::of<float2>(), CallOp::MAKE_FLOAT2, {f32(0.f), f32(1.f)}));

            auto d = cur.local(Type::of<uint>());
            cur.assign(d, u32(0u));
            auto corner_loop = cur.loop_();
            cur.with(corner_loop->body(), [&] {
                when(b(BinaryOp::GREATER_EQUAL, d, level), [&] { cur.break_(); });
                // shift = 2 * (level - 1 - d): the first subdivision level is the
                // most significant base-4 digit.
                auto shift = u(BinaryOp::MUL, u(BinaryOp::SUB,
                                                 u(BinaryOp::SUB, level, u32(1u)), d),
                               u32(2u));
                auto digit = u(BinaryOp::BIT_AND,
                               u(BinaryOp::SHR, k, shift), u32(3u));
                // The three edge midpoints have to be materialized *before* any branch
                // overwrites a corner: the branches assign to (ca, cb, cc) and every
                // midpoint reads two of them.
                auto t01 = cur.local(Type::of<float2>());
                auto t12 = cur.local(Type::of<float2>());
                auto t20 = cur.local(Type::of<float2>());
                cur.assign(t01, v2(BinaryOp::MUL, v2(BinaryOp::ADD, ca, cb), half2));
                cur.assign(t12, v2(BinaryOp::MUL, v2(BinaryOp::ADD, cb, cc), half2));
                cur.assign(t20, v2(BinaryOp::MUL, v2(BinaryOp::ADD, cc, ca), half2));
                when(b(BinaryOp::EQUAL, digit, u32(0u)), [&] {
                    cur.assign(cb, t01);
                    cur.assign(cc, t20);
                });
                when(b(BinaryOp::EQUAL, digit, u32(1u)), [&] {
                    // middle triangle, reversed: (m20, m12, m01)
                    cur.assign(ca, t20);
                    cur.assign(cb, t12);
                    cur.assign(cc, t01);
                });
                when(b(BinaryOp::EQUAL, digit, u32(2u)), [&] {
                    cur.assign(ca, t01);
                    cur.assign(cc, t12);
                });
                when(b(BinaryOp::EQUAL, digit, u32(3u)), [&] {
                    // nearest vertex 2, reversed: (v2, m12, m20)
                    cur.assign(ca, cc);
                    cur.assign(cb, t12);
                    cur.assign(cc, t20);
                });
                cur.assign(d, u(BinaryOp::ADD, d, u32(1u)));
            });

            // ---- the triangle's three vertex UVs ----
            auto index0 = cur.call(Type::of<uint>(), CallOp::BUFFER_READ,
                                   {triangle_buffer, u(BinaryOp::MUL, triangle, u32(3u))});
            auto index1 = cur.call(Type::of<uint>(), CallOp::BUFFER_READ,
                                   {triangle_buffer, u(BinaryOp::ADD,
                                                       u(BinaryOp::MUL, triangle, u32(3u)), u32(1u))});
            auto index2 = cur.call(Type::of<uint>(), CallOp::BUFFER_READ,
                                   {triangle_buffer, u(BinaryOp::ADD,
                                                       u(BinaryOp::MUL, triangle, u32(3u)), u32(2u))});
            auto uv0 = cur.call(Type::of<float2>(), CallOp::BUFFER_READ, {uv_buffer, index0});
            auto uv1 = cur.call(Type::of<float2>(), CallOp::BUFFER_READ, {uv_buffer, index1});
            auto uv2 = cur.call(Type::of<float2>(), CallOp::BUFFER_READ, {uv_buffer, index2});

            // ---- sample the mask at one corner ----
            auto size2 = cur.call(Type::of<float2>(), CallOp::MAKE_FLOAT2,
                                  {cur.cast(Type::of<float>(), CastOp::STATIC, mask_width),
                                   cur.cast(Type::of<float>(), CastOp::STATIC, mask_height)});
            auto max_texel = cur.call(Type::of<int2>(), CallOp::MAKE_INT2,
                                      {cur.cast(Type::of<int>(), CastOp::STATIC,
                                                u(BinaryOp::SUB, mask_width, u32(1u))),
                                       cur.cast(Type::of<int>(), CastOp::STATIC,
                                                u(BinaryOp::SUB, mask_height, u32(1u)))});
            auto zero2 = cur.call(Type::of<int2>(), CallOp::MAKE_INT2,
                                  {cur.literal(Type::of<int>(), 0), cur.literal(Type::of<int>(), 0)});
            // texel = clamp(int(uv * size), 0, size - 1), mip level 0, no sampler.
            auto sample = [&](const Expression *corner, const Expression *uv0,
                              const Expression *uv1, const Expression *uv2) noexcept {
                auto cu = component(corner, 0u);
                auto cv = component(corner, 1u);
                auto w1 = splat2(cu);
                auto w2 = splat2(cv);
                auto w0 = splat2(cur.binary(Type::of<float>(), BinaryOp::SUB,
                                            f32(1.0f),
                                            cur.binary(Type::of<float>(), BinaryOp::ADD, cu, cv)));
                auto uv = v2(BinaryOp::ADD,
                             v2(BinaryOp::ADD,
                                v2(BinaryOp::MUL, uv0, w0),
                                v2(BinaryOp::MUL, uv1, w1)),
                             v2(BinaryOp::MUL, uv2, w2));
                auto texel = cur.call(
                    Type::of<int2>(), CallOp::CLAMP,
                    {cur.cast(Type::of<int2>(), CastOp::STATIC,
                              v2(BinaryOp::MUL, uv, size2)),
                     zero2, max_texel});
                auto alpha = cur.local(Type::of<float>());
                if (mask_is_image) {
                    auto coord = cur.cast(Type::of<uint2>(), CastOp::STATIC, texel);
                    auto texel_value = cur.call(Type::of<float4>(), CallOp::TEXTURE_READ,
                                                {mask_texture, coord});
                    cur.assign(alpha, component(texel_value, 3u));
                    for (auto c = 0u; c < 3u; c++) {
                        when(b(BinaryOp::EQUAL, channel, u32(c)), [&, c] {
                            cur.assign(alpha, component(texel_value, c));
                        });
                    }
                } else {
                    // flat RGBA: ((y * width) + x) * 4 + channel
                    auto flat_index = u(
                        BinaryOp::ADD,
                        u(BinaryOp::MUL,
                          u(BinaryOp::ADD,
                            u(BinaryOp::MUL, cur.cast(Type::of<uint>(), CastOp::STATIC,
                                                      cur.swizzle(Type::of<int>(), texel, 1u, 1ull)),
                              mask_width),
                            cur.cast(Type::of<uint>(), CastOp::STATIC,
                                     cur.swizzle(Type::of<int>(), texel, 1u, 0ull))),
                          u32(4u)),
                        channel);
                    cur.assign(alpha, cur.call(Type::of<float>(), CallOp::BUFFER_READ,
                                               {mask_buffer, flat_index}));
                }
                // The caller compares this against `cutoff`, so that "opaque" and
                // "transparent" stay exact complements of each other.
                return alpha;
            };

                auto alpha0 = sample(ca, uv0, uv1, uv2);
                auto alpha1 = sample(cb, uv0, uv1, uv2);
                auto alpha2 = sample(cc, uv0, uv1, uv2);
                auto centroid = v2(BinaryOp::DIV,
                                   v2(BinaryOp::ADD, v2(BinaryOp::ADD, ca, cb), cc),
                                   splat2(f32(3.0f)));
                // One texel fetch, two predicates: `opaque` and `transparent` must be the
                // exact complements of each other, so both are derived from the same alpha.
                auto opaque = [&](const Expression *alpha) noexcept {
                    return cur.binary(Type::of<bool>(), BinaryOp::GREATER_EQUAL,
                                      alpha, cutoff);
                };
                auto transparent = [&](const Expression *alpha) noexcept {
                    return cur.binary(Type::of<bool>(), BinaryOp::LESS, alpha, cutoff);
                };
                auto bool_and = [&](const Expression *lhs, const Expression *rhs) noexcept {
                    return cur.binary(Type::of<bool>(), BinaryOp::BIT_AND, lhs, rhs);
                };
                auto bool_or = [&](const Expression *lhs, const Expression *rhs) noexcept {
                    return cur.binary(Type::of<bool>(), BinaryOp::BIT_OR, lhs, rhs);
                };
                auto any_opaque = bool_or(opaque(alpha0),
                                          bool_or(opaque(alpha1), opaque(alpha2)));
                auto any_transparent = bool_or(transparent(alpha0),
                                               bool_or(transparent(alpha1), transparent(alpha2)));
                // value: 0 = transparent (the traversal discards the micro triangle),
                //        1 = opaque,
                //        2 = "unknown", four-state only: the any-hit shader resolves the
                //            boundary against the original mask texture.
                auto value = cur.local(Type::of<uint>());
                cur.assign(value, u32(0u));
                // 1. The conservative rule: a micro triangle is cut out only when *all three*
                //    of its corners are transparent, so a cut-out never grows past the mask's
                //    own boundary and a grazing angle cannot open a hole (a mixed micro
                //    triangle is opaque).
                when(any_opaque, [&] { cur.assign(value, u32(1u)); });
                // 2. The fast path: one centroid sample decides the whole micro triangle,
                //    which is cheaper but also cuts the micro triangles whose corners happen
                //    to be opaque. Emitted inside the branch so the extra fetch only runs
                //    for this option.
                when(b(BinaryOp::EQUAL, conservative, u32(0u)), [&] {
                    cur.assign(value, u32(0u));
                    when(opaque(sample(centroid, uv0, uv1, uv2)),
                         [&] { cur.assign(value, u32(1u)); });
                });
                // 3. Four-state with an any-hit resolver: a boundary micro triangle is
                //    reported as "unknown" instead of being rounded to opaque, so the
                //    shader can sample the original mask and decide.
                when(bool_and(b(BinaryOp::NOT_EQUAL, four_state, u32(0u)),
                              b(BinaryOp::NOT_EQUAL, resolve_boundary, u32(0u))),
                     [&] {
                         when(bool_and(any_opaque, any_transparent),
                              [&] { cur.assign(value, u32(2u)); });
                     });

            auto shift_in_word = u(BinaryOp::MUL, i, bits_per_unit);
            cur.assign(acc, u(BinaryOp::BIT_OR, acc,
                              u(BinaryOp::SHL, value, shift_in_word)));
            cur.assign(i, u(BinaryOp::ADD, i, u32(1u)));
        });

        cur.call(CallOp::BUFFER_WRITE, {out_buffer, word, acc});
        cur.set_block_size(uint3{128u, 1u, 1u});
        cur.set_name("opacity_micromap_bake");
    });
}

}// namespace opacity_micromap_detail

// ---------------------------------------------------------------------------
// Host-only helpers
// ---------------------------------------------------------------------------

OpacityMicromap::MicroTriangle OpacityMicromap::micro_triangle(
    uint32_t level, uint64_t index) noexcept {
    auto level_clamped = level > max_representable_subdivision_level ?
                             max_representable_subdivision_level :
                             level;
    // Corner barycentrics as (u, v) = (weight of vertex 1, weight of vertex 2).
    auto a = make_float2(0.0f, 0.0f);// vertex 0
    auto b = make_float2(1.0f, 0.0f);// vertex 1
    auto c = make_float2(0.0f, 1.0f);// vertex 2
    for (auto d = static_cast<int32_t>(level_clamped) - 1; d >= 0; d--) {
        auto digit = (index >> (2u * static_cast<uint32_t>(d))) & 3ull;
        auto m01 = (a + b) * 0.5f;
        auto m12 = (b + c) * 0.5f;
        auto m20 = (c + a) * 0.5f;
        // The children are labelled with the corner order the next level's digits
        // refer to: the two "outer" children keep the parent's winding, while the
        // middle child and the child at vertex 2 are *reversed* ("the middle triangle
        // with ordering flipped ... nearest vertex 2 with ordering flipped"). That
        // reversal is what makes the numbering continue along one space-filling curve
        // instead of restarting at each level; ignoring it permutes the micro
        // triangles of every level below the first.
        switch (digit) {
            case 0u:// nearest vertex 0: (v0, m01, m20)
                b = m01;
                c = m20;
                break;
            case 1u:// middle triangle, reversed: (m20, m12, m01)
                a = m20;
                c = m01;
                b = m12;
                break;
            case 2u:// nearest vertex 1: (m01, v1, m12)
                a = m01;
                c = m12;
                break;
            default:// nearest vertex 2, reversed: (v2, m12, m20)
                a = c;
                b = m12;
                c = m20;
                break;
        }
    }
    return MicroTriangle{a, b, c};
}

uint64_t OpacityMicromap::micro_triangle_index(uint32_t level, float2 bary) noexcept {
    auto level_clamped = level > max_representable_subdivision_level ?
                             max_representable_subdivision_level :
                             level;
    auto w = make_float3(1.0f - bary.x - bary.y, bary.x, bary.y);
    auto index = uint64_t{0u};
    for (auto d = static_cast<int32_t>(level_clamped) - 1; d >= 0; d--) {
        uint64_t digit;
        if (w.x > 0.5f) {// nearest vertex 0: (v0, m01, m20)
            digit = 0u;
            w = make_float3(2.0f * w.x - 1.0f, 2.0f * w.y, 2.0f * w.z);
        } else if (w.y > 0.5f) {// nearest vertex 1: (m01, v1, m12)
            digit = 2u;
            w = make_float3(2.0f * w.x, 2.0f * w.y - 1.0f, 2.0f * w.z);
        } else if (w.z > 0.5f) {// nearest vertex 2, reversed: (v2, m12, m20)
            digit = 3u;
            w = make_float3(2.0f * w.z - 1.0f, 2.0f * w.y, 2.0f * w.x);
        } else {// middle triangle, reversed: (m20, m12, m01)
            digit = 1u;
            w = make_float3(w.x + w.z - w.y,
                            w.y + w.z - w.x,
                            w.x + w.y - w.z);
        }
        index |= digit << (2u * static_cast<uint32_t>(d));
    }
    return index;
}

bool OpacityMicromap::is_supported(const Device &device) noexcept {
    return device.query("opacity_micromap") == "1";
}

uint32_t OpacityMicromap::max_subdivision_level(const Device &device, Format format) noexcept {
    auto key = format == Format::FourState ?
                   luisa::string_view{"opacity_micromap_max_four_state_level"} :
                   luisa::string_view{"opacity_micromap_max_two_state_level"};
    return opacity_micromap_detail::parse_uint(device.query(key));
}

// ---------------------------------------------------------------------------
// The bake shader cache
// ---------------------------------------------------------------------------

struct OpacityMicromap::Baker {
    // 4 bindings + 12 uniforms (mask size, level, micro count, channel, unit count,
    // units/bits per word, four-state, conservative, resolve-boundary, cutoff).
    using ImageShader = Shader<1, Image<float>, Buffer<float2>, Buffer<uint>, Buffer<uint>,
                               uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, float>;
    using FlatShader = Shader<1, Buffer<float>, Buffer<float2>, Buffer<uint>, Buffer<uint>,
                              uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, float>;
    luisa::shared_ptr<const detail::FunctionBuilder> image_kernel;
    luisa::shared_ptr<const detail::FunctionBuilder> flat_kernel;
    luisa::unique_ptr<ImageShader> image_shader;
    luisa::unique_ptr<FlatShader> flat_shader;
};

// ---------------------------------------------------------------------------
// Resource
// ---------------------------------------------------------------------------

OpacityMicromap::OpacityMicromap(
    DeviceInterface *device, const ResourceCreationInfo &info,
    const Option &option, uint32_t triangle_count,
    Buffer<uint> data, Buffer<uint> index) noexcept
    : Resource{device, Resource::Tag::OPACITY_MICROMAP, info},
      _data{std::move(data)}, _index{std::move(index)},
      _option{option}, _triangle_count{triangle_count} {}

OpacityMicromap::OpacityMicromap() noexcept = default;
OpacityMicromap::OpacityMicromap(OpacityMicromap &&) noexcept = default;

OpacityMicromap::~OpacityMicromap() noexcept = default;

OpacityMicromap::Format OpacityMicromap::format() const noexcept {
    return _option.format;
}

uint32_t OpacityMicromap::subdivision_level() const noexcept {
    return _option.subdivision_level;
}

uint32_t OpacityMicromap::triangle_count() const noexcept {
    return _triangle_count;
}

OpacityMicromap::IndexFormat OpacityMicromap::index_format() const noexcept {
    return _option.index_format;
}

const OpacityMicromap::Option &OpacityMicromap::option() const noexcept {
    return _option;
}

bool OpacityMicromap::ready() const noexcept {
    return static_cast<bool>(*this) && _ready;
}

BufferView<uint> OpacityMicromap::data() const noexcept {
    _check_is_valid();
    return _data.view();
}

BufferView<uint> OpacityMicromap::index() const noexcept {
    _check_is_valid();
    return _index.view();
}

size_t OpacityMicromap::_check_bits_size(size_t size_bytes, const char *what) const noexcept {
    _check_is_valid();
    // The payload buffer is padded to whole 32-bit words (the baker writes words),
    // so the command-level size is the padded one; `data_byte_size()` remains the
    // number of *meaningful* bytes Vulkan requires.
    auto expected = _data.view().size_bytes();
    if (size_bytes != expected) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "OpacityMicromap::{}() got {} bytes of packed micro-triangle bits, but "
            "{} triangles at subdivision level {} ({}-state) occupy {} bytes "
            "({} bytes of padding included).",
            what, size_bytes, _triangle_count, _option.subdivision_level,
            _option.format == Format::FourState ? 4 : 2, expected,
            expected - data_byte_size());
    }
    return expected;
}

luisa::unique_ptr<Command> OpacityMicromap::upload(luisa::span<const uint> bits) noexcept {
    _check_bits_size(bits.size_bytes(), "upload");
    _ready = true;
    return _data.view().copy_from(bits);
}

luisa::unique_ptr<Command> OpacityMicromap::upload(BufferView<uint> bits) noexcept {
    _check_bits_size(bits.size_bytes(), "upload");
    _ready = true;
    return _data.view().copy_from(bits);
}

luisa::unique_ptr<Command> OpacityMicromap::readback(luisa::span<uint> bits) const noexcept {
    _check_bits_size(bits.size_bytes(), "readback");
    return _data.view().copy_to(bits);
}

luisa::unique_ptr<Command> OpacityMicromap::readback(BufferView<uint> bits) const noexcept {
    _check_bits_size(bits.size_bytes(), "readback");
    return _data.view().copy_to(bits);
}

CommandList OpacityMicromap::bake(ImageView<float> mask,
                                  BufferView<float2> uv,
                                  BufferView<Triangle> triangles,
                                  uint32_t mask_channel) const {
    if (mask.level() != 0u) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "OpacityMicromap::bake() reads the mask at mip level 0 without a sampler, "
            "so the given ImageView is at level {} instead.",
            mask.level());
    }
    return _bake(true, &mask, {}, mask.size(), uv, triangles, mask_channel);
}

CommandList OpacityMicromap::bake(BufferView<float> mask, uint2 mask_size,
                                  BufferView<float2> uv,
                                  BufferView<Triangle> triangles,
                                  uint32_t mask_channel) const {
    // The flat mask has the same RGBA layout as the image read, so both overloads
    // share one kernel and one channel selection.
    auto required = static_cast<size_t>(mask_size.x) * mask_size.y * 4u;
    if (mask_size.x == 0u || mask_size.y == 0u || mask.size() < required) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "OpacityMicromap::bake() needs a flat RGBA mask of {}x{}x4 = {} floats, "
            "but the buffer view holds {} elements.",
            mask_size.x, mask_size.y, required, mask.size());
    }
    return _bake(false, nullptr, mask, mask_size, uv, triangles, mask_channel);
}

CommandList OpacityMicromap::_bake(bool mask_is_image,
                                   const ImageView<float> *image_mask,
                                   BufferView<float> flat_mask,
                                   uint2 mask_size,
                                   BufferView<float2> uv,
                                   BufferView<Triangle> triangles,
                                   uint32_t mask_channel) const {
    _check_is_valid();
    if (mask_channel > 3u) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "OpacityMicromap::bake() selects mask channel {}, but a texel has four.",
            mask_channel);
    }
    if (uv.size() == 0u || triangles.size() == 0u) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "OpacityMicromap::bake() needs a non-empty UV buffer and triangle buffer "
            "(got {} UVs and {} triangles).",
            uv.size(), triangles.size());
    }
    auto level = _option.subdivision_level;
    auto micro = micro_triangle_count(level);
    auto unit_count = static_cast<uint64_t>(_triangle_count) * micro;
    if (unit_count > 0xFFFFFFFFull) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "OpacityMicromap::bake() would dispatch {} micro triangles, which does not "
            "fit the 32-bit kernel index space ({} triangles at subdivision level {}).",
            unit_count, _triangle_count, level);
    }
    auto bits_per_unit = opacity_micromap_detail::bits_per_micro_triangle(_option.format);
    auto units_per_word = 32u / bits_per_unit;
    auto words = opacity_micromap_detail::word_count(_option.format, level, _triangle_count);
    if (words > 0xFFFFFFFFull) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "OpacityMicromap::bake() would dispatch {} output words, which does not fit "
            "a 32-bit dispatch size.",
            words);
    }
    auto *device = this->device();
    if (_baker == nullptr) { _baker = luisa::make_unique<Baker>(); }
    ShaderOption shader_option;
    shader_option.enable_fast_math = true;
    shader_option.enable_debug_info = false;
    if (mask_is_image) {
        if (_baker->image_shader == nullptr) {
            if (_baker->image_kernel == nullptr) {
                _baker->image_kernel = opacity_micromap_detail::make_bake_kernel(true);
            }
            _baker->image_shader = luisa::make_unique<Baker::ImageShader>(
                device, _baker->image_kernel->function(), shader_option);
        }
    } else if (_baker->flat_shader == nullptr) {
        if (_baker->flat_kernel == nullptr) {
            _baker->flat_kernel = opacity_micromap_detail::make_bake_kernel(false);
        }
        _baker->flat_shader = luisa::make_unique<Baker::FlatShader>(
            device, _baker->flat_kernel->function(), shader_option);
    }
    // The triangle buffer is read as three uint indices per triangle; `Triangle` is
    // exactly `{uint, uint, uint}`, and the AST has no registered struct type for it.
    auto triangle_indices = triangles.as<uint>();
    auto out = _data.view();
    auto list = CommandList::create();
    auto four_state = _option.format == Format::FourState ? 1u : 0u;
    auto conservative = _option.conservative ? 1u : 0u;
    auto resolve_boundary = _option.resolve_boundary_in_any_hit ? 1u : 0u;
    if (mask_is_image) {
        list << (*_baker->image_shader)(
            *image_mask, uv, triangle_indices, out,
            mask_size.x, mask_size.y, level, static_cast<uint>(micro), mask_channel,
            static_cast<uint>(unit_count), units_per_word, bits_per_unit,
            four_state, conservative, resolve_boundary, _option.alpha_cutoff)
                    .dispatch(static_cast<uint>(words));
    } else {
        list << (*_baker->flat_shader)(
            flat_mask, uv, triangle_indices, out,
            mask_size.x, mask_size.y, level, static_cast<uint>(micro), mask_channel,
            static_cast<uint>(unit_count), units_per_word, bits_per_unit,
            four_state, conservative, resolve_boundary, _option.alpha_cutoff)
                    .dispatch(static_cast<uint>(words));
    }
    // The bits are final from here on: `Mesh::build()` only attaches a ready
    // micromap, so an un-baked one can never make a mesh silently disappear.
    _ready = true;
    return list;
}

// ---------------------------------------------------------------------------
// Device entry point
// ---------------------------------------------------------------------------

OpacityMicromap Device::create_opacity_micromap(
    uint32_t triangle_count, const OpacityMicromap::Option &option) noexcept {
    if (triangle_count == 0u) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "An opacity micromap needs at least one triangle (got 0).");
    }
    if (option.subdivision_level > OpacityMicromap::max_representable_subdivision_level) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Opacity micromap subdivision level {} is not representable; the largest "
            "level is {}.",
            option.subdivision_level,
            OpacityMicromap::max_representable_subdivision_level);
    }
    switch (option.index_format) {
        case OpacityMicromap::IndexFormat::None:
        case OpacityMicromap::IndexFormat::UInt8:
        case OpacityMicromap::IndexFormat::UInt16:
        case OpacityMicromap::IndexFormat::UInt32: break;
        default: [[unlikely]] LUISA_ERROR_WITH_LOCATION(
            "Invalid opacity micromap index format {}.",
            static_cast<uint32_t>(option.index_format));
    }
    // The backend validates the rest (extension negotiated, feature enabled, level
    // within the implementation limit) and raises with an actionable message, so a
    // device without micromap support never hands out a handle.
    auto bits = OpacityMicromap::data_byte_size(
        option.format, option.subdivision_level, triangle_count);
    auto data_words = static_cast<size_t>((bits + 3u) / 4u);
    auto data = this->_create<Buffer<uint>>(data_words);
    Buffer<uint> index;
    auto index_bytes = uint64_t{0u};
    if (option.index_format != OpacityMicromap::IndexFormat::None) {
        index_bytes = static_cast<uint64_t>(static_cast<uint32_t>(option.index_format)) *
                      triangle_count;
        index = this->_create<Buffer<uint>>(static_cast<size_t>((index_bytes + 3u) / 4u));
    }
    OpacityMicromapOption pod;
    pod.format = static_cast<uint32_t>(option.format);
    pod.subdivision_level = option.subdivision_level;
    pod.triangle_count = triangle_count;
    pod.index_format = static_cast<uint32_t>(option.index_format);
    pod.data_buffer = data.handle();
    pod.data_offset = 0u;
    pod.data_size = data_words * sizeof(uint);
    pod.index_buffer = index ? index.handle() : invalid_resource_handle;
    pod.index_offset = 0u;
    pod.index_size = static_cast<size_t>((index_bytes + 3u) / 4u * 4u);
    auto info = this->_impl->create_opacity_micromap(pod);
    if (!info.valid()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to create an opacity micromap with {} triangles at subdivision "
            "level {} on this device.",
            triangle_count, option.subdivision_level);
    }
    return OpacityMicromap{this->_impl.get(), info, option, triangle_count,
                           std::move(data), std::move(index)};
}

}// namespace luisa::compute

