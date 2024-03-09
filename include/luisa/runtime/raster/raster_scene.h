#pragma once
#include <luisa/backends/ext/raster_cmd.h>
#include <luisa/backends/ext/raster_ext_interface.h>
#include <luisa/runtime/buffer.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/runtime/raster/raster_shader.h>
#include <luisa/runtime/raster/depth_buffer.h>
namespace lc::validation {
class Stream;
}// namespace lc::validation
namespace luisa::compute {
namespace detail {
template<typename T>
    requires(is_buffer_view_v<T>)
VertexBufferView make_vbv(T const &buffer_view) noexcept {
    return VertexBufferView{
        .handle = buffer_view.handle(),
        .offset = buffer_view.offset_bytes(),
        .size = static_cast<uint>(buffer_view.size()),
        .stride = static_cast<uint>(buffer_view.stride()),
    };
}

template<typename T>
    requires(is_buffer_v<T>)
VertexBufferView make_vbv(T const &buffer) noexcept {
    return VertexBufferView{
        .handle = buffer.handle(),
        .offset = 0,
        .size = buffer.size_bytes(),
        .stride = buffer.stride(),
    };
}
struct RasterMesh {
    friend class lc::validation::Stream;
    luisa::fixed_vector<VertexBufferView, 2> _vertex_buffers{};
    luisa::variant<BufferView<uint>, uint> _index_buffer;
    RasterState _state;

    template<typename T>
    RasterMesh(
        luisa::span<BufferView<T>> vertex_buffers,
        BufferView<uint> index_buffer) noexcept
        : _index_buffer(index_buffer) {
        _vertex_buffers.reserve(vertex_buffers.size());
        for (auto &i : vertex_buffers) {
            _vertex_buffers.emplace_back(detail::make_vbv(i));
        }
    }
    RasterMesh() noexcept = default;
    RasterMesh(RasterMesh &&) noexcept = default;
    RasterMesh(RasterMesh const &) noexcept = delete;
    RasterMesh &operator=(RasterMesh &&) noexcept = default;
    RasterMesh &operator=(RasterMesh const &) noexcept = delete;
    template<typename T>
    RasterMesh(
        luisa::span<BufferView<T>> vertex_buffers,
        uint vertex_count) noexcept
        : _index_buffer(vertex_count) {
        _vertex_buffers.reserve(vertex_buffers.size());
        for (auto &i : vertex_buffers) {
            _vertex_buffers.emplace_back(detail::make_vbv(i));
        }
    }
};
}// namespace detail
class LC_RUNTIME_API RasterScene : public Resource {
    friend class RasterExt;
public:
    using Modification = BuildRasterSceneCommand::Modification;
private:
    luisa::fixed_vector<PixelFormat, 8> _render_formats;
    DepthFormat _depth_format;
    luisa::unordered_map<size_t, Modification> _modifications;
    mutable luisa::spin_mutex _mtx;
    size_t _instance_count{};
    RasterScene(
        DeviceInterface *device,
        luisa::span<const PixelFormat> render_formats,
        DepthFormat depth_format) noexcept;
    template<typename... Args>
    static auto encode_binds(RasterShader<Args...> const &shader, Args &&...args) noexcept {
        using invoke_type = detail::RasterInvokeBase;
        auto arg_count = (0u + ... + detail::shader_argument_encode_count<Args>::value);
        invoke_type invoke{shader.handle(), arg_count, RasterShader<Args...>::uniform_size()};
        static_cast<void>((invoke << ... << args));
        return std::move(invoke.encoder);
    }
    template<typename T>
    static IndexBufferView make_idx_view(BufferView<T> b) {
        return IndexBufferView{
            .handle = b.handle(),
            .size = static_cast<uint>(b.size()),
            .format = std::is_same_v<T, uint16_t> ? IndexFormat::UInt16 : IndexFormat::UInt32};
    }
public:
    using Resource::operator bool;
    RasterScene(RasterScene &&) noexcept;
    RasterScene(RasterScene const &) noexcept = delete;
    RasterScene &operator=(RasterScene &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }

    template<typename Vert, typename Index, typename... Args>
        requires(std::is_same_v<Index, uint16_t> || std::is_same_v<Index, uint32_t>)
    void emplace_back(
        luisa::span<BufferView<Vert> const> vertex_buffers,
        BufferView<Index> index_buffer,
        uint instance_count,
        RasterState const &state,
        RasterShader<Args...> const &shader,
        Args &&...args) noexcept {
        Modification mod{
            .index_buffer = make_idx_view(index_buffer),
            .encoder = encode_binds(shader, std::forward<Args>(args)...),
            .instance = instance_count,
            .state = state,
            .flag = Modification::flag_all};
        mod.vertex_buffers.reserve(vertex_buffers.size());
        for (auto &i : vertex_buffers) {
            mod.vertex_buffers.emplace_back(detail::make_vbv(i));
        }
        _modifications.force_emplace(_instance_count, std::move(mod));
        _instance_count++;
    }
    template<typename Vert, typename... Args>
    void emplace_back(
        luisa::span<BufferView<Vert> const> vertex_buffers,
        uint draw_index_count,
        uint instance_count,
        RasterState const &state,
        RasterShader<Args...> const &shader,
        Args &&...args) noexcept {
        Modification mod{
            .index_buffer = draw_index_count,
            .encoder = encode_binds(shader, std::forward<Args>(args)...),
            .instance = instance_count,
            .state = state,
            .flag = Modification::flag_all};
        mod.vertex_buffers.reserve(vertex_buffers.size());
        for (auto &i : vertex_buffers) {
            mod.vertex_buffers.emplace_back(detail::make_vbv(i));
        }
        _modifications.force_emplace(_instance_count, std::move(mod));
        _instance_count++;
    }
    template<typename Vert, typename Index, typename... Args>
        requires(std::is_same_v<Index, uint16_t> || std::is_same_v<Index, uint32_t>)
    void set(
        size_t idx,
        luisa::span<BufferView<Vert> const> vertex_buffers,
        BufferView<Index> index_buffer,
        uint instance_count,
        RasterState const &state,
        RasterShader<Args...> const &shader,
        Args &&...args) noexcept {
        Modification mod{
            .index_buffer = make_idx_view(index_buffer),
            .encoder = encode_binds(shader, std::forward<Args>(args)...),
            .instance = instance_count,
            .state = state,
            .flag = Modification::flag_all};
        mod.vertex_buffers.reserve(vertex_buffers.size());
        for (auto &i : vertex_buffers) {
            mod.vertex_buffers.emplace_back(detail::make_vbv(i));
        }
        _modifications.force_emplace(idx, std::move(mod));
    }
    template<typename Vert, typename... Args>
    void set(
        size_t idx,
        luisa::span<BufferView<Vert> const> vertex_buffers,
        uint draw_index_count,
        uint instance_count,
        RasterState const &state,
        RasterShader<Args...> const &shader,
        Args &&...args) noexcept {
        Modification mod{
            .index_buffer = draw_index_count,
            .encoder = encode_binds(shader, std::forward<Args>(args)...),
            .instance = instance_count,
            .state = state,
            .flag = Modification::flag_all};
        mod.vertex_buffers.reserve(vertex_buffers.size());
        for (auto &i : vertex_buffers) {
            mod.vertex_buffers.emplace_back(detail::make_vbv(i));
        }
        std::lock_guard lck{_mtx};
        _modifications.force_emplace(idx, std::move(mod));
    }
    template<typename Vert>
    void set_vertex(
        size_t idx,
        luisa::span<BufferView<Vert> const> vertex_buffers) noexcept {
        std::lock_guard lck{_mtx};
        auto &mod = _modifications.try_emplace(idx).first->second;
        mod.vertex_buffers.clear();
        mod.vertex_buffers.reserve(vertex_buffers.size());
        for (auto &i : vertex_buffers) {
            mod.vertex_buffers.emplace_back(detail::make_vbv(i));
        }
        mod.flag |= Modification::flag_vertex_buffer;
    }
    template<typename Index>
        requires(std::is_same_v<Index, uint16_t> || std::is_same_v<Index, uint32_t>)
    void set_index(
        size_t idx,
        BufferView<Index> index_buffer) noexcept {
        std::lock_guard lck{_mtx};
        auto &mod = _modifications.try_emplace(idx).first->second;
        mod.index_buffer = make_idx_view(index_buffer);
        mod.flag |= Modification::flag_index_buffer;
    }
    void set_index(
        size_t idx,
        uint draw_index_count) noexcept {
        std::lock_guard lck{_mtx};
        auto &mod = _modifications.try_emplace(idx).first->second;
        mod.index_buffer = draw_index_count;
        mod.flag |= Modification::flag_index_buffer;
    }
    void set_instance_count(
        size_t idx,
        uint instance_count) noexcept {
        std::lock_guard lck{_mtx};
        auto &mod = _modifications.try_emplace(idx).first->second;
        mod.instance = instance_count;
        mod.flag |= Modification::flag_instance;
    }
    template<typename... Args>
    void set_shader(
        size_t idx,
        RasterState const &state,
        RasterShader<Args...> const &shader,
        Args &&...args) noexcept {
        std::lock_guard lck{_mtx};
        auto &mod = _modifications.try_emplace(idx).first->second;
        mod.encoder = encode_binds(shader, std::forward<Args>(args)...);
        mod.flag |= Modification::flag_shader;
    }
    // [[nodiscard]] luisa::unique_ptr<Command> draw(
    //     float4x4 const &view,
    //     float4x4 const &projection,
    //     luisa::span<const ImageView<float>> rtv_texs,
    //     DepthBuffer const *depth,
    //     Viewport viewport) const noexcept;
    [[nodiscard]] luisa::unique_ptr<Command> build() noexcept;
    [[nodiscard]] luisa::unique_ptr<Command> build(
        luisa::span<const PixelFormat> render_formats,
        DepthFormat depth_format) noexcept;
    RasterScene &operator=(RasterScene const &) noexcept = delete;
    ~RasterScene() noexcept;
};
}// namespace luisa::compute