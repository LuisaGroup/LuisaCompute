#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string_view>

#include <stb/stb_image_write.h>

#include <luisa/luisa-compute.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;

namespace {

constexpr auto request_schema =
    "luisa.coro.example.virtual-texture.request";
constexpr uint page_grid_size = 8u;
constexpr uint virtual_page_count = page_grid_size * page_grid_size;
constexpr uint physical_page_count = 8u;

struct Options {
    uint dimension{256u};
    uint page_size{32u};
    bool write_image{true};
};

[[nodiscard]] Options parse_options(int argc, char *argv[]) noexcept {
    Options options;
    for (auto i = 2; i < argc; ++i) {
        if (std::string_view{argv[i]} == "--test") {
            options.dimension = 64u;
            options.page_size = 8u;
            options.write_image = false;
        }
    }
    return options;
}

[[nodiscard]] luisa::vector<float4> make_virtual_texture(
    uint dimension) noexcept {
    luisa::vector<float4> pixels(
        static_cast<size_t>(dimension) * dimension);
    for (auto y = 0u; y < dimension; ++y) {
        for (auto x = 0u; x < dimension; ++x) {
            auto u = (static_cast<float>(x) + .5f) /
                     static_cast<float>(dimension);
            auto v = (static_cast<float>(y) + .5f) /
                     static_cast<float>(dimension);
            auto checker = ((x / 4u) ^ (y / 4u)) & 1u;
            pixels[x + y * dimension] = make_float4(
                .15f + .75f * u,
                .12f + .78f * v,
                checker == 0u ? .2f : .85f, 1.0f);
        }
    }
    return pixels;
}

class OnDemandTextureCache final {

private:
    class Handler final
        : public WavefrontCoroSchedulerExtensionHandler {

    private:
        OnDemandTextureCache *_owner;
        Shader1D<ByteBuffer, Buffer<uint>, uint, uint, Buffer<uint>>
            _shader;

    public:
        Handler(
            OnDemandTextureCache *owner,
            Shader1D<ByteBuffer, Buffer<uint>, uint, uint, Buffer<uint>>
                shader) noexcept
            : _owner{owner}, _shader{std::move(shader)} {}

        [[nodiscard]] luisa::string_view name() const noexcept override {
            return "on-demand-texture-cache";
        }

        void dispatch(
            const WavefrontCoroExtensionDispatchContext &context) noexcept override;
    };

    Options _options;
    uint _page_texel_count{0u};
    Buffer<uint> _requests;
    BufferView<uint> _page_table;
    BufferView<float4> _physical_cache;
    luisa::vector<float4> _virtual_texture;
    luisa::vector<uint> _host_page_table;
    luisa::vector<int> _slot_pages;
    luisa::vector<float4> _host_cache;
    luisa::vector<uint> _host_requests;
    size_t _page_load_count{0u};
    size_t _round_count{0u};
    bool _initialized{false};

private:
    void _dispatch(
        const WavefrontCoroExtensionDispatchContext &context,
        Shader1D<ByteBuffer, Buffer<uint>, uint, uint, Buffer<uint>>
            &kernel) noexcept {
        auto request_span = luisa::span{
            _host_requests.data(), context.frame_count};
        context.stream
            << kernel(
                   context.frame_buffer, context.frame_indices,
                   context.frame_capacity, context.frame_count,
                   _requests)
                   .dispatch(context.frame_count)
            << _requests.view()
                   .subview(0u, context.frame_count)
                   .copy_to(request_span)
            << synchronize();

        luisa::vector<bool> requested(virtual_page_count, false);
        for (auto encoded : request_span) {
            LUISA_ASSERT(
                encoded != 0u && encoded <= virtual_page_count,
                "Invalid virtual-texture page request {}.", encoded);
            requested[encoded - 1u] = true;
        }
        LUISA_ASSERT(
            std::any_of(
                requested.begin(), requested.end(),
                [](auto value) noexcept { return value; }),
            "Selected texture Extension queue contains no page request.");

        for (auto page = 0u; page < virtual_page_count; ++page) {
            if (!requested[page] || _host_page_table[page] != 0u) {
                continue;
            }
            auto slot = physical_page_count;
            for (auto i = 0u; i < physical_page_count; ++i) {
                if (_slot_pages[i] < 0) {
                    slot = i;
                    break;
                }
            }
            if (slot == physical_page_count) {
                for (auto i = 0u; i < physical_page_count; ++i) {
                    auto resident =
                        static_cast<uint>(_slot_pages[i]);
                    if (!requested[resident]) {
                        slot = i;
                        break;
                    }
                }
            }
            // All physical slots contain pages needed by this resume batch.
            // Remaining misses re-enter the same suspend point next round.
            if (slot == physical_page_count) { break; }
            if (_slot_pages[slot] >= 0) {
                _host_page_table[
                    static_cast<uint>(_slot_pages[slot])] = 0u;
            }
            _slot_pages[slot] = static_cast<int>(page);
            _host_page_table[page] = slot + 1u;
            auto page_x = page % page_grid_size;
            auto page_y = page / page_grid_size;
            for (auto y = 0u; y < _options.page_size; ++y) {
                for (auto x = 0u; x < _options.page_size; ++x) {
                    auto source_x =
                        page_x * _options.page_size + x;
                    auto source_y =
                        page_y * _options.page_size + y;
                    _host_cache[
                        slot * _page_texel_count +
                        x + y * _options.page_size] =
                        _virtual_texture[
                            source_x + source_y * _options.dimension];
                }
            }
            _page_load_count++;
        }
        _round_count++;
        context.stream
            << _page_table.copy_from(luisa::span{_host_page_table})
            << _physical_cache.copy_from(luisa::span{_host_cache});
    }

public:
    OnDemandTextureCache(
        Options options,
        luisa::vector<float4> virtual_texture,
        BufferView<uint> page_table,
        BufferView<float4> physical_cache) noexcept
        : _options{options},
          _page_texel_count{options.page_size * options.page_size},
          _page_table{page_table},
          _physical_cache{physical_cache},
          _virtual_texture{std::move(virtual_texture)},
          _host_page_table(virtual_page_count, 0u),
          _slot_pages(physical_page_count, -1),
          _host_cache(
              physical_page_count * _page_texel_count,
              make_float4(0.0f)) {}

    [[nodiscard]] luisa::unique_ptr<
        WavefrontCoroSchedulerExtensionHandler>
    operator()(
        WavefrontCoroExtensionPrepareContext &context,
        const WavefrontCoroExtensionStage &stage) noexcept {
        if (stage.extension->schema() != request_schema ||
            stage.extension->version() != 1u) {
            return nullptr;
        }
        LUISA_ASSERT(
            stage.extension->is_annotation() &&
                stage.extension->fallback() ==
                    CoroSuspendFallback::reject &&
                stage.dataflow->def.slots.empty(),
            "A required virtual-texture request must be a read-only "
            "coroutine annotation.");
        if (!_initialized) {
            context.stream
                << _page_table.copy_from(luisa::span{_host_page_table})
                << _physical_cache.copy_from(luisa::span{_host_cache});
            _initialized = true;
        }
        if (!_requests) {
            _requests = context.device.create_buffer<uint>(
                context.frame_capacity);
            _host_requests.resize(context.frame_capacity);
        }
        auto reconstruct_slots = stage.dataflow->reconstruct_slots;
        auto *page = &stage.binding("page");
        auto *desc = &context.frame_desc;
        Kernel1D collect = [desc, page,
                            layout = context.frame_layout,
                            soa = context.global_memory_soa,
                            reconstruct_slots](
                               ByteBufferVar frame_storage,
                               BufferUInt frame_indices,
                               UInt frame_capacity, UInt count,
                               BufferUInt requests) noexcept {
            auto x = dispatch_x();
            $if (x >= count) { $return(); };
            auto frame_index = frame_indices.read(x);
            auto frame = CoroFrame::create(desc);
            coro_frame_load_into(
                frame, frame_storage, frame_index, frame_capacity,
                layout, soa, luisa::span{reconstruct_slots},
                false, false);
            // Zero denotes no request in host-side diagnostics.
            requests.write(x, page->read<uint>(frame) + 1u);
        };
        auto label = luisa::format(
            "wavefront_extension_texture_request_{}",
            stage.queue_index);
        auto shader = coro::detail::coro_scheduler_label_shader(
            context.device.compile(
                collect,
                coro::detail::coro_scheduler_shader_option(
                    context.shader_option, label)),
            label);
        return luisa::make_unique<Handler>(this, std::move(shader));
    }

    [[nodiscard]] size_t page_load_count() const noexcept {
        return _page_load_count;
    }
    [[nodiscard]] size_t round_count() const noexcept {
        return _round_count;
    }
    [[nodiscard]] luisa::span<const float4>
    virtual_texture() const noexcept {
        return _virtual_texture;
    }
};

void OnDemandTextureCache::Handler::dispatch(
    const WavefrontCoroExtensionDispatchContext &context) noexcept {
    _owner->_dispatch(context, _shader);
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        LUISA_INFO("Usage: {} <backend> [--test]", argv[0]);
        return 1;
    }
    auto options = parse_options(argc, argv);
    LUISA_ASSERT(options.dimension == options.page_size * page_grid_size,
                 "Virtual-texture dimensions must contain an 8x8 page grid.");
    auto pixel_count = options.dimension * options.dimension;
    auto page_texel_count = options.page_size * options.page_size;

    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto output = device.create_image<float>(
        PixelStorage::FLOAT4, options.dimension, options.dimension);
    auto ldr = device.create_image<float>(
        PixelStorage::BYTE4, options.dimension, options.dimension);
    auto page_table = device.create_buffer<uint>(virtual_page_count);
    auto physical_cache = device.create_buffer<float4>(
        physical_page_count * page_texel_count);
    luisa::vector<uint> test_page_table_poison;
    if (!options.write_image) {
        // Prove that handler preparation initializes the external cache before
        // the coroutine entry kernel observes it.
        test_page_table_poison.assign(
            virtual_page_count, 0xffffffffu);
        stream << page_table.copy_from(
            luisa::span{test_page_table_poison});
    }

    Coroutine<void(Image<float>, Buffer<uint>, Buffer<float4>)> coroutine =
        [options, page_texel_count](
            ImageFloat output_image, BufferUInt table,
            BufferVar<float4> cache) noexcept {
            auto coord = dispatch_id().xy();
            auto page_coord = coord / options.page_size;
            auto local_coord = coord % options.page_size;
            Var page = page_coord.x + page_coord.y * page_grid_size;

            // The scheduler repeatedly routes this exact static stage until
            // the handler makes the requested page resident.
            $while (table.read(page) == 0u) {
                $suspend(
                    "texture_miss",
                    coro_annotation(request_schema)
                        .fallback(CoroSuspendFallback::reject)
                        .read("page", page));
            };

            auto physical_page = table.read(page) - 1u;
            auto local_index =
                local_coord.x + local_coord.y * options.page_size;
            auto texel = cache.read(
                physical_page * page_texel_count + local_index);
            output_image.write(coord, texel);
        };

    WavefrontCoroScheduler<
        Image<float>, Buffer<uint>, Buffer<float4>> scheduler{
        device, coroutine,
        WavefrontCoroSchedulerConfig{
            .thread_count = pixel_count,
            .global_memory_soa = true,
            .gather_by_sorting = true,
            .frame_buffer_compaction = false,
            .report_stats = true,
            .execution_block_size = 256u,
            .largest_continuation_first = true,
            .incremental_continuation_counts = true}};
    auto texture_cache = luisa::make_shared<OnDemandTextureCache>(
        options, make_virtual_texture(options.dimension),
        page_table.view(), physical_cache.view());
    scheduler.register_extension_handler(
        stream,
        [texture_cache](
            WavefrontCoroExtensionPrepareContext &prepare_context,
            const WavefrontCoroExtensionStage &stage) noexcept {
            return (*texture_cache)(prepare_context, stage);
        });
    stream << scheduler(output, page_table, physical_cache)
                  .dispatch(options.dimension, options.dimension);

    LUISA_ASSERT(
        texture_cache->round_count() ==
            virtual_page_count / physical_page_count,
        "Expected {} cache-fault rounds, got {}.",
        virtual_page_count / physical_page_count,
        texture_cache->round_count());
    LUISA_ASSERT(
        texture_cache->page_load_count() == virtual_page_count,
        "Expected every virtual page to be loaded once, got {} loads for "
        "{} pages.",
        texture_cache->page_load_count(), virtual_page_count);

    luisa::vector<float> host_output(
        static_cast<size_t>(pixel_count) * 4u);
    stream << output.copy_to(luisa::span{host_output}) << synchronize();
    float max_error = 0.0f;
    auto reference = texture_cache->virtual_texture();
    for (auto i = 0u; i < pixel_count; ++i) {
        auto expected = reference[i];
        for (auto channel = 0u; channel < 4u; ++channel) {
            auto actual = host_output[i * 4u + channel];
            max_error = std::max(
                max_error, std::abs(actual - expected[channel]));
        }
    }
    LUISA_ASSERT(max_error <= 1e-6f,
                 "On-demand texture output mismatch: max error {}.",
                 max_error);

    if (options.write_image) {
        Kernel2D encode = [](ImageFloat source, ImageFloat destination) {
            auto coord = dispatch_id().xy();
            destination.write(coord, source.read(coord));
        };
        auto encode_shader = device.compile(encode);
        luisa::vector<uint8_t> host_ldr(
            static_cast<size_t>(pixel_count) * 4u);
        stream << encode_shader(output, ldr)
                      .dispatch(options.dimension, options.dimension)
               << ldr.copy_to(luisa::span{host_ldr}) << synchronize();
        LUISA_ASSERT(
            stbi_write_png("coro_on_demand_texture.png", options.dimension,
                           options.dimension, 4, host_ldr.data(), 0) != 0,
            "Failed to write 'coro_on_demand_texture.png'.");
    }

    LUISA_INFO(
        "On-demand coroutine texture passed on '{}': {}x{}, 64 virtual "
        "pages / 8 physical pages, {} handler rounds, {} loads, max error "
        "{:.9f}{}.",
        argv[1], options.dimension, options.dimension,
        texture_cache->round_count(), texture_cache->page_load_count(),
        max_error,
        options.write_image ? ", wrote coro_on_demand_texture.png" : "");
    return 0;
}
