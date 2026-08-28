#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string_view>

#include <stb/stb_image_write.h>

#include <luisa/luisa-compute.h>
#include <luisa/coro/coro_frame_storage.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>

#include "coro/external_stage_common.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace luisa::compute::coro::example;

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
    auto requests = device.create_buffer<uint>(pixel_count);

    Coroutine<void(Image<float>, Buffer<uint>, Buffer<float4>)> coroutine =
        [options, page_texel_count](
            ImageFloat output_image, BufferUInt table,
            BufferVar<float4> cache) noexcept {
            auto coord = dispatch_id().xy();
            auto page_coord = coord / options.page_size;
            auto local_coord = coord % options.page_size;
            Var page = page_coord.x + page_coord.y * page_grid_size;

            // This is the intended source spelling: the coroutine remains
            // suspended while the page is absent. The annotation carries a
            // queued snapshot of `page`; a host cache handler services it.
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

    auto request_views = find_external_stages(
        coroutine.graph(), request_schema);
    for (auto &&view : request_views) {
        LUISA_ASSERT(
            view.extension->is_annotation() &&
                view.extension->fallback() == CoroSuspendFallback::reject,
            "Virtual-texture request must be a required annotation.");
        LUISA_ASSERT(
            view.stage->def.frame_values.empty() &&
                view.stage->required_writeback_slot_span().empty(),
            "A texture request must not mutate coroutine frame values.");
    }

    auto layout = CoroFrameStorageLayout::make_aos(
        coroutine.frame(), pixel_count);
    auto frames = device.create_byte_buffer(layout.size_bytes);
    auto routes = device.create_buffer<uint>(pixel_count);
    auto scheduled_routes = device.create_buffer<uint>(pixel_count);
    // Route tokens live in the scheduler's side buffer, so field 6 is not
    // duplicated in CoroFrame storage.
    auto io_plan = coro_frame_make_io_plan(
        coroutine.graph(), coroutine.frame().frame_field_count());
    auto outgoing_routes = [&](size_t source) noexcept {
        return collect_external_stage_routes(coroutine.graph(), source);
    };

    auto entry_outputs = io_plan.transition_output_fields[0u];
    auto entry_routes = outgoing_routes(0u);
    Kernel1D generate = [&coroutine, layout, entry_outputs, entry_routes,
                         options, pixel_count](
                            ByteBufferVar frame_storage, ImageFloat image,
                            BufferUInt table, BufferVar<float4> cache,
                            BufferUInt route_buffer) noexcept {
        auto index = dispatch_x();
        $if (index >= pixel_count) { $return(); };
        auto x = index % options.dimension;
        auto y = index / options.dimension;
        auto frame = coroutine.instantiate(
            make_uint3(x, y, 0u),
            make_uint3(options.dimension, options.dimension, 1u));
        frame.target_token = 0u;
        coroutine.entry()(frame, image, table, cache);
        Var next = 0u;
        Var next_route = 0u;
        for (auto route : entry_routes) {
            $if (frame.target_token == route.token) {
                next = route.target;
                next_route = route.boundary;
            };
        }
        route_buffer.write(index, next_route);
        for (auto target = 0u; target < entry_outputs.size(); ++target) {
            $if (next == static_cast<uint>(target)) {
                coro_frame_store(
                    frame_storage, index, frame, layout, false,
                    luisa::span{entry_outputs[target]}, false, false);
            };
        }
    };
    auto generate_shader = device.compile(generate);

    Kernel1D copy_routes = [](BufferUInt source,
                              BufferUInt destination) noexcept {
        auto index = dispatch_x();
        destination.write(index, source.read(index));
    };
    Kernel1D clear_requests = [](BufferUInt request_buffer) noexcept {
        request_buffer.write(dispatch_x(), 0u);
    };
    auto copy_routes_shader = device.compile(copy_routes);
    auto clear_requests_shader = device.compile(clear_requests);

    luisa::vector<Shader1D<ByteBuffer, Buffer<uint>, Buffer<uint>>>
        request_stages;
    request_stages.reserve(request_views.size());
    for (auto &&view : request_views) {
        auto reconstruct_slots = merge_stage_slots(
            view.stage->reconstruct_slot_span());
        auto page = &view.binding("page");
        auto route = static_cast<uint>(view.boundary->index + 1u);
        Kernel1D request = [&coroutine, layout, reconstruct_slots, page,
                            route, pixel_count](
                               ByteBufferVar frame_storage,
                               BufferUInt scheduled_route_buffer,
                               BufferUInt request_buffer) noexcept {
            auto index = dispatch_x();
            $if (index >= pixel_count) { $return(); };
            $if (scheduled_route_buffer.read(index) != route) { $return(); };
            auto frame = CoroFrame::create(&coroutine.frame());
            coro_frame_load_into(
                frame, frame_storage, index, layout, false,
                luisa::span{reconstruct_slots}, false, false);
            // Zero means no request, so encode the virtual page as page + 1.
            request_buffer.write(index, page->read<uint>(frame) + 1u);
            // Read-only annotation: no frame writeback at all.
        };
        request_stages.emplace_back(device.compile(request));
    }

    using ResumeShader =
        Shader1D<ByteBuffer, Image<float>, Buffer<uint>, Buffer<float4>,
                 Buffer<uint>, Buffer<uint>>;
    luisa::vector<ResumeShader> resume_shaders;
    resume_shaders.reserve(request_views.size());
    for (auto &&view : request_views) {
        auto node = view.boundary->to_index;
        auto route = static_cast<uint>(view.boundary->index + 1u);
        auto input_slots = io_plan.input_fields[node];
        auto output_slots = io_plan.transition_output_fields[node];
        auto next_routes = outgoing_routes(node);
        Kernel1D resume = [&coroutine, layout, node, route, input_slots,
                           output_slots, next_routes, pixel_count](
                              ByteBufferVar frame_storage, ImageFloat image,
                              BufferUInt table, BufferVar<float4> cache,
                              BufferUInt route_buffer,
                              BufferUInt scheduled_route_buffer) noexcept {
            auto index = dispatch_x();
            $if (index >= pixel_count) { $return(); };
            $if (scheduled_route_buffer.read(index) != route) { $return(); };
            auto frame = CoroFrame::create(&coroutine.frame());
            coro_frame_load_into(
                frame, frame_storage, index, layout, false,
                luisa::span{input_slots}, false, false);
            frame.target_token = CoroFrame::TERMINAL_TOKEN;
            coroutine[node](frame, image, table, cache);
            Var next = 0u;
            Var next_route = 0u;
            for (auto candidate : next_routes) {
                $if (frame.target_token == candidate.token) {
                    next = candidate.target;
                    next_route = candidate.boundary;
                };
            }
            route_buffer.write(index, next_route);
            for (auto target = 0u; target < output_slots.size(); ++target) {
                $if (next == static_cast<uint>(target)) {
                    coro_frame_store(
                        frame_storage, index, frame, layout, false,
                        luisa::span{output_slots[target]}, false, false);
                };
            }
        };
        resume_shaders.emplace_back(device.compile(resume));
    }

    auto virtual_texture = make_virtual_texture(options.dimension);
    luisa::vector<uint> host_page_table(virtual_page_count, 0u);
    luisa::vector<int> slot_pages(physical_page_count, -1);
    luisa::vector<float4> host_cache(
        physical_page_count * page_texel_count, make_float4(0.0f));
    luisa::vector<uint> host_requests(pixel_count, 0u);
    luisa::vector<uint> host_routes(pixel_count, 0u);
    stream << page_table.copy_from(luisa::span{host_page_table})
           << physical_cache.copy_from(luisa::span{host_cache})
           << generate_shader(frames, output, page_table, physical_cache,
                              routes)
                  .dispatch(pixel_count);

    size_t page_load_count = 0u;
    size_t round_count = 0u;
    for (; round_count < virtual_page_count; ++round_count) {
        stream << copy_routes_shader(routes, scheduled_routes)
                      .dispatch(pixel_count)
               << clear_requests_shader(requests).dispatch(pixel_count);
        for (auto &&shader : request_stages) {
            stream << shader(frames, scheduled_routes, requests)
                          .dispatch(pixel_count);
        }
        stream << requests.copy_to(luisa::span{host_requests})
               << synchronize();

        luisa::vector<bool> requested(virtual_page_count, false);
        for (auto encoded : host_requests) {
            if (encoded != 0u) {
                LUISA_ASSERT(encoded <= virtual_page_count,
                             "Invalid virtual-texture page request {}.",
                             encoded);
                requested[encoded - 1u] = true;
            }
        }
        LUISA_ASSERT(std::any_of(requested.begin(), requested.end(),
                                 [](auto value) noexcept { return value; }),
                     "Texture scheduler has live routes but no page request.");

        for (auto page = 0u; page < virtual_page_count; ++page) {
            if (!requested[page] || host_page_table[page] != 0u) { continue; }
            auto slot = physical_page_count;
            for (auto i = 0u; i < physical_page_count; ++i) {
                if (slot_pages[i] < 0) {
                    slot = i;
                    break;
                }
            }
            if (slot == physical_page_count) {
                for (auto i = 0u; i < physical_page_count; ++i) {
                    auto resident = static_cast<uint>(slot_pages[i]);
                    if (!requested[resident]) {
                        slot = i;
                        break;
                    }
                }
            }
            // All physical slots contain pages needed by this resume batch.
            // Defer remaining faults until the next host round.
            if (slot == physical_page_count) { break; }
            if (slot_pages[slot] >= 0) {
                host_page_table[static_cast<uint>(slot_pages[slot])] = 0u;
            }
            slot_pages[slot] = static_cast<int>(page);
            host_page_table[page] = slot + 1u;
            auto page_x = page % page_grid_size;
            auto page_y = page / page_grid_size;
            for (auto y = 0u; y < options.page_size; ++y) {
                for (auto x = 0u; x < options.page_size; ++x) {
                    auto source_x = page_x * options.page_size + x;
                    auto source_y = page_y * options.page_size + y;
                    host_cache[slot * page_texel_count +
                               x + y * options.page_size] =
                        virtual_texture[source_x +
                                        source_y * options.dimension];
                }
            }
            page_load_count++;
        }

        stream << page_table.copy_from(luisa::span{host_page_table})
               << physical_cache.copy_from(luisa::span{host_cache});
        for (auto &&shader : resume_shaders) {
            stream << shader(frames, output, page_table, physical_cache,
                             routes, scheduled_routes)
                          .dispatch(pixel_count);
        }
        stream << routes.copy_to(luisa::span{host_routes}) << synchronize();
        if (std::all_of(host_routes.begin(), host_routes.end(),
                        [](auto route) noexcept { return route == 0u; })) {
            round_count++;
            break;
        }
    }

    LUISA_ASSERT(round_count ==
                     virtual_page_count / physical_page_count,
                 "Expected {} cache-fault rounds, got {}.",
                 virtual_page_count / physical_page_count, round_count);
    LUISA_ASSERT(page_load_count == virtual_page_count,
                 "Expected every virtual page to be loaded once, got {} "
                 "loads for {} pages.",
                 page_load_count, virtual_page_count);

    luisa::vector<float> host_output(
        static_cast<size_t>(pixel_count) * 4u);
    stream << output.copy_to(luisa::span{host_output}) << synchronize();
    float max_error = 0.0f;
    for (auto i = 0u; i < pixel_count; ++i) {
        auto expected = virtual_texture[i];
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
        "pages / 8 physical pages, {} fault rounds, {} loads, max error "
        "{:.9f}{}.",
        argv[1], options.dimension, options.dimension, round_count,
        page_load_count, max_error,
        options.write_image ? ", wrote coro_on_demand_texture.png" : "");
    return 0;
}
