#include <cstdlib>

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

int main(int argc, char *argv[]) {
    if (argc < 2) {
        LUISA_INFO("Usage: {} <backend>", argv[0]);
        return 1;
    }

    constexpr uint frame_count = 256u;
    constexpr auto debug_schema = "luisa.coro.debug.structured-watch";

    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto output = device.create_buffer<uint>(frame_count);
    // Two uints per record: logical coroutine id and watched value. This is a
    // structured observer buffer, not device_log or printf instrumentation.
    auto records = device.create_buffer<uint>(frame_count * 2u);

    Coroutine<void(Buffer<uint>)> coroutine =
        [](BufferUInt output_buffer) noexcept {
            auto id = dispatch_x();
            auto watched = id * 3u + 7u;
            $suspend(
                "after_transform",
                coro_annotation(debug_schema)
                    .fallback(CoroSuspendFallback::reject)
                    .read("value", watched));
            output_buffer.write(id, watched * 2u + 1u);
        };

    auto stage_view = find_external_stage(
        coroutine.graph(), debug_schema);
    auto &&boundary = *stage_view.boundary;
    auto &&stage = *stage_view.stage;
    auto &&watched = stage_view.binding("value");
    LUISA_ASSERT(boundary.from_index == 0u,
                 "Debug example expects its stage after the entry scope.");
    LUISA_ASSERT(stage.def.frame_values.empty() &&
                     stage.required_writeback_slot_span().empty(),
                 "A read-only debug observer must not write frame state.");

    auto layout = CoroFrameStorageLayout::make_aos(
        coroutine.frame(), frame_count);
    auto frames = device.create_byte_buffer(layout.size_bytes);
    auto io_plan = coro_frame_make_io_plan(
        coroutine.graph(), coroutine.frame().frame_field_count());

    // Step 1 is user-owned: this debugger additionally needs coro_id_x for
    // record attribution, so it combines that application requirement with
    // the stage's exact reconstruction certificate.
    auto reconstruct_slots = merge_stage_slots(
        stage.reconstruct_slot_span(), {0u});
    // The same application-required id must also be initialized by the source
    // frame transport. Ordinary continuation IO comes from the graph-edge
    // plan; adding field zero is an explicit debugger policy, not a hidden
    // whole-header spill.
    auto source_store_slots = merge_stage_slots(
        io_plan.output(boundary.from_index, boundary.to_index), {0u});
    auto resume_slots = luisa::vector<size_t>{
        io_plan.input(boundary.to_index).begin(),
        io_plan.input(boundary.to_index).end()};

    Kernel1D generate = [&coroutine, layout, source_store_slots](
                            ByteBufferVar frame_storage,
                            BufferUInt output_buffer) noexcept {
        auto id = dispatch_x();
        auto frame = coroutine.instantiate(
            make_uint3(id, 0u, 0u), make_uint3(frame_count, 1u, 1u));
        frame.target_token = 0u;
        coroutine.entry()(frame, output_buffer);
        coro_frame_store(
            frame_storage, id, frame, layout, false,
            luisa::span{source_store_slots}, false, false);
    };

    Kernel1D observe = [&coroutine, &watched, layout, reconstruct_slots](
                           ByteBufferVar frame_storage,
                           BufferUInt record_buffer) noexcept {
        auto id = dispatch_x();
        auto frame = CoroFrame::create(&coroutine.frame());
        // Load precisely compiler uses/RMW carriers plus the debugger's id.
        // include_reserved_fields=false is essential: no implicit whole-header
        // load is hidden behind this partial reconstruction.
        coro_frame_load_into(
            frame, frame_storage, id, layout, false,
            luisa::span{reconstruct_slots}, false, false);

        // Step 2 is exclusively binding-callable access. The observer does
        // not know a generated frame-field name or physical type.
        auto value = watched.read<uint>(frame);
        record_buffer.write(id * 2u, frame.coro_id_x);
        record_buffer.write(id * 2u + 1u, value);

        // Step 3 intentionally emits no store: required_writeback is empty.
    };

    Kernel1D resume = [&coroutine, layout, resume_slots,
                       resume_index = boundary.to_index](
                          ByteBufferVar frame_storage,
                          BufferUInt output_buffer) noexcept {
        auto id = dispatch_x();
        auto frame = CoroFrame::create(&coroutine.frame());
        coro_frame_load_into(
            frame, frame_storage, id, layout, false,
            luisa::span{resume_slots}, false, false);
        frame.target_token = CoroFrame::TERMINAL_TOKEN;
        coroutine[resume_index](frame, output_buffer);
    };

    auto generate_shader = device.compile(generate);
    auto observe_shader = device.compile(observe);
    auto resume_shader = device.compile(resume);

    luisa::vector<uint> host_records(frame_count * 2u);
    luisa::vector<uint> host_output(frame_count);
    stream << generate_shader(frames, output).dispatch(frame_count)
           << observe_shader(frames, records).dispatch(frame_count)
           << resume_shader(frames, output).dispatch(frame_count)
           << records.copy_to(luisa::span{host_records})
           << output.copy_to(luisa::span{host_output})
           << synchronize();

    for (auto i = 0u; i < frame_count; ++i) {
        auto expected_watch = i * 3u + 7u;
        LUISA_ASSERT(host_records[i * 2u] == i &&
                         host_records[i * 2u + 1u] == expected_watch,
                     "Structured debug record {} is invalid: ({}, {}).",
                     i, host_records[i * 2u],
                     host_records[i * 2u + 1u]);
        LUISA_ASSERT(host_output[i] == expected_watch * 2u + 1u,
                     "Coroutine result {} is invalid: {}.", i,
                     host_output[i]);
    }

    LUISA_INFO(
        "Structured coroutine debug passed on '{}': {} frames, {} "
        "reconstructed slot(s), zero frame writeback slots.",
        argv[1], frame_count, reconstruct_slots.size());
    return 0;
}
