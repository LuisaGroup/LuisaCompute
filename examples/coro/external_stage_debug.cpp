#include <cstdlib>

#include <luisa/luisa-compute.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;

namespace {

constexpr auto debug_schema = "luisa.coro.debug.structured-watch";

class StructuredDebugger final {

private:
    class Handler final
        : public WavefrontCoroSchedulerExtensionHandler {

    private:
        Shader1D<ByteBuffer, Buffer<uint>, uint, uint, Buffer<uint>>
            _shader;
        BufferView<uint> _records;

    public:
        Handler(
            Shader1D<ByteBuffer, Buffer<uint>, uint, uint, Buffer<uint>>
                shader,
            BufferView<uint> records) noexcept
            : _shader{std::move(shader)}, _records{records} {}

        [[nodiscard]] luisa::string_view name() const noexcept override {
            return "structured-debugger";
        }

        void dispatch(
            const WavefrontCoroExtensionDispatchContext &context) noexcept override {
            context.stream << _shader(
                                  context.frame_buffer,
                                  context.frame_indices,
                                  context.frame_capacity,
                                  context.frame_count, _records)
                                  .dispatch(context.frame_count);
        }
    };

    Buffer<uint> _records;

public:
    StructuredDebugger(Device &device, uint frame_count)
        : _records{device.create_buffer<uint>(frame_count * 2u)} {}

    [[nodiscard]] luisa::unique_ptr<
        WavefrontCoroSchedulerExtensionHandler>
    operator()(
        WavefrontCoroExtensionPrepareContext &context,
        const WavefrontCoroExtensionStage &stage) noexcept {
        if (stage.extension->schema() != debug_schema ||
            stage.extension->version() != 1u) {
            return nullptr;
        }
        LUISA_ASSERT(
            stage.dataflow->required_writeback_slot_span().empty(),
            "A structured debug observer must not write coroutine state.");
        auto reconstruct_slots = stage.dataflow->reconstruct_slots;
        auto *id = &stage.binding("id");
        auto *value = &stage.binding("value");
        auto *desc = &context.frame_desc;
        Kernel1D observe = [desc, id, value,
                            layout = context.frame_layout,
                            soa = context.global_memory_soa,
                            reconstruct_slots](
                               ByteBufferVar frame_storage,
                               BufferUInt frame_indices,
                               UInt frame_capacity, UInt count,
                               BufferUInt records) noexcept {
            auto x = dispatch_x();
            $if (x >= count) { $return(); };
            auto frame_index = frame_indices.read(x);
            auto frame = CoroFrame::create(desc);
            coro_frame_load_into(
                frame, frame_storage, frame_index, frame_capacity,
                layout, soa, luisa::span{reconstruct_slots},
                false, false);
            auto logical_id = id->read<uint>(frame);
            records.write(logical_id * 2u, logical_id);
            records.write(logical_id * 2u + 1u,
                          value->read<uint>(frame));
        };
        auto label = luisa::format(
            "wavefront_extension_debug_{}", stage.queue_index);
        auto shader = coro::detail::coro_scheduler_label_shader(
            context.device.compile(
                observe,
                coro::detail::coro_scheduler_shader_option(
                    context.shader_option, label)),
            label);
        return luisa::make_unique<Handler>(
            std::move(shader), _records.view());
    }

    [[nodiscard]] const Buffer<uint> &records() const noexcept {
        return _records;
    }
};

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        LUISA_INFO("Usage: {} <backend>", argv[0]);
        return 1;
    }

    constexpr uint frame_count = 256u;
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto output = device.create_buffer<uint>(frame_count);

    Coroutine<void(Buffer<uint>)> coroutine =
        [](BufferUInt output_buffer) noexcept {
            auto id = dispatch_x();
            auto watched = id * 3u + 7u;
            $suspend(
                "after_transform",
                coro_annotation(debug_schema)
                    .fallback(CoroSuspendFallback::reject)
                    .read("id", id)
                    .read("value", watched));
            output_buffer.write(id, watched * 2u + 1u);
        };

    WavefrontCoroScheduler<Buffer<uint>> scheduler{
        device, coroutine,
        WavefrontCoroSchedulerConfig{
            .thread_count = frame_count,
            .global_memory_soa = true,
            .gather_by_sorting = true,
            .frame_buffer_compaction = true,
            .report_stats = true,
            .execution_block_size = 32u,
            .largest_continuation_first = true}};
    StructuredDebugger debugger{device, frame_count};
    scheduler.register_extension_handler(
        stream,
        [&debugger](WavefrontCoroExtensionPrepareContext &prepare_context,
                    const WavefrontCoroExtensionStage &stage) noexcept {
            return debugger(prepare_context, stage);
        });

    luisa::vector<uint> host_records(frame_count * 2u);
    luisa::vector<uint> host_output(frame_count);
    stream << scheduler(output).dispatch(frame_count)
           << debugger.records().copy_to(luisa::span{host_records})
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

    auto &&stats = scheduler.last_dispatch_stats();
    LUISA_ASSERT(stats.extensions.size() == 1u &&
                     stats.extensions.front().dispatch_count == 1u &&
                     stats.extensions.front().executed_count == frame_count,
                 "Structured debugger was not scheduled as one exact "
                 "Extension queue.");
    LUISA_INFO(
        "Structured coroutine debug passed on '{}': {} frames through "
        "handler '{}', with zero frame writeback slots.",
        argv[1], frame_count, stats.extensions.front().handler);
    return 0;
}
