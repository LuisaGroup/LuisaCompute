// Shared suspending callables: repeated and nested calls, live caller state,
// conditional suspend, early return, and all three scheduling strategies.
#include "ut/ut.hpp"
#include "coro_test_utils.h"
#include <luisa/luisa-compute.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/coro/schedulers/state_machine.h>
#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/coro/schedulers/persistent.h>
#include <luisa/coro/coro_frame_storage.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {
class AliasHandler final : public WavefrontCoroSchedulerExtensionHandler {
    Shader1D<ByteBuffer, Buffer<uint>, uint, uint> _shader;
public:
    explicit AliasHandler(Shader1D<ByteBuffer, Buffer<uint>, uint, uint> shader) noexcept : _shader{std::move(shader)} {}
    luisa::string_view name() const noexcept override { return "shared-callable-alias"; }
    void dispatch(const WavefrontCoroExtensionDispatchContext &ctx) noexcept override {
        ctx.stream << _shader(ctx.frame_buffer, ctx.frame_indices, ctx.frame_capacity, ctx.frame_count).dispatch(ctx.frame_count);
    }
};
luisa::unique_ptr<WavefrontCoroSchedulerExtensionHandler> prepare_alias_handler(
    WavefrontCoroExtensionPrepareContext &ctx, const WavefrontCoroExtensionStage &stage) {
    expect(stage.extension->bindings().size() == 2u) << "compiler bindings must stay private";
    auto *a = &stage.binding("a");
    auto *b = &stage.binding("b");
    auto write_only = stage.extension->schema() == "luisa.test.alias.write";
    auto boolean = a->type() == Type::of<bool>();
    auto aggregate = a->type() == Type::of<uint2>();
    auto soa = ctx.global_memory_soa;
    auto *desc = &ctx.frame_desc;
    auto reconstruct = stage.dataflow->reconstruct_slots;
    auto writeback = stage.dataflow->required_def.slots;
    Kernel1D kernel = [=, layout = ctx.frame_layout](ByteBufferVar storage, BufferUInt indices, UInt capacity, UInt n) {
        auto i = dispatch_x();
        $if (i >= n) { $return(); };
        auto index = indices.read(i);
        auto frame = CoroFrame::create(desc);
        coro_frame_load_into(frame, storage, index, capacity, layout, soa, luisa::span{reconstruct}, false, false);
        if (boolean) {
            a->write<bool>(frame, !a->read<bool>(frame));
            b->write<bool>(frame, b->read<bool>(frame) != a->read<bool>(frame));
        } else if (aggregate) {
            a->write<uint2>(frame, a->read<uint2>(frame) + make_uint2(3u, 5u));
            b->write<uint>(frame, b->read<uint>(frame) + a->read<uint2>(frame).x);
        } else {
            if (write_only) {
                a->write<uint>(frame, b->read<uint>(frame) + 9u);
            } else {
                a->write<uint>(frame, a->read<uint>(frame) + 3u);
            }
            b->write<uint>(frame, b->read<uint>(frame) + (write_only ? b->read<uint>(frame) : a->read<uint>(frame)));
        }
        coro_frame_store(storage, index, capacity, frame, layout, soa, luisa::span{writeback}, false, false);
    };
    return luisa::make_unique<AliasHandler>(ctx.device.compile(kernel, ctx.shader_option));
}

}// namespace

int main(int argc, char *argv[]) {
    auto options = luisa::test::coro_test::parse_options(argc, argv);
    auto dc = luisa::test::coro_test::create_device(options);
    auto &device = dc.device;
    auto stream = device.create_stream();
    "shared_suspending_callable_execution"_test = [&] {
        constexpr uint32_t count = 256u;
        auto output = device.create_buffer<uint32_t>(count);
        Callable<uint32_t(uint32_t)> leaf = [](UInt x) {
            luisa::compute::detail::FunctionBuilder::current()->mark_noinline();
            $if ((x & 1u) != 0u) { $suspend("leaf"); };
            return x * 3u + 1u;
        };
        Callable<uint32_t(uint32_t)> helper = [&](UInt x) {
            auto saved = x + 7u;
            auto first = leaf(x);
            $if (x % 3u == 0u) { $return(first + saved); };
            auto second = leaf(x + 1u);
            return first + second + saved;
        };
        Coroutine<void(Buffer<uint32_t>)> coroutine = [&](BufferUInt out) {
            auto x = dispatch_id().x;
            auto a = helper(x);
            auto b = helper(x + 2u);
            out.write(x, a * 11u + b);
        };
        expect(coroutine.graph().node_count() == 2u) << "one shared leaf resume node";
        expect(coroutine.graph().call_graph().functions.size() == 3u);
        expect(coroutine.graph().call_graph().analysis_state_count != 0u);
        expect(coroutine.graph().call_graph().edges.size() == 4u);
        auto reference = [](uint32_t x) {
            auto first = x * 3u + 1u;
            return first + x + 7u + (x % 3u == 0u ? 0u : (x + 1u) * 3u + 1u);
        };
        auto run = [&](auto &task, auto &&expected, auto &&...arguments) {
            auto check = [&] {
                luisa::vector<uint32_t> values(count);
                stream << output.copy_to(luisa::span{values}) << synchronize();
                for (uint32_t i = 0u; i < count; ++i) {
                    expect(values[i] == expected(i)) << "at " << i << " got " << values[i] << " expected " << expected(i);
                }
            };
            {
                StateMachineCoroScheduler scheduler{device, task};
                stream << scheduler(arguments...).dispatch(count);
                check();
            }
            for (bool soa : {false, true}) {
                WavefrontCoroSchedulerConfig config;
                config.thread_count = count;
                config.global_memory_soa = soa;
                config.frame_buffer_compaction = true;
                WavefrontCoroScheduler scheduler{device, task, config};
                stream << scheduler(arguments...).dispatch(count);
                check();
            }
            {
                PersistentThreadsCoroSchedulerConfig config;
                config.thread_count = count;
                config.block_size = 128u;
                PersistentThreadsCoroScheduler scheduler{device, task, config};
                stream << scheduler(arguments...).dispatch(count);
                check();
            }
        };
        run(coroutine, [&](uint32_t i) { return reference(i) * 11u + reference(i + 2u); }, output);

        auto source_a = device.create_buffer<uint32_t>(count);
        auto source_b = device.create_buffer<uint32_t>(count);
        luisa::vector<uint32_t> data_a(count), data_b(count);
        for (uint32_t i = 0u; i < count; ++i) {
            data_a[i] = i + 3u;
            data_b[i] = i * 2u + 5u;
        }
        stream << source_a.copy_from(luisa::span{data_a}) << source_b.copy_from(luisa::span{data_b});
        Callable<uint32_t(uint32_t &, uint32_t &, Buffer<uint32_t>, uint32_t)> access =
            [](UInt &a, UInt &b, BufferUInt source, UInt index) {
                a += source.read(index);
                $suspend("alias");
                b += a;
                return a + b;
            };
        Callable<uint32_t(uint32_t &, uint32_t &, Buffer<uint32_t>, uint32_t)> wrapper =
            [&](UInt &a, UInt &b, BufferUInt source, UInt index) {
                return access(a, b, source, index);
            };
        Coroutine<void(Buffer<uint32_t>, Buffer<uint32_t>, Buffer<uint32_t>)> aliases =
            [&](BufferUInt out, BufferUInt a, BufferUInt b) {
                auto x = dispatch_id().x;
                Var<std::array<uint32_t, 4u>> values;
                for (uint32_t i = 0u; i < 4u; ++i) { values[i] = x + i + 1u; }
                UInt k = x % 2u;
                auto first = wrapper(values[k], values[k], a, x);
                k = (x + 1u) % 2u;
                auto second = wrapper(values[k], values[k + 1u], b, x);
                out.write(x, first * 11u + second + values[0u]);
            };
        expect(aliases.graph().node_count() == 2u);
        expect(aliases.graph().call_graph().edges.size() == 3u);
        run(aliases, [&](uint32_t x) {
        std::array<uint32_t, 4u> values{x + 1u, x + 2u, x + 3u, x + 4u};
        auto k = x % 2u;
        values[k] += data_a[x];
        values[k] += values[k];
        auto first = values[k] * 2u;
        k = (x + 1u) % 2u;
        values[k] += data_b[x];
        values[k + 1u] += values[k];
        auto second = values[k] + values[k + 1u];
        return first * 11u + second + values[0u]; }, output, source_a, source_b);

        Callable<uint(uint &, uint &)> external = [](UInt &a, UInt &b) {
            luisa::compute::detail::FunctionBuilder::current()->mark_noinline();
            $suspend("external_alias",
                     coro_stage("luisa.test.alias.add").read_write("a", a).read_write("b", b),
                     coro_stage("luisa.test.alias.write").write("a", a).read_write("b", b));
            return a * 7u + b;
        };
        Callable<uint(uint &, uint &)> external_wrapper = [&](UInt &a, UInt &b) { return external(a, b); };
        Coroutine<void(Buffer<uint>)> external_task = [&](BufferUInt out) {
            auto x = dispatch_x();
            Var<std::array<uint, 4>> values;
            for (uint i = 0; i < 4; ++i) { values[i] = x + i + 1u; }
            UInt k = x % 2u;
            auto first = external_wrapper(values[k], values[k]);
            k = (x + 1u) % 2u;
            auto second = external_wrapper(values[k], values[k + 1u]);
            out.write(x, first * 11u + second + values[0u] + values[3u]);
        };
        expect(external_task.graph().node_count() == 2u);
        for (bool soa : {false, true}) {
            WavefrontCoroSchedulerConfig config;
            config.thread_count = 17u;
            config.shader_option.enable_cache = false;
            config.global_memory_soa = soa;
            config.frame_buffer_compaction = true;
            WavefrontCoroScheduler scheduler{device, external_task, config};
            scheduler.register_extension_handler(stream, prepare_alias_handler);
            stream << scheduler(output).dispatch(count);
            luisa::vector<uint> host(count);
            stream << output.copy_to(luisa::span{host}) << synchronize();
            for (uint x = 0; x < count; ++x) {
                std::array<uint, 4> values{x + 1, x + 2, x + 3, x + 4};
                auto simulate = [](uint &a, uint &b) { a += 3; b += a; a = b + 9; b += b; return a * 7 + b; };
                auto k = x % 2;
                auto first = simulate(values[k], values[k]);
                k = (x + 1) % 2;
                auto second = simulate(values[k], values[k + 1]);
                expect(host[x] == first * 11 + second + values[0] + values[3]) << "scheduler alias at " << x;
            }
        }

        Callable<void(bool &, bool &)> boolean_access = [](Bool &a, Bool &b) {
            $suspend("packed_alias", coro_stage("luisa.test.alias.bool").read_write("a", a).read_write("b", b));
        };
        Coroutine<void(Buffer<uint>)> boolean_task = [&](BufferUInt out) {
            auto x = dispatch_x();
            Bool4 values;
            for (uint i = 0; i < 4; ++i) { values[i] = (x & (1u << i)) != 0u; }
            UInt k = x % 2u;
            boolean_access(values[k], values[k]);
            k = (x + 1u) % 2u;
            boolean_access(values[k], values[k + 1u]);
            UInt encoded = 0u;
            for (uint i = 0; i < 4; ++i) { encoded |= values[i].cast<uint>() << i; }
            out.write(x, encoded);
        };
        Callable<void(uint2 &, uint &)> aggregate_access = [](UInt2 &a, UInt &b) {
            $suspend("aggregate_alias", coro_stage("luisa.test.alias.aggregate").read_write("a", a).read_write("b", b));
        };
        Coroutine<void(Buffer<uint>)> aggregate_task = [&](BufferUInt out) {
            auto x = dispatch_x();
            Var<std::array<uint2, 2>> values;
            values[0u] = make_uint2(x + 1u, x + 2u);
            values[1u] = make_uint2(x + 3u, x + 4u);
            UInt k = x % 2u;
            aggregate_access(values[k], values[k].x);
            k = (x + 1u) % 2u;
            aggregate_access(values[k], values[k].y);
            out.write(x, values[0u].x * 3u + values[0u].y * 5u + values[1u].x * 7u + values[1u].y * 11u);
        };
        auto check_external = [&](auto &task, auto &&expected) {
            expect(task.graph().node_count() == 2u);
            for (bool soa : {false, true}) {
                WavefrontCoroSchedulerConfig config;
                config.thread_count = 17u;
                config.shader_option.enable_cache = false;
                config.global_memory_soa = soa;
                config.frame_buffer_compaction = true;
                WavefrontCoroScheduler scheduler{device, task, config};
                scheduler.register_extension_handler(stream, prepare_alias_handler);
                stream << scheduler(output).dispatch(count);
                luisa::vector<uint> host(count);
                stream << output.copy_to(luisa::span{host}) << synchronize();
                for (uint x = 0; x < count; ++x) { expect(host[x] == expected(x)) << "projected binding at " << x; }
            }
        };
        Coroutine<void(Buffer<uint>)> disjoint_task = [&](BufferUInt out) {
            auto x = dispatch_x();
            Var<std::array<uint, 2>> left, right;
            $if ((x & 1u) == 0u) {
                left[0u] = x + 1u;
                left[1u] = x + 2u;
                out.write(x, external_wrapper(left[0u], left[1u]));
            }
            $else {
                right[0u] = x + 3u;
                right[1u] = x + 4u;
                out.write(x, external_wrapper(right[1u], right[1u]));
            };
        };
        check_external(disjoint_task, [](uint x) {
            auto apply = [](uint &a, uint &b) { a += 3; b += a; a = b + 9; b += b; return a * 7 + b; };
            uint a = x + 1, b = x + 2, c = x + 4;
            return (x & 1u) == 0 ? apply(a, b) : apply(c, c);
        });
        check_external(boolean_task, [](uint x) {
            std::array<bool, 4> values;
            for (uint i = 0; i < 4; ++i) { values[i] = (x & (1u << i)) != 0; }
            auto apply = [](bool &a, bool &b) { a = !a; b = b != a; };
            auto k = x % 2;
            apply(values[k], values[k]);
            k = (x + 1) % 2;
            apply(values[k], values[k + 1]);
            uint result = 0;
            for (uint i = 0; i < 4; ++i) { result |= uint(values[i]) << i; }
            return result;
        });
        check_external(aggregate_task, [](uint x) {
            std::array<uint2, 2> values{make_uint2(x + 1, x + 2), make_uint2(x + 3, x + 4)};
            auto apply = [](uint2 &a, uint &b) { a += make_uint2(3u, 5u); b += a.x; };
            auto k = x % 2;
            apply(values[k], values[k].x);
            k = (x + 1) % 2;
            apply(values[k], values[k].y);
            return values[0].x * 3 + values[0].y * 5 + values[1].x * 7 + values[1].y * 11;
        });
    };

    "shared_callable_dynamic_alias_depth_scaling"_test = [&] {
        constexpr uint32_t count = 37u;
        using Access = Callable<uint32_t(uint32_t &, uint32_t)>;
        luisa::vector<Access> chain;
        chain.emplace_back([](UInt &value, UInt seed) {
            value += seed * 3u + 1u;
            $if ((seed & 1u) != 0u) { $suspend("deep_alias"); };
            value += seed + 2u;
            return value ^ seed;
        });
        auto output = device.create_buffer<uint32_t>(count);
        size_t first_size = 0u;
        for (uint32_t depth : {1u, 4u, 8u}) {
            while (chain.size() < depth) {
                auto level = static_cast<uint32_t>(chain.size());
                Access next = [&](UInt &value, UInt seed) {
                    auto first = chain.back()(value, seed + level);
                    auto second = chain.back()(value, seed + level + 1u);
                    return first * 17u + second + value;
                };
                chain.emplace_back(std::move(next));
            }
            Coroutine<void(Buffer<uint32_t>)> task = [&](BufferUInt out) {
                auto x = dispatch_x();
                Var<std::array<uint32_t, 4u>> left, right;
                for (uint32_t i = 0u; i < 4u; ++i) {
                    left[i] = x * 5u + i + 1u;
                    right[i] = x * 7u + i + 9u;
                }
                UInt index = x % 4u;
                auto first = chain.back()(left[index], x + 1u);
                index = (x + 1u) % 4u;
                auto second = chain.back()(left[index], x + 2u);
                auto third = chain.back()(right[index], x + 3u);
                auto fourth = chain.back()(left[2u], x + 4u);
                UInt result = first * 3u + second * 5u + third * 7u + fourth * 11u;
                for (uint32_t i = 0u; i < 4u; ++i) { result += left[i] * (i + 13u) + right[i] * (i + 19u); }
                out.write(x, result);
            };
            auto bytes = task.frame_desc().frame_type()->size();
            if (depth == 1u) { first_size = bytes; }
            LUISA_INFO("Dynamic alias depth scaling: depth={}, callsites={}, bytes={}, fields={}",
                       depth, task.graph().call_graph().edges.size(), bytes, task.frame_desc().frame_field_count());
            expect(task.graph().node_count() == 2u);
            expect(task.graph().call_graph().functions.size() == depth + 1u);
            expect(task.graph().call_graph().edges.size() == 4u + 2u * (depth - 1u));
            size_t token_count = 0u;
            for (auto &function : task.graph().call_graph().functions) { token_count += function.resume_tokens.size(); }
            expect(token_count == 1u);
            // Each level has two static callsites, but one activation and one
            // set of dynamic indices. Path counts must not double per level.
            expect(bytes <= first_size + 32u * (depth - 1u)) << "dynamic reference capture frame grows faster than call depth";
            auto reference = [depth](uint32_t x) {
                std::array<uint32_t, 4u> left, right;
                for (uint32_t i = 0u; i < 4u; ++i) {
                    left[i] = x * 5u + i + 1u;
                    right[i] = x * 7u + i + 9u;
                }
                auto apply = [](auto &&self, uint32_t level, uint32_t &value, uint32_t seed) -> uint32_t {
                    if (level == 0u) {
                        value += seed * 3u + 1u;
                        value += seed + 2u;
                        return value ^ seed;
                    }
                    auto first = self(self, level - 1u, value, seed + level);
                    auto second = self(self, level - 1u, value, seed + level + 1u);
                    return first * 17u + second + value;
                };
                auto first = apply(apply, depth - 1u, left[x % 4u], x + 1u);
                auto second = apply(apply, depth - 1u, left[(x + 1u) % 4u], x + 2u);
                auto third = apply(apply, depth - 1u, right[(x + 1u) % 4u], x + 3u);
                auto fourth = apply(apply, depth - 1u, left[2u], x + 4u);
                auto result = first * 3u + second * 5u + third * 7u + fourth * 11u;
                for (uint32_t i = 0u; i < 4u; ++i) { result += left[i] * (i + 13u) + right[i] * (i + 19u); }
                return result;
            };
            auto run = [&](auto &scheduler, luisa::string_view name) {
                luisa::vector<uint32_t> values(count, 0xdeadbeefu);
                stream << output.copy_from(luisa::span{values});
                stream << scheduler(output).dispatch(count);
                stream << output.copy_to(luisa::span{values}) << synchronize();
                for (uint32_t i = 0u; i < count; ++i) {
                    expect(values[i] == reference(i)) << name << " depth " << depth << " at " << i;
                }
            };
            {
                StateMachineCoroSchedulerConfig config;
                config.shader_option.enable_cache = false;
                StateMachineCoroScheduler scheduler{device, task, config};
                run(scheduler, "state_machine");
            }
            for (bool soa : {false, true}) {
                WavefrontCoroSchedulerConfig config;
                config.thread_count = 17u;
                config.frame_buffer_compaction = true;
                config.global_memory_soa = soa;
                config.shader_option.enable_cache = false;
                WavefrontCoroScheduler scheduler{device, task, config};
                run(scheduler, soa ? "wavefront_soa" : "wavefront_aos");
            }
            {
                PersistentThreadsCoroSchedulerConfig config;
                config.thread_count = 64u;
                config.block_size = 64u;
                config.shader_option.enable_cache = false;
                PersistentThreadsCoroScheduler scheduler{device, task, config};
                run(scheduler, "persistent");
            }
        }
    };
}
