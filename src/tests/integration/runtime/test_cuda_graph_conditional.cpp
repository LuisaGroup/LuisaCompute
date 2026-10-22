// Tests for the conditional-node API of the CUDA Graph extension
// (`luisa/backends/ext/cuda/cuda_graph_ext.h`, Phase 2 / SA-2):
//
// - create_conditional_handle / add_conditional_node / build_body_graph
//   compose the CUDA 12.3+ IF/WHILE/SWITCH container nodes;
// - a WHILE loop runs the exact deterministic trip count, once with do-while
//   semantics (a non-zero launch default is assigned) and once as a true
//   while-loop whose condition is seeded by a kernel inside the parent graph
//   (zero-trip and positive-trip variants);
// - an IF/ELSE runs the branch selected by a kernel-written handle value;
// - SWITCH runs the value-th body (CUDA 12.8+), or is refused loudly on older
//   toolkits;
// - the conditional-body restrictions are enforced loudly: a texture command
//   lowers to a CUDA-array memcpy, which is forbidden inside a body, and must
//   be refused before any node is created (so the body stays reusable);
// - whole-graph update() on a graph containing a conditional fails up-front
//   with GraphUpdateErrorType::UNSUPPORTED (its topology is frozen at
//   instantiation).
//
// A note on handle plumbing. `create_graph` builds a whole graph from a
// CommandList in one shot, so a *parent-graph* kernel that writes the condition
// cannot embed the handle (it does not exist yet). The tests therefore pass the
// handle through a device buffer that the host fills between instantiate() and
// launch(), which is the only expressible form of the "preceded by a kernel
// that writes the handle" pattern of research doc 2.3. Conditional *body*
// kernels are built after the handle exists and embed it directly.
//
// Requires the CUDA backend (the extension only exists there); any other
// backend silently skips the test. Everything is additionally guarded by
// `capabilities()` so the test never assumes a driver/toolkit feature the
// machine lacks.
#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/stream.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr uint32_t kCount = 64u;

/// The extension refuses empty command lists, so every graph built here starts
/// from a host -> device upload of `data` into `view`.
[[nodiscard]] CudaGraphInstance make_upload_graph(CudaGraphExt *ext, BufferView<uint32_t> view,
                                                  const luisa::vector<uint32_t> &data) {
    auto cmdlist = CommandList::create();
    cmdlist << view.copy_from(luisa::span{data});
    return ext->create_graph(cmdlist.commit().command_list());
}

/// Fills a device buffer (holding an opaque conditional handle) from the host.
template<typename T>
void write_host(Stream &stream, BufferView<T> view, const luisa::vector<T> &data) {
    stream << view.copy_from(luisa::span{data}) << synchronize();
}

/// Reads a device buffer back to the host.
template<typename T>
[[nodiscard]] luisa::vector<T> read_back(Stream &stream, BufferView<T> view, size_t count) {
    luisa::vector<T> data(count);
    stream << view.copy_to(luisa::span{data}) << synchronize();
    return data;
}

void test_capabilities(CudaGraphExt *ext) {
    auto caps = ext->capabilities();
    // The backend itself only builds against CUDA >= 12.1.
    expect(caps.cuda_version >= 12010u)
        << "capabilities().cuda_version must report the toolkit version";
    if (caps.conditional_nodes) {
        expect(caps.cuda_version >= 12030u) << "conditional nodes need CUDA >= 12.3";
    }
    if (caps.switch_conditional) {
        expect(caps.cuda_version >= 12080u) << "switch conditions need CUDA >= 12.8";
        expect(caps.conditional_nodes) << "SWITCH implies conditional nodes";
    }
    LUISA_INFO("CudaGraphExt conditional capabilities: cuda_version = {}, "
               "conditional_nodes = {}, switch_conditional = {}.",
               caps.cuda_version, caps.conditional_nodes, caps.switch_conditional);
}

/// A graph with a single upload node followed by a wired WHILE conditional
/// node. `handle_out` receives the conditional handle, `body_out` the (empty)
/// body graph the caller populates with build_body_graph() and
/// `conditional_index_out` the conditional node's creation index. All outputs
/// are left invalid when the setup fails; the caller checks them.
[[nodiscard]] CudaGraphInstance make_while_graph(
    CudaGraphExt *ext, BufferView<uint32_t> view, const luisa::vector<uint32_t> &data,
    CudaGraphExt::ConditionalHandle *handle_out, CudaGraphExt::GraphHandle *body_out,
    size_t *conditional_index_out) {
    *handle_out = 0u;
    *body_out = CudaGraphExt::invalid_handle;
    *conditional_index_out = ~size_t{0u};
    auto graph = make_upload_graph(ext, view, data);
    auto graph_handle = graph.handle().handle;
    if (graph_handle == CudaGraphExt::invalid_handle) { return graph; }
    // Default 0: the loop is zero-trip unless the body re-arms it, so these
    // helper graphs can never spin forever even if they are launched.
    auto handle = ext->create_conditional_handle(graph_handle, 0u, true);
    if (handle == 0u) { return graph; }
    luisa::vector<CudaGraphExt::GraphHandle> bodies(1u, CudaGraphExt::invalid_handle);
    size_t node_index = ~size_t{0u};
    if (!ext->add_conditional_node(graph_handle, CudaGraphExt::ConditionalType::WHILE,
                                   handle, 1u, luisa::span{bodies}, &node_index) ||
        !ext->add_dependency(graph_handle, 0u, node_index)) {
        return graph;
    }
    *handle_out = handle;
    *body_out = bodies[0];
    *conditional_index_out = node_index;
    return graph;
}

/// WHILE with do-while semantics: the handle carries the launch default 1, so
/// the body runs at least once. The body decrements `counter`, counts its own
/// executions and re-arms the condition until the counter reaches the target,
/// which makes the trip count deterministic (counter 5 -> 2 = 3 executions).
void test_while_do_while(Device &device, CudaGraphExt *ext) {
    auto stream = device.create_stream();

    Kernel1D countdown = [](BufferVar<uint32_t> counter, BufferVar<uint32_t> iterations,
                            Var<uint64_t> handle, Var<uint32_t> target) noexcept {
        set_block_size(32u);
        $if (thread_id().x == 0u) {
            auto next = counter.read(0u) - 1u;
            counter.write(0u, next);
            iterations.write(0u, iterations.read(0u) + 1u);
            set_graph_conditional(handle, (next > target).cast<uint>());
        };
    };
    auto shader = device.compile(countdown);

    auto counter = device.create_buffer<uint32_t>(1u);
    auto iterations = device.create_buffer<uint32_t>(1u);
    luisa::vector<uint32_t> initial_counter{5u};
    luisa::vector<uint32_t> initial_iterations{0u};
    auto cmdlist = CommandList::create();
    cmdlist << counter.view().copy_from(luisa::span{initial_counter});
    cmdlist << iterations.view().copy_from(luisa::span{initial_iterations});
    auto graph = ext->create_graph(cmdlist.commit().command_list());
    expect(graph.handle().handle != CudaGraphExt::invalid_handle)
        << "create_graph from two uploads should succeed";
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto graph_handle = graph.handle().handle;

    auto handle = ext->create_conditional_handle(graph_handle, 1u, true);
    expect(handle != 0u) << "create_conditional_handle should succeed on CUDA >= 12.3";
    if (handle == 0u) { return; }

    luisa::vector<CudaGraphExt::GraphHandle> bodies(1u, CudaGraphExt::invalid_handle);
    size_t conditional_index = ~size_t{0u};
    expect(ext->add_conditional_node(graph_handle, CudaGraphExt::ConditionalType::WHILE,
                                     handle, 1u, luisa::span{bodies}, &conditional_index))
        << "add_conditional_node(WHILE) should succeed";
    expect(conditional_index == 2u) << "the WHILE node keeps its creation order";
    expect(bodies[0] != CudaGraphExt::invalid_handle) << "CUDA must hand back a body graph";
    if (bodies[0] == CudaGraphExt::invalid_handle) { return; }
    expect(ext->add_dependency(graph_handle, 0u, conditional_index))
        << "the WHILE node must be wired after the counter upload";
    expect(ext->add_dependency(graph_handle, 1u, conditional_index))
        << "the WHILE node must be wired after the iteration upload";

    auto body_cmdlist = CommandList::create();
    body_cmdlist << shader(counter.view(), iterations.view(), handle, 2u).dispatch(1u);
    expect(ext->build_body_graph(bodies[0], body_cmdlist.commit().command_list()))
        << "build_body_graph should accept a single-kernel body";

    auto exec = ext->instantiate(graph_handle);
    expect(exec.handle().handle != CudaGraphExt::invalid_handle) << "instantiate should succeed";
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }

    // The assigned default is re-applied on every launch, so a replay must
    // reproduce the same trip count.
    for (auto replay = 0; replay < 2; replay++) {
        ext->launch(exec.handle().handle, stream.handle());
        stream << synchronize();
        auto final_counter = read_back(stream, counter.view(), 1u);
        auto final_iterations = read_back(stream, iterations.view(), 1u);
        expect(final_counter[0] == 2u)
            << "the WHILE body must run until the counter reaches the target";
        expect(final_iterations[0] == 3u)
            << "counter 5 -> 2 is exactly 3 body executions";
    }
}

/// Zero-trip WHILE: the assigned default is 0, so the body never runs.
void test_while_zero_trip(Device &device, CudaGraphExt *ext) {
    auto stream = device.create_stream();

    Kernel1D bump = [](BufferVar<uint32_t> iterations) noexcept {
        set_block_size(32u);
        $if (thread_id().x == 0u) {
            iterations.write(0u, iterations.read(0u) + 1u);
        };
    };
    auto shader = device.compile(bump);

    auto iterations = device.create_buffer<uint32_t>(1u);
    luisa::vector<uint32_t> initial_iterations{0u};
    auto graph = make_upload_graph(ext, iterations.view(), initial_iterations);
    expect(graph.handle().handle != CudaGraphExt::invalid_handle);
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto graph_handle = graph.handle().handle;

    auto handle = ext->create_conditional_handle(graph_handle, 0u, true);
    expect(handle != 0u);
    if (handle == 0u) { return; }
    luisa::vector<CudaGraphExt::GraphHandle> bodies(1u, CudaGraphExt::invalid_handle);
    size_t conditional_index = ~size_t{0u};
    expect(ext->add_conditional_node(graph_handle, CudaGraphExt::ConditionalType::WHILE,
                                     handle, 1u, luisa::span{bodies}, &conditional_index))
        << "add_conditional_node(WHILE) with a zero default should succeed";
    if (bodies[0] == CudaGraphExt::invalid_handle) { return; }
    expect(ext->add_dependency(graph_handle, 0u, conditional_index));

    auto body_cmdlist = CommandList::create();
    body_cmdlist << shader(iterations.view()).dispatch(1u);
    expect(ext->build_body_graph(bodies[0], body_cmdlist.commit().command_list()));

    auto exec = ext->instantiate(graph_handle);
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }
    ext->launch(exec.handle().handle, stream.handle());
    stream << synchronize();
    auto final_iterations = read_back(stream, iterations.view(), 1u);
    expect(final_iterations[0] == 0u)
        << "a WHILE whose launch default is 0 must not execute its body";
}

/// True while-loop whose condition is written by a kernel inside the parent
/// graph: the kernel loads the (host-supplied) handle from a device buffer and
/// seeds the condition from the counter, so the body runs until the counter
/// reaches the target. This is the "preceded by a kernel that writes the
/// handle" pattern of research doc 2.3.
void test_while_kernel_seeded(Device &device, CudaGraphExt *ext) {
    auto stream = device.create_stream();

    Kernel1D seed = [](BufferVar<uint64_t> handle_buf, BufferVar<uint32_t> source) noexcept {
        set_block_size(32u);
        $if (thread_id().x == 0u) {
            set_graph_conditional(handle_buf.read(0u), source.read(0u));
        };
    };
    auto seed_shader = device.compile(seed);
    Kernel1D countdown = [](BufferVar<uint32_t> counter, BufferVar<uint32_t> iterations,
                            Var<uint64_t> handle, Var<uint32_t> target) noexcept {
        set_block_size(32u);
        $if (thread_id().x == 0u) {
            auto next = counter.read(0u) - 1u;
            counter.write(0u, next);
            iterations.write(0u, iterations.read(0u) + 1u);
            set_graph_conditional(handle, (next > target).cast<uint>());
        };
    };
    auto countdown_shader = device.compile(countdown);

    auto counter = device.create_buffer<uint32_t>(1u);
    auto iterations = device.create_buffer<uint32_t>(1u);
    auto handle_buf = device.create_buffer<uint64_t>(1u);
    luisa::vector<uint32_t> initial_counter{5u};
    luisa::vector<uint32_t> initial_iterations{0u};
    auto cmdlist = CommandList::create();
    cmdlist << counter.view().copy_from(luisa::span{initial_counter});
    cmdlist << iterations.view().copy_from(luisa::span{initial_iterations});
    cmdlist << seed_shader(handle_buf.view(), counter.view()).dispatch(1u);
    auto graph = ext->create_graph(cmdlist.commit().command_list());
    expect(graph.handle().handle != CudaGraphExt::invalid_handle)
        << "create_graph with a condition-seeding kernel should succeed";
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto graph_handle = graph.handle().handle;

    // No assigned default: the seed kernel (wired before the WHILE node) is the
    // only writer of the condition.
    auto handle = ext->create_conditional_handle(graph_handle, 0u, false);
    expect(handle != 0u) << "create_conditional_handle without a default should succeed";
    if (handle == 0u) { return; }

    luisa::vector<CudaGraphExt::GraphHandle> bodies(1u, CudaGraphExt::invalid_handle);
    size_t conditional_index = ~size_t{0u};
    expect(ext->add_conditional_node(graph_handle, CudaGraphExt::ConditionalType::WHILE,
                                     handle, 1u, luisa::span{bodies}, &conditional_index))
        << "add_conditional_node(WHILE) should succeed";
    expect(conditional_index == 3u)
        << "the WHILE node follows the two uploads and the seed kernel";
    if (bodies[0] == CudaGraphExt::invalid_handle) { return; }
    expect(ext->add_dependency(graph_handle, 2u, conditional_index))
        << "the WHILE node must be wired after the seed kernel";
    expect(ext->add_dependency(graph_handle, 1u, conditional_index))
        << "the WHILE node must be wired after the iteration upload";

    // The body is built after the handle exists, so it embeds the handle value.
    auto body_cmdlist = CommandList::create();
    body_cmdlist << countdown_shader(counter.view(), iterations.view(), handle, 2u).dispatch(1u);
    expect(ext->build_body_graph(bodies[0], body_cmdlist.commit().command_list()));

    auto exec = ext->instantiate(graph_handle);
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }
    // The seed kernel reads the opaque handle from device memory; hand it over
    // now that the graph has created it, before the launch that consumes it.
    luisa::vector<uint64_t> handle_value{handle};
    write_host(stream, handle_buf.view(), handle_value);

    ext->launch(exec.handle().handle, stream.handle());
    stream << synchronize();
    auto final_counter = read_back(stream, counter.view(), 1u);
    auto final_iterations = read_back(stream, iterations.view(), 1u);
    expect(final_counter[0] == 2u)
        << "the kernel-seeded WHILE body must run until the counter reaches the target";
    expect(final_iterations[0] == 3u)
        << "counter 5 -> 2 is exactly 3 body executions";
}

/// IF with an ELSE body: body[0] runs when the kernel-written handle is
/// non-zero (adds 1), body[1] otherwise (adds 100). The branch value comes from
/// an uploaded flag that the seeding kernel forwards into the condition.
void test_if_else(Device &device, CudaGraphExt *ext) {
    auto stream = device.create_stream();

    Kernel1D seed = [](BufferVar<uint64_t> handle_buf, BufferVar<uint32_t> flag) noexcept {
        set_block_size(32u);
        $if (thread_id().x == 0u) {
            set_graph_conditional(handle_buf.read(0u), flag.read(0u));
        };
    };
    auto seed_shader = device.compile(seed);
    Kernel1D add = [](BufferVar<uint32_t> result, Var<uint32_t> amount) noexcept {
        set_block_size(32u);
        $if (thread_id().x == 0u) {
            result.write(0u, result.read(0u) + amount);
        };
    };
    auto add_shader = device.compile(add);

    auto run_if = [&](uint32_t flag_value, uint32_t expected_result) {
        auto flag = device.create_buffer<uint32_t>(1u);
        auto result = device.create_buffer<uint32_t>(1u);
        auto handle_buf = device.create_buffer<uint64_t>(1u);
        luisa::vector<uint32_t> initial_flag{flag_value};
        luisa::vector<uint32_t> initial_result{0u};
        auto cmdlist = CommandList::create();
        cmdlist << result.view().copy_from(luisa::span{initial_result});
        cmdlist << flag.view().copy_from(luisa::span{initial_flag});
        cmdlist << seed_shader(handle_buf.view(), flag.view()).dispatch(1u);
        auto graph = ext->create_graph(cmdlist.commit().command_list());
        expect(graph.handle().handle != CudaGraphExt::invalid_handle);
        if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
        auto graph_handle = graph.handle().handle;

        auto handle = ext->create_conditional_handle(graph_handle, 0u, false);
        expect(handle != 0u);
        if (handle == 0u) { return; }
        luisa::vector<CudaGraphExt::GraphHandle> bodies(2u, CudaGraphExt::invalid_handle);
        size_t conditional_index = ~size_t{0u};
        expect(ext->add_conditional_node(graph_handle, CudaGraphExt::ConditionalType::IF,
                                         handle, 2u, luisa::span{bodies}, &conditional_index))
            << "add_conditional_node(IF, 2 bodies) should succeed";
        expect(conditional_index == 3u) << "the IF node keeps its creation order";
        if (bodies[0] == CudaGraphExt::invalid_handle ||
            bodies[1] == CudaGraphExt::invalid_handle) {
            return;
        }
        expect(ext->add_dependency(graph_handle, 2u, conditional_index))
            << "the IF node must be wired after the seed kernel";
        expect(ext->add_dependency(graph_handle, 0u, conditional_index))
            << "the IF node must be wired after the result upload";

        auto then_cmdlist = CommandList::create();
        then_cmdlist << add_shader(result.view(), 1u).dispatch(1u);
        expect(ext->build_body_graph(bodies[0], then_cmdlist.commit().command_list()))
            << "the THEN body should build";
        auto else_cmdlist = CommandList::create();
        else_cmdlist << add_shader(result.view(), 100u).dispatch(1u);
        expect(ext->build_body_graph(bodies[1], else_cmdlist.commit().command_list()))
            << "the ELSE body should build";

        auto exec = ext->instantiate(graph_handle);
        if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }
        luisa::vector<uint64_t> handle_value{handle};
        write_host(stream, handle_buf.view(), handle_value);

        ext->launch(exec.handle().handle, stream.handle());
        stream << synchronize();
        auto final_result = read_back(stream, result.view(), 1u);
        expect(final_result[0] == expected_result)
            << "the IF branch selected by the kernel-written handle value must have run";
    };
    run_if(1u, 1u);   // non-zero -> THEN adds 1
    run_if(0u, 100u); // zero -> ELSE adds 100
}

/// SWITCH (CUDA 12.8+): the value-th body runs. On older toolkits the SWITCH
/// node type must be refused loudly.
void test_switch(Device &device, CudaGraphExt *ext, bool supported) {
    if (!supported) {
        auto flag = device.create_buffer<uint32_t>(1u);
        luisa::vector<uint32_t> initial{1u};
        auto graph = make_upload_graph(ext, flag.view(), initial);
        if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
        auto graph_handle = graph.handle().handle;
        auto handle = ext->create_conditional_handle(graph_handle, 0u, true);
        if (handle == 0u) { return; }
        luisa::vector<CudaGraphExt::GraphHandle> bodies(1u, CudaGraphExt::invalid_handle);
        size_t conditional_index = ~size_t{0u};
        expect(!ext->add_conditional_node(graph_handle, CudaGraphExt::ConditionalType::SWITCH,
                                          handle, 1u, luisa::span{bodies}, &conditional_index))
            << "SWITCH must be refused before CUDA 12.8";
        return;
    }
    auto stream = device.create_stream();

    Kernel1D seed = [](BufferVar<uint64_t> handle_buf, BufferVar<uint32_t> selector) noexcept {
        set_block_size(32u);
        $if (thread_id().x == 0u) {
            set_graph_conditional(handle_buf.read(0u), selector.read(0u));
        };
    };
    auto seed_shader = device.compile(seed);
    Kernel1D add = [](BufferVar<uint32_t> result, Var<uint32_t> amount) noexcept {
        set_block_size(32u);
        $if (thread_id().x == 0u) {
            result.write(0u, result.read(0u) + amount);
        };
    };
    auto add_shader = device.compile(add);

    auto flag = device.create_buffer<uint32_t>(1u);
    auto result = device.create_buffer<uint32_t>(1u);
    auto handle_buf = device.create_buffer<uint64_t>(1u);
    luisa::vector<uint32_t> initial_flag{1u};// condition value 1 -> body[1]
    luisa::vector<uint32_t> initial_result{0u};
    auto cmdlist = CommandList::create();
    cmdlist << result.view().copy_from(luisa::span{initial_result});
    cmdlist << flag.view().copy_from(luisa::span{initial_flag});
    cmdlist << seed_shader(handle_buf.view(), flag.view()).dispatch(1u);
    auto graph = ext->create_graph(cmdlist.commit().command_list());
    if (graph.handle().handle == CudaGraphExt::invalid_handle) { return; }
    auto graph_handle = graph.handle().handle;

    auto handle = ext->create_conditional_handle(graph_handle, 0u, false);
    if (handle == 0u) { return; }
    luisa::vector<CudaGraphExt::GraphHandle> bodies(3u, CudaGraphExt::invalid_handle);
    size_t conditional_index = ~size_t{0u};
    expect(ext->add_conditional_node(graph_handle, CudaGraphExt::ConditionalType::SWITCH,
                                     handle, 3u, luisa::span{bodies}, &conditional_index))
        << "add_conditional_node(SWITCH, 3 bodies) should succeed on CUDA >= 12.8";
    if (bodies[0] == CudaGraphExt::invalid_handle ||
        bodies[1] == CudaGraphExt::invalid_handle ||
        bodies[2] == CudaGraphExt::invalid_handle) {
        return;
    }
    // A SWITCH whose output array is too small must still be refused.
    {
        luisa::vector<CudaGraphExt::GraphHandle> too_small(1u, CudaGraphExt::invalid_handle);
        size_t rejected_index = ~size_t{0u};
        expect(!ext->add_conditional_node(graph_handle, CudaGraphExt::ConditionalType::SWITCH,
                                          handle, 3u, luisa::span{too_small}, &rejected_index))
            << "a SWITCH whose body_graphs_out is too small must be refused";
    }
    expect(ext->add_dependency(graph_handle, 2u, conditional_index));
    expect(ext->add_dependency(graph_handle, 0u, conditional_index));

    for (auto i = 0u; i < bodies.size(); i++) {
        auto body_cmdlist = CommandList::create();
        body_cmdlist << add_shader(result.view(), (i + 1u) * 1000u).dispatch(1u);
        expect(ext->build_body_graph(bodies[i], body_cmdlist.commit().command_list()))
            << "each SWITCH body should build";
    }

    auto exec = ext->instantiate(graph_handle);
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }
    luisa::vector<uint64_t> handle_value{handle};
    write_host(stream, handle_buf.view(), handle_value);

    ext->launch(exec.handle().handle, stream.handle());
    stream << synchronize();
    auto final_result = read_back(stream, result.view(), 1u);
    expect(final_result[0] == 2000u) << "value 1 must select body[1] (which adds 2000)";
}

/// The conditional-body restrictions must be enforced loudly, and a refused
/// build must leave the body empty and reusable.
void test_body_restrictions(Device &device, CudaGraphExt *ext) {
    auto counter = device.create_buffer<uint32_t>(kCount);
    luisa::vector<uint32_t> initial(kCount, 1u);
    CudaGraphExt::ConditionalHandle handle = 0u;
    CudaGraphExt::GraphHandle body = CudaGraphExt::invalid_handle;
    size_t conditional_index = ~size_t{0u};
    auto graph = make_while_graph(ext, counter.view(), initial, &handle, &body, &conditional_index);
    if (graph.handle().handle == CudaGraphExt::invalid_handle ||
        handle == 0u || body == CudaGraphExt::invalid_handle) {
        return;
    }
    // A top-level graph handle (or anything that is not a conditional body) is
    // refused.
    {
        auto not_a_body = CommandList::create();
        not_a_body << counter.view().copy_from(luisa::span{initial});
        expect(!ext->build_body_graph(graph.handle().handle, not_a_body.commit().command_list()))
            << "build_body_graph must refuse a non-body graph handle";
    }
    // A texture command lowers to a CUDA-array memcpy, which is forbidden in a
    // conditional body: it must be refused before any node is created.
    {
        constexpr uint32_t kSize = 4u;
        auto image = device.create_image<float>(PixelStorage::FLOAT4, make_uint2(kSize, kSize));
        luisa::vector<float> pixels(kSize * kSize * 4u, 1.0f);
        auto texture_cmdlist = CommandList::create();
        texture_cmdlist << image.view(0u).copy_from(luisa::span{pixels});
        expect(!ext->build_body_graph(body, texture_cmdlist.commit().command_list()))
            << "a texture upload must be refused inside a conditional body";
    }
    // The body stays empty and usable after the refusal.
    {
        Kernel1D bump = [](BufferVar<uint32_t> value) noexcept {
            set_block_size(32u);
            $if (thread_id().x == 0u) {
                value.write(0u, value.read(0u) + 1u);
            };
        };
        auto shader = device.compile(bump);
        auto good_cmdlist = CommandList::create();
        good_cmdlist << shader(counter.view()).dispatch(1u);
        expect(ext->build_body_graph(body, good_cmdlist.commit().command_list()))
            << "a refused body build must leave the body empty and reusable";
    }
    // Handle/graph consistency and body-count validation.
    {
        auto other = make_upload_graph(ext, counter.view(), initial);
        if (other.handle().handle != CudaGraphExt::invalid_handle) {
            luisa::vector<CudaGraphExt::GraphHandle> one(1u, CudaGraphExt::invalid_handle);
            size_t rejected_index = ~size_t{0u};
            expect(!ext->add_conditional_node(other.handle().handle,
                                              CudaGraphExt::ConditionalType::WHILE, handle,
                                              1u, luisa::span{one}, &rejected_index))
                << "a handle created for another graph must be refused";
        }
        luisa::vector<CudaGraphExt::GraphHandle> one(1u, CudaGraphExt::invalid_handle);
        size_t rejected_index = ~size_t{0u};
        expect(!ext->add_conditional_node(graph.handle().handle,
                                          CudaGraphExt::ConditionalType::WHILE, handle,
                                          2u, luisa::span{one}, &rejected_index))
            << "WHILE with 2 bodies must be refused";
        expect(!ext->add_conditional_node(graph.handle().handle,
                                          CudaGraphExt::ConditionalType::IF, handle,
                                          3u, luisa::span{one}, &rejected_index))
            << "IF with 3 bodies must be refused";
        expect(ext->create_conditional_handle(CudaGraphExt::invalid_handle, 0u, true) == 0u)
            << "create_conditional_handle on an invalid graph must fail";
    }
}

/// Whole-graph update() on a graph containing a conditional node must fail
/// up-front with UNSUPPORTED: CUDA freezes conditional topology at
/// instantiation.
void test_update_unsupported(Device &device, CudaGraphExt *ext) {
    auto counter = device.create_buffer<uint32_t>(1u);
    luisa::vector<uint32_t> initial{1u};
    CudaGraphExt::ConditionalHandle handle = 0u;
    CudaGraphExt::GraphHandle body = CudaGraphExt::invalid_handle;
    size_t conditional_index = ~size_t{0u};
    auto graph = make_while_graph(ext, counter.view(), initial, &handle, &body, &conditional_index);
    if (graph.handle().handle == CudaGraphExt::invalid_handle || body == CudaGraphExt::invalid_handle) {
        return;
    }
    // Populate the body so the graph is a valid, instantiable conditional graph.
    {
        Kernel1D bump = [](BufferVar<uint32_t> value) noexcept {
            set_block_size(32u);
            $if (thread_id().x == 0u) {
                value.write(0u, value.read(0u) + 1u);
            };
        };
        auto shader = device.compile(bump);
        auto body_cmdlist = CommandList::create();
        body_cmdlist << shader(counter.view()).dispatch(1u);
        expect(ext->build_body_graph(body, body_cmdlist.commit().command_list()));
    }
    auto exec = ext->instantiate(graph.handle().handle);
    expect(exec.handle().handle != CudaGraphExt::invalid_handle)
        << "a graph with a conditional node must instantiate";
    if (exec.handle().handle == CudaGraphExt::invalid_handle) { return; }

    luisa::vector<uint32_t> updated{7u};
    auto new_cmdlist = CommandList::create();
    new_cmdlist << counter.view().copy_from(luisa::span{updated});
    CudaGraphExt::GraphUpdateError error{};
    auto updated_ok = ext->update(exec.handle().handle, new_cmdlist.commit().command_list(), &error);
    expect(!updated_ok)
        << "whole-graph update must fail for a graph containing conditional nodes";
    expect(error.type == CudaGraphExt::GraphUpdateErrorType::UNSUPPORTED)
        << "the failure must be reported as UNSUPPORTED";
    expect(!error.detail.empty()) << "the failure must carry a human-readable detail";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    auto *ext = device.extension<CudaGraphExt>();
    if (ext == nullptr) {
        LUISA_INFO("CudaGraphExt not available, skipping test.");
        return 0;
    }
    test_capabilities(ext);
    auto caps = ext->capabilities();
    if (!caps.conditional_nodes) {
        LUISA_INFO("Conditional CUDA graph nodes require CUDA >= 12.3 "
                   "(toolkit/driver {}), skipping the conditional test.",
                   caps.cuda_version);
        return 0;
    }
    // NOTE (driver robustness, verified on WSL2 + Blackwell, driver 617.42):
    // instantiating a conditional graph AFTER other conditional graphs were
    // built AND launched in the same process intermittently segfaults deep
    // inside libcuda (every scenario below passes in isolation and in the
    // reversed order, and plain accumulation of unlaunched conditional graphs
    // is clean, so this is a driver issue, not an extension bug). The
    // not-launched update_unsupported graph therefore runs FIRST, before any
    // conditional graph launch, which is empirically stable.
    test_update_unsupported(device, ext);
    test_while_do_while(device, ext);
    test_while_zero_trip(device, ext);
    test_while_kernel_seeded(device, ext);
    test_if_else(device, ext);
    test_switch(device, ext, caps.switch_conditional);
    test_body_restrictions(device, ext);
    LUISA_INFO("CUDA graph conditional node test passed.");
    return 0;
}
