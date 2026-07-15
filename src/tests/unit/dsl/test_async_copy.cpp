// Test for CallOp::ASYNC_COPY through the C++ DSL.
// This test covers:
// - Building an ASYNC_COPY builtin call in a DSL kernel
// - Dispatching the kernel on a real backend
// - Verifying the copied buffer contents on the host

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/ast/function_builder.h>
#include <luisa/ast/op.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Mirrors the signature of __builtin_spirv_group_async_copy in
// D:/DirectXShaderCompiler/tools/clang/test/CodeGenSPIRV/spv.async-copy.hlsl:
//   uint __builtin_spirv_group_async_copy(
//       uint execution_scope,
//       [[vk::ext_reference]] inout uint destination,
//       [[vk::ext_reference]] in uint source,
//       uint element_num_bytes,
//       uint num_elements,
//       uint stride,
//       uint event);
// Returns the new event token (uint) produced by OpUntypedGroupAsyncCopyKHR.
[[nodiscard]] auto async_copy(
    Expr<uint> execution_scope,
    Expr<uint> destination,
    Expr<uint> source,
    Expr<uint> element_num_bytes,
    Expr<uint> num_elements,
    Expr<uint> stride,
    Expr<uint> event) noexcept {
    auto fb = luisa::compute::detail::FunctionBuilder::current();
    return def<uint>(fb->call(
        Type::of<uint>(), CallOp::ASYNC_COPY,
        {execution_scope.expression(),
         destination.expression(),
         source.expression(),
         element_num_bytes.expression(),
         num_elements.expression(),
         stride.expression(),
         event.expression()}));
}

}// namespace

int test_async_copy(Device &device) {
    constexpr uint N = 64u;

    Stream stream = device.create_stream();
    Buffer<uint> src = device.create_buffer<uint>(N);
    Buffer<uint> dst = device.create_buffer<uint>(N);

    luisa::vector<uint> host_src(N);
    for (auto i = 0u; i < N; ++i) {
        host_src[i] = i;
    }
    stream << src.copy_from(luisa::span{host_src});
    stream << synchronize();

    Kernel1D kernel = [&](BufferVar<uint> src_buf, BufferVar<uint> dst_buf) noexcept {
        set_block_size(N, 1u, 1u);
        Shared<uint> s_src{N};
        Shared<uint> s_dst{N};

        auto tid = thread_x();
        s_src[tid] = src_buf.read(dispatch_x());
        sync_block();

        $if (tid == 0u) {
            [[maybe_unused]] auto new_event = async_copy(
                /* execution_scope   */ 2u,
                /* destination       */ s_dst[0u],
                /* source            */ s_src[0u],
                /* element_num_bytes */ 4u,
                /* num_elements      */ N,
                /* stride            */ 4u,
                /* event             */ 0u);
        };

        sync_block();
        dst_buf.write(dispatch_x(), s_dst[tid]);
    };

    // Sanity-check that the kernel references the ASYNC_COPY builtin.
    Function f{kernel.function().get()};
    expect(f.propagated_builtin_callables().test(CallOp::ASYNC_COPY));

    auto shader = device.compile(kernel);
    stream << shader(src, dst).dispatch(N);
    stream << synchronize();

    luisa::vector<uint> host_dst(N);
    stream << dst.copy_to(luisa::span{host_dst});
    stream << synchronize();

    bool all_correct = true;
    for (auto i = 0u; i < N; ++i) {
        if (host_dst[i] != host_src[i]) {
            LUISA_WARNING("async_copy mismatch at [{}]: got {} expected {}", i, host_dst[i], host_src[i]);
            all_correct = false;
        }
    }
    expect(all_correct) << "ASYNC_COPY kernel result must match the source buffer";

    return 0;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_async_copy(device);
}
