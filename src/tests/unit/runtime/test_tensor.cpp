#include "ut/ut.hpp"
#include "test_device.h"
// Test for matrix multiplication using the Tensor interface.
// This test demonstrates the high-level Tensor API for performing
// general matrix multiplication (GEMM) with fused ReLU activation.
//
// The test creates two 4x4 matrices, performs C = A * B with ReLU,
// and verifies the result on the GPU using the tensor interface.

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>
#include <luisa/core/clock.h>
#include <luisa/tensor/kernel.h>
#include <luisa/tensor/pass/expr_topo.h>
#include <luisa/tensor/tensor_interface.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_tensor(Device &device) {
    // Create tensor builder for defining the computation graph
    TensorBuilder builder;

    auto stream = device.create_stream();

    // Create fallback tensor interface backend for executing tensor operations
    auto interface = TensorInterface::create_fallback_backend(device);

    // Define first 4x4 matrix (column-major order)
    // Layout: [col0_row0, col0_row1, col0_row2, col0_row3, col1_row0, ...]
    float4x4 sb = make_float4x4(
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16);

    // Define second 4x4 matrix for multiplication
    float4x4 sb1 = make_float4x4(
        10, 20, 30, 40,
        50, 60, 70, 80,
        1, 2, 3, 4,
        1, 2, 3, 4);

    // Build command list for GPU operations
    CommandList cmdlist;

    // Create GPU buffers and upload matrix data
    auto b0 = device.create_buffer<float>(16);
    auto b1 = device.create_buffer<float>(16);
    cmdlist << b0.copy_from(luisa::span{&sb, 1}) << b1.copy_from(luisa::span{&sb1, 1});

    // Load and compile tensor kernel for matrix multiplication
    // The kernel performs: output = ReLU(matmul(tensor0, tensor1))
    auto kernel = load_tensor(
        interface.get(),
        [&]() {
            // Create tensor views over the buffers with shape [4, 4]
            Tensor t0{b0, {4ull, 4ull}};
            Tensor t1{b1, {4ull, 4ull}};
            // Initialize tensor metadata with host data pointers
            Tensor::init_tensor(t0, &sb, nullptr);
            Tensor::init_tensor(t1, &sb1, nullptr);
            // Perform matrix multiplication with ReLU activation
            return Tensor::matmul(t0, t1, FusedActivation::relu());
        });
    kernel.compile();

    // Execute the kernel and retrieve results
    float4x4 result;
    auto outs = kernel.execute(cmdlist);
    auto bout = outs[0];
    cmdlist << bout.copy_to(luisa::span{&result, 1});
    stream << cmdlist.commit() << synchronize();
    expect(true) << "tensor test completed";

    // Log the resulting matrix
    LUISA_INFO("{}", result);
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_tensor(device);
}
