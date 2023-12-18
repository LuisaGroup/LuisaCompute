# Next-Test Framework

The Next Stage Test Framework will be 

- built on `doctest`
- on branch: `next-test`

WIP: currently next-test only support xmake build sys, CMAKE is in progress.

`src/tests/xmake.lua`

dummy test is not required, we can just start for 
- main feature -> `test_feat`
- with io (e.g. stb-image) -> `test_with_io`
- with window (enable_gui) -> `test_with_gui`
- all test suite -> `test_all`
- all examples -> `examples`

and we can write a python script / lua script to output the report


## Test Cases

a typical test cases can be written like this:

```cpp
TEST_SUITE("example") {
    TEST_CASE("write_image") {
        Context context{luisa::test::argv()[0]};
        const luisa::vector<luisa::string> device_names = {"dx", "cuda"};
        for (auto &device_name: device_names) {
            SUBCASE(device_name.c_str()) {
                Device device = context.create_device(device_name.c_str());
                REQUIRE(luisa::test::write_image(device) == 0);
            }
        }
    }
}
```

Then you can filter what you want to run by setting

`xmake run <target> -ts=<test_suite> -tc=<test_case> -sc=<sub_case>`

e.g. if you want to test the 'write_image' example on dx device, you can simply type `xmake run example -tc=write_image -sc=dx` after building successfully.

Then you may found write_image_dx.png in your /bin folder

<!-- WIP -->
- [ ] ast2ir_headless
- [ ] ast2ir_ir2ast
- [ ] ast2ir
- [ ] autodiff
- [ ] autodiff_full
- [ ] path_tracing_ir
### Test

##### test_feat
suite "ast" The Core Feature, Kernel and Shader
- [x] ast_basic: basic ast procedure
suite "dsl" The Semantic
- [x] dsl_calc: calculation
- [x] dsl_callable: DSL callable
- [x] dsl_var: DSL variable
- [x] dsl_matrix_float2x2
- [x] dsl_soa_simple: SOA simple situation

suite "runtime"
- [x] context: info output the installed backends, make sure there is at least one backend available
- [x] buffer
    - [x] buffer_float3x3
    - [x] buffer_float3x3_order
    - [x] buffer_float4x4
    - [x] buffer_float4
    - [x] buffer_float3
    - [x] buffer_float2
- [x] buffer_view
- [x] external_buffer
- [x] device
    - [x] device_create
    - [x] device_wrapped
- [x] shared_memory
- [ ] thread_pool
- [ ] type
- [ ] runtime
- [ ] command_reorder
- [ ] copy
- [ ] dml -> where is CPP_params.txt?
- [ ] indirect
- [ ] kernel_ir
- [ ] mipmap
- [ ] native_include
- [ ] normal_encoding
- [ ] shader_visuals_present

suite "swapchain"
- [ ] swapchain_qt
- [ ] swapchain_static
- [ ] swapchain_wx
- [ ] swapchain
suite "texture"
- [ ] texture_compression
- [ ] texture_io
- [ ] texture_3d
- [ ] sparse_texture
suite "bindless"
- [ ] binding_group
- [ ] bindless_buffer
- [ ] bindless
suite "autodiff"
- [ ] atomic_queue
- [ ] atomic

##### test_ext

The Test Cases for built-in extensions

Core
- [ ] dstorage
- [ ] dstorage_decompression
CUDA
- [ ] cuda-lcub
DX
- [ ] dx_supersampling


### Example

##### target "example_feat"
The minimum examples to show how to use a feature, which is useful in tutorials.
- [x] helloworld: this is an example on how to write an image in parallel and output with stb-image
- [x] printer: -> example/use_printer
- [ ] raster
- [ ] select_device
- [ ] sampler
- [ ] denoiser
##### target "example_gallary"

- [ ] shader_toy
- [ ] game_of_life

suite "fluid_sim"
- [x] mpm3d -> example/gallary/fluid_sim/mpm3d.cpp
- [ ] mpm88
suite "procedure"
- [ ] procedural_callable
- [ ] procedural
suite "renderer"
- [ ] path_tracing_camera
- [ ] path_tracing_cutout
- [x] path_tracing -> example/gallary/render/path_tracer.cpp
- [ ] rtx
- [ ] indirect_rtx
- [ ] photon_mapping
- [ ] sdf_renderer_ir
- [x] sdf_renderer

