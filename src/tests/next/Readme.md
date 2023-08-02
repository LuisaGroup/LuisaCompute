# Next-Test Framework

The Next Stage Test Framework will be 

- built on `doctest`
- on branch: `next-test`

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

<!-- Hot Updating -->
ir is updating...
- [ ] ast2ir_headless
- [ ] ast2ir_ir2ast
- [ ] ast2ir
- [ ] autodiff
- [ ] autodiff_full
- [ ] path_tracing_ir
### Test

##### test_feat
suite "common"
- [x] context: info output the installed backends, make sure there is at least one backend available
- [x] ast: the simplest test case, write 42 in position 1 in a buffer of length 10
- [x] callable: a simple add callable test
- [ ] thread_pool
- [ ] type
- [ ] runtime
- [ ] command_reorder
- [ ] copy
- [ ] dml
- [ ] dstorage_decompression
- [ ] dx_supersampling
- [ ] indirect
- [ ] kernel_ir
- [ ] mipmap
- [ ] native_include
- [ ] normal_encoding
- [ ] shader_visuals_present
- [ ] shared_memory
suite "dsl"
- [ ] dsl
- [ ] dsl_multithread
- [ ] dsl_sugar
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

##### test_io


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

