#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <luisa/runtime/rhi/command_encoder.h>

namespace py = pybind11;
using namespace luisa;
using namespace luisa::compute;
constexpr auto pyref = py::return_value_policy::reference;// object lifetime is managed on C++ side
void export_commands(py::module &m) {
    // commands
    py::class_<Command>(m, "Command");
    py::class_<ShaderDispatchCommand, Command>(m, "ShaderDispatchCommand");
    py::class_<ComputeDispatchCmdEncoder>(m, "ComputeDispatchCmdEncoder")
        .def_static(
            "create", [](size_t arg_size, uint64_t handle, Function func) {
                auto uniform_size = ComputeDispatchCmdEncoder::compute_uniform_size(func.arguments());
                return make_unique<ComputeDispatchCmdEncoder>(handle, arg_size, uniform_size).release();
            },
            pyref)
        .def("set_dispatch_size", [](ComputeDispatchCmdEncoder &self, uint32_t sx, uint32_t sy, uint32_t sz) { self.set_dispatch_size(uint3{sx, sy, sz}); })
        .def("set_dispatch_buffer", [](ComputeDispatchCmdEncoder &self, uint64_t handle, uint32_t offset, uint32_t size) { self.set_dispatch_size(IndirectDispatchArg{handle, offset, size}); })
        .def("encode_buffer", &ComputeDispatchCmdEncoder::encode_buffer)
        .def("encode_texture", &ComputeDispatchCmdEncoder::encode_texture)
        .def("encode_uniform", [](ComputeDispatchCmdEncoder &self, char *buf, size_t size) { self.encode_uniform(buf, size); })
        .def("encode_bindless_array", &ComputeDispatchCmdEncoder::encode_bindless_array)
        .def("encode_accel", &ComputeDispatchCmdEncoder::encode_accel)
        .def(
            "build", [](ComputeDispatchCmdEncoder &c) { return std::move(c).build().release(); }, pyref);
    // buffer operation commands
    // Pybind can't deduce argument list of the create function, so using lambda to inform it
    py::class_<BufferUploadCommand, Command>(m, "BufferUploadCommand")
        .def_static(
            "create", [](uint64_t handle, size_t offset_bytes, size_t size_bytes, py::buffer const &buf) {
                return luisa::make_unique<BufferUploadCommand>(handle, offset_bytes, size_bytes, buf.request().ptr).release();
            },
            pyref);
    py::class_<BufferDownloadCommand, Command>(m, "BufferDownloadCommand")
        .def_static(
            "create", [](uint64_t handle, size_t offset_bytes, size_t size_bytes, py::buffer const &buf) {
                return luisa::make_unique<BufferDownloadCommand>(handle, offset_bytes, size_bytes, buf.request().ptr).release();
            },
            pyref);
    py::class_<BufferCopyCommand, Command>(m, "BufferCopyCommand")
        .def_static(
            "create", [](uint64_t src, uint64_t dst, size_t src_offset, size_t dst_offset, size_t size) {
                return luisa::make_unique<BufferCopyCommand>(src, dst, src_offset, dst_offset, size).release();
            },
            pyref);
    // texture operation commands
    py::class_<TextureUploadCommand, Command>(m, "TextureUploadCommand")
        .def_static(
            "create", [](uint64_t handle, PixelStorage storage, uint32_t level, uint3 size, py::buffer const &buf) {
                return luisa::make_unique<TextureUploadCommand>(handle, storage, level, size, buf.request().ptr).release();
            },
            pyref);
    py::class_<TextureDownloadCommand, Command>(m, "TextureDownloadCommand")
        .def_static(
            "create", [](uint64_t handle, PixelStorage storage, uint32_t level, uint3 size, py::buffer const &buf) {
                return luisa::make_unique<TextureDownloadCommand>(handle, storage, level, size, buf.request().ptr).release();
            },
            pyref);
    py::class_<TextureCopyCommand, Command>(m, "TextureCopyCommand")
        .def_static(
            "create", [](PixelStorage storage, uint64_t src_handle, uint64_t dst_handle, uint32_t src_level, uint32_t dst_level, uint3 size) {
                return luisa::make_unique<TextureCopyCommand>(storage, src_handle, dst_handle, src_level, dst_level, size).release();
            },
            pyref);
    py::class_<BufferToTextureCopyCommand, Command>(m, "BufferToTextureCopyCommand")
        .def_static(
            "create", [](uint64_t buffer, size_t buffer_offset, uint64_t texture, PixelStorage storage, uint32_t level, uint3 size) {
                return luisa::make_unique<BufferToTextureCopyCommand>(buffer, buffer_offset, texture, storage, level, size).release();
            },
            pyref);
    py::class_<TextureToBufferCopyCommand, Command>(m, "TextureToBufferCopyCommand")
        .def_static(
            "create", [](uint64_t buffer, size_t buffer_offset, uint64_t texture, PixelStorage storage, uint32_t level, uint3 size) {
                return luisa::make_unique<TextureToBufferCopyCommand>(buffer, buffer_offset, texture, storage, level, size).release();
            },
            pyref);
}
