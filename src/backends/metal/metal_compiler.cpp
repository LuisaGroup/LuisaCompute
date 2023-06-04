//
// Created by Mike Smith on 2023/4/15.
//

#include <fstream>

#include <core/clock.h>
#include <core/logging.h>
#include <backends/metal/metal_device.h>
#include <backends/metal/metal_compiler.h>

#define LUISA_METAL_BACKEND_DUMP_SOURCE 1

namespace luisa::compute::metal {

namespace detail {

[[nodiscard]] static auto temp_unique_file_path() noexcept {
    std::error_code ec;
    auto temp_dir = std::filesystem::temp_directory_path(ec);
    std::filesystem::path temp_path;
    if (ec) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to find temporary directory: {}.",
            ec.message());
    } else {
        auto uuid = CFUUIDCreate(nullptr);
        auto uuid_string = CFUUIDCreateString(nullptr, uuid);
        temp_path = std::filesystem::absolute(
            temp_dir / CFStringGetCStringPtr(uuid_string, kCFStringEncodingUTF8));
        CFRelease(uuid);
        CFRelease(uuid_string);
    }
    return temp_path;
}

}// namespace detail

MetalCompiler::MetalCompiler(const MetalDevice *device) noexcept
    : _device{device}, _cache{max_cache_item_count} {}

void MetalCompiler::_store_disk_archive(luisa::string_view name, bool is_aot,
                                        const PipelineDescriptorHandle &desc,
                                        const MetalShaderMetadata &metadata) const noexcept {

    // create a binary archive
    NS::Error *error = nullptr;
    auto archive_desc = MTL::BinaryArchiveDescriptor::alloc()->init();
    auto archive = NS::TransferPtr(_device->handle()->newBinaryArchive(archive_desc, &error));
    archive_desc->release();
    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal shader "
            "archive for '{}': {}.",
            name, error->localizedDescription()->utf8String());
        return;
    }
    archive->addComputePipelineFunctions(desc.entry.get(), &error);
    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal shader "
            "archive for '{}': {}.",
            name, error->localizedDescription()->utf8String());
        return;
    }
    archive->addComputePipelineFunctions(desc.indirect_entry.get(), &error);
    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal shader (indirect dispatch version) "
            "archive for '{}': {}.",
            name, error->localizedDescription()->utf8String());
        return;
    }

    // dump library
    auto temp_file_path = detail::temp_unique_file_path();
    if (temp_file_path.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal shader "
            "archive for '{}': failed to create temporary file.",
            name);
        return;
    }
    auto url = NS::URL::fileURLWithPath(NS::String::string(
        temp_file_path.c_str(), NS::UTF8StringEncoding));
    archive->serializeToURL(url, &error);
    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal shader "
            "archive for '{}': {}.",
            name, error->localizedDescription()->utf8String());
        return;
    }

    // read the dumped library
    std::error_code ec;
    auto file_size = std::filesystem::file_size(temp_file_path, ec);
    if (ec) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal shader "
            "archive for '{}': {}.",
            name, ec.message());
        return;
    }
    auto metadata_str = serialize_metal_shader_metadata(metadata);
    auto metadata_size = metadata_str.size();
    luisa::vector<std::byte> buffer;
    buffer.resize(sizeof(size_t) + metadata_size + file_size);
    std::memcpy(buffer.data(), &metadata_size, sizeof(size_t));
    std::memcpy(buffer.data() + sizeof(size_t), metadata_str.data(), metadata_size);
    std::ifstream file{temp_file_path, std::ios::binary};
    if (!file.is_open()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal shader "
            "archive for '{}': failed to open temporary file.",
            name);
        return;
    }
    file.read(reinterpret_cast<char *>(buffer.data() + sizeof(size_t) + metadata_size),
              static_cast<ssize_t>(file_size));
    file.close();

    // store the binary archive
    auto io = _device->io();
    if (is_aot) {
        io->write_shader_bytecode(name, buffer);
    } else {
        io->write_shader_cache(name, buffer);
    }
}

MetalShaderHandle
MetalCompiler::_load_disk_archive(luisa::string_view name, bool is_aot,
                                  MetalShaderMetadata &metadata) const noexcept {

    Clock clk;

    // open file stream
    auto io = _device->io();
    auto stream = is_aot ? io->read_shader_bytecode(name) :
                           io->read_shader_cache(name);
    if (stream == nullptr || stream->length() == 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load Metal shader "
            "archive for '{}': file not found.",
            name);
        return {};
    }

    // load data
    luisa::vector<std::byte> buffer(stream->length());
    stream->read(buffer);
    stream.reset();

    // check hash
    size_t metadata_size;
    if (buffer.size() < sizeof(size_t)) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load Metal shader "
            "archive for '{}': invalid file size.",
            name);
        return {};
    }
    std::memcpy(&metadata_size, buffer.data(), sizeof(size_t));
    if (buffer.size() < sizeof(size_t) + metadata_size) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load Metal shader "
            "archive for '{}': invalid file size.",
            name);
        return {};
    }
    luisa::string_view metadata_str{
        reinterpret_cast<const char *>(buffer.data() + sizeof(size_t)),
        metadata_size};
    auto file_metadata = deserialize_metal_shader_metadata(metadata_str);
    if (!file_metadata) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load Metal shader "
            "archive for '{}': invalid metadata.",
            name);
        return {};
    }

    // check metadata (or complete it)
    if (metadata.checksum == 0ull) { metadata.checksum = file_metadata->checksum; }
    if (all(metadata.block_size == 0u)) { metadata.block_size = file_metadata->block_size; }
    if (metadata.checksum != file_metadata->checksum ||
        any(metadata.block_size != file_metadata->block_size)) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load Metal shader "
            "archive for '{}': metadata mismatch.",
            name);
        return {};
    }
    metadata.argument_types = std::move(file_metadata->argument_types);
    metadata.argument_usages = std::move(file_metadata->argument_usages);

    // load library
    auto library_data = luisa::span{buffer}.subspan(sizeof(size_t) + metadata_size);
    auto temp_file_path = detail::temp_unique_file_path();
    if (temp_file_path.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load Metal shader "
            "archive for '{}': failed to create temporary file.",
            name);
        return {};
    }
    std::ofstream library_dump{temp_file_path, std::ios::binary};
    if (!library_dump.is_open()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load Metal shader "
            "archive for '{}': failed to open temporary file.",
            name);
        return {};
    }
    library_dump.write(reinterpret_cast<const char *>(library_data.data()),
                       static_cast<ssize_t>(library_data.size()));
    library_dump.close();

    auto url = NS::URL::fileURLWithPath(NS::String::string(
        temp_file_path.string().c_str(), NS::UTF8StringEncoding));
    NS::Error *error = nullptr;
    auto library = NS::TransferPtr(_device->handle()->newLibrary(url, &error));
    std::filesystem::remove(temp_file_path);
    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load Metal shader "
            "archive for '{}': {}.",
            name, error->localizedDescription()->utf8String());
        return {};
    }

    // load kernel
    auto [pipeline_desc, pipeline] = _load_kernels_from_library(library.get(), metadata.block_size);
    if (pipeline.entry && pipeline.indirect_entry) {
        LUISA_INFO("Loaded Metal shader archive for '{}' in {} ms.", name, clk.toc());
    }
    return pipeline;
}

std::pair<MetalCompiler::PipelineDescriptorHandle, MetalShaderHandle>
MetalCompiler::_load_kernels_from_library(MTL::Library *library, uint3 block_size) const noexcept {

    auto load = [&](NS::String *name, bool is_indirect) noexcept -> std::pair<NS::SharedPtr<MTL::ComputePipelineDescriptor>,
                                                                              NS::SharedPtr<MTL::ComputePipelineState>> {
        auto compute_pipeline_desc = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
        compute_pipeline_desc->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
        compute_pipeline_desc->setMaxTotalThreadsPerThreadgroup(block_size.x * block_size.y * block_size.z);
        compute_pipeline_desc->setSupportIndirectCommandBuffers(is_indirect);
        NS::Error *error = nullptr;
        auto function_desc = MTL::FunctionDescriptor::alloc()->init();
        function_desc->setName(name);
        function_desc->setOptions(MTL::FunctionOptionCompileToBinary);
        auto function = NS::TransferPtr(library->newFunction(function_desc, &error));
        function_desc->release();
        if (error != nullptr) {
            LUISA_WARNING_WITH_LOCATION(
                "Error during creating Metal compute function: {}.",
                error->localizedDescription()->utf8String());
            return {};
        }
        compute_pipeline_desc->setComputeFunction(function.get());
        auto pipeline = NS::TransferPtr(_device->handle()->newComputePipelineState(
            compute_pipeline_desc.get(), MTL::PipelineOptionNone, nullptr, &error));
        if (error != nullptr) {
            LUISA_WARNING_WITH_LOCATION(
                "Error during creating Metal compute pipeline: {}.",
                error->localizedDescription()->utf8String());
        }
        return std::make_pair(std::move(compute_pipeline_desc),
                              std::move(pipeline));
    };
    auto [compute_pipeline_desc, pipeline] = load(MTLSTR("kernel_main"), false);
    auto [compute_pipeline_desc_indirect, pipeline_indirect] = load(MTLSTR("kernel_main_indirect"), true);
    return std::make_pair(
        PipelineDescriptorHandle{std::move(compute_pipeline_desc),
                                 std::move(compute_pipeline_desc_indirect)},
        MetalShaderHandle{std::move(pipeline),
                          std::move(pipeline_indirect)});
}

MetalShaderHandle MetalCompiler::compile(luisa::string_view src,
                                         const ShaderOption &option,
                                         MetalShaderMetadata &metadata) const noexcept {

    return with_autorelease_pool([&] {
        auto src_hash = luisa::hash_value(src);
        auto opt_hash = luisa::hash_value(option);
        auto hash = luisa::hash_combine({src_hash, opt_hash});
        metadata.checksum = hash;

        // try memory cache
        if (auto pso = _cache.fetch(hash)) { return *pso; }

        // name
        auto name = option.name.empty() ?
                        luisa::format("metal_kernel_{:016x}", hash) :
                        option.name;

        auto is_aot = !option.name.empty();
        auto uses_cache = is_aot || option.enable_cache;

        // try disk cache
        if (uses_cache) {
            if (auto pso = _load_disk_archive(name, is_aot, metadata);
                pso.entry && pso.indirect_entry) {
                _cache.update(hash, pso);
                return pso;
            }
            LUISA_INFO("Failed to load Metal shader archive for '{}'. "
                       "Falling back to compilation from source.",
                       name);
        }

#if LUISA_METAL_BACKEND_DUMP_SOURCE
        auto src_dump_name = luisa::format("{}.metal", name);
        luisa::span src_dump{reinterpret_cast<const std::byte *>(src.data()), src.size()};
        if (is_aot) {
            _device->io()->write_shader_bytecode(src_dump_name, src_dump);
        } else if (option.enable_cache) {
            _device->io()->write_shader_cache(src_dump_name, src_dump);
        }
#endif

        // no cache found, compile from source
        auto source = NS::String::alloc()->init(const_cast<char *>(src.data()),
                                                src.size(), NS::UTF8StringEncoding, false);
        auto options = MTL::CompileOptions::alloc()->init();
        options->setFastMathEnabled(option.enable_fast_math);
        options->setLanguageVersion(MTL::LanguageVersion3_0);
        options->setLibraryType(MTL::LibraryTypeExecutable);
        options->setMaxTotalThreadsPerThreadgroup(metadata.block_size.x *
                                                  metadata.block_size.y *
                                                  metadata.block_size.z);
        NS::Error *error;
        auto library = NS::TransferPtr(_device->handle()->newLibrary(source, options, &error));
        source->release();
        options->release();
        if (error != nullptr) {
            LUISA_WARNING_WITH_LOCATION(
                "Error during compiling Metal shader '{}': {}.",
                name, error->localizedDescription()->utf8String());
        }
        LUISA_ASSERT(library, "Failed to compile Metal shader '{}'.", name);

        auto [pso_desc, pso] = _load_kernels_from_library(
            library.get(), metadata.block_size);

        // create pso
        LUISA_ASSERT(pso.entry && pso.indirect_entry,
                     "Failed to create Metal compute pipeline for '{}'.", name);

        // store the library
        if (uses_cache) {
            _store_disk_archive(name, is_aot, pso_desc, metadata);
        }
        _cache.update(hash, pso);
        return pso;
    });
}

MetalShaderHandle MetalCompiler::load(luisa::string_view name,
                                      MetalShaderMetadata &metadata) const noexcept {
    return with_autorelease_pool([&] {
        auto pso = _load_disk_archive(name, true, metadata);
        LUISA_ASSERT(pso.entry && pso.indirect_entry,
                     "Failed to load Metal shader archive for '{}'.", name);
        _cache.update(metadata.checksum, pso);
        return pso;
    });
}

}// namespace luisa::compute::metal
