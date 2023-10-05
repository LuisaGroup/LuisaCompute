#include <fstream>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include "metal_device.h"
#include "metal_compiler.h"

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

[[nodiscard]] static auto get_bool_env(const char *name) noexcept {
    if (auto env_c_str = getenv(name)) {
        luisa::string env{env_c_str};
        for (auto &c : env) { c = static_cast<char>(toupper(c)); }
        using namespace std::string_view_literals;
        return env != "0"sv &&
               env != "OFF"sv &&
               env != "FALSE"sv &&
               env != "NO"sv &&
               env != "DISABLE"sv &&
               env != "DISABLED"sv;
    }
    return false;
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
        static_cast<void>(io->write_shader_bytecode(name, buffer));
    } else {
        static_cast<void>(io->write_shader_cache(name, buffer));
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

    auto should_dump_metallib =
        MTL::CaptureManager::sharedCaptureManager()->isCapturing() ||
        detail::get_bool_env("METAL_CAPTURE_ENABLED") ||
        detail::get_bool_env("MTL_DEBUG_LAYER") ||
        detail::get_bool_env("MTL_SHADER_VALIDATION") ||
        detail::get_bool_env("LUISA_DUMP_METAL_LIBRARY");

    if (should_dump_metallib) {
        LUISA_VERBOSE(
            "Metal shader archive for '{}' dumped to '{}'.",
            name, temp_file_path.string());
    } else {
        std::filesystem::remove(temp_file_path);
    }

    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load Metal shader "
            "archive for '{}': {}.",
            name, error->localizedDescription()->utf8String());
        return {};
    }

    // load kernel
    auto ns_name = NS::String::alloc()->init(
        const_cast<char *>(name.data()), name.size(),
        NS::UTF8StringEncoding, false);
    library->setLabel(ns_name);
    ns_name->release();
    auto [pipeline_desc, pipeline] = _load_kernels_from_library(library.get(), metadata.block_size);
    if (pipeline.entry && pipeline.indirect_entry) {
        LUISA_VERBOSE(
            "Loaded Metal shader archive for '{}' in {} ms.",
            name, clk.toc());
    }
    return pipeline;
}

std::pair<MetalCompiler::PipelineDescriptorHandle, MetalShaderHandle>
MetalCompiler::_load_kernels_from_library(MTL::Library *library, uint3 block_size) const noexcept {

    auto load = [&](NS::String *name, bool is_indirect) noexcept -> std::pair<NS::SharedPtr<MTL::ComputePipelineDescriptor>,
                                                                              NS::SharedPtr<MTL::ComputePipelineState>> {
        auto label = is_indirect ?
                         library->label()->stringByAppendingString(MTLSTR(" (indirect)")) :
                         library->label();
        auto compute_pipeline_desc = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
        compute_pipeline_desc->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
        compute_pipeline_desc->setMaxTotalThreadsPerThreadgroup(block_size.x * block_size.y * block_size.z);
        compute_pipeline_desc->setSupportIndirectCommandBuffers(is_indirect);
        compute_pipeline_desc->setLabel(label);
        NS::Error *error = nullptr;
        auto function_desc = MTL::FunctionDescriptor::alloc()->init();
        function_desc->setName(name);
        function_desc->setOptions(MTL::FunctionOptionCompileToBinary);
        auto function = NS::TransferPtr(library->newFunction(function_desc, &error));
        function->setLabel(label);
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

        if (option.enable_debug_info || detail::get_bool_env("LUISA_DUMP_SOURCE")) {
            auto src_dump_name = luisa::format("{}.metal", name);
            luisa::span src_dump{reinterpret_cast<const std::byte *>(src.data()), src.size()};
            luisa::filesystem::path src_dump_path;
            if (is_aot) {
                src_dump_path = _device->io()->write_shader_bytecode(src_dump_name, src_dump);
            } else if (option.enable_cache) {
                src_dump_path = _device->io()->write_shader_cache(src_dump_name, src_dump);
            }
            // TODO: attach shader source to Metal shader archive for debugging.
            //       Is it possible without using the command line?
            if (!src_dump_path.empty()) {
                LUISA_VERBOSE(
                    "Dumped Metal shader source for '{}' to '{}'.",
                    name, src_dump_path.string());
            }
        }

        // try disk cache
        if (uses_cache) {
            if (option.enable_debug_info) {
                LUISA_WARNING_WITH_LOCATION(
                    "Debug information is enabled for Metal shader '{}'. "
                    "The disk cache will not be loaded.",
                    name);
            } else {
                if (auto pso = _load_disk_archive(name, is_aot, metadata);
                    pso.entry && pso.indirect_entry) {
                    _cache.update(hash, pso);
                    return pso;
                }
                LUISA_VERBOSE(
                    "Failed to load Metal shader archive for '{}'. "
                    "Falling back to compilation from source.",
                    name);
            }
        }

        // no cache found, compile from source
        auto source = NS::String::alloc()->init(const_cast<char *>(src.data()),
                                                src.size(), NS::UTF8StringEncoding, false);
        auto options = MTL::CompileOptions::alloc()->init();
        options->setFastMathEnabled(option.enable_fast_math);
        options->setLanguageVersion(MTL::LanguageVersion3_0);
        options->setLibraryType(MTL::LibraryTypeExecutable);

        // this requires iOS 16.4+, iPadOS 16.4+, macOS 13.3+, Mac Catalyst 16.4+, tvOS 16.4+
        if (__builtin_available(iOS 16.4, macOS 13.3, tvOS 16.4, macCatalyst 16.4, *)) {
            options->setMaxTotalThreadsPerThreadgroup(metadata.block_size.x *
                                                      metadata.block_size.y *
                                                      metadata.block_size.z);
        }

        NS::Error *error;
        auto library = NS::TransferPtr(_device->handle()->newLibrary(source, options, &error));
        library->setLabel(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
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
