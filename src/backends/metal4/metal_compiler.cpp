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

[[nodiscard]] static auto metal_validation_enabled() noexcept {
    return get_bool_env("MTL_DEBUG_LAYER") ||
           get_bool_env("MTL_SHADER_VALIDATION");
}

}// namespace detail

MetalCompiler::MetalCompiler(const MetalDevice *device) noexcept
    : _device{device}, _cache{max_cache_item_count} {}

namespace {

constexpr auto metal4_pipeline_archive_magic =
    0x4c55495341344d41ull;// "LUISA4MA"

}// namespace

void MetalCompiler::_store_disk_archive(luisa::string_view name, bool is_aot,
                                        const PipelineDescriptorHandle &desc,
                                        const MetalShaderMetadata &metadata) const noexcept {

    if (!desc.entry || !desc.indirect_entry) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal 4 shader archive for '{}': "
            "archive descriptors were not created.",
            name);
        return;
    }

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

void MetalCompiler::_store_metal4_archive(
    luisa::string_view name, bool is_aot,
    luisa::span<const std::byte> metallib,
    MTL4::PipelineDataSetSerializer *serializer,
    const MetalShaderMetadata &metadata) const noexcept {

    LUISA_ASSERT(serializer != nullptr,
                 "Metal 4 pipeline archive serializer must not be null.");
    auto temp_file_path = detail::temp_unique_file_path();
    if (temp_file_path.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal 4 pipeline archive for '{}': "
            "failed to create temporary file.",
            name);
        return;
    }
    auto url = NS::URL::fileURLWithPath(NS::String::string(
        temp_file_path.c_str(), NS::UTF8StringEncoding));
    NS::Error *error = nullptr;
    if (!serializer->serializeAsArchiveAndFlushToURL(url, &error) ||
        error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal 4 pipeline archive for '{}': {}.",
            name, error == nullptr ? "unknown error" :
                                     error->localizedDescription()
                                         ->utf8String());
        return;
    }

    std::error_code ec;
    auto archive_size = std::filesystem::file_size(temp_file_path, ec);
    if (ec) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal 4 pipeline archive for '{}': {}.",
            name, ec.message());
        return;
    }
    auto metadata_str = serialize_metal_shader_metadata(metadata);
    auto metadata_size = metadata_str.size();
    auto metallib_size = static_cast<uint64_t>(metallib.size_bytes());
    luisa::vector<std::byte> buffer;
    buffer.resize(sizeof(size_t) + metadata_size +
                  sizeof(metal4_pipeline_archive_magic) +
                  sizeof(metallib_size) + metallib.size_bytes() +
                  archive_size);
    auto *destination = buffer.data();
    std::memcpy(destination, &metadata_size, sizeof(size_t));
    destination += sizeof(size_t);
    std::memcpy(destination, metadata_str.data(), metadata_size);
    destination += metadata_size;
    std::memcpy(destination, &metal4_pipeline_archive_magic,
                sizeof(metal4_pipeline_archive_magic));
    destination += sizeof(metal4_pipeline_archive_magic);
    std::memcpy(destination, &metallib_size, sizeof(metallib_size));
    destination += sizeof(metallib_size);
    std::memcpy(destination, metallib.data(), metallib.size_bytes());
    destination += metallib.size_bytes();
    std::ifstream file{temp_file_path, std::ios::binary};
    if (!file.is_open()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to store Metal 4 pipeline archive for '{}': "
            "failed to open temporary file.",
            name);
        return;
    }
    file.read(reinterpret_cast<char *>(destination),
              static_cast<ssize_t>(archive_size));
    file.close();
    std::filesystem::remove(temp_file_path, ec);

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
    auto buffer = stream->read(~0ull);
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
    if (metadata.curve_bases.none()) { metadata.curve_bases = file_metadata->curve_bases; }
    if (all(metadata.block_size == 0u)) { metadata.block_size = file_metadata->block_size; }
    if (metadata.checksum != file_metadata->checksum ||
        metadata.curve_bases != file_metadata->curve_bases ||
        any(metadata.block_size != file_metadata->block_size)) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load Metal shader "
            "archive for '{}': metadata mismatch.",
            name);
        return {};
    }
    if (!file_metadata->intersection_functions.empty() &&
        detail::metal_validation_enabled()) {
        LUISA_VERBOSE(
            "Skipping Metal 4 pipeline archive '{}' because Metal shader "
            "validation does not support this MTL4Archive.",
            name);
        return {};
    }
    metadata.argument_types = std::move(file_metadata->argument_types);
    metadata.argument_usages = std::move(file_metadata->argument_usages);
    metadata.argument_sampled = std::move(file_metadata->argument_sampled);
    metadata.format_types = std::move(file_metadata->format_types);
    metadata.intersection_functions =
        std::move(file_metadata->intersection_functions);

    auto archive_payload = luisa::span<std::byte>{buffer}.subspan(
        sizeof(size_t) + metadata_size);
    if (!metadata.intersection_functions.empty()) {
        constexpr auto header_size =
            sizeof(metal4_pipeline_archive_magic) + sizeof(uint64_t);
        if (archive_payload.size_bytes() < header_size) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to load Metal 4 pipeline archive for '{}': "
                "invalid file size.",
                name);
            return {};
        }
        uint64_t magic{};
        uint64_t metallib_size{};
        std::memcpy(&magic, archive_payload.data(), sizeof(magic));
        std::memcpy(&metallib_size,
                    archive_payload.data() + sizeof(magic),
                    sizeof(metallib_size));
        if (magic != metal4_pipeline_archive_magic ||
            metallib_size == 0u ||
            metallib_size > archive_payload.size_bytes() - header_size) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to load Metal 4 pipeline archive for '{}': "
                "invalid header.",
                name);
            return {};
        }
        auto metallib = archive_payload.subspan(
            header_size, static_cast<size_t>(metallib_size));
        auto archive_data = archive_payload.subspan(
            header_size + static_cast<size_t>(metallib_size));
        if (archive_data.empty()) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to load Metal 4 pipeline archive for '{}': "
                "missing archive data.",
                name);
            return {};
        }

        auto library_dispatch_data = dispatch_data_create(
            metallib.data(), metallib.size_bytes(), nullptr,
            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NS::Error *error = nullptr;
        auto library = NS::TransferPtr(
            _device->handle()->newLibrary(
                library_dispatch_data, &error));
        dispatch_release(library_dispatch_data);
        if (error != nullptr || !library) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to load Metal library for pipeline archive '{}': "
                "{}.",
                name, error == nullptr ? "unknown error" :
                                         error->localizedDescription()
                                             ->utf8String());
            return {};
        }
        auto ns_name = NS::String::alloc()->init(
            const_cast<char *>(name.data()), name.size(),
            NS::UTF8StringEncoding, false);
        library->setLabel(ns_name);
        ns_name->release();

        auto temp_file_path = detail::temp_unique_file_path();
        if (temp_file_path.empty()) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to load Metal 4 pipeline archive for '{}': "
                "failed to create temporary file.",
                name);
            return {};
        }
        std::ofstream archive_dump{temp_file_path, std::ios::binary};
        if (!archive_dump.is_open()) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to load Metal 4 pipeline archive for '{}': "
                "failed to open temporary file.",
                name);
            return {};
        }
        archive_dump.write(
            reinterpret_cast<const char *>(archive_data.data()),
            static_cast<ssize_t>(archive_data.size_bytes()));
        archive_dump.close();
        auto url = NS::URL::fileURLWithPath(NS::String::string(
            temp_file_path.string().c_str(), NS::UTF8StringEncoding));
        error = nullptr;
        auto archive = NS::TransferPtr(
            _device->handle()->newArchive(url, &error));
        std::error_code remove_error;
        std::filesystem::remove(temp_file_path, remove_error);
        if (error != nullptr || !archive) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to open Metal 4 pipeline archive '{}': {}.",
                name, error == nullptr ? "unknown error" :
                                         error->localizedDescription()
                                             ->utf8String());
            return {};
        }
        auto [pipeline_desc, pipeline] = _load_kernels_from_library(
            library.get(), metadata.block_size,
            metadata.intersection_functions, false,
            nullptr, archive.get());
        if (pipeline.entry && pipeline.indirect_entry) {
            LUISA_VERBOSE(
                "Loaded Metal 4 pipeline archive for '{}' in {} ms.",
                name, clk.toc());
        }
        return pipeline;
    }

    // load library
    auto library_data = archive_payload;
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
        detail::get_bool_env("MTL_ENABLE_CAPTURE") ||
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
    auto [pipeline_desc, pipeline] = _load_kernels_from_library(
        library.get(), metadata.block_size,
        metadata.intersection_functions, false);
    if (pipeline.entry && pipeline.indirect_entry) {
        LUISA_VERBOSE(
            "Loaded Metal shader archive for '{}' in {} ms.",
            name, clk.toc());
    }
    return pipeline;
}

std::pair<MetalCompiler::PipelineDescriptorHandle, MetalShaderHandle>
MetalCompiler::_load_kernels_from_library(
    MTL::Library *library, uint3 block_size,
    luisa::span<const luisa::string> intersection_functions,
    bool create_archive_descriptors,
    MTL4::Compiler *compiler,
    MTL4::Archive *archive) const noexcept {

    LUISA_ASSERT(compiler == nullptr || archive == nullptr,
                 "Metal 4 pipeline creation cannot use a compiler and an "
                 "archive simultaneously.");
    if (compiler == nullptr && archive == nullptr) {
        compiler = _device->metal4_compiler();
    }

    struct LoadedPipeline {
        NS::SharedPtr<MTL::ComputePipelineDescriptor> archive_descriptor;
        NS::SharedPtr<MTL::ComputePipelineState> pipeline;
        luisa::vector<NS::SharedPtr<MTL::IntersectionFunctionTable>> tables;
    };
    auto load = [&](NS::String *name,
                    bool is_indirect) noexcept -> LoadedPipeline {
        auto label = is_indirect ?
                         library->label()->stringByAppendingString(MTLSTR(" (indirect)")) :
                         library->label();

        auto function_desc = NS::TransferPtr(
            MTL4::LibraryFunctionDescriptor::alloc()->init());
        function_desc->setLibrary(library);
        function_desc->setName(name);
        auto metal4_pipeline_desc = NS::TransferPtr(
            MTL4::ComputePipelineDescriptor::alloc()->init());
        metal4_pipeline_desc->setComputeFunctionDescriptor(function_desc.get());
        luisa::vector<NS::SharedPtr<MTL4::LibraryFunctionDescriptor>>
            linked_function_descriptors;
        NS::SharedPtr<MTL4::StaticLinkingDescriptor> static_linking_descriptor;
        if (!intersection_functions.empty()) {
            linked_function_descriptors.reserve(intersection_functions.size());
            luisa::vector<const NS::Object *> descriptor_objects;
            descriptor_objects.reserve(intersection_functions.size());
            for (auto &&function_name : intersection_functions) {
                auto descriptor = NS::TransferPtr(
                    MTL4::LibraryFunctionDescriptor::alloc()->init());
                descriptor->setLibrary(library);
                descriptor->setName(NS::String::string(
                    function_name.c_str(), NS::UTF8StringEncoding));
                descriptor_objects.emplace_back(descriptor.get());
                linked_function_descriptors.emplace_back(
                    std::move(descriptor));
            }
            static_linking_descriptor = NS::TransferPtr(
                MTL4::StaticLinkingDescriptor::alloc()->init());
            static_linking_descriptor->setFunctionDescriptors(
                NS::Array::array(descriptor_objects.data(),
                                 descriptor_objects.size()));
            metal4_pipeline_desc->setStaticLinkingDescriptor(
                static_linking_descriptor.get());
            metal4_pipeline_desc->setSupportBinaryLinking(true);
        }
        metal4_pipeline_desc->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
        metal4_pipeline_desc->setMaxTotalThreadsPerThreadgroup(
            block_size.x * block_size.y * block_size.z);
        metal4_pipeline_desc->setSupportIndirectCommandBuffers(
            is_indirect ?
                MTL4::IndirectCommandBufferSupportStateEnabled :
                MTL4::IndirectCommandBufferSupportStateDisabled);
        metal4_pipeline_desc->setLabel(label);
        NS::Error *error = nullptr;
        auto pipeline = NS::TransferPtr(
            archive == nullptr ?
                compiler->newComputePipelineState(
                    metal4_pipeline_desc.get(), nullptr, &error) :
                archive->newComputePipelineState(
                    metal4_pipeline_desc.get(), &error));
        if (error != nullptr || !pipeline) {
            LUISA_WARNING_WITH_LOCATION(
                "Error while creating Metal 4 compute pipeline '{}': {}.",
                name->utf8String(),
                error == nullptr ? "unknown error" :
                                   error->localizedDescription()->utf8String());
            return {};
        }

        luisa::vector<NS::SharedPtr<MTL::IntersectionFunctionTable>>
            intersection_tables;
        intersection_tables.reserve(intersection_functions.size());
        for (auto &&function_name : intersection_functions) {
            auto table_descriptor = NS::TransferPtr(
                MTL::IntersectionFunctionTableDescriptor::alloc()->init());
            table_descriptor->setFunctionCount(1u);
            auto table = NS::TransferPtr(
                pipeline->newIntersectionFunctionTable(
                    table_descriptor.get()));
            auto ns_function_name = NS::String::string(
                function_name.c_str(), NS::UTF8StringEncoding);
            auto function_handle = pipeline->functionHandle(
                ns_function_name);
            if (!table) {
                LUISA_WARNING_WITH_LOCATION(
                    "Failed to create Metal 4 intersection function table "
                    "storage for '{}'.",
                    function_name);
                return {};
            }
            if (function_handle == nullptr) {
                LUISA_WARNING_WITH_LOCATION(
                    "Metal 4 compute pipeline did not expose an intersection "
                    "function handle for '{}'.",
                    function_name);
                return {};
            }
            table->setFunction(function_handle, 0u);
            intersection_tables.emplace_back(std::move(table));
        }

        NS::SharedPtr<MTL::ComputePipelineDescriptor> archive_desc;
        if (create_archive_descriptors) {
            archive_desc = NS::TransferPtr(
                MTL::ComputePipelineDescriptor::alloc()->init());
            archive_desc->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
            archive_desc->setMaxTotalThreadsPerThreadgroup(
                block_size.x * block_size.y * block_size.z);
            archive_desc->setSupportIndirectCommandBuffers(is_indirect);
            archive_desc->setLabel(label);
            auto archive_function_desc = NS::TransferPtr(
                MTL::FunctionDescriptor::alloc()->init());
            archive_function_desc->setName(name);
            archive_function_desc->setOptions(MTL::FunctionOptionCompileToBinary);
            error = nullptr;
            auto archive_function = NS::TransferPtr(
                library->newFunction(archive_function_desc.get(), &error));
            if (error != nullptr || !archive_function) {
                LUISA_WARNING_WITH_LOCATION(
                    "Error while preparing Metal 4 archive function '{}': {}.",
                    name->utf8String(),
                    error == nullptr ? "unknown error" :
                                       error->localizedDescription()->utf8String());
                return {};
            }
            archive_function->setLabel(label);
            archive_desc->setComputeFunction(archive_function.get());
        }
        return LoadedPipeline{
            .archive_descriptor = std::move(archive_desc),
            .pipeline = std::move(pipeline),
            .tables = std::move(intersection_tables)};
    };
    auto direct = load(MTLSTR("kernel_main"), false);
    auto indirect = load(MTLSTR("kernel_main_indirect"), true);
    return std::make_pair(
        PipelineDescriptorHandle{
            std::move(direct.archive_descriptor),
            std::move(indirect.archive_descriptor)},
        MetalShaderHandle{
            std::move(direct.pipeline),
            std::move(indirect.pipeline),
            std::move(direct.tables),
            std::move(indirect.tables)});
}

MetalShaderHandle MetalCompiler::compile(luisa::span<const std::byte> metallib,
                                         const ShaderOption &option,
                                         MetalShaderMetadata &metadata) const noexcept {

    return with_autorelease_pool([&] {
        // MetalDevice supplies a semantic checksum so LLVM 14 bitcode
        // serialization details do not perturb shader-cache identity. Keep a
        // byte-derived fallback for direct compiler users.
        auto hash = metadata.checksum;
        if (hash == 0u) {
            auto library_hash = luisa::hash64(
                metallib.data(), metallib.size_bytes(),
                luisa::hash64_default_seed);
            auto option_hash = luisa::hash_value(option);
            hash = luisa::hash_combine(
                {library_hash, option_hash, 0x4d544c425f414952ull});
        }
        metadata.checksum = hash;

        if (auto pso = _cache.fetch(hash)) { return *pso; }

        auto name = option.name.empty() ?
                        luisa::format("metal_air_kernel_{:016x}", hash) :
                        option.name;
        auto is_aot = !option.name.empty();
        auto uses_cache = is_aot || option.enable_cache;

        if (option.enable_debug_info || detail::get_bool_env("LUISA_DUMP_SOURCE")) {
            auto dump_name = luisa::format("{}.metallib", name);
            luisa::filesystem::path dump_path;
            if (is_aot) {
                dump_path = _device->io()->write_shader_bytecode(dump_name, metallib);
            } else if (option.enable_cache) {
                dump_path = _device->io()->write_shader_source(dump_name, metallib);
            }
            if (!dump_path.empty()) {
                LUISA_VERBOSE("Dumped Metal AIR library for '{}' to '{}'.",
                              name, dump_path.string());
            }
        }

        if (uses_cache) {
            if (option.enable_debug_info) {
                LUISA_WARNING_WITH_LOCATION(
                    "Debug information is enabled for Metal AIR shader '{}'. "
                    "The disk cache will not be loaded.",
                    name);
            } else if (auto pso = _load_disk_archive(name, is_aot, metadata);
                       pso.entry && pso.indirect_entry) {
                _cache.update(hash, pso);
                return pso;
            }
        }

        auto library_data = dispatch_data_create(
            metallib.data(), metallib.size_bytes(), nullptr,
            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NS::Error *error = nullptr;
        auto library = NS::TransferPtr(
            _device->handle()->newLibrary(library_data, &error));
        dispatch_release(library_data);
        if (error != nullptr) {
            LUISA_WARNING_WITH_LOCATION(
                "Error while loading Metal AIR library '{}': {}.",
                name, error->localizedDescription()->utf8String());
        }
        if (!library) { return MetalShaderHandle{}; }
        library->setLabel(NS::String::string(name.c_str(), NS::UTF8StringEncoding));

        auto uses_metal4_archive =
            uses_cache && !metadata.intersection_functions.empty() &&
            !detail::metal_validation_enabled();
        auto uses_legacy_archive =
            uses_cache && metadata.intersection_functions.empty();
        NS::SharedPtr<MTL4::PipelineDataSetSerializer> serializer;
        NS::SharedPtr<MTL4::Compiler> archive_compiler;
        if (uses_metal4_archive) {
            auto serializer_desc = NS::TransferPtr(
                MTL4::PipelineDataSetSerializerDescriptor::alloc()->init());
            serializer_desc->setConfiguration(
                MTL4::PipelineDataSetSerializerConfigurationCaptureBinaries);
            serializer = NS::TransferPtr(
                _device->handle()->newPipelineDataSetSerializer(
                    serializer_desc.get()));
            if (!serializer) {
                LUISA_WARNING_WITH_LOCATION(
                    "Failed to create Metal 4 pipeline archive serializer "
                    "for '{}'; continuing without a disk cache.",
                    name);
                uses_metal4_archive = false;
            } else {
                auto compiler_desc = NS::TransferPtr(
                    MTL4::CompilerDescriptor::alloc()->init());
                compiler_desc->setLabel(NS::String::string(
                    name.c_str(), NS::UTF8StringEncoding));
                compiler_desc->setPipelineDataSetSerializer(
                    serializer.get());
                error = nullptr;
                archive_compiler = NS::TransferPtr(
                    _device->handle()->newCompiler(
                        compiler_desc.get(), &error));
                if (error != nullptr || !archive_compiler) {
                    LUISA_WARNING_WITH_LOCATION(
                        "Failed to create Metal 4 pipeline archive compiler "
                        "for '{}': {}; continuing without a disk cache.",
                        name, error == nullptr ? "unknown error" :
                                               error->localizedDescription()
                                                   ->utf8String());
                    serializer.reset();
                    uses_metal4_archive = false;
                }
            }
        }

        auto [pipeline_desc, pipeline] = _load_kernels_from_library(
            library.get(), metadata.block_size,
            metadata.intersection_functions, uses_legacy_archive,
            archive_compiler.get());
        if (!pipeline.entry || !pipeline.indirect_entry) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to create Metal AIR compute pipeline for '{}'.", name);
            return MetalShaderHandle{};
        }

        if (uses_metal4_archive) {
            _store_metal4_archive(
                name, is_aot, metallib,
                archive_compiler->pipelineDataSetSerializer(), metadata);
        } else if (uses_legacy_archive) {
            _store_disk_archive(name, is_aot, pipeline_desc, metadata);
        }
        _cache.update(hash, pipeline);
        return pipeline;
    });
}

MetalShaderHandle MetalCompiler::load_cached(
    const ShaderOption &option, uint64_t checksum,
    MetalShaderMetadata &metadata) const noexcept {
    return with_autorelease_pool([&] {
        auto is_aot = !option.name.empty();
        auto uses_cache = is_aot || option.enable_cache;
        if (!uses_cache || option.enable_debug_info) {
            return MetalShaderHandle{};
        }
        metadata.checksum = checksum;
        if (auto pipeline = _cache.fetch(checksum)) { return *pipeline; }
        auto name = is_aot ? option.name :
                             luisa::format(
                                 "metal_air_kernel_{:016x}", checksum);
        auto pipeline = _load_disk_archive(
            name, is_aot, metadata);
        if (pipeline.entry && pipeline.indirect_entry) {
            _cache.update(checksum, pipeline);
        }
        return pipeline;
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
