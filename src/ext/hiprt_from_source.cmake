# hiprt_from_source.cmake
# Custom CMake wrapper to build HIPRT from source as a subdirectory target.
# This avoids using HIPRT's own CMakeLists.txt which is not suitable for add_subdirectory().
#
# Usage: include() this file from the HIP backend CMakeLists.txt.
# Expects HIPRT_SOURCE_DIR to be set to the root of the HIPRT repo.
# Produces:
#   - Target: hiprt_from_source (SHARED library)
#   - Variables: HIPRT_FROM_SOURCE_INCLUDE_DIR, HIPRT_FROM_SOURCE_LIB_DIR
#   - Custom commands to compile bitcode (.bc) and fat binaries (.hipfb)

if(NOT DEFINED HIPRT_SOURCE_DIR)
    message(FATAL_ERROR "HIPRT_SOURCE_DIR must be set before including hiprt_from_source.cmake")
endif()

# ---------------------------------------------------------------------------
# 1. Read version info
# ---------------------------------------------------------------------------
file(STRINGS "${HIPRT_SOURCE_DIR}/version.txt" _hiprt_version_lines)
list(GET _hiprt_version_lines 0 HIPRT_MAJOR_VERSION)
list(GET _hiprt_version_lines 1 HIPRT_MINOR_VERSION)
list(GET _hiprt_version_lines 2 HIPRT_PATCH_VERSION)
list(GET _hiprt_version_lines 3 HIPRT_HASH_VERSION)

math(EXPR HIPRT_API_VERSION "${HIPRT_MAJOR_VERSION} * 1000 + ${HIPRT_MINOR_VERSION}")

# Zero-pad to 5 digits
string(LENGTH "${HIPRT_API_VERSION}" _hiprt_ver_len)
if(_hiprt_ver_len LESS 5)
    math(EXPR _pad_count "5 - ${_hiprt_ver_len}")
    string(REPEAT "0" ${_pad_count} _pad_str)
    set(HIPRT_VERSION_STR "${_pad_str}${HIPRT_API_VERSION}")
else()
    set(HIPRT_VERSION_STR "${HIPRT_API_VERSION}")
endif()

set(HIPRT_LIB_NAME "hiprt${HIPRT_VERSION_STR}")
message(STATUS "[HIPRT from source] Version: ${HIPRT_MAJOR_VERSION}.${HIPRT_MINOR_VERSION}.${HIPRT_PATCH_VERSION} (${HIPRT_HASH_VERSION})")
message(STATUS "[HIPRT from source] Library name: ${HIPRT_LIB_NAME}")

# ---------------------------------------------------------------------------
# 2. Get HIP SDK version for bitcode naming
# ---------------------------------------------------------------------------
execute_process(
    COMMAND hipcc --version
    OUTPUT_VARIABLE _hipcc_version_output
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _hip_ver_match "${_hipcc_version_output}")
set(HIP_VERSION_STR "${_hip_ver_match}")
string(REGEX MATCH "([0-9]+)" HIP_VERSION_MAJOR "${_hip_ver_match}")
string(REGEX REPLACE "^[0-9]+\\." "" HIP_VERSION_MINOR "${_hip_ver_match}")
math(EXPR HIP_SDK_VERSION_NUM "${HIP_VERSION_MAJOR} * 10 + ${HIP_VERSION_MINOR}")
message(STATUS "[HIPRT from source] HIP SDK version: ${HIP_VERSION_STR} (num=${HIP_SDK_VERSION_NUM})")

# ---------------------------------------------------------------------------
# 3. Generate hiprt.h and hiprtew.h from templates
# ---------------------------------------------------------------------------
set(_hiprt_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/hiprt_generated")
file(MAKE_DIRECTORY "${_hiprt_generated_dir}/hiprt")

# We need to generate into a separate dir so we don't pollute the source tree.
# The generated headers will be in ${_hiprt_generated_dir}/hiprt/hiprt.h etc.
foreach(_template hiprt.h hiprtew.h)
    file(READ "${HIPRT_SOURCE_DIR}/hiprt/${_template}.in" _template_content)
    string(REPLACE "@HIPRT_MAJOR_VERSION@" "${HIPRT_MAJOR_VERSION}" _template_content "${_template_content}")
    string(REPLACE "@HIPRT_MINOR_VERSION@" "${HIPRT_MINOR_VERSION}" _template_content "${_template_content}")
    string(REPLACE "@HIPRT_PATCH_VERSION@" "${HIPRT_PATCH_VERSION}" _template_content "${_template_content}")
    string(REPLACE "@HIPRT_HASH_VERSION@" "0x${HIPRT_HASH_VERSION}" _template_content "${_template_content}")
    string(REPLACE "@HIPRT_VERSION_STR@" "\"${HIPRT_VERSION_STR}\"" _template_content "${_template_content}")
    string(REPLACE "@HIPRT_API_VERSION@" "${HIPRT_API_VERSION}" _template_content "${_template_content}")
    string(REPLACE "@HIP_VERSION_STR@" "\"${HIP_VERSION_STR}\"" _template_content "${_template_content}")
    file(WRITE "${_hiprt_generated_dir}/hiprt/${_template}" "${_template_content}")
    message(STATUS "[HIPRT from source] Generated ${_template}")
endforeach()

# ---------------------------------------------------------------------------
# Generate a compatibility overlay rather than mutating the HIPRT submodule.
# New hipcc supplies device placement new even when emitting LLVM bitcode.
function(_hiprt_write_overlay_if_changed _dst _content)
    set(_needs_write TRUE)
    if(EXISTS "${_dst}")
        file(SHA256 "${_dst}" _old_hash)
        string(SHA256 _new_hash "${_content}")
        if(_old_hash STREQUAL _new_hash)
            set(_needs_write FALSE)
        endif()
    endif()
    if(_needs_write)
        get_filename_component(_dst_dir "${_dst}" DIRECTORY)
        file(MAKE_DIRECTORY "${_dst_dir}")
        file(WRITE "${_dst}" "${_content}")
    endif()
endfunction()

file(READ "${HIPRT_SOURCE_DIR}/hiprt/impl/hiprt_device_impl.h" _hiprt_device_impl)
string(REPLACE
    "#if defined( __HIPCC_RTC__ ) || ( defined( HIPRT_BITCODE_LINKING ) && defined( __KERNELCC__ ) )"
    "#if !defined( __CLANG_CUDA_WRAPPERS_NEW ) && ( defined( __HIPCC_RTC__ ) || ( defined( HIPRT_BITCODE_LINKING ) && defined( __KERNELCC__ ) ) )"
    _hiprt_device_impl "${_hiprt_device_impl}")
file(MAKE_DIRECTORY "${_hiprt_generated_dir}/hiprt/impl")
file(CONFIGURE
    OUTPUT "${_hiprt_generated_dir}/hiprt/impl/hiprt_device_impl.h"
    CONTENT "${_hiprt_device_impl}" @ONLY)

# Fix BatchBuilder::getStorageBufferSize for RTIP >= 3.1 TrianglePacketNode:
# upstream allocates for primCount instead of getMaxPrimNodeCount(...).
file(READ "${HIPRT_SOURCE_DIR}/hiprt/impl/BatchBuilder.cpp" _hiprt_batch_builder_cpp)
string(REPLACE "\r\n" "\n" _hiprt_batch_builder_cpp "${_hiprt_batch_builder_cpp}")
set(_hiprt_batch_builder_cpp_orig "${_hiprt_batch_builder_cpp}")
string(REPLACE
"size_t BatchBuilder::getStorageBufferSize(
\tContext& context, const hiprtGeometryBuildInput& buildInput, [[maybe_unused]] const hiprtBuildOptions buildOptions )
{
\tconst size_t primCount\t  = getPrimCount( buildInput );
\tconst size_t boxNodeCount = getMaxBoxNodeCount( buildInput, context.getRtip(), primCount );
\treturn getGeometryStorageBufferSize(
\t\tprimCount, boxNodeCount, getPrimNodeSize( buildInput, context.getTriangleNodeSize() ), context.getBoxNodeSize() );
}"
"size_t BatchBuilder::getStorageBufferSize(
\tContext& context, const hiprtGeometryBuildInput& buildInput, [[maybe_unused]] const hiprtBuildOptions buildOptions )
{
\tconst size_t primCount\t   = getPrimCount( buildInput );
\tconst size_t primNodeCount = getMaxPrimNodeCount( buildInput, context.getRtip(), primCount );
\tconst size_t boxNodeCount  = getMaxBoxNodeCount( buildInput, context.getRtip(), primCount );
\treturn getGeometryStorageBufferSize(
\t\tprimNodeCount, boxNodeCount, getPrimNodeSize( buildInput, context.getTriangleNodeSize() ), context.getBoxNodeSize() );
}"
    _hiprt_batch_builder_cpp "${_hiprt_batch_builder_cpp}")
if(_hiprt_batch_builder_cpp STREQUAL _hiprt_batch_builder_cpp_orig)
    message(FATAL_ERROR "[HIPRT from source] Failed to patch BatchBuilder.cpp overlay.")
endif()
_hiprt_write_overlay_if_changed("${_hiprt_generated_dir}/hiprt/impl/BatchBuilder.cpp" "${_hiprt_batch_builder_cpp}")

# Replace 2N synchronous oroMemcpyDtoH stalls in Context::compactGeometries with
# one batched pinned DtoHAsync + stream sync, then reuse headers for DtoD/HtoDAsync.
file(READ "${HIPRT_SOURCE_DIR}/hiprt/impl/Context.cpp" _hiprt_context_cpp)
string(REPLACE "\r\n" "\n" _hiprt_context_cpp "${_hiprt_context_cpp}")
set(_hiprt_context_cpp_orig "${_hiprt_context_cpp}")
string(REPLACE
"std::vector<hiprtGeometry> Context::compactGeometries( const std::vector<hiprtGeometry>& geometriesIn, oroStream stream )
{
\tcheckOro( oroCtxSetCurrent( m_ctxt ) );

\tsize_t\t\t\t\tsize = 0;
\tstd::vector<size_t> sizes( geometriesIn.size() );
\tfor ( size_t i = 0; i < geometriesIn.size(); ++i )
\t{
\t\tGeomHeader header;
\t\tcheckOro( oroMemcpyDtoH( &header, reinterpret_cast<oroDeviceptr>( geometriesIn[i] ), sizeof( GeomHeader ) ) );
\t\tconst size_t primNodeSize = header.m_geomType & 1 ? getTriangleNodeSize() : sizeof( CustomNode );
\t\tsizes[i] =
\t\t\tgetGeometryStorageBufferSize( header.m_primNodeCount, header.m_boxNodeCount, primNodeSize, getBoxNodeSize() );
\t\tsize += sizes[i];
\t}

\toroDeviceptr buffer;
\tcheckOro( oroMalloc( &buffer, size ) );

\tstd::vector<hiprtGeometry> geometriesOut( geometriesIn.size() );
\tfor ( size_t i = 0; i < geometriesIn.size(); ++i )
\t{
\t\tGeomHeader header;
\t\tcheckOro( oroMemcpyDtoH( &header, reinterpret_cast<oroDeviceptr>( geometriesIn[i] ), sizeof( GeomHeader ) ) );
\t\tconst size_t primNodeSize = header.m_geomType & 1 ? getTriangleNodeSize() : sizeof( CustomNode );
\t\tconst size_t boxNodeSize  = getBoxNodeSize();

\t\tgeometriesOut[i] = reinterpret_cast<hiprtGeometry>( buffer );
\t\tMemoryArena storageMemoryArena( geometriesOut[i], sizes[i], DefaultAlignment );
\t\tstorageMemoryArena.allocate<GeomHeader>();
\t\tvoid* boxNodes\t= storageMemoryArena.allocate<uint8_t>( boxNodeSize * header.m_boxNodeCount );
\t\tvoid* primNodes = storageMemoryArena.allocate<uint8_t>( primNodeSize * header.m_primNodeCount );

\t\tcheckOro( oroMemcpyDtoDAsync(
\t\t\treinterpret_cast<oroDeviceptr>( boxNodes ),
\t\t\treinterpret_cast<oroDeviceptr>( header.m_boxNodes ),
\t\t\tboxNodeSize * header.m_boxNodeCount,
\t\t\tstream ) );

\t\tcheckOro( oroMemcpyDtoDAsync(
\t\t\treinterpret_cast<oroDeviceptr>( primNodes ),
\t\t\treinterpret_cast<oroDeviceptr>( header.m_primNodes ),
\t\t\tprimNodeSize * header.m_primNodeCount,
\t\t\tstream ) );

\t\theader.m_boxNodes  = boxNodes;
\t\theader.m_primNodes = primNodes;
\t\theader.m_size\t   = sizes[i];
\t\tcheckOro(
\t\t\toroMemcpyHtoDAsync( reinterpret_cast<oroDeviceptr>( geometriesOut[i] ), &header, sizeof( GeomHeader ), stream ) );

\t\tbuffer = static_cast<uint8_t*>( buffer ) + sizes[i];
\t}

\t{
\t\tstd::lock_guard<std::mutex> lockMutex( m_poolMutex );
\t\tm_poolHeads[{ reinterpret_cast<oroDeviceptr>( geometriesOut.front() ), size }] =
\t\t\tstatic_cast<uint32_t>( geometriesOut.size() );
\t}

\tcheckOro( oroStreamSynchronize( stream ) );
\tdestroyGeometries( geometriesIn );

\treturn geometriesOut;
}"
"std::vector<hiprtGeometry> Context::compactGeometries( const std::vector<hiprtGeometry>& geometriesIn, oroStream stream )
{
\tif ( geometriesIn.empty() ) return {};

\tcheckOro( oroCtxSetCurrent( m_ctxt ) );

\tstruct PinnedHeadersGuard
\t{
\t\tvoid* pinned = nullptr;
\t\t~PinnedHeadersGuard()
\t\t{
\t\t\tif ( pinned != nullptr ) oroHostFree( pinned );
\t\t}
\t} pinnedGuard;

\tGeomHeader smallHeaders[4];
\tstd::vector<GeomHeader> fallbackHeaders;
\tGeomHeader* headers = nullptr;
\tif ( geometriesIn.size() < 4 )
\t{
\t\theaders = smallHeaders;
\t}
\telse if ( oroHostMalloc( &pinnedGuard.pinned, geometriesIn.size() * sizeof( GeomHeader ), 0 ) == oroSuccess &&
\t\t pinnedGuard.pinned != nullptr )
\t{
\t\theaders = reinterpret_cast<GeomHeader*>( pinnedGuard.pinned );
\t}
\telse
\t{
\t\tpinnedGuard.pinned = nullptr;
\t\tfallbackHeaders.resize( geometriesIn.size() );
\t\theaders = fallbackHeaders.data();
\t}

\tfor ( size_t i = 0; i < geometriesIn.size(); ++i )
\t{
\t\tcheckOro( oroMemcpyDtoHAsync(
\t\t\t&headers[i], reinterpret_cast<oroDeviceptr>( geometriesIn[i] ), sizeof( GeomHeader ), stream ) );
\t}
\tcheckOro( oroStreamSynchronize( stream ) );

\tsize_t size = 0;
\tstd::vector<size_t> sizes( geometriesIn.size() );
\tconst size_t triangleNodeSize = getTriangleNodeSize();
\tconst size_t boxNodeSize      = getBoxNodeSize();
\tfor ( size_t i = 0; i < geometriesIn.size(); ++i )
\t{
\t\tconst size_t primNodeSize = headers[i].m_geomType & 1 ? triangleNodeSize : sizeof( CustomNode );
\t\tsizes[i] =
\t\t\tgetGeometryStorageBufferSize( headers[i].m_primNodeCount, headers[i].m_boxNodeCount, primNodeSize, boxNodeSize );
\t\tsize += sizes[i];
\t}

\toroDeviceptr buffer{};
\tcheckOro( oroMalloc( &buffer, size ) );

\tstd::vector<hiprtGeometry> geometriesOut( geometriesIn.size() );
\tfor ( size_t i = 0; i < geometriesIn.size(); ++i )
\t{
\t\tconst size_t primNodeSize = headers[i].m_geomType & 1 ? triangleNodeSize : sizeof( CustomNode );

\t\tgeometriesOut[i] = reinterpret_cast<hiprtGeometry>( buffer );
\t\tMemoryArena storageMemoryArena( geometriesOut[i], sizes[i], DefaultAlignment );
\t\tstorageMemoryArena.allocate<GeomHeader>();
\t\tvoid* boxNodes  = storageMemoryArena.allocate<uint8_t>( boxNodeSize * headers[i].m_boxNodeCount );
\t\tvoid* primNodes = storageMemoryArena.allocate<uint8_t>( primNodeSize * headers[i].m_primNodeCount );

\t\tcheckOro( oroMemcpyDtoDAsync(
\t\t\treinterpret_cast<oroDeviceptr>( boxNodes ),
\t\t\treinterpret_cast<oroDeviceptr>( headers[i].m_boxNodes ),
\t\t\tboxNodeSize * headers[i].m_boxNodeCount,
\t\t\tstream ) );

\t\tcheckOro( oroMemcpyDtoDAsync(
\t\t\treinterpret_cast<oroDeviceptr>( primNodes ),
\t\t\treinterpret_cast<oroDeviceptr>( headers[i].m_primNodes ),
\t\t\tprimNodeSize * headers[i].m_primNodeCount,
\t\t\tstream ) );

\t\theaders[i].m_boxNodes  = boxNodes;
\t\theaders[i].m_primNodes = primNodes;
\t\theaders[i].m_size      = sizes[i];
\t\tcheckOro(
\t\t\toroMemcpyHtoDAsync( reinterpret_cast<oroDeviceptr>( geometriesOut[i] ), &headers[i], sizeof( GeomHeader ), stream ) );

\t\tbuffer = static_cast<uint8_t*>( buffer ) + sizes[i];
\t}

\t{
\t\tstd::lock_guard<std::mutex> lockMutex( m_poolMutex );
\t\tm_poolHeads[{ reinterpret_cast<oroDeviceptr>( geometriesOut.front() ), size }] =
\t\t\tstatic_cast<uint32_t>( geometriesOut.size() );
\t}

\tcheckOro( oroStreamSynchronize( stream ) );
\tdestroyGeometries( geometriesIn );

\treturn geometriesOut;
}"
    _hiprt_context_cpp "${_hiprt_context_cpp}")
if(_hiprt_context_cpp STREQUAL _hiprt_context_cpp_orig)
    message(FATAL_ERROR "[HIPRT from source] Failed to patch Context.cpp overlay.")
endif()
_hiprt_write_overlay_if_changed("${_hiprt_generated_dir}/hiprt/impl/Context.cpp" "${_hiprt_context_cpp}")

# Fix BatchBuilderKernels.h bugs:
# 1. taskQueue is uint3*, so taskQueue + 1 * blockDim.x advances 3 * blockDim.x uint32s
#    and overflows __shared__ BatchBuildCacheSize when blockDim.x == 512.
# 2. Collapse() contains __syncthreads(); round up the loop limit so all block threads
#    participate uniformly across every barrier iteration.
file(READ "${HIPRT_SOURCE_DIR}/hiprt/impl/BatchBuilderKernels.h" _hiprt_batch_builder_kernels_h)
string(REPLACE "\r\n" "\n" _hiprt_batch_builder_kernels_h "${_hiprt_batch_builder_kernels_h}")
set(_hiprt_batch_builder_kernels_h_before_sort "${_hiprt_batch_builder_kernels_h}")
string(REPLACE
"\tuint32_t* mortonCodeKeys   = reinterpret_cast<uint32_t*>( taskQueue + 0 * blockDim.x );
\tuint32_t* mortonCodeValues = reinterpret_cast<uint32_t*>( taskQueue + 1 * blockDim.x );"
"\tuint32_t* mortonCodeKeys   = reinterpret_cast<uint32_t*>( taskQueue );
\tuint32_t* mortonCodeValues = mortonCodeKeys + blockDim.x;"
    _hiprt_batch_builder_kernels_h "${_hiprt_batch_builder_kernels_h}")
if(_hiprt_batch_builder_kernels_h STREQUAL _hiprt_batch_builder_kernels_h_before_sort)
    message(FATAL_ERROR "[HIPRT from source] Failed to patch BatchBuilderKernels.h mortonCodeValues offset.")
endif()
set(_hiprt_batch_builder_kernels_h_before_collapse "${_hiprt_batch_builder_kernels_h}")
string(REPLACE
"\tfor ( uint32_t i = index; i < BranchingFactor * maxBoxNodeCount; i += blockDim.x )
\t{
\t\tCollapse<PrimitiveNode>(
\t\t\ti, maxBoxNodeCount, primCount, header, scratchNodes, references, boxNodes, taskQueue, referenceIndices );
\t\t__syncthreads();
\t}"
"\tconst uint32_t collapseLimit = RoundUp( BranchingFactor * maxBoxNodeCount, static_cast<uint32_t>( blockDim.x ) );
\tfor ( uint32_t i = index; i < collapseLimit; i += blockDim.x )
\t{
\t\tCollapse<PrimitiveNode>(
\t\t\ti, maxBoxNodeCount, primCount, header, scratchNodes, references, boxNodes, taskQueue, referenceIndices );
\t\t__syncthreads();
\t}"
    _hiprt_batch_builder_kernels_h "${_hiprt_batch_builder_kernels_h}")
if(_hiprt_batch_builder_kernels_h STREQUAL _hiprt_batch_builder_kernels_h_before_collapse)
    message(FATAL_ERROR "[HIPRT from source] Failed to patch BatchBuilderKernels.h Collapse barrier loop.")
endif()
_hiprt_write_overlay_if_changed("${_hiprt_generated_dir}/hiprt/impl/BatchBuilderKernels.h" "${_hiprt_batch_builder_kernels_h}")

# 4. Determine GPU architectures for bitcode compilation
# ---------------------------------------------------------------------------
# When HIPRT_GPU_ARCHS is set externally (e.g. -DHIPRT_GPU_ARCHS="gfx1201"),
# use only those architectures. Otherwise, build for all supported archs.
if(DEFINED HIPRT_GPU_ARCHS AND NOT HIPRT_GPU_ARCHS STREQUAL "")
    string(REPLACE "," ";" _hiprt_gpu_archs "${HIPRT_GPU_ARCHS}")
    string(REPLACE " " ";" _hiprt_gpu_archs "${_hiprt_gpu_archs}")
else()
    set(_hiprt_gpu_archs
        gfx1100 gfx1101 gfx1102 gfx1103
        gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036
        gfx1010 gfx1011 gfx1012 gfx1013
        gfx900 gfx902 gfx904 gfx906 gfx908 gfx909 gfx90a gfx90c gfx942)

    if(HIP_SDK_VERSION_NUM GREATER_EQUAL 61)
        list(APPEND _hiprt_gpu_archs gfx1150 gfx1151)
    endif()
    if(HIP_SDK_VERSION_NUM GREATER_EQUAL 63)
        list(APPEND _hiprt_gpu_archs gfx1152)
    endif()
    if(HIP_SDK_VERSION_NUM GREATER_EQUAL 64)
        list(APPEND _hiprt_gpu_archs gfx1200 gfx1201 gfx1153)
    endif()
endif()

set(_hiprt_offload_args "")
foreach(_arch ${_hiprt_gpu_archs})
    list(APPEND _hiprt_offload_args "--offload-arch=${_arch}")
endforeach()

message(STATUS "[HIPRT from source] Target architectures: ${_hiprt_gpu_archs}")

# ---------------------------------------------------------------------------
# 5. Compile bitcode and fat binaries (3 artifacts)
# ---------------------------------------------------------------------------
set(_hiprt_bitcode_dir "${CMAKE_CURRENT_BINARY_DIR}/hiprt_bitcodes")
file(MAKE_DIRECTORY "${_hiprt_bitcode_dir}")

if(WIN32)
    set(HIPRT_FROM_SOURCE_PLATFORM win)
    set(_hiprt_os_define -DHIPCC_OS_WINDOWS)
else()
    set(HIPRT_FROM_SOURCE_PLATFORM linux)
    set(_hiprt_os_define -DHIPCC_OS_LINUX)
endif()

set(_hiprt_bc_file "${_hiprt_bitcode_dir}/${HIPRT_LIB_NAME}_${HIP_VERSION_STR}_amd_lib_${HIPRT_FROM_SOURCE_PLATFORM}.bc")
set(_hiprt_hipfb_file "${_hiprt_bitcode_dir}/${HIPRT_LIB_NAME}_${HIP_VERSION_STR}_amd.hipfb")
set(_hiprt_oro_hipfb_file "${_hiprt_bitcode_dir}/oro_compiled_kernels.hipfb")

find_path(_LIBCXX_INCLUDE_DIR NAMES __config
    PATHS
        "${hip_INCLUDE_DIRS}/../lib/llvm/lib/c++/v1"
        "$ENV{ROCM_PATH}/lib/llvm/lib/c++/v1"
        "$ENV{ROCM_PATH}/include/c++/v1"
    PATH_SUFFIXES c++/v1
)
if(NOT _LIBCXX_INCLUDE_DIR)
    message(WARNING "[HIPRT] libc++ headers not found; hipcc device compilation may fail. "
                    "Set ROCM_PATH or ensure libc++ is installed.")
endif()

set(_hiprt_common_flags -O3 -std=c++17 -ffast-math -parallel-jobs=15
    -DHIPRT_MATRIX_FRAME=1
    -I "${_hiprt_generated_dir}"
    -I "${hip_INCLUDE_DIRS}")
if(_LIBCXX_INCLUDE_DIR)
    list(APPEND _hiprt_common_flags -stdlib=libc++ "-isystem${_LIBCXX_INCLUDE_DIR}")
endif()

# hipcc does not emit a depfile for these header-as-translation-unit commands.
# Track the transitive source trees explicitly so edits to included HIPRT/Orochi
# kernels cannot leave the precompiled runtime artifacts stale.
file(GLOB_RECURSE _hiprt_kernel_dependencies CONFIGURE_DEPENDS
    "${HIPRT_SOURCE_DIR}/hiprt/*.h"
    "${HIPRT_SOURCE_DIR}/hiprt/*.inl")
file(GLOB_RECURSE _orochi_kernel_dependencies CONFIGURE_DEPENDS
    "${HIPRT_SOURCE_DIR}/contrib/Orochi/Orochi/*.h"
    "${HIPRT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/*.h")

# 5a. Bitcode bundle (.bc) - for hiprtBuildTraceKernelsFromBitcode
add_custom_command(
    OUTPUT "${_hiprt_bc_file}"
    COMMAND hipcc
        -x hip "${HIPRT_SOURCE_DIR}/hiprt/impl/hiprt_kernels_bitcode.h"
        ${_hiprt_common_flags}
        ${_hiprt_offload_args}
        -fgpu-rdc -c --gpu-bundle-output -c -emit-llvm
        -I "${HIPRT_SOURCE_DIR}/contrib/Orochi/"
        -I "${HIPRT_SOURCE_DIR}"
        -DHIPRT_BITCODE_LINKING ${_hiprt_os_define}
        -o "${_hiprt_bc_file}"
    DEPENDS ${_hiprt_kernel_dependencies}
            ${_orochi_kernel_dependencies}
    WORKING_DIRECTORY "${HIPRT_SOURCE_DIR}"
    COMMENT "[HIPRT] Compiling bitcode bundle: ${HIPRT_LIB_NAME}_${HIP_VERSION_STR}_amd_lib_${HIPRT_FROM_SOURCE_PLATFORM}.bc"
    VERBATIM)

# 5b. Fat binary (.hipfb) - precompiled BVH build kernels
add_custom_command(
    OUTPUT "${_hiprt_hipfb_file}"
    COMMAND hipcc
        -x hip "${HIPRT_SOURCE_DIR}/hiprt/impl/hiprt_kernels.h"
        ${_hiprt_common_flags}
        ${_hiprt_offload_args}
        -mllvm -amdgpu-early-inline-all=false
        -mllvm -amdgpu-function-calls=true
        --genco
        -I "${HIPRT_SOURCE_DIR}"
        -DHIPRT_BITCODE_LINKING ${_hiprt_os_define}
        -o "${_hiprt_hipfb_file}"
    DEPENDS ${_hiprt_kernel_dependencies}
            "${_hiprt_generated_dir}/hiprt/impl/BatchBuilderKernels.h"
    WORKING_DIRECTORY "${HIPRT_SOURCE_DIR}"
    COMMENT "[HIPRT] Compiling fat binary: ${HIPRT_LIB_NAME}_${HIP_VERSION_STR}_amd.hipfb"
    VERBATIM)

# 5c. Orochi radix sort kernels (.hipfb)
add_custom_command(
    OUTPUT "${_hiprt_oro_hipfb_file}"
    COMMAND hipcc
        -x hip "${HIPRT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/RadixSortKernels.h"
        ${_hiprt_common_flags}
        ${_hiprt_offload_args}
        --genco
        -I "${HIPRT_SOURCE_DIR}/contrib/Orochi/"
        -include hip/hip_runtime.h
        -DHIPRT_BITCODE_LINKING ${_hiprt_os_define}
        -o "${_hiprt_oro_hipfb_file}"
    DEPENDS ${_orochi_kernel_dependencies}
    WORKING_DIRECTORY "${HIPRT_SOURCE_DIR}"
    COMMENT "[HIPRT] Compiling Orochi radix sort kernels: oro_compiled_kernels.hipfb"
    VERBATIM)

add_custom_target(hiprt_compile_bitcodes
    DEPENDS "${_hiprt_bc_file}" "${_hiprt_hipfb_file}" "${_hiprt_oro_hipfb_file}"
    COMMENT "[HIPRT] All bitcode/hipfb artifacts compiled")

# ---------------------------------------------------------------------------
# 6. Collect source files
# ---------------------------------------------------------------------------
file(GLOB_RECURSE _hiprt_sources
    "${HIPRT_SOURCE_DIR}/hiprt/*.h"
    "${HIPRT_SOURCE_DIR}/hiprt/*.cpp"
    "${HIPRT_SOURCE_DIR}/hiprt/*.inl")
list(FILTER _hiprt_sources EXCLUDE REGEX "hiprt/bitcodes/.*")
list(FILTER _hiprt_sources EXCLUDE REGEX "hiprt/impl/(BatchBuilder|Context)\\.cpp$")
# Exclude .h.in templates
list(FILTER _hiprt_sources EXCLUDE REGEX "\\.h\\.in$")
list(APPEND _hiprt_sources
    "${_hiprt_generated_dir}/hiprt/impl/BatchBuilder.cpp"
    "${_hiprt_generated_dir}/hiprt/impl/Context.cpp")

file(GLOB_RECURSE _orochi_sources
    "${HIPRT_SOURCE_DIR}/contrib/Orochi/Orochi/*.h"
    "${HIPRT_SOURCE_DIR}/contrib/Orochi/Orochi/*.cpp"
    "${HIPRT_SOURCE_DIR}/contrib/Orochi/contrib/cuew/*.h"
    "${HIPRT_SOURCE_DIR}/contrib/Orochi/contrib/cuew/*.cpp"
    "${HIPRT_SOURCE_DIR}/contrib/Orochi/contrib/hipew/*.h"
    "${HIPRT_SOURCE_DIR}/contrib/Orochi/contrib/hipew/*.cpp"
    "${HIPRT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/*.h"
    "${HIPRT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/*.cpp")

# ---------------------------------------------------------------------------
# 7. Build shared library
# ---------------------------------------------------------------------------
add_library(hiprt_from_source SHARED ${_hiprt_sources} ${_orochi_sources})

set_target_properties(hiprt_from_source PROPERTIES
    OUTPUT_NAME "${HIPRT_LIB_NAME}64"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

if(UNIX)
    target_compile_options(hiprt_from_source PRIVATE -fvisibility=hidden)
endif()

# Generated headers take priority (hiprt.h, hiprtew.h)
target_include_directories(hiprt_from_source PRIVATE
    "${_hiprt_generated_dir}"
    "${HIPRT_SOURCE_DIR}"
    "${HIPRT_SOURCE_DIR}/contrib/Orochi")

# Also expose generated headers for consumers
target_include_directories(hiprt_from_source PUBLIC
    "${_hiprt_generated_dir}")

target_compile_definitions(hiprt_from_source PRIVATE
    $<$<PLATFORM_ID:Windows>:__WINDOWS__>
    HIPRT_EXPORTS
    __USE_HIP__
    HIPRT_PUBLIC_REPO
    HIPRT_BITCODE_LINKING
    HIPRT_MATRIX_FRAME=1
    ORO_PRECOMPILED)

if(UNIX)
    target_link_libraries(hiprt_from_source PRIVATE pthread dl)
elseif(WIN32)
    target_link_libraries(hiprt_from_source PRIVATE version)
endif()

# Depend on bitcode compilation
add_dependencies(hiprt_from_source hiprt_compile_bitcodes)

# ---------------------------------------------------------------------------
# 8. Export variables for consumers
# ---------------------------------------------------------------------------
set(HIPRT_FROM_SOURCE_INCLUDE_DIR "${_hiprt_generated_dir}" PARENT_SCOPE)
set(HIPRT_FROM_SOURCE_LIB_TARGET hiprt_from_source PARENT_SCOPE)
set(HIPRT_FROM_SOURCE_BITCODE_DIR "${_hiprt_bitcode_dir}" PARENT_SCOPE)
set(HIPRT_FROM_SOURCE_LIB_NAME "${HIPRT_LIB_NAME}" PARENT_SCOPE)

# Also set the variables at current scope for immediate use
set(HIPRT_FROM_SOURCE_INCLUDE_DIR "${_hiprt_generated_dir}")
set(HIPRT_FROM_SOURCE_LIB_TARGET hiprt_from_source)
set(HIPRT_FROM_SOURCE_BITCODE_DIR "${_hiprt_bitcode_dir}")
set(HIPRT_FROM_SOURCE_LIB_NAME "${HIPRT_LIB_NAME}")
