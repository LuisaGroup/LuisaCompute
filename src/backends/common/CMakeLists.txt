# Apple metal-cpp is shared by the legacy Metal and Metal 4 backends. Keep the
# wrapper symbols private to each backend module so loading both plugins cannot
# interpose their Objective-C bridge globals.
if (APPLE)
    add_library(luisa-compute-metal-cpp INTERFACE)
    target_include_directories(luisa-compute-metal-cpp INTERFACE
            "${CMAKE_CURRENT_LIST_DIR}/metal-cpp")
    target_compile_definitions(luisa-compute-metal-cpp INTERFACE
            METALCPP_SYMBOL_VISIBILITY_HIDDEN=1)
endif ()

# OIDN Denoiser Extension
if (LUISA_COMPUTE_DOWNLOAD_OIDN)

    set(oidn_DOWNLOAD_VERSION 2.2.2)
    if (WIN32)
        set(oidn_DOWNLOAD_URL "https://github.com/RenderKit/oidn/releases/download/v${oidn_DOWNLOAD_VERSION}/oidn-${oidn_DOWNLOAD_VERSION}.x64.windows.zip")
    elseif (UNIX AND NOT APPLE)
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(oidn_DOWNLOAD_URL "https://github.com/RenderKit/oidn/releases/download/v${oidn_DOWNLOAD_VERSION}/oidn-${oidn_DOWNLOAD_VERSION}.aarch64.linux.tar.gz")
        else ()
            set(oidn_DOWNLOAD_URL "https://github.com/RenderKit/oidn/releases/download/v${oidn_DOWNLOAD_VERSION}/oidn-${oidn_DOWNLOAD_VERSION}.x86_64.linux.tar.gz")
        endif ()
    elseif (APPLE)
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
            set(oidn_DOWNLOAD_URL "https://github.com/RenderKit/oidn/releases/download/v${oidn_DOWNLOAD_VERSION}/oidn-${oidn_DOWNLOAD_VERSION}.x86_64.macos.tar.gz")
        else ()
            set(oidn_DOWNLOAD_URL "https://github.com/RenderKit/oidn/releases/download/v${oidn_DOWNLOAD_VERSION}/oidn-${oidn_DOWNLOAD_VERSION}.arm64.macos.tar.gz")
        endif ()
    endif ()

    message(STATUS "Downloading OpenImageDenoise from ${oidn_DOWNLOAD_URL}")

    include(FetchContent)
    FetchContent_Declare(OpenImageDenoise
            URL ${oidn_DOWNLOAD_URL}
            DOWNLOAD_EXTRACT_TIMESTAMP ON)
    FetchContent_MakeAvailable(OpenImageDenoise)
    FetchContent_GetProperties(OpenImageDenoise)
    message(STATUS "OpenImageDenoise binary downloaded to ${openimagedenoise_SOURCE_DIR}")
    list(APPEND CMAKE_PREFIX_PATH ${openimagedenoise_SOURCE_DIR})
    find_package(OpenImageDenoise CONFIG REQUIRED)
    file(GLOB oidn_RUNTIME_LIBS
            "${openimagedenoise_SOURCE_DIR}/bin/*.dll"
            "${openimagedenoise_SOURCE_DIR}/lib/*.so"
            "${openimagedenoise_SOURCE_DIR}/lib/*.so.*"
            "${openimagedenoise_SOURCE_DIR}/lib/*.dylib")

    add_library(luisa-compute-oidn-ext STATIC oidn_denoiser.cpp)
    set_target_properties(luisa-compute-oidn-ext PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # fix the rpaths of the OpenImageDenoise libraries
    if (APPLE) # recent version already has the correct RPATH
        # find_program(INSTALL_NAME_TOOL_EXE install_name_tool)
        # if (INSTALL_NAME_TOOL_EXE)
        #     foreach (oidn_lib ${oidn_RUNTIME_LIBS})
        #         # get file name without path
        #         get_filename_component(oidn_lib_name ${oidn_lib} NAME)
        #         add_custom_command(TARGET luisa-compute-oidn-ext POST_BUILD
        #                 COMMENT "Fixing rpath of ${oidn_lib_name}"
        #                 COMMAND ${INSTALL_NAME_TOOL_EXE} -id "@rpath/${oidn_lib_name}" "${oidn_lib}"
        #                 COMMAND ${INSTALL_NAME_TOOL_EXE} -add_rpath "@loader_path" "${oidn_lib}"
        #                 BYPRODUCTS ${oidn_lib}
        #                 VERBATIM)
        #     endforeach ()
        # else ()
        #     message(FATAL_ERROR "install_name_tool not found, rpath of OpenImageDenoise libraries will not be fixed. You may either install Xcode or disable the OIDN extension with -DLUISA_COMPUTE_DOWNLOAD_OIDN=OFF when configuring.")
        # endif ()
    elseif (UNIX)
        find_program(PATCHELF_EXE patchelf)
        if (PATCHELF_EXE)
            foreach (oidn_lib ${oidn_RUNTIME_LIBS})
                add_custom_command(TARGET luisa-compute-oidn-ext POST_BUILD
                        COMMENT "Fixing rpath of ${oidn_lib}"
                        COMMAND ${PATCHELF_EXE} --set-rpath "$ORIGIN" "${oidn_lib}"
                        BYPRODUCTS ${oidn_lib}
                        VERBATIM)
            endforeach ()
        else ()
            message(FATAL_ERROR "patchelf not found, rpath of OpenImageDenoise libraries will not be fixed. You may either install patchelf or disable the OIDN extension with -DLUISA_COMPUTE_DOWNLOAD_OIDN=OFF when configuring.")
        endif ()
    endif ()
    target_compile_definitions(luisa-compute-oidn-ext
            PUBLIC LUISA_BACKEND_ENABLE_OIDN=1
            PRIVATE LUISA_BACKEND_EXPORT_DLL)
    target_link_libraries(luisa-compute-oidn-ext PUBLIC
            luisa-compute-runtime
            OpenImageDenoise)

    # copy the OpenImageDenoise dll to the output directory
    message(STATUS "OpenImageDenoise runtime libs: ${oidn_RUNTIME_LIBS}")
    foreach (oidn_RUNTIME_LIB ${oidn_RUNTIME_LIBS})
        add_custom_command(TARGET luisa-compute-oidn-ext POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${oidn_RUNTIME_LIB}"
                "$<TARGET_FILE_DIR:luisa-compute-core>")
    endforeach ()
    install(FILES ${oidn_RUNTIME_LIBS} DESTINATION ${CMAKE_INSTALL_BINDIR})

endif ()

function(luisa_compute_link_wayland_if_enabled backend)
    if (UNIX AND NOT APPLE AND LUISA_COMPUTE_ENABLE_WAYLAND)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(Wayland REQUIRED wayland-client)
        target_include_directories(${backend} PRIVATE ${Wayland_INCLUDE_DIRS})
        target_link_libraries(${backend} PRIVATE ${Wayland_LIBRARIES})
    endif ()
endfunction()

# luisa_compute_link_llvm_into_backend(
#     backend [TARGET target] [REQUIRED] COMPONENTS llvm_components...)
function(luisa_compute_link_llvm_into_backend backend)

    cmake_parse_arguments(
            LUISA_LINK_LLVM "REQUIRED" "TARGET" "COMPONENTS;DEFINITIONS" ${ARGN})
    if (NOT LUISA_LINK_LLVM_TARGET)
        set(LUISA_LINK_LLVM_TARGET luisa-compute-backend-${backend})
    endif ()

    # find LLVM
    if (NOT LUISA_LINK_LLVM_REQUIRED)
        find_package(LLVM QUIET CONFIG)
    else ()
        find_package(LLVM CONFIG REQUIRED)
    endif ()

    if (NOT LLVM_FOUND)
        message(STATUS "LLVM not found, skipping LLVM linking for backend ${backend}")
        return()
    endif ()

    # FindZLIB can resolve Apple's SDK copy and publish
    # <SDK>/usr/include through ZLIB::ZLIB. Propagating that directory as an
    # ordinary -isystem argument breaks non-Apple Clang toolchains on macOS:
    # it is searched before Clang's libc++ overlay headers and mixes two SDK
    # header stacks (for example, <cstddef> then reports a missing size_t).
    # SDK headers are already part of the compiler's implicit/sysroot search,
    # so keep the link dependency but remove only this redundant include.
    if (APPLE AND TARGET ZLIB::ZLIB)
        get_target_property(_luisa_zlib_include_dirs
                ZLIB::ZLIB INTERFACE_INCLUDE_DIRECTORIES)
        if (_luisa_zlib_include_dirs)
            set(_luisa_zlib_filtered_include_dirs "")
            set(_luisa_zlib_removed_sdk_include FALSE)
            foreach (_luisa_zlib_include_dir IN LISTS _luisa_zlib_include_dirs)
                if (_luisa_zlib_include_dir MATCHES
                        "\\.sdk[/\\\\]usr[/\\\\]include[/\\\\]?$")
                    set(_luisa_zlib_removed_sdk_include TRUE)
                else ()
                    list(APPEND _luisa_zlib_filtered_include_dirs
                            "${_luisa_zlib_include_dir}")
                endif ()
            endforeach ()
            if (_luisa_zlib_removed_sdk_include)
                set_property(TARGET ZLIB::ZLIB PROPERTY
                        INTERFACE_INCLUDE_DIRECTORIES
                        "${_luisa_zlib_filtered_include_dirs}")
                message(STATUS
                        "Removed redundant Apple SDK include directory from "
                        "LLVM's ZLIB dependency")
            endif ()
        endif ()
    endif ()

    # LLVM's generic Unix configuration appends librt to LLVMSupport. Apple
    # platforms provide the corresponding APIs from libSystem, and iOS does
    # not ship a standalone librt archive. Keep the imported target otherwise
    # intact so the compiler stack remains linked exactly as exported by the
    # selected LLVM package.
    if (CMAKE_SYSTEM_NAME STREQUAL "iOS" AND TARGET LLVMSupport)
        get_target_property(_luisa_llvm_support_link_libraries
                LLVMSupport INTERFACE_LINK_LIBRARIES)
        if (_luisa_llvm_support_link_libraries)
            list(FILTER _luisa_llvm_support_link_libraries
                    EXCLUDE REGEX "^rt$")
            set_property(TARGET LLVMSupport PROPERTY
                    INTERFACE_LINK_LIBRARIES
                    "${_luisa_llvm_support_link_libraries}")
        endif ()
    endif ()

    include(HandleLLVMOptions)

    message(STATUS "Linking LLVM to target ${LUISA_LINK_LLVM_TARGET} with components: ${LUISA_LINK_LLVM_COMPONENTS}")

    # LLVM
    target_include_directories(${LUISA_LINK_LLVM_TARGET} AFTER PRIVATE ${LLVM_INCLUDE_DIRS})
    llvm_map_components_to_libnames(LLVM_LIBS ${LUISA_LINK_LLVM_COMPONENTS})

    # LLVM's C++ ABI includes its RTTI mode. In particular, polymorphic helper
    # classes implemented partly in headers (for example CallbackVH, used by
    # MergeFunctions) make a consumer compiled with RTTI emit references to
    # typeinfo that an RTTI-disabled static LLVM deliberately does not define.
    # Match the selected LLVM package instead of relying on which inline
    # templates a backend happens to instantiate today. LLVM with RTTI enabled
    # needs no override: LuisaCompute's normal compiler policy remains valid.
    if (DEFINED LLVM_ENABLE_RTTI AND NOT LLVM_ENABLE_RTTI)
        target_compile_options(${LUISA_LINK_LLVM_TARGET} PRIVATE
                "$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang,GNU>:-fno-rtti>"
                "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/GR->")
    endif ()

    # llvm_map_components_to_libnames() deliberately turns unknown component
    # names into LLVM<name> and leaves the error to a later link step. Validate
    # the result here so a typo cannot make us silently select a monolithic
    # library from an unrelated LLVM installation. A backend must be compiled
    # and linked against one LLVM package: its C++ enums and ABI are not stable
    # across downstream branches that advertise the same major version.
    set(_luisa_llvm_available_libs ${LLVM_AVAILABLE_LIBS})
    if (NOT _luisa_llvm_available_libs)
        get_property(_luisa_llvm_available_libs
                GLOBAL PROPERTY LLVM_LIBS)
    endif ()
    string(TOUPPER "${_luisa_llvm_available_libs}"
            _luisa_llvm_available_libs_upper)
    set(_luisa_llvm_unknown_libs "")
    foreach (_luisa_llvm_lib IN LISTS LLVM_LIBS)
        string(TOUPPER "${_luisa_llvm_lib}"
                _luisa_llvm_lib_upper)
        list(FIND _luisa_llvm_available_libs_upper
                "${_luisa_llvm_lib_upper}"
                _luisa_llvm_available_index)
        if (NOT TARGET "${_luisa_llvm_lib}" AND
                _luisa_llvm_available_index LESS 0)
            list(APPEND _luisa_llvm_unknown_libs
                    "${_luisa_llvm_lib}")
        endif ()
    endforeach ()
    if (_luisa_llvm_unknown_libs)
        message(FATAL_ERROR
                "Unknown LLVM components for backend '${backend}' mapped to: "
                "${_luisa_llvm_unknown_libs}. Requested components: "
                "${LUISA_LINK_LLVM_COMPONENTS}.")
    endif ()

    if (UNIX AND NOT APPLE)
        if (LLVM_LINK_LLVM_DYLIB AND TARGET LLVM)
            # Respect LLVM's exported shared-library policy. Recent official
            # Ubuntu packages expose component archives whose transitive
            # imported dependencies (for example zstd::libzstd_shared) cannot
            # be reconstructed by this try_compile project, while the exported
            # monolithic LLVM target is complete and relocatable.
            message(STATUS "Linking with LLVM shared library.")
            set(LLVM_LIBS LLVM)
        else ()
            # check if the libraries exist
            # map library dirs to link directories options
            set(LLVM_LINK_DIR_OPTIONS "")
            foreach (dir ${LLVM_LIBRARY_DIRS})
                list(APPEND LLVM_LINK_DIR_OPTIONS "-L${dir}")
            endforeach ()
            string(TOUPPER "${backend}" _luisa_llvm_backend_upper)
            set(_luisa_llvm_probe_var
                    "LUISA_COMPUTE_${_luisa_llvm_backend_upper}_LLVM_LIBS_EXIST")
            unset(${_luisa_llvm_probe_var} CACHE)
            unset(${_luisa_llvm_probe_var})
            try_compile(${_luisa_llvm_probe_var}
                    SOURCE_FROM_CONTENT "try_link_llvm.cpp" "int main() {}"
                    LINK_OPTIONS ${LLVM_LINK_DIR_OPTIONS}
                    LINK_LIBRARIES ${LLVM_LIBS})
            if (${_luisa_llvm_probe_var})
                # libzstd_shared might cause problem on Ubuntu
                if (NOT TARGET zstd::libzstd_shared)
                    get_target_property(LLVM_SUPPORT_LINK_LIBS LLVMSupport INTERFACE_LINK_LIBRARIES)
                    list(FILTER LLVM_SUPPORT_LINK_LIBS EXCLUDE REGEX "(zstd::libzstd_shared)")
                    set_target_properties(LLVMSupport PROPERTIES INTERFACE_LINK_LIBRARIES "${LLVM_SUPPORT_LINK_LIBS}")
                endif ()
            else ()
                # Some distributions export incomplete static targets but do
                # ship a monolithic LLVM library. It is a valid fallback only
                # when it comes from the very same package directories as the
                # headers selected by find_package(LLVM).
                unset(_luisa_llvm_dylib CACHE)
                unset(_luisa_llvm_dylib)
                find_library(_luisa_llvm_dylib
                        NAMES LLVM "LLVM-${LLVM_VERSION_MAJOR}"
                        HINTS ${LLVM_LIBRARY_DIRS}
                        NO_DEFAULT_PATH)
                if (NOT _luisa_llvm_dylib)
                    message(FATAL_ERROR
                            "LLVM component libraries for backend '${backend}' "
                            "failed to link, and no monolithic LLVM library "
                            "exists in LLVM_LIBRARY_DIRS='${LLVM_LIBRARY_DIRS}'. "
                            "Refusing an unscoped system-library fallback that "
                            "could mix incompatible LLVM headers and binaries.")
                endif ()
                message(STATUS
                        "Linking with LLVM shared library from the selected "
                        "package: ${_luisa_llvm_dylib}")
                set(LLVM_LIBS "${_luisa_llvm_dylib}")
            endif ()
        endif ()
    endif ()

    message(STATUS "Linking with LLVM libraries: ${LLVM_LIBS}")
    if (UNIX AND NOT APPLE)
        # avoid re-exporting symbols from LLVM
        target_link_libraries(${LUISA_LINK_LLVM_TARGET} PRIVATE "-Wl,--exclude-libs,ALL" ${LLVM_LIBS})
    else ()
        target_link_libraries(${LUISA_LINK_LLVM_TARGET} PRIVATE ${LLVM_LIBS})
    endif ()

    target_compile_definitions(${LUISA_LINK_LLVM_TARGET} PRIVATE ${LUISA_LINK_LLVM_DEFINITIONS})

    # return LUISA_COMPUTE_${backend.upper}_LINKED_LLVM=ON to the caller
    string(TOUPPER ${backend} backend_upper)
    set(LUISA_COMPUTE_${backend_upper}_LINKED_LLVM ON PARENT_SCOPE)

endfunction()

if (LUISA_COMPUTE_ENABLE_CUDA OR
        LUISA_COMPUTE_ENABLE_FALLBACK OR
        LUISA_COMPUTE_ENABLE_HIP)

    if (UNIX AND NOT APPLE)
        find_package(X11)
    endif ()
    set(LUISA_COMPUTE_VULKAN_SWAPCHAIN_SOURCES
            vulkan_instance.cpp vulkan_instance.h
            vulkan_swapchain.cpp)

    if (APPLE AND CMAKE_C_COMPILER_ID MATCHES "Clang")
        enable_language(OBJC)
        enable_language(OBJCXX)
        list(APPEND LUISA_COMPUTE_VULKAN_SWAPCHAIN_SOURCES moltenvk_surface.mm)
    endif ()

    add_library(luisa-compute-vulkan-swapchain OBJECT
            ${LUISA_COMPUTE_VULKAN_SWAPCHAIN_SOURCES})
    target_link_libraries(luisa-compute-vulkan-swapchain
            PUBLIC
            luisa-compute-runtime
            luisa-compute-ext-volk
            ${X11_LIBRARIES})
    target_compile_definitions(luisa-compute-vulkan-swapchain
            PUBLIC
            LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN=1)

    if (UNIX AND NOT APPLE)
        set_target_properties(luisa-compute-vulkan-swapchain PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif ()

    if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_link_libraries(luisa-compute-vulkan-swapchain
                PUBLIC "-framework QuartzCore" "-framework AppKit")
    endif ()

    # wayland support
    luisa_compute_link_wayland_if_enabled(luisa-compute-vulkan-swapchain)
    set_target_properties(luisa-compute-vulkan-swapchain PROPERTIES
            OUTPUT_NAME luisa-vulkan-swapchain)
endif ()

add_subdirectory(hlsl)

if (LUISA_COMPUTE_ENABLE_VULKAN)
    add_subdirectory(spirv)
endif ()
if (LUISA_COMPUTE_ENABLE_VULKAN AND
        LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV)
    add_subdirectory(spirv_llvm)
endif ()
