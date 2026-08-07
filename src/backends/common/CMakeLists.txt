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

# luisa_compute_backend_link_llvm(backend REQUIRED COMPONENTS llvm_components...)
function(luisa_compute_link_llvm_into_backend backend)

    set(LUISA_LINK_LLVM_TARGET luisa-compute-backend-${backend})
    cmake_parse_arguments(LUISA_LINK_LLVM "REQUIRED" "" "COMPONENTS;DEFINITIONS" ${ARGN})

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

    include(HandleLLVMOptions)

    message(STATUS "Linking LLVM to target ${LUISA_LINK_LLVM_TARGET} with components: ${LUISA_LINK_LLVM_COMPONENTS}")

    # LLVM
    target_include_directories(${LUISA_LINK_LLVM_TARGET} AFTER PRIVATE ${LLVM_INCLUDE_DIRS})
    llvm_map_components_to_libnames(LLVM_LIBS ${LUISA_LINK_LLVM_COMPONENTS})
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
            try_compile(LLVM_LIBS_EXIST
                    SOURCE_FROM_CONTENT "try_link_llvm.cpp" "int main() {}"
                    LINK_OPTIONS ${LLVM_LINK_DIR_OPTIONS}
                    LINK_LIBRARIES ${LLVM_LIBS})
            if (LLVM_LIBS_EXIST)
                # libzstd_shared might cause problem on Ubuntu
                if (NOT TARGET zstd::libzstd_shared)
                    get_target_property(LLVM_SUPPORT_LINK_LIBS LLVMSupport INTERFACE_LINK_LIBRARIES)
                    list(FILTER LLVM_SUPPORT_LINK_LIBS EXCLUDE REGEX "(zstd::libzstd_shared)")
                    set_target_properties(LLVMSupport PROPERTIES INTERFACE_LINK_LIBRARIES "${LLVM_SUPPORT_LINK_LIBS}")
                endif ()
            else ()
                message(WARNING "LLVM libraries not found. Trying to link with LLVM shared libraries.")
                try_compile(LLVM_DYLIB_EXISTS
                        SOURCE_FROM_CONTENT "try_link_llvm_dylib.cpp" "int main() {}"
                        LINK_LIBRARIES "LLVM")
                if (LLVM_DYLIB_EXISTS)
                    message(STATUS "Linking with LLVM shared library.")
                    set(LLVM_LIBS "LLVM")
                else ()
                    find_library(LLVM_DYLIB LLVM REQUIRED)
                    message(STATUS "LLVM libraries found at ${LLVM_DYLIB}")
                    set(LLVM_LIBS ${LLVM_DYLIB})
                endif ()
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

if (LUISA_COMPUTE_ENABLE_VULKAN AND
        (LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV OR LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV))
    add_subdirectory(spirv)
endif ()
if (LUISA_COMPUTE_ENABLE_VULKAN AND
        LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV)
    add_subdirectory(spirv_llvm)
endif ()
