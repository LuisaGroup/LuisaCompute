set(LUISA_COMPUTE_CPU_SOURCES
        ../common/rust_device_common.cpp ../common/rust_device_common.h
        cpu_device.h cpu_device.cpp)

if(LUISA_COMPUTE_ENABLE_TENSOR)
        find_package(MKL CONFIG REQUIRED QUIET)
        message(STATUS "Luisa Compute Tensor CPU backend: MKL")
        message(STATUS "MKL imported targets: ${MKL_IMPORTED_TARGETS}") # Provides available list of targets based on input
        file(GLOB_RECURSE TENSOR_SOURCES "tensor/*.cpp" "tensor/*.h")
        set(LUISA_COMPUTE_CPU_SOURCES "${LUISA_COMPUTE_CPU_SOURCES};${TENSOR_SOURCES}")
endif()

luisa_compute_add_backend(cpu SOURCES ${LUISA_COMPUTE_CPU_SOURCES})

target_link_libraries(luisa-compute-backend-cpu PRIVATE
        luisa-compute-vulkan-swapchain
        luisa-compute-rust-meta
        luisa_compute_backend_impl)

if(LUISA_COMPUTE_ENABLE_TENSOR)
        target_compile_options(luisa-compute-backend-cpu PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
        target_include_directories(luisa-compute-backend-cpu PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
        target_link_libraries(luisa-compute-backend-cpu PRIVATE 
                $<LINK_ONLY:MKL::MKL>
                luisa-compute-tensor)
endif()