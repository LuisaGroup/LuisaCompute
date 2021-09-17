add_executable(test_type test_type.cpp)
target_link_libraries(test_type PRIVATE luisa::compute)

add_executable(test_dsl test_dsl.cpp)
target_link_libraries(test_dsl PRIVATE luisa::compute)

add_executable(test_dsl_multithread test_dsl_multithread.cpp)
target_link_libraries(test_dsl_multithread PRIVATE luisa::compute)

add_executable(test_dsl_sugar test_dsl_sugar.cpp)
target_link_libraries(test_dsl_sugar PRIVATE luisa::compute)

add_executable(test_runtime test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE luisa::compute)

add_executable(test_simple test_simple.cpp)
target_link_libraries(test_simple PRIVATE luisa::compute)

add_executable(test_texture_io test_texture_io.cpp)
target_link_libraries(test_texture_io PRIVATE luisa::compute)

add_executable(test_builtin test_builtin.cpp)
target_link_libraries(test_builtin PRIVATE luisa::compute)

add_executable(test_atomic test_atomic.cpp)
target_link_libraries(test_atomic PRIVATE luisa::compute)

add_executable(test_bindless test_bindless.cpp)
target_link_libraries(test_bindless PRIVATE luisa::compute)

add_executable(test_rtx test_rtx.cpp)
target_link_libraries(test_rtx PRIVATE luisa::compute)

add_executable(test_path_tracing test_path_tracing.cpp)
target_link_libraries(test_path_tracing PRIVATE luisa::compute)

add_executable(test_wavefront_pt test_wavefront_pt.cpp)
target_link_libraries(test_wavefront_pt PRIVATE luisa::compute)

add_executable(test_playground test_playground.cpp)
target_link_libraries(test_playground PRIVATE luisa::compute)

add_executable(test_network_master test_network_master.cpp)
target_link_libraries(test_network_master PRIVATE luisa::compute)

add_executable(test_network_client test_network_client.cpp)
target_link_libraries(test_network_client PRIVATE luisa::compute)

add_executable(test_network_worker test_network_worker.cpp)
target_link_libraries(test_network_worker PRIVATE luisa::compute)

add_executable(test_serialize test_serialize.cpp)
target_link_libraries(test_serialize PRIVATE luisa-compute-vstl luisa-compute-serialize)

find_package(OpenCV CONFIG)
if (OpenCV_FOUND)
    add_executable(test_shader_toy test_shader_toy.cpp)
    target_link_libraries(test_shader_toy PRIVATE luisa::compute ${OpenCV_LIBS})
    add_executable(test_game_of_life test_game_of_life.cpp)
    target_link_libraries(test_game_of_life PRIVATE luisa::compute ${OpenCV_LIBS})
    add_executable(test_path_tracing_display test_path_tracing_display.cpp)
    target_link_libraries(test_path_tracing_display PRIVATE luisa::compute ${OpenCV_LIBS})
endif ()
