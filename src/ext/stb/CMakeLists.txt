add_library(stb SHARED stb.c)
target_include_directories(stb PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
set_target_properties(stb PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
