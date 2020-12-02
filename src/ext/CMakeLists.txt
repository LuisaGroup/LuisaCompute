find_package(OpenImageIO CONFIG REQUIRED)
add_library(luisa-ext-oiio INTERFACE)
target_link_libraries(luisa-ext-oiio INTERFACE OpenImageIO::OpenImageIO)

find_package(OpenCV CONFIG REQUIRED)
add_library(luisa-ext-opencv INTERFACE)
target_link_libraries(luisa-ext-opencv INTERFACE ${OpenCV_LIBS})
