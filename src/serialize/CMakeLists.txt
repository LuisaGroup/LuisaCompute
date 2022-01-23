set(LUISA_COMPUTE_SERIALIZE_SOURCES
astserializer.cpp
funcserializer.cpp
config.h
funcserializer.h
serialize.h
)
add_library(luisa-compute-serialize SHARED ${LUISA_COMPUTE_SERIALIZE_SOURCES})
target_link_libraries(luisa-compute-serialize PUBLIC luisa-compute-vstl luisa-compute-ast)

