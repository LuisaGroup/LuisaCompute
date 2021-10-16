#set(LUISA_COMPUTE_SERIALIZE_SOURCES
#)

#add_library(luisa-compute-serialize-json MODULE ${LUISA_COMPUTE_SERIALIZE_SOURCES})
#target_link_libraries(luisa-compute-serialize-json PRIVATE luisa-compute-vstl)
#set_target_properties(luisa-compute-serialize-json PROPERTIES UNITY_BUILD ON)

#add_library(luisa-compute-serialize INTERFACE)
#add_dependencies(luisa-compute-serialize luisa-compute-serialize-json)
