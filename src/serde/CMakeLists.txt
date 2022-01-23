set(SOURCEFILE
NoSqlPython.cpp
PythonLib.cpp
SerdeExtern.cpp
SimpleBinaryJson.cpp
SimpleJsonLoader.cpp
SimpleJsonValue.cpp
SimpleParser.cpp
config.h
DatabaseInclude.h
IJsonDatabase.h
IJsonObject.h
JsonNetServerRPC.h
NoSqlExample.h
Parser/StateRecorder.h
ParserException.h
PythonLib.h
SimpleBinaryJson.h
SimpleJsonLoader.h
SimpleJsonObject.h
SimpleJsonValue.h

)

add_library(VEngine_Database MODULE ${SOURCEFILE})
target_link_libraries(VEngine_Database PRIVATE xxx yyy)
set_target_properties(VEngine_Database PROPERTIES
        UNITY_BUILD ON)
