cmake_minimum_required(VERSION 3.25...3.29)

include(FetchContent)

option(OUTPUT_DIR "Output directory for downloaded SDKs" "${CMAKE_CURRENT_LIST_DIR}/../downloaded_sdks")
if (NOT IS_ABSOLUTE ${OUTPUT_DIR})
    set(OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${OUTPUT_DIR})
endif ()
file(MAKE_DIRECTORY ${OUTPUT_DIR})
file(REAL_PATH ${OUTPUT_DIR} OUTPUT_DIR EXPAND_TILDE)
message(STATUS "Output directory for downloaded SDKs: ${OUTPUT_DIR}")

file(LOCK ${OUTPUT_DIR} DIRECTORY)

function(download_sdk name url sha1)
    message(STATUS "Downloading ${name} from ${url}")
    FetchContent_Populate(${name}
            URL ${url}
            URL_HASH SHA1=${sha1}
            SOURCE_DIR ${OUTPUT_DIR}/${name}-src
            SUBBUILD_DIR ${OUTPUT_DIR}/${name}-build
            BINARY_DIR ${OUTPUT_DIR}/${name}-build)
endfunction()

option(COMPONENTS "Components to download" "")
string(TOLOWER "${COMPONENTS}" COMPONENTS)
message(STATUS "Downloading SDKs: ${COMPONENTS}")

set(LUISA_COMPUTE_DOWNLOADED_SDKS)
foreach (sdk ${COMPONENTS})
    set(valid TRUE)
    if (sdk STREQUAL "dx")
        download_sdk(${sdk} "https://github.com/LuisaGroup/SDKs/releases/download/sdk/dx_sdk_20240920.zip" "4c8390d674f375e6676ba15ce452db59df88da8f")
    else ()
        set(valid FALSE)
        message(WARNING "Unknown SDK: ${sdk}")
    endif ()
    if (valid)
        list(APPEND LUISA_COMPUTE_DOWNLOADED_SDKS ${sdk})
    endif ()
endforeach ()

# write the downloaded SDKs to a file
message(STATUS "Downloaded SDKs: ${LUISA_COMPUTE_DOWNLOADED_SDKS}")
set(output_variables)
foreach (sdk ${LUISA_COMPUTE_DOWNLOADED_SDKS})
    string(TOUPPER ${sdk} upper_name)
    string(APPEND output_variables "set(LUISA_COMPUTE_${upper_name}_SDK_DIR \"${OUTPUT_DIR}/${sdk}-src\")\n")
endforeach ()

set(CONFIG_PATH ${CMAKE_CURRENT_LIST_DIR}/downloaded_sdks.cmake)
message(STATUS "Writing downloaded SDKs to ${CONFIG_PATH}")
file(WRITE ${CONFIG_PATH} ${output_variables})
