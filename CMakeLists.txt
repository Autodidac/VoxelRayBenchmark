cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{VCPKG_ROOT})
        file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" EPOCH_VCPKG_TOOLCHAIN)
        if(EXISTS "${EPOCH_VCPKG_TOOLCHAIN}")
            set(CMAKE_TOOLCHAIN_FILE "${EPOCH_VCPKG_TOOLCHAIN}" CACHE FILEPATH "vcpkg toolchain file")
        endif()
    endif()
endif()

project(epoch_voxel_hybrid_benchmark VERSION 0.6.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(EPOCH_VISUALIZER_USE_EPOCHGUI "Use EpochGui for live inspector layout and bitmap text." ON)
set(EPOCHGUI_SOURCE_DIR "" CACHE PATH "Optional local EpochGui checkout; otherwise FetchContent uses Autodidac/EpochGui.")

if(EPOCH_VISUALIZER_USE_EPOCHGUI)
    include(FetchContent)
    if(EPOCHGUI_SOURCE_DIR)
        if(NOT EXISTS "${EPOCHGUI_SOURCE_DIR}/CMakeLists.txt")
            message(FATAL_ERROR "EPOCHGUI_SOURCE_DIR does not contain EpochGui CMakeLists.txt: ${EPOCHGUI_SOURCE_DIR}")
        endif()
        add_subdirectory("${EPOCHGUI_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/epochgui")
    else()
        FetchContent_Declare(
            EpochGui
            GIT_REPOSITORY https://github.com/Autodidac/EpochGui.git
            GIT_TAG 347ad52e8fc27deb08dea97e56a9b6d8c0db3af2
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(EpochGui)
    endif()
endif()

if(NOT DEFINED VCPKG_INSTALLED_DIR)
    message(FATAL_ERROR
        "This project uses vcpkg manifest mode. Configure with a supplied preset, "
        "or set VCPKG_ROOT and pass its scripts/buildsystems/vcpkg.cmake toolchain.")
endif()

find_package(Vulkan REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

set(EPOCH_GLSLC_HINTS)
if(DEFINED VCPKG_HOST_TRIPLET)
    list(APPEND EPOCH_GLSLC_HINTS
        "${VCPKG_INSTALLED_DIR}/${VCPKG_HOST_TRIPLET}/tools/shaderc")
endif()
file(GLOB EPOCH_GLSLC_FALLBACK_HINTS
    LIST_DIRECTORIES true
    "${VCPKG_INSTALLED_DIR}/*/tools/shaderc")
list(APPEND EPOCH_GLSLC_HINTS ${EPOCH_GLSLC_FALLBACK_HINTS})

find_program(EPOCH_GLSLC_EXECUTABLE
    NAMES glslc glslc.exe
    HINTS ${EPOCH_GLSLC_HINTS}
    REQUIRED)

message(STATUS "vcpkg installed directory: ${VCPKG_INSTALLED_DIR}")
message(STATUS "Vulkan library: ${Vulkan_LIBRARY}")
message(STATUS "glslc: ${EPOCH_GLSLC_EXECUTABLE}")

set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/trace.comp")
set(SHADER_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIRECTORY}")

function(epoch_compile_shader output_name)
    set(definitions ${ARGN})
    set(output_path "${SHADER_OUTPUT_DIRECTORY}/${output_name}.spv")
    set(definition_args)
    foreach(definition IN LISTS definitions)
        list(APPEND definition_args "-D${definition}")
    endforeach()

    add_custom_command(
        OUTPUT "${output_path}"
        COMMAND "${EPOCH_GLSLC_EXECUTABLE}"
                --target-env=vulkan1.3
                -O
                ${definition_args}
                "${SHADER_SOURCE}"
                -o "${output_path}"
        DEPENDS "${SHADER_SOURCE}"
        VERBATIM
        COMMENT "Compiling ${output_name}.spv")
    set(EPOCH_SHADER_OUTPUTS ${EPOCH_SHADER_OUTPUTS} "${output_path}" PARENT_SCOPE)
endfunction()

epoch_compile_shader(trace_bvh_debug TRACE_BVH=1 TRACE_VOXEL=0 INSTRUMENT=1)
epoch_compile_shader(trace_voxel_debug TRACE_BVH=0 TRACE_VOXEL=1 INSTRUMENT=1)
epoch_compile_shader(trace_bvh_benchmark TRACE_BVH=1 TRACE_VOXEL=0 INSTRUMENT=0)
epoch_compile_shader(trace_voxel_benchmark TRACE_BVH=0 TRACE_VOXEL=1 INSTRUMENT=0)

add_custom_target(epoch_voxel_shaders ALL DEPENDS ${EPOCH_SHADER_OUTPUTS})

add_executable(epoch_voxel_vulkan_demo
    src/main_vulkan.cpp
    src/ray_inspector.cpp
    src/ray_inspector.hpp
    src/benchmark_common.hpp)
add_dependencies(epoch_voxel_vulkan_demo epoch_voxel_shaders)
target_link_libraries(epoch_voxel_vulkan_demo PRIVATE Vulkan::Vulkan SDL3::SDL3)
target_compile_definitions(epoch_voxel_vulkan_demo PRIVATE
    EPOCH_VOXEL_SHADER_DIRECTORY="${SHADER_OUTPUT_DIRECTORY}")

add_executable(epoch_voxel_opengl_demo
    src/main_opengl.cpp
    src/ray_inspector.cpp
    src/ray_inspector.hpp
    src/benchmark_common.hpp)
target_link_libraries(epoch_voxel_opengl_demo PRIVATE SDL3::SDL3 glad::glad OpenGL::GL)
target_compile_definitions(epoch_voxel_opengl_demo PRIVATE
    EPOCH_VOXEL_SHADER_SOURCE="${SHADER_SOURCE}")

foreach(target IN ITEMS epoch_voxel_vulkan_demo epoch_voxel_opengl_demo)
    target_compile_definitions(${target} PRIVATE
        EPOCH_VISUALIZER_USE_EPOCHGUI=$<BOOL:${EPOCH_VISUALIZER_USE_EPOCHGUI}>)
    if(EPOCH_VISUALIZER_USE_EPOCHGUI)
        target_link_libraries(${target} PRIVATE EpochGui)
    endif()
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /Zc:preprocessor /EHsc)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
endforeach()
