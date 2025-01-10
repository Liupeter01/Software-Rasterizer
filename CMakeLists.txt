cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if(MSVC)
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ox")
  else()
  endif()
endif()

if(NOT MSVC)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Found CCache: ${CCACHE_PROGRAM}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_PROGRAM})
  endif()
endif()

project(
  Software-Rasterizer
  VERSION 0.0.16
  LANGUAGES CXX)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

file(GLOB_RECURSE source_file CONFIGURE_DEPENDS src/*.cpp src/*.cc src/*.c)
file(GLOB_RECURSE header_file CONFIGURE_DEPENDS include/*.hpp include/*.h)

add_subdirectory(eigen)
add_subdirectory(spdlog)

add_executable(Software-Rasterizer ${source_file} ${header_file})
target_include_directories(Software-Rasterizer PUBLIC include
                                                      ${OpenCV_INCLUDE_DIRS})
target_link_libraries(Software-Rasterizer PUBLIC ${OpenCV_LIBRARIES} eigen
                                                 spdlog)

# Enable OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(Software-Rasterizer PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(
  Software-Rasterizer PUBLIC -DCONFIG_HOME=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
