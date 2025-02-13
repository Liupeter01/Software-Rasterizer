cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(TBB_TEST OFF)

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

if(PROJECT_BINARY_DIR EQUAL PROJECT_SOURCE_DIR)
  message(
    WARNING "Binary directory of CMake CANNOT be the same as Source Directory!")
endif()

if(MSVC) # 1. fix string literial cannot include unicode characters; 2. rid min
         #   and max macros; 3. get back M_PI, M_E macros
  add_compile_options(
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8$<SEMICOLON>/DNOMINMAX$<SEMICOLON>/D_USE_MATH_DEFINES>
  )
  add_compile_options(
    $<$<COMPILE_LANG_AND_ID:C,MSVC>:/utf-8$<SEMICOLON>/DNOMINMAX$<SEMICOLON>/D_USE_MATH_DEFINES>
  )
elseif(CMAKE_COMPILER_IS_GNUCXX) # Add necessary checks and warnings
  if(CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]")
    add_compile_options(
      $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wall$<SEMICOLON>-Wextra>)
    add_compile_options(
      $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-ftrapv$<SEMICOLON>-fbounds-check$<SEMICOLON>-fsanitize-trap>
    )
  endif()
endif()

project(
  Software-Rasterizer
  VERSION 0.6.0
  LANGUAGES C CXX)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

file(GLOB_RECURSE source_file CONFIGURE_DEPENDS src/*.cpp src/*.cc src/*.c)
file(GLOB_RECURSE header_file CONFIGURE_DEPENDS include/*.hpp include/*.h)

add_subdirectory(external/spdlog)
add_subdirectory(external/tinyobjloader)
add_subdirectory(external/glm)
add_subdirectory(external/oneTBB)

add_executable(Software-Rasterizer ${source_file} ${header_file})
target_include_directories(
  Software-Rasterizer PUBLIC include external/simde/simde external/sse2neon
                             ${OpenCV_INCLUDE_DIRS})
target_link_libraries(Software-Rasterizer PUBLIC ${OpenCV_LIBRARIES} spdlog
                                                 tinyobjloader glm TBB::tbb)

# Enable OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(Software-Rasterizer PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(
  Software-Rasterizer PUBLIC -DCONFIG_HOME=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
