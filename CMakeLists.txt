cmake_minimum_required(VERSION 3.10)
project(SlotMeasurement CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES
    src/main.cpp
    src/locator.cpp
    src/measurement.cpp
    src/geometric_fit.cpp
    src/pipeline.cpp
    src/visualization.cpp
)

add_executable(slot_measure ${SOURCES})
target_link_libraries(slot_measure ${OpenCV_LIBS})
