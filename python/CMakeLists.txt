cmake_minimum_required(VERSION 3.27)

project(pydbow2 VERSION 1.0
                  DESCRIPTION "python bindings to DBoW2"
                  LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(OpenCV 4.4)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 4.4 not found.")
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Found Python interpreter: ${Python3_EXECUTABLE}")
message(STATUS "Python include dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python library: ${Python3_LIBRARIES}")

find_package(pybind11 CONFIG REQUIRED)
message(STATUS "pybind11_DIR: ${pybind11_DIR}")
message(STATUS "pybind11_FOUND: ${pybind11_FOUND}")
message(STATUS "pybind11_INCLUDE_DIRS: ${pybind11_INCLUDE_DIRS}")


include_directories(${PROJECT_SOURCE_DIR}/../DBoW2/include/DBoW2)


pybind11_add_module(${PROJECT_NAME} src/dbow2.cpp )
target_link_libraries(${PROJECT_NAME} PRIVATE
             ${OpenCV_LIBS}
             DBoW2
             ${Python3_LIBRARIES}
              )

