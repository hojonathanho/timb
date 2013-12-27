cmake_minimum_required(VERSION 2.8)
project(timb)

include(cmake/main.cmake)

# http://cmake.3232098.n2.nabble.com/Default-value-for-CMAKE-BUILD-TYPE-td7550756.html
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++ -fPIC")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# external libraries
find_package(Boost COMPONENTS system python filesystem program_options REQUIRED)
find_package(Cholmod)
find_package(BLAS)
find_package(LAPACK)
# libraries packaged with this source tree
set(EIGEN_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/ext/eigen-3.2.0)

#set(CERES_DIR ${CMAKE_SOURCE_DIR}/ext/ceres-solver-1.8.0)
#set(CERES_INCLUDE_DIRS ${CERES_DIR}/include)

#find_package(Ceres 1.8 REQUIRED)

add_subdirectory(src)
#add_subdirectory(ext)
