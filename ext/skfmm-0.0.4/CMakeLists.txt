find_package(PythonLibs REQUIRED)
find_package(Numpy REQUIRED)
include_directories(
  ${PYTHON_INCLUDE_PATH}
  ${PYTHON_NUMPY_INCLUDE_DIR}
)
add_library(timb_cfmm SHARED
  fmm.cpp
  heap.cpp
  base_marcher.cpp
  distance_marcher.cpp
  travel_time_marcher.cpp
  extension_velocity_marcher.cpp
)

set_target_properties(timb_cfmm
  PROPERTIES
  OUTPUT_NAME timb_cfmm
  COMPILE_FLAGS "${FASTIDIOUS_FLAGS}"
  LINK_FLAGS -dynamic
  PREFIX ""
)

if( APPLE OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # on mac osx, python cannot import libraries with .dylib extension
  set_target_properties(timb_cfmm PROPERTIES SUFFIX ".so")
endif()  

target_link_libraries(timb_cfmm ${PYTHON_LIBRARIES})
