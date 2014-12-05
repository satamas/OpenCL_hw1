cmake_minimum_required(VERSION 2.8.4)
project(Convolution)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")

find_package( OpenCL REQUIRED )
include_directories( ${OPENCL_INCLUDE_DIRS} )

if( OPENCL_HAS_CPP_BINDINGS )
	message( "OpenCL has CPP bindings. Full include is: " ${OPENCL_INCLUDE_DIRS} )
else( OPENCL_HAS_CPP_BINDINGS )
	message( "No OpenCL CPP bindings found" )
endif( OPENCL_HAS_CPP_BINDINGS )

configure_file(prefix_sum.cl ${CMAKE_CURRENT_BINARY_DIR}/prefix_sum.cl COPYONLY)
configure_file(input.txt ${CMAKE_CURRENT_BINARY_DIR}/input.txt COPYONLY)

set(SOURCE_FILES main.cpp)

add_executable(Convolution ${SOURCE_FILES})
target_link_libraries( Convolution ${OPENCL_LIBRARIES} )