cmake_minimum_required(VERSION 3.14)
project(diffu_inpainting)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
if (OPENCV_FOUND)
    message("OpenCV Found")
endif()

find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    message("Eigen3 Found")
endif()
message("${EIGEN3_INCLUDE_DIR}")
include_directories(include
                    ${EIGEN3_INCLUDE_DIR}
                    ${OpenCV_INCLUDE_DIRS}
                    )
#message("${VTK_INCLUDE_DIRS}")
#include(${VTK_USE_FILE})

aux_source_directory(src SOURCE_FILES)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
#target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
#message("${VTK_LIBRARIES}")