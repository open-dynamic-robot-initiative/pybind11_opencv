cmake_minimum_required(VERSION 3.5)
project(pybind11_opencv)

# Specify C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

find_package(ament_cmake REQUIRED)
find_package(mpi_cmake_modules REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)


###########
## Build ##
###########

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} INTERFACE
    ${OpenCV_LIBRARIES}
    pybind11::pybind11
)


#############
## Install ##
#############

ament_export_interfaces(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

# Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION include/${PROJECT_NAME})

install(
    TARGETS ${PROJECT_NAME}
    EXPORT export_${PROJECT_NAME}
    LIBRARY DESTINATION lib
)

#############
## Testing ##
#############

if (BUILD_TESTING)
    find_package(ament_cmake_nose REQUIRED)

    add_pybind11_module(cvbind_test srcpy/cvbind_test.cpp
        LINK_LIBRARIES ${PROJECT_NAME}
    )

    ament_add_nose_test(test_cvbind test/test_cvbind.py)
endif()


ament_export_include_directories(include)
ament_export_dependencies(pybind11 OpenCV)
ament_package()
