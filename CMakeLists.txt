cmake_minimum_required(VERSION 3.5)
project(pybind11_opencv)

# Specify C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# pybind11 needs to be first, otherwise other packages which also search for
# Python can cause an 'Unknown CMake command "python3_add_library"' error.
# Probably related to how Python is found, see
# https://github.com/pybind/pybind11/issues/3996
find_package(pybind11 REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(mpi_cmake_modules REQUIRED)
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
    find_package(ament_cmake_pytest REQUIRED)

    # Need to install a dummy package with __init__.py, otherwise the package
    # will not be added to PYTHONPATH in the setup.bash and the pybind11 module
    # installed below will not be found.
    ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR test/dummy_pkg/${PROJECT_NAME})
    add_pybind11_module(cvbind_test srcpy/cvbind_test.cpp
        LINK_LIBRARIES ${PROJECT_NAME}
    )

    ament_add_pytest_test(test_cvbind test/test_cvbind.py)
endif()


ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(pybind11 OpenCV)
ament_package()
