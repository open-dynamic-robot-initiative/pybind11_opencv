cmake_minimum_required(VERSION 3.1)
project(pybind11_opencv)

# Specify C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

find_package(catkin REQUIRED COMPONENTS
  pybind11_catkin
)

find_package(OpenCV REQUIRED)


catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS pybind11_catkin
  DEPENDS OpenCV
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

catkin_add_pybind11_module(cvbind_test ${OpenCV_LIBRARIES})

#############
## Install ##
#############

# Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

#############
## Testing ##
#############

# Add folders to be run by python nosetests
catkin_add_nosetests(test)
