cmake_minimum_required(VERSION 3.8)
project(local_sensing_node LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra)
option(USE_GPU "Build the OpenGL renderer" OFF)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem iostreams program_options system serialization)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

set(renderer_sources
  include/ikd-Tree/ikd_Tree.cpp
  include/FOV_Checker/FOV_Checker.cpp)
set(renderer_dependencies
  ament_index_cpp cv_bridge geometry_msgs nav_msgs pcl_conversions rclcpp
  sensor_msgs tf2_ros)

add_executable(pcl_render_node src/pointcloud_render_node.cpp ${renderer_sources})
target_include_directories(pcl_render_node PRIVATE
  ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_compile_options(pcl_render_node PRIVATE
  "-iquote${CMAKE_CURRENT_SOURCE_DIR}/include")
ament_target_dependencies(pcl_render_node ${renderer_dependencies})
target_link_libraries(pcl_render_node
  Eigen3::Eigen ${PCL_LIBRARIES} ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

set(renderer_targets pcl_render_node)
if(USE_GPU)
  find_package(glm REQUIRED)   # only the OpenGL renderer uses glm
  find_package(GLEW REQUIRED)
  find_package(OpenGL REQUIRED)
  find_package(glfw3 REQUIRED)
  add_executable(opengl_render_node src/opengl_render_node.cpp ${renderer_sources})
  target_include_directories(opengl_render_node PRIVATE
    ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
  target_compile_options(opengl_render_node PRIVATE
    "-iquote${CMAKE_CURRENT_SOURCE_DIR}/include")
  ament_target_dependencies(opengl_render_node ${renderer_dependencies})
  target_link_libraries(opengl_render_node
    Eigen3::Eigen glm::glm ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES}
    OpenMP::OpenMP_CXX OpenGL::GL GLEW::GLEW glfw)
  list(APPEND renderer_targets opengl_render_node)
endif()

install(TARGETS ${renderer_targets} DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY params DESTINATION share/${PROJECT_NAME})
install(FILES include/camera.vs include/camera.fs include/360camera.vs
  DESTINATION share/${PROJECT_NAME}/shaders)
ament_package()
