cmake_minimum_required(VERSION 3.22)
project(vulkanApp)

set(CMAKE_CXX_STANDARD 20)

add_executable(vulkanApp main.cpp)

include_directories(${CMAKE_SOURCE_DIR}/libs/GLFW/include
        ${CMAKE_SOURCE_DIR}/libs/glm/include
        ${CMAKE_SOURCE_DIR}/libs/vulkan/Include
        ${CMAKE_SOURCE_DIR}/libs/stb/include)

target_link_directories(vulkanApp PRIVATE ${CMAKE_SOURCE_DIR}/libs/GLFW/lib
        ${CMAKE_SOURCE_DIR}/libs/vulkan/lib)

target_link_libraries(vulkanApp PRIVATE glfw3 libvulkan-1.dll.a)

# Copy shaders directory to the build directory
add_custom_command(TARGET vulkanApp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders $<TARGET_FILE_DIR:vulkanApp>/shaders)
# Copy textures directory
add_custom_command(TARGET vulkanApp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/textures $<TARGET_FILE_DIR:vulkanApp>/textures)




