cmake_minimum_required (VERSION 3.5)

# Setup projectname
project (Yolocv VERSION 1.0.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -ldl")
# Compile and link main.cpp and foo.cpp


set(OPENCV_INCLUDE_DIR /home/ali/installation/opencv4/include/opencv4)
set(OPENCV_LIB_DIR /home/ali/installation/opencv4/lib)
set(CMAKE_PREFIX_PATH /home/ali/installation/opencv4/lib/cmake/opencv4)

find_package( OpenCV REQUIRED )

# set(NUMCPP_INCLUDES /home/ali/installations/NumCpp/include)
# set(Boost_INCLUDE_DIR /home/ali/installation/boost_1_62_0/)
# set(Boost_LIBRARY_DIRS /home/ali/installation/boost_1_62_0/libs)
include_directories( include )
include_directories( weights )

add_library( nmslib
    SHARED
    include/nms.cpp
    include/nms.hpp
    include/utils.cpp
    include/utils.hpp
)


# include_directories	(${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBS})
# link_directories(${Boost_LIBRARY_DIRS})


add_executable( ${PROJECT_NAME} src/main.cpp )

target_link_libraries(  ${PROJECT_NAME}  nmslib  ${OpenCV_LIBS}  ) # ${Boost_LIBRARY_DIRS}
# target_include_directories( ${PROJECT_NAME} PUBLIC ${NUMCPP_INCLUDES})
# target_include_directories( ${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIRS}) 
