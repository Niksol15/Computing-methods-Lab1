cmake_minimum_required(VERSION 3.17)
project(PageRank)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED)

if(Boost_FOUND)
    set(SOURCES_DIR sources)
    set(HEADERS_DIR headers)

    list(APPEND SOURCES ${SOURCES_DIR}/main.cpp ${SOURCES_DIR}/PageRank.cpp)
    list(APPEND HEADERS ${HEADERS_DIR}/PageRank.h)

    add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES})

    target_include_directories(${PROJECT_NAME} PUBLIC ${HEADERS_DIR})
    target_include_directories(${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIRS})
else()
    message(Boost doesnt found)
endif()

#set(Boost_USE_STATIC_LIBS    ON)
#set(Boost_USE_MULTITHREADED  ON)
#set(Boost_USE_STATIC_RUNTIME OFF)
