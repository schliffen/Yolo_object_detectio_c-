# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/372/bin/cmake

# The command to remove a file.
RM = /snap/cmake/372/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ali/ProjLAB/YOLO_tvm/yoloOCV

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ali/ProjLAB/YOLO_tvm/yoloOCV/build

# Include any dependencies generated for this target.
include CMakeFiles/foolib.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/foolib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/foolib.dir/flags.make

CMakeFiles/foolib.dir/include/foo.cpp.o: CMakeFiles/foolib.dir/flags.make
CMakeFiles/foolib.dir/include/foo.cpp.o: ../include/foo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ali/ProjLAB/YOLO_tvm/yoloOCV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/foolib.dir/include/foo.cpp.o"
	/usr/bin/g++-8  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/foolib.dir/include/foo.cpp.o -c /home/ali/ProjLAB/YOLO_tvm/yoloOCV/include/foo.cpp

CMakeFiles/foolib.dir/include/foo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/foolib.dir/include/foo.cpp.i"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ali/ProjLAB/YOLO_tvm/yoloOCV/include/foo.cpp > CMakeFiles/foolib.dir/include/foo.cpp.i

CMakeFiles/foolib.dir/include/foo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/foolib.dir/include/foo.cpp.s"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ali/ProjLAB/YOLO_tvm/yoloOCV/include/foo.cpp -o CMakeFiles/foolib.dir/include/foo.cpp.s

# Object files for target foolib
foolib_OBJECTS = \
"CMakeFiles/foolib.dir/include/foo.cpp.o"

# External object files for target foolib
foolib_EXTERNAL_OBJECTS =

libfoolib.so: CMakeFiles/foolib.dir/include/foo.cpp.o
libfoolib.so: CMakeFiles/foolib.dir/build.make
libfoolib.so: CMakeFiles/foolib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ali/ProjLAB/YOLO_tvm/yoloOCV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libfoolib.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/foolib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/foolib.dir/build: libfoolib.so

.PHONY : CMakeFiles/foolib.dir/build

CMakeFiles/foolib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/foolib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/foolib.dir/clean

CMakeFiles/foolib.dir/depend:
	cd /home/ali/ProjLAB/YOLO_tvm/yoloOCV/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ali/ProjLAB/YOLO_tvm/yoloOCV /home/ali/ProjLAB/YOLO_tvm/yoloOCV /home/ali/ProjLAB/YOLO_tvm/yoloOCV/build /home/ali/ProjLAB/YOLO_tvm/yoloOCV/build /home/ali/ProjLAB/YOLO_tvm/yoloOCV/build/CMakeFiles/foolib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/foolib.dir/depend

