# Makefile for the CUDA Connect 4 Project

# Compiler
CXX = nvcc

# Executable name
TARGET = connect4_gpu

# Source file
SRC = connect4_cuda_ai.cu

# Compiler flags
# -std=c++17: Use the C++17 standard
# -lcurand: Link the CUDA Random Number Generation library
CXXFLAGS = -std=c++17 -lcurand

# Default target executed when you just type "make"
all: build

# Rule to build the executable
build: $(SRC)
	$(CXX) $(SRC) -o $(TARGET) $(CXXFLAGS)

# Rule to run the executable with optional arguments
# Example: make run ARGS="-d 500"
run: all
	./$(TARGET) $(ARGS)

# Rule to clean up the build files
clean:
	rm -f $(TARGET) output.txt

# Phony targets are not actual files
.PHONY: all build run clean
