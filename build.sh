#!/bin/bash

# Build script for NMPC ACADOS C++ project

set -e

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

echo "Build completed successfully!"
