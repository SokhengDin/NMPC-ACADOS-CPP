FROM ubuntu:22.04

# Set timezone to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install required dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgsl-dev \
    liblapack-dev \
    libopenblas-dev \
    pkg-config \
    python3-dev \
    python3-matplotlib \
    python3-numpy \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install  dependencies
RUN pip3 install casadi matplotlib

# Clone and build acados
RUN git clone https://github.com/acados/acados.git /opt/acados
WORKDIR /opt/acados
RUN mkdir -p build && cd build && \
    cmake .. && \
    make install


ENV ACADOS_INSTALL_DIR /opt/acados

# Copy
COPY . /app
WORKDIR /app

# Build
RUN mkdir -p build && cd build && \
    cmake .. && \
    make

# Executable
ENTRYPOINT ["/app/build/differential_drive_solver"]