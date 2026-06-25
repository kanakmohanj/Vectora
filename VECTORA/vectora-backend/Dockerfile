# Use a lightweight, stable Linux base image
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install the C++ compiler, build tools, and OpenSSL libraries (required for httplib HTTPS)
RUN apt-get update && apt-get install -y \
    g++ \
    build-essential \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the source code and headers into the container
COPY main.cpp httplib.h ./

# Compile the C++ database for Linux architecture
# -lpthread handles multithreading
# -lssl -lcrypto handles the HTTPS connection to Google Gemini
# -mavx2 -mfma unlocks SIMD hardware acceleration for your distance metrics
RUN g++ -std=c++17 -O2 -mavx2 -mfma main.cpp -o vectora_db -lpthread -lssl -lcrypto

# Expose the port your httplib server runs on
EXPOSE 8080

# Run the compiled binary when the container boots
CMD ["./vectora_db"]