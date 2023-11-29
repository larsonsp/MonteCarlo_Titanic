#!/bin/bash

# Define the number of modes to profile
NUM_MODES=6
N_VALUE=1000000  # Set the value of n (e.g., 1 million)

# Loop through each mode
for MODE in $(seq 0 $NUM_MODES)
do
    # Compile the CUDA program to PTX
    nvcc -ptx -o "data_mode_${MODE}.ptx" data_creation.cu

    # Temporary nvvp file for profiling
    TEMP_NVVP_FILE="temp_data_mode_${MODE}.nvvp"

    # Profile the CUDA program with nvprof
    nvprof --analysis-metrics -o $TEMP_NVVP_FILE ./data $MODE $N_VALUE > "data_mode_${MODE}_profile.txt" 2>&1

    # Remove the temporary nvvp file
    rm $TEMP_NVVP_FILE
done

echo "Profiling completed. Output files: data_mode_{0..$NUM_MODES}.ptx and data_mode_{0..$NUM_MODES}_profile.txt"
