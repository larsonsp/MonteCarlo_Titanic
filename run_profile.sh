#!/bin/bash

# Define the number of runs (in this case, 6 for modes 0 to 5)
NUM_RUNS=6
N_VALUE=1000000

# Loop through each mode
for (( MODE=0; MODE<=$NUM_RUNS; MODE++ ))
do
    # Define the output file name with the mode number
    OUTPUT_FILE="profile_mode_${MODE}.txt"

    # Run nvprof with the specified mode and n value, outputting to the defined file
    nvprof --print-summary --log-file $OUTPUT_FILE ./data $MODE $N_VALUE
done
