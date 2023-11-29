#include <curand.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>


__global__ void init(curandState_t* states, unsigned long seed, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

//IS NOT THE BEST FOR LOG WSAS WORSE FOR LOG BUT HAD MORE BETTER FOR NORAML
__device__ inline float roundToDecimal(float value, int decimalPlaces) {
    float scale = powf(10.0f, decimalPlaces); // Efficient scaling using powf
    return roundf(value * scale) / scale; // Round and scale back
}


//ALWAYS REMBER FOR THIS KERANL YOU HAVE OTO USE THE OTHER DOCUMENT TO GET ACCURATE PARAMATERS
__global__ void generateLognormal_deprecated(curandState_t* states, float* results, int n, float mean, float stddev, int decimalPlaces) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float z = curand_normal(&states[idx]);
        z = stddev * z + mean;
        float lognormal = exp(z);
        results[idx] = roundToDecimal(lognormal, decimalPlaces);
    }
}

__global__ void generateLognormal(curandState_t* states, float* results, int n, float mean, float stddev, int decimalPlaces) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    float z = curand_normal(&states[idx]);
    float lognormal = expf(fma(stddev, z, mean)); // Using FMA and expf for efficiency
    results[idx] = roundToDecimal(lognormal, decimalPlaces);
}

__global__ void generateCategorical3_deprecated(curandState_t* states, int* results, int n, float ratio1, float ratio2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float randVal = curand_uniform(&states[idx]);
        if (randVal < ratio1) {
            results[idx] = 1;
        } else if (randVal < ratio1 + ratio2) {
            results[idx] = 2;
        } else {
            results[idx] = 3;
        }
    }
}

//MINMIZE BRANCH DIVERGENCE
__global__ void generateCategorical3(curandState_t* states, int* results, int n, float ratio1, float ratio2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    float randVal = curand_uniform(&states[idx]);
    float combinedRatio = ratio1 + ratio2; // Pre-calculate the sum of the ratios

    // Minimize branch divergence
    int category = 3; // Default category
    category = (randVal < combinedRatio) ? ((randVal < ratio1) ? 1 : 2) : category;

    results[idx] = category;
}


__global__ void generateCategorical2_deprecated(curandState_t* states, int* results, int n, float ratio) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        results[idx] = (curand_uniform(&states[idx]) < ratio) ? 0 : 1;
    }
}

__global__ void generateCategorical2(curandState_t* states, int* results, int n, float ratio) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    results[idx] = (curand_uniform(&states[idx]) < ratio) ? 0 : 1;
}




__global__ void generateNormal_deprecated(curandState_t* states, float* results, int n, float mean, float stddev, int decimalPlaces) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        curandState localState = states[idx];
        float u1 = curand_uniform(&localState);
        float u2 = curand_uniform(&localState);
        float normalValue = mean + stddev * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        results[idx] = roundToDecimal(normalValue, decimalPlaces);
        states[idx] = localState;
    }
}

__constant__ float TWO_PI = 2.0f * M_PI;
__global__ void generateNormal(curandState_t* states, float* results, int n, float mean, float stddev, int decimalPlaces) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    curandState localState = states[idx];
    float u1 = curand_uniform(&localState);
    float u2 = curand_uniform(&localState);

    // Using FMA for efficiency
    float radius = sqrtf(-2.0f * logf(u1));
    float angle = TWO_PI * M_PI * u2;
    float normalValue = fma(radius, cosf(angle), mean);
    normalValue = fma(normalValue, stddev, 0.0f);

    results[idx] = roundToDecimal(normalValue, decimalPlaces);
    states[idx] = localState;
}


__global__ void generateNormalNoRounding(curandState_t* states, float* results, int n, float mean, float stddev, int decimalPlaces) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    curandState localState = states[idx];
    float u1 = curand_uniform(&localState);
    float u2 = curand_uniform(&localState);

    // Using FMA for efficiency
    float radius = sqrtf(-2.0f * logf(u1));
    float angle = TWO_PI * M_PI * u2;
    float normalValue = fma(radius, cosf(angle), mean);
    normalValue = fma(normalValue, stddev, 0.0f);

    results[idx] = normalValue;
    states[idx] = localState;
}




// Function to calculate mean
float calculateMean(float* data, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / n;
}

// Function to calculate standard deviation
float calculateStdDev(float* data, int n, float mean) {
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (data[i] - mean) * (data[i] - mean);
    }
    return sqrt(sum / n);
}

// Function to calculate and print ratios for categorical data
void printCategoricalRatios(int* data, int n) {
    int count1 = 0, count2 = 0, count3 = 0;
    for (int i = 0; i < n; i++) {
        if (data[i] == 1) count1++;
        else if (data[i] == 2) count2++;
        else count3++;
    }
    printf("Ratio 1: %f\n", (float)count1 / n);
    printf("Ratio 2: %f\n", (float)count2 / n);
    printf("Ratio 3: %f\n", (float)count3 / n);
}

void writeCSV(const char* filename, int n, float* ageResults, float* fareResults, float* sibspResults, float* parchResults, int* survivedResults, int* pclassResults, int* sexResults, int* embarkedResults) {
    std::ofstream file;
    file.open(filename);

    // Write column headers
    file << "Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked\n";

    // Write data
    for (int i = 0; i < n; ++i) {
        file << survivedResults[i] << ",";
        file << pclassResults[i] << ",";
        file << sexResults[i] << ",";
        file << ageResults[i] << ",";
        file << sibspResults[i] << ",";
        file << parchResults[i] << ",";
        file << fareResults[i] << ",";
        file << embarkedResults[i] << "\n";
    }

    file.close();
}

// Define your methods here
void debugLognormal(int n);
void debugCategorical3(int n);
void debugCategorical2(int n);

void debugNormal(int n);
void profile_Init(int n);
void profile_info(int n);
void runActualProcess(int n);
void profile_round(int n);

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <mode> <n>\n", argv[0]);
        return 1;
    }

    int mode = atoi(argv[1]);
    int n = atoi(argv[2]);

    if (n <= 0) {
        printf("Error: 'n' must be a positive integer\n");
        return 1;
    }

    switch (mode) {
        case 0:
            runActualProcess(n);
            break;
        case 1:
            debugLognormal(n);
            break;
        case 2:
            debugCategorical3(n);
            break;
        case 3:
            debugCategorical2(n);
            break;
        case 4:
            debugNormal(n);
            break;
        case 5:
            profile_Init(n);
            break;
        case 6:
            profile_info(n);
            break;
        case 7:
            profile_round(n);
        default:
            printf("Invalid mode. Please enter a mode between 0 and 7.\n");
            return 1;
    }

    return 0;
}



void debugLognormal_deprecated(int n) {
    // Parameters calculated from Python script
    float mu = -0.020202707317519466;
    float sigma = 0.48995747991485767;

    // Allocate memory for curand states and results
    curandState_t* states;
    cudaMalloc((void**)&states, n * sizeof(curandState_t));

    float* lognormalResults;
    cudaMalloc((void**)&lognormalResults, n * sizeof(float));

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Initialize curand states (assuming init kernel is defined elsewhere)
    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    cudaDeviceSynchronize();

    // Generate lognormal distribution
    generateLognormal<<<numBlocks, blockSize>>>(states, lognormalResults, n, mu, sigma,2);
    cudaDeviceSynchronize();

    // Copy results back to host
    float* hostLognormalResults = (float*)malloc(n * sizeof(float));
    cudaMemcpy(hostLognormalResults, lognormalResults, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate and print mean and standard deviation
    float mean = calculateMean(hostLognormalResults, n);
    float stddev = calculateStdDev(hostLognormalResults, n, mean);
    printf(" ATUAL Log-Normal Distribution: Mean = 1, StdDev = .5\n");
    printf("Log-Normal Distribution: Mean = %f, StdDev = %f\n", mean, stddev);

    // Print sample results
    printf("Sample log-normal data:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", hostLognormalResults[i]);
    }
    printf("\n");

    // Free resources
    free(hostLognormalResults);
    cudaFree(lognormalResults);
    cudaFree(states);
}




void debugLognormal_deprecated1(int n) {
    // Parameters calculated from Python script
    float mu = -0.020202707317519466;
    float sigma = 0.48995747991485767;

    // Allocate memory for curand states and results
    curandState_t* states;
    cudaMalloc((void**)&states, n * sizeof(curandState_t));

    float* lognormalResults;
    cudaMalloc((void**)&lognormalResults, n * sizeof(float));

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Create CUDA events for profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Initialize curand states
    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    cudaDeviceSynchronize();

    // Record event for kernel start
    cudaEventRecord(start, 0);

    // Generate lognormal distribution
    generateLognormal<<<numBlocks, blockSize>>>(states, lognormalResults, n, mu, sigma, 2);
    cudaDeviceSynchronize();

    // Record event for kernel stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time for kernel execution
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution Time: %f ms\n", milliseconds);

    // Record event for memory transfer start
    cudaEventRecord(start, 0);

    // Copy results back to host
    float* hostLognormalResults = (float*)malloc(n * sizeof(float));
    cudaMemcpy(hostLognormalResults, lognormalResults, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Record event for memory transfer stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time for memory transfer
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Memory Transfer Time: %f ms\n", milliseconds);

    // Calculate and print mean and standard deviation
    float mean = calculateMean(hostLognormalResults, n);
    float stddev = calculateStdDev(hostLognormalResults, n, mean);
    printf("Actual Log-Normal Distribution: Mean = 1, StdDev = .5\n");
    printf("Log-Normal Distribution: Mean = %f, StdDev = %f\n", mean, stddev);

    // Print sample results
    printf("Sample log-normal data:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", hostLognormalResults[i]);
    }
    printf("\n");

    // Free resources
    free(hostLognormalResults);
    cudaFree(lognormalResults);
    cudaFree(states);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void debugLognormal(int n) {
    // Parameters calculated from Python script
    float mu = -0.020202707317519466;
    float sigma = 0.48995747991485767;

    // Allocate memory for curand states and results
    curandState_t* states;
    cudaMalloc((void**)&states, n * sizeof(curandState_t));

    float* lognormalResults;
    cudaMalloc((void**)&lognormalResults, n * sizeof(float));

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Create CUDA events for profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event for initialization
    cudaEventRecord(start, 0);

    // Initialize curand states
    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    cudaDeviceSynchronize();

    // Record the stop event for initialization
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time for initialization
    float initTime = 0;
    cudaEventElapsedTime(&initTime, start, stop);
    printf("Initialization Time: %f ms\n", initTime);

    // Record the start event for kernel execution
    cudaEventRecord(start, 0);

    // Generate lognormal distribution
    generateLognormal<<<numBlocks, blockSize>>>(states, lognormalResults, n, mu, sigma, 2);
    cudaDeviceSynchronize();

    // Record the stop event for kernel execution
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time for kernel execution
    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, start, stop);
    printf("Kernel Execution Time: %f ms\n", kernelTime);

    // Record the start event for memory transfer
    cudaEventRecord(start, 0);

    // Copy results back to host
    float* hostLognormalResults = (float*)malloc(n * sizeof(float));
    cudaMemcpy(hostLognormalResults, lognormalResults, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Record the stop event for memory transfer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time for memory transfer
    float transferTime = 0;
    cudaEventElapsedTime(&transferTime, start, stop);
    printf("Memory Transfer Time: %f ms\n", transferTime);

    // Calculate and print mean and standard deviation
    float mean = calculateMean(hostLognormalResults, n);
    float stddev = calculateStdDev(hostLognormalResults, n, mean);
    printf("Actual Log-Normal Distribution: Mean = 1, StdDev = .5\n");
    printf("Log-Normal Distribution: Mean = %f, StdDev = %f\n", mean, stddev);

    // Print sample results
    printf("Sample log-normal data:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", hostLognormalResults[i]);
    }
    printf("\n");

    // Free resources
    free(hostLognormalResults);
    cudaFree(lognormalResults);
    cudaFree(states);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}




void debugCategorical3_deprecated(int n) {
    // Probabilities for the three categories
    float p1 = 0.3f; // Probability of category 1
    float p2 = 0.5f; // Probability of category 2

    // Allocate memory for curand states and results
    curandState_t* states;
    cudaMalloc((void**)&states, n * sizeof(curandState_t));
    int* categoricalResults;
    cudaMalloc((void**)&categoricalResults, n * sizeof(int));

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Initialize curand states

    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    cudaDeviceSynchronize();

    // Generate categorical distribution
    generateCategorical3<<<numBlocks, blockSize>>>(states, categoricalResults, n, p1, p2);
    cudaDeviceSynchronize();

    // Copy results back to host
    int* hostCategoricalResults = (int*)malloc(n * sizeof(int));
    cudaMemcpy(hostCategoricalResults, categoricalResults, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate and print ratios
    printCategoricalRatios(hostCategoricalResults, n);

    // Print sample results for verification
    printf("Sample categorical data:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", hostCategoricalResults[i]);
    }
    printf("\n");

    // Free resources
    free(hostCategoricalResults);
    cudaFree(categoricalResults);
    cudaFree(states);
}

void debugCategorical3_deprecated1(int n) {
    // Probabilities for the three categories
    float p1 = 0.3f; // Probability of category 1
    float p2 = 0.5f; // Probability of category 2

    // Allocate memory for curand states and results
    curandState_t* states;
    cudaMalloc((void**)&states, n * sizeof(curandState_t));
    int* categoricalResults;
    cudaMalloc((void**)&categoricalResults, n * sizeof(int));

    // Create CUDA events for profiling
    cudaEvent_t start, stop, memStart, memStop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&memStart);
    cudaEventCreate(&memStop);

    // Record the start event for memory transfer
    cudaEventRecord(memStart, 0);

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Initialize curand states
    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    cudaDeviceSynchronize();

    // Record the stop event for memory transfer
    cudaEventRecord(memStop, 0);
    cudaEventSynchronize(memStop);

    // Record the start event for kernel execution
    cudaEventRecord(start, 0);

    // Generate categorical distribution
    generateCategorical3<<<numBlocks, blockSize>>>(states, categoricalResults, n, p1, p2);
    cudaDeviceSynchronize();

    // Record the stop event for kernel execution
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time for kernel execution
    float elapsedKernelMilliseconds;
    cudaEventElapsedTime(&elapsedKernelMilliseconds, start, stop);

    // Calculate elapsed time for memory transfer
    float elapsedMemMilliseconds;
    cudaEventElapsedTime(&elapsedMemMilliseconds, memStart, memStop);

    // Output the results
    printf("Kernel Execution Time: %f ms\n", elapsedKernelMilliseconds);
    printf("Memory Transfer Time: %f ms\n", elapsedMemMilliseconds);

    // Copy results back to host
    int* hostCategoricalResults = (int*)malloc(n * sizeof(int));
    cudaMemcpy(hostCategoricalResults, categoricalResults, n * sizeof(int), cudaMemcpyDeviceToHost);
    // Print actual probabilities
    printf("Actual probabilities:\n");
    printf("Category 1: %f\n", p1);
    printf("Category 2: %f\n", p2);
    printf("Category 3: %f\n", 1-p2-p1);
    // Calculate and print ratios
    printCategoricalRatios(hostCategoricalResults, n);

    // Print sample results for verification
    printf("Sample categorical data:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", hostCategoricalResults[i]);
    }
    printf("\n");

    // Free resources
    free(hostCategoricalResults);
    cudaFree(categoricalResults);
    cudaFree(states);
}

void debugCategorical3(int n) {
    // Probabilities for the three categories
    float p1 = 0.3f; // Probability of category 1
    float p2 = 0.5f; // Probability of category 2

    // Allocate memory for curand states and results
    curandState_t* states;
    cudaMalloc((void**)&states, n * sizeof(curandState_t));
    int* categoricalResults;
    cudaMalloc((void**)&categoricalResults, n * sizeof(int));

    // Create CUDA events for profiling
    cudaEvent_t start, stop, memStart, memStop, initStart, initStop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&memStart);
    cudaEventCreate(&memStop);
    cudaEventCreate(&initStart);
    cudaEventCreate(&initStop);

    // Record the start event for initialization
    cudaEventRecord(initStart, 0);

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Initialize curand states
    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    cudaDeviceSynchronize();

    // Record the stop event for initialization
    cudaEventRecord(initStop, 0);
    cudaEventSynchronize(initStop);

    // Calculate elapsed time for initialization
    float elapsedInitMilliseconds;
    cudaEventElapsedTime(&elapsedInitMilliseconds, initStart, initStop);

    // Record the start event for memory transfer
    cudaEventRecord(memStart, 0);

    // Generate categorical distribution
    generateCategorical3<<<numBlocks, blockSize>>>(states, categoricalResults, n, p1, p2);
    cudaDeviceSynchronize();

    // Record the stop event for memory transfer
    cudaEventRecord(memStop, 0);
    cudaEventSynchronize(memStop);

    // Record the start event for kernel execution
    cudaEventRecord(start, 0);

    // Record the stop event for kernel execution
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time for kernel execution
    float elapsedKernelMilliseconds;
    cudaEventElapsedTime(&elapsedKernelMilliseconds, start, stop);

    // Calculate elapsed time for memory transfer
    float elapsedMemMilliseconds;
    cudaEventElapsedTime(&elapsedMemMilliseconds, memStart, memStop);

    // Output the results
    printf("Initialization Time: %f ms\n", elapsedInitMilliseconds);
    printf("Kernel Execution Time: %f ms\n", elapsedKernelMilliseconds);
    printf("Memory Transfer Time: %f ms\n", elapsedMemMilliseconds);

    // Copy results back to host
    int* hostCategoricalResults = (int*)malloc(n * sizeof(int));
    cudaMemcpy(hostCategoricalResults, categoricalResults, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print actual probabilities
    printf("Actual probabilities:\n");
    printf("Category 1: %f\n", p1);
    printf("Category 2: %f\n", p2);
    printf("Category 3: %f\n", 1-p2-p1);

    // Calculate and print ratios
    printCategoricalRatios(hostCategoricalResults, n);

    // Print sample results for verification
    printf("Sample categorical data:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", hostCategoricalResults[i]);
    }
    printf("\n");

    // Free resources
    free(hostCategoricalResults);
    cudaFree(categoricalResults);
    cudaFree(states);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(memStart);
    cudaEventDestroy(memStop);
    cudaEventDestroy(initStart);
    cudaEventDestroy(initStop);
}




void debugCategorical2_deprecated(int n) {
    // Probability for the first category
    float p = 0.5f; // Example probability, adjust as needed

    // Allocate memory for curand states and results
    curandState_t* states;
    cudaMalloc((void**)&states, n * sizeof(curandState_t));
    int* categoricalResults;
    cudaMalloc((void**)&categoricalResults, n * sizeof(int));

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Initialize curand states
    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    cudaDeviceSynchronize();

    // Generate categorical distribution
    generateCategorical2<<<numBlocks, blockSize>>>(states, categoricalResults, n, p);
    cudaDeviceSynchronize();

    // Copy results back to host
    int* hostCategoricalResults = (int*)malloc(n * sizeof(int));
    cudaMemcpy(hostCategoricalResults, categoricalResults, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate and print ratios
    int count0 = 0, count1 = 0;
    for (int i = 0; i < n; i++) {
        if (hostCategoricalResults[i] == 0) count0++;
        else count1++;
    }
    printf("Ratio for Category 0: %f\n", (float)count0 / n);
    printf("Ratio for Category 1: %f\n", (float)count1 / n);

    // Print sample results for verification
    printf("Sample categorical data:\n");
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%d ", hostCategoricalResults[i]);
    }
    printf("\n");

    // Free resources
    free(hostCategoricalResults);
    cudaFree(categoricalResults);
    cudaFree(states);
}


void debugCategorical2(int n) {
    // Probability for the first category
    float p = 0.5f; // Example probability, adjust as needed

    // Allocate memory for curand states and results
    curandState_t* states;
    cudaMalloc((void**)&states, n * sizeof(curandState_t));
    int* categoricalResults;
    cudaMalloc((void**)&categoricalResults, n * sizeof(int));

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Create CUDA events for profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize curand states
    cudaEventRecord(start);
    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float initTime = 0;
    cudaEventElapsedTime(&initTime, start, stop);

    // Generate categorical distribution
    cudaEventRecord(start);
    generateCategorical2<<<numBlocks, blockSize>>>(states, categoricalResults, n, p);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, start, stop);

    // Copy results back to host
    cudaEventRecord(start);
    int* hostCategoricalResults = (int*)malloc(n * sizeof(int));
    cudaMemcpy(hostCategoricalResults, categoricalResults, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float memoryTransferTime = 0;
    cudaEventElapsedTime(&memoryTransferTime, start, stop);

    // Calculate and print ratios
    int count0 = 0, count1 = 0;
    for (int i = 0; i < n; i++) {
        if (hostCategoricalResults[i] == 0) count0++;
        else count1++;
    }
    printf("Ratio for Category 0: %f\n", (float)count0 / n);
    printf("Ratio for Category 1: %f\n", (float)count1 / n);

    // Print profiling results
    printf("Initialization Time: %f ms\n", initTime);
    printf("Kernel Execution Time: %f ms\n", kernelTime);
    printf("Memory Transfer Time: %f ms\n", memoryTransferTime);

    // Print sample results for verification
    printf("Sample categorical data:\n");
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%d ", hostCategoricalResults[i]);
    }
    printf("\n");

    // Free resources
    free(hostCategoricalResults);
    cudaFree(categoricalResults);
    cudaFree(states);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}




void debugNormal_deprecated(int n) {
    // Parameters for the normal distribution
    float mean = 0.0f; // Mean of the normal distribution
    float stddev = 1.0f; // Standard deviation of the normal distribution
    int decimalPlaces = 2; // Number of decimal places for rounding

    // Allocate memory for curand states and results
    curandState_t* states;
    cudaMalloc((void**)&states, n * sizeof(curandState_t));

    float* normalResults;
    cudaMalloc((void**)&normalResults, n * sizeof(float));

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Initialize curand states
    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    cudaDeviceSynchronize();

    // Generate normal distribution with rounding
    generateNormal<<<numBlocks, blockSize>>>(states, normalResults, n, mean, stddev, decimalPlaces);
    cudaDeviceSynchronize();

    // Copy results back to host
    float* hostNormalResults = (float*)malloc(n * sizeof(float));
    cudaMemcpy(hostNormalResults, normalResults, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate and print mean and standard deviation
    float calculatedMean = calculateMean(hostNormalResults, n);
    float calculatedStdDev = calculateStdDev(hostNormalResults, n, calculatedMean);
    printf("Normal Distribution: Mean = %f, StdDev = %f\n", calculatedMean, calculatedStdDev);

    // Print sample results
    printf("Sample normal data:\n");
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%f ", hostNormalResults[i]);
    }
    printf("\n");

    // Free resources
    free(hostNormalResults);
    cudaFree(normalResults);
    cudaFree(states);
}

void debugNormal(int n) {
    // Parameters for the normal distribution
    float mean = 0.0f; // Mean of the normal distribution
    float stddev = 1.0f; // Standard deviation of the normal distribution
    int decimalPlaces = 2; // Number of decimal places for rounding

    // Allocate memory for curand states and results
    curandState_t* states;
    cudaMalloc((void**)&states, n * sizeof(curandState_t));

    float* normalResults;
    cudaMalloc((void**)&normalResults, n * sizeof(float));

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Create CUDA events for profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize curand states
    cudaEventRecord(start);
    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float initTime = 0;
    cudaEventElapsedTime(&initTime, start, stop);

    // Generate normal distribution with rounding
    cudaEventRecord(start);
    generateNormal<<<numBlocks, blockSize>>>(states, normalResults, n, mean, stddev, decimalPlaces);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, start, stop);

    // Copy results back to host
    cudaEventRecord(start);
    float* hostNormalResults = (float*)malloc(n * sizeof(float));
    cudaMemcpy(hostNormalResults, normalResults, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float memoryTransferTime = 0;
    cudaEventElapsedTime(&memoryTransferTime, start, stop);

    // Calculate and print mean and standard deviation
    float calculatedMean = calculateMean(hostNormalResults, n);
    float calculatedStdDev = calculateStdDev(hostNormalResults, n, calculatedMean);
    printf("Normal Distribution: Mean = %f, StdDev = %f\n", calculatedMean, calculatedStdDev);

    // Print profiling results
    printf("Initialization Time: %f ms\n", initTime);
    printf("Kernel Execution Time: %f ms\n", kernelTime);
    printf("Memory Transfer Time: %f ms\n", memoryTransferTime);

    // Print sample results
    printf("Sample normal data:\n");
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%f ", hostNormalResults[i]);
    }
    printf("\n");

    // Free resources
    free(hostNormalResults);
    cudaFree(normalResults);
    cudaFree(states);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}



void profile_Init_dep(int n) {
    // Allocate memory for curand states
    curandState_t* states;
    cudaMalloc((void**)&states, n * sizeof(curandState_t));

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Start CUDA event recording
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Initialize curand states
    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    cudaDeviceSynchronize();

    // Stop CUDA event recording
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Initialization time: %f milliseconds\n", milliseconds);

    // Calculate the amount of data transferred
    size_t bytesTransferred = n * sizeof(curandState_t); // Size of the allocated memory
    printf("Data transferred: %zu bytes\n", bytesTransferred);

    // Calculate bandwidth
    float bandwidth = (bytesTransferred / (milliseconds / 1000.0f)) / (1024.0f * 1024.0f); // Convert to MB/s
    printf("Bandwidth: %f MB/s\n", bandwidth);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(states);
}

void profile_Init(int n) {
    // Allocate memory for curand states
    curandState_t* states;
    cudaMalloc((void**)&states, n * sizeof(curandState_t));

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Start CUDA event recording for initialization
    cudaEvent_t startInit, stopInit;
    cudaEventCreate(&startInit);
    cudaEventCreate(&stopInit);
    cudaEventRecord(startInit);

    // Initialize curand states
    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    cudaDeviceSynchronize();

    // Stop CUDA event recording for initialization
    cudaEventRecord(stopInit);
    cudaEventSynchronize(stopInit);

    // Calculate initialization time
    float initTime = 0;
    cudaEventElapsedTime(&initTime, startInit, stopInit);
    printf("Initialization Time: %f ms\n", initTime);

    // Start CUDA event recording for kernel execution
    cudaEvent_t startKernel, stopKernel;
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);
    cudaEventRecord(startKernel);

    // Initialize curand states
    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    cudaDeviceSynchronize();

    // Stop CUDA event recording for kernel execution
    cudaEventRecord(stopKernel);
    cudaEventSynchronize(stopKernel);

    // Calculate kernel execution time
    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, startKernel, stopKernel);
    printf("Kernel Execution Time: %f ms\n", kernelTime);

    // Start CUDA event recording for memory transfer
    cudaEvent_t startTransfer, stopTransfer;
    cudaEventCreate(&startTransfer);
    cudaEventCreate(&stopTransfer);
    cudaEventRecord(startTransfer);

    

    // Stop CUDA event recording for memory transfer
    cudaEventRecord(stopTransfer);
    cudaEventSynchronize(stopTransfer);

    // Calculate memory transfer time
    float transferTime = 0;
    cudaEventElapsedTime(&transferTime, startTransfer, stopTransfer);
    printf("Memory Transfer Time: %f ms\n", transferTime);

    

    // Clean up
    cudaEventDestroy(startInit);
    cudaEventDestroy(stopInit);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);
    cudaEventDestroy(startTransfer);
    cudaEventDestroy(stopTransfer);
    cudaFree(states);
}

void profile_info(int n) {
    // Device properties
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    printf("Profiling on Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory on GPU: %lu bytes\n", prop.totalGlobalMem);
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Memory clock rate (KHz): %d\n", prop.memoryClockRate);
    printf("Memory bus width (bits): %d\n", prop.memoryBusWidth);
    //printf("L2 cache size: %d bytes\n", prop.l2CacheSiz);
    printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
    printf("Maximum grid size: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Concurrent kernels: %s\n", (prop.concurrentKernels ? "Yes" : "No"));
    printf("ECC Enabled: %s\n", (prop.ECCEnabled ? "Yes" : "No"));

    // Profiling events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory and initialize states as needed
    curandState_t* states;
    cudaMalloc((void**)&states, n * sizeof(curandState_t));
    float* lognormalResults;
    cudaMalloc((void**)&lognormalResults, n * sizeof(float));

    // Initialize curand states
    int blockSize = 256; // Adjust as needed
    int numBlocks = (n + blockSize - 1) / blockSize;
    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    cudaDeviceSynchronize();

    // Profile generateLognormal
    cudaEventRecord(start);
    generateLognormal<<<numBlocks, blockSize>>>(states, lognormalResults, n, /* mean */ -0.0202027, /* stddev */ 0.489957,0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for generateLognormal: %f ms\n", milliseconds);

    // Calculate bandwidth and size transferred
    size_t bytesTransferred = n * sizeof(float); // Assuming only the results are transferred
    float bandwidth = (bytesTransferred / (milliseconds / 1000.0f)) / (1024.0f * 1024.0f); // Convert to MB/s
    printf("Data transferred: %zu bytes\n", bytesTransferred);
    printf("Bandwidth: %f MB/s\n", bandwidth);



     // Allocate memory and initialize states for normal_transform
    curandState_t* normalStates;
    cudaMalloc((void**)&normalStates, n * sizeof(curandState_t));
    float* normalResults;
    cudaMalloc((void**)&normalResults, n * sizeof(float));

    // Initialize curand states for normal_transform
    init<<<numBlocks, blockSize>>>(normalStates, /* seed */ 1234, n);
    cudaDeviceSynchronize();

    // Profile normal_transform
    cudaEventRecord(start);
    generateNormal<<<numBlocks, blockSize>>>(states, normalResults, n, /* mean */ 0.0, /* stddev */ 1.0,  /* decimalPlaces */ 2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for normal_transform: %f ms\n", milliseconds);

    // Calculate bandwidth and size transferred for normal_transform
    bytesTransferred = n * sizeof(float); // Assuming only the results are transferred
    bandwidth = (bytesTransferred / (milliseconds / 1000.0f)) / (1024.0f * 1024.0f); // Convert to MB/s
    printf("Data transferred for normal_transform: %zu bytes\n", bytesTransferred);
    printf("Bandwidth for normal_transform: %f MB/s\n", bandwidth);
    
        // Allocate memory and initialize states for generateCategorical3
    curandState_t* catStates;
    cudaMalloc((void**)&catStates, n * sizeof(curandState_t));
    int* catResults;
    cudaMalloc((void**)&catResults, n * sizeof(int));

    // Initialize curand states for generateCategorical3
    init<<<numBlocks, blockSize>>>(catStates, /* seed */ 1234, n);
    cudaDeviceSynchronize();

    // Profile generateCategorical3
    cudaEventRecord(start);
    generateCategorical3<<<numBlocks, blockSize>>>(catStates, catResults, n, /* ratio1 */ 0.3, /* ratio2 */ 0.5);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for generateCategorical3: %f ms\n", milliseconds);

    // Calculate bandwidth and size transferred for generateCategorical3
    bytesTransferred = n * sizeof(int); // Assuming only the results are transferred
    bandwidth = (bytesTransferred / (milliseconds / 1000.0f)) / (1024.0f * 1024.0f); // Convert to MB/s
    printf("Data transferred for generateCategorical3: %zu bytes\n", bytesTransferred);
    printf("Bandwidth for generateCategorical3: %f MB/s\n", bandwidth);

    // Profile generateCategorical2
    cudaEventRecord(start);
    generateCategorical2<<<numBlocks, blockSize>>>(catStates, catResults, n, 0.8);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for generateCategorical2: %f ms\n", milliseconds);

    // Calculate bandwidth
    bytesTransferred = n * sizeof(int); // Assuming only the results are transferred
    bandwidth = (bytesTransferred / (milliseconds / 1000.0f)) / (1024.0f * 1024.0f); // Convert to MB/s
    printf("Data transferred: %zu bytes\n", bytesTransferred);
    printf("Bandwidth: %f MB/s\n", bandwidth);

    
    

    // Clean up
    cudaFree(lognormalResults);
    cudaFree(states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // Free any other allocated memory
    // Calculate the maximum safe N
    size_t totalGlobalMem = prop.totalGlobalMem;
    size_t memoryPerOperation = 4; // 4 bytes per element for float and int
    size_t totalOperations = 8; // 2 instances each for 4 operations
    size_t maxSafeN = totalGlobalMem / (memoryPerOperation * totalOperations);

    printf("Maximum safe N for simultaneous operations: %zu\n", maxSafeN);
}


void runActualProcess(int n) {
    // Set up CUDA environment and memory allocations...
    // Initialize curand states
    int blockSize = 256; // Adjust as needed
    int numBlocks = (n + blockSize - 1) / blockSize;
    curandState_t* states;
    cudaMalloc((void**)&states, n * sizeof(curandState_t));
    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    
\

    // Allocate memory for results
    float* ageResults;
    float* fareResults;
    float* sibspResults;
    float* parchResults;
    // CATEGORICAL
    int* survivedResults;
    int* pclassResults;
    int* sexResults;
    int* embarkedResults;

    // Allocate memory on GPU
    cudaMalloc((void**)&ageResults, n * sizeof(float));
    cudaMalloc((void**)&fareResults, n * sizeof(float));
    cudaMalloc((void**)&sibspResults, n * sizeof(float));
    cudaMalloc((void**)&parchResults, n * sizeof(float));
    // CATEGORICAL
    cudaMalloc((void**)&survivedResults, n * sizeof(int));
    cudaMalloc((void**)&pclassResults, n * sizeof(int));
    cudaMalloc((void**)&sexResults, n * sizeof(int));
    cudaMalloc((void**)&embarkedResults, n * sizeof(int));

    // Parameters for each distribution
    float age_mu = 3.284686886076619, age_sigma = 0.4629353527760947;
    float fare_mu = 2.9120556444921593, fare_sigma = 0.9831822209722102;
    float sibsp_mean = 0.523008, sibsp_stddev = 1.102743;
    float parch_mean = 0.381594, parch_stddev = 0.805605;

    // Generate data for each variable
    generateLognormal<<<numBlocks, blockSize>>>(states, ageResults, n, age_mu, age_sigma, 0); // For Age
    generateLognormal<<<numBlocks, blockSize>>>(states, fareResults, n, fare_mu, fare_sigma, 2); // For Fare
    generateNormal<<<numBlocks, blockSize>>>(states, sibspResults, n, sibsp_mean, sibsp_stddev,0); // For SibSp
    generateNormal<<<numBlocks, blockSize>>>(states, parchResults, n, parch_mean, parch_stddev,0); // For Parch

    // Generate categorical data
    generateCategorical2<<<numBlocks, blockSize>>>(states, survivedResults, n, 342.0f / (342 + 549)); // Survived
    generateCategorical3<<<numBlocks, blockSize>>>(states, pclassResults, n, 216.0f / (216 + 184 + 491), 184.0f / (216 + 184 + 491)); // Pclass
    generateCategorical2<<<numBlocks, blockSize>>>(states, sexResults, n, 577.0f / (577 + 314)); // Sex
    generateCategorical3<<<numBlocks, blockSize>>>(states, embarkedResults, n, 644.0f / (644 + 168 + 77), 168.0f / (644 + 168 + 77)); // Embarked

    cudaDeviceSynchronize();

    // Copy results back to host
    float* hostAgeResults = new float[n];
    float* hostFareResults = new float[n];
    float* hostSibspResults = new float[n];
    float* hostParchResults = new float[n];
    int* hostSurvivedResults = new int[n];
    int* hostPclassResults = new int[n];
    int* hostSexResults = new int[n];
    int* hostEmbarkedResults = new int[n];

    cudaMemcpy(hostAgeResults, ageResults, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostFareResults, fareResults, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostSibspResults, sibspResults, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostParchResults, parchResults, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostSurvivedResults, survivedResults, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostPclassResults, pclassResults, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostSexResults, sexResults, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostEmbarkedResults, embarkedResults, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Write data to CSV file
    writeCSV("output.csv", n, hostAgeResults, hostFareResults, hostSibspResults, hostParchResults, hostSurvivedResults, hostPclassResults, hostSexResults, hostEmbarkedResults);

    // Clean up resources
    cudaFree(ageResults);
    cudaFree(fareResults);
    cudaFree(sibspResults);
    cudaFree(parchResults);
    cudaFree(survivedResults);
    cudaFree(pclassResults);
    cudaFree(sexResults);
    cudaFree(embarkedResults);

    delete[] hostAgeResults;
    delete[] hostFareResults;
    delete[] hostSibspResults;
    delete[] hostParchResults;
    delete[] hostSurvivedResults;
    delete[] hostPclassResults;
    delete[] hostSexResults;
    delete[] hostEmbarkedResults;
}


void profile_round(int n) {
    // Allocate memory and initialize states
    curandState_t* states;
    cudaMalloc((void**)&states, n * sizeof(curandState_t));
    float* results;
    cudaMalloc((void**)&results, n * sizeof(float));

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    init<<<numBlocks, blockSize>>>(states, /* seed */ 1234, n);
    cudaDeviceSynchronize();

    // Profiling events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Profile generateNormal with rounding
    cudaEventRecord(start);
    generateNormal<<<numBlocks, blockSize>>>(states, results, n, /* mean */ 0.0, /* stddev */ 1.0, /* decimalPlaces */ 2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisecondsWithRounding = 0;
    cudaEventElapsedTime(&millisecondsWithRounding, start, stop);

    // Profile generateNormal without rounding
    cudaEventRecord(start);
    generateNormalNoRounding<<<numBlocks, blockSize>>>(states, results, n, /* mean */ 0.0, /* stddev */ 1.0, /* decimalPlaces */ 2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisecondsWithoutRounding = 0;
    cudaEventElapsedTime(&millisecondsWithoutRounding, start, stop);

    // Output the results
    printf("Time with rounding: %f ms\n", millisecondsWithRounding);
    printf("Time without rounding: %f ms\n", millisecondsWithoutRounding);

    // Clean up
    cudaFree(results);
    cudaFree(states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}



