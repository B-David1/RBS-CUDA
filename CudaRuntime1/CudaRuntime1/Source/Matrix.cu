#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Core.h"

#ifdef USE_CUDA
    #include "..\Includes\CUDAMatrix.h"
#else
    #include "..\Includes\Matrix.h"
#endif
#include <stdio.h>

typedef struct {
    ufast16 width;
    ufast16 height;
    bool* elements;
} CUDAMatrix;

#ifdef USE_CUDA
__global__ void MatIsEqualKernel(CUDAMatrix A, const bool* const B, bool* const value)
{
    uint32 row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32 col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= A.height || col >= A.width) return;

    if (A.elements[row * A.width + col] ^ B[row * A.width + col])
        *value = false;
}

bool MatIsEqual(const BoolMatrix& A, const BoolMatrix& B)
{
    CUDAMatrix a;
    a.elements = A.getElements();
    a.height = A.getRows();
    a.width = A.getColumns();

    bool* value;
    cudaMalloc(&value, sizeof(*value));
    cudaMemset(value, 1, sizeof(*value));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.getColumns() + dimBlock.x - 1) / dimBlock.x, (B.getRows() + dimBlock.y - 1) / dimBlock.y);
    MatIsEqualKernel << <dimGrid, dimBlock >> > (a, B.getElements(), value);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    bool v;
    cudaMemcpy(&v, value, sizeof(*value), cudaMemcpyDeviceToHost);
    cudaFree(value);

    return v;
}

__global__ void MatNegateKernel(CUDAMatrix A)
{
    uint32 row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32 col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= A.height || col >= A.width) return;

    A.elements[row * A.width + col] = !A.elements[row * A.width + col];
}

void MatNegate(BoolMatrix& A)
{
    CUDAMatrix a;
    a.elements = A.getElements();
    a.width = A.getColumns();
    a.height = A.getRows();

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((A.getColumns() + dimBlock.x - 1) / dimBlock.x, (A.getRows() + dimBlock.y - 1) / dimBlock.y);
    MatNegateKernel<<<dimGrid, dimBlock>>>(a);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
}

__global__ void MatAndKernel(const CUDAMatrix A, const bool* const B, bool* const C)
{
    uint32 row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32 col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= A.height || col >= A.width) return;

    C[row * A.width + col] = A.elements[row * A.width + col] & B[row * A.width + col];
}

BoolMatrix MatAnd(const BoolMatrix& A, const BoolMatrix& B)
{
    BoolMatrix C(A.getRows(), A.getColumns());

    CUDAMatrix a;
    a.elements = A.getElements();
    a.height = A.getRows();
    a.width = A.getColumns();

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.getColumns() + dimBlock.x - 1) / dimBlock.x, (B.getRows() + dimBlock.y - 1) / dimBlock.y);
    MatAndKernel << <dimGrid, dimBlock >> > (a, B.getElements(), C.getElements());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    return C;
}

__global__ void MatOrKernel(const CUDAMatrix A, const bool* const B, bool* const C)
{
    uint32 row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32 col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= A.height || col >= A.width) return;

    C[row * A.width + col] = A.elements[row * A.width + col] | B[row * A.width + col];
}

void MatOr(const BoolMatrix& A, const BoolMatrix& B, BoolMatrix& C)
{
    CUDAMatrix a;
    a.elements = A.getElements();
    a.height = A.getRows();
    a.width = A.getColumns();

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.getColumns() + dimBlock.x - 1) / dimBlock.x, (B.getRows() + dimBlock.y - 1) / dimBlock.y);
    MatOrKernel << <dimGrid, dimBlock >> > (a, B.getElements(), C.getElements());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
}

__global__ void MatAndProductKernel(const CUDAMatrix A, const bool* const B, bool* const out)
{
    uint32 row = blockIdx.y * blockDim.y + threadIdx.y;

    out[row] = true;
    for (ufast16 e = 0; e < A.height; e++)
        out[row] &= A.elements[row * A.width + e] | B[e];
}

void MatAndProduct(const BoolMatrix& A, const BoolMatrix& B, BoolMatrix& C)
{
    CUDAMatrix a;
    a.elements = A.getElements();
    a.height = A.getRows();
    a.width = A.getColumns();

    // Invoke kernel
    dim3 dimBlock(1, BLOCK_SIZE * 2);
    dim3 dimGrid(dimBlock.x, (B.getRows() + dimBlock.y - 1) / dimBlock.y);
    MatAndProductKernel<<<dimGrid, dimBlock >>>(a, B.getElements(), C.getElements());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
}

__global__ void MatOrProductKernel(const CUDAMatrix A, const bool* const B, bool* const out)
{
    uint32 row = blockIdx.y * blockDim.y + threadIdx.y;

    out[row] = false;
    for (ufast16 e = 0; e < A.height; e++)
        out[row] |= A.elements[row * A.width + e] & B[e];
}

void MatOrProduct(const BoolMatrix& A, const BoolMatrix& B, BoolMatrix& C)
{
    CUDAMatrix a;
    a.elements = A.getElements();
    a.height = A.getRows();
    a.width = A.getColumns();

    // Invoke kernel
    dim3 dimBlock(1, BLOCK_SIZE * 2);
    dim3 dimGrid(dimBlock.x, (B.getRows() + dimBlock.y - 1) / dimBlock.y);
    MatOrProductKernel << <dimGrid, dimBlock >> > (a, B.getElements(), C.getElements());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
}
#endif