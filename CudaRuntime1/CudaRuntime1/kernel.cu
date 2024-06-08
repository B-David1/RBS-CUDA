
#include "Core.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef USE_CUDA
    #include "Includes\CUDAMatrix.h"
#else
    #include "Includes\Matrix.h"
#endif
#include <stdio.h>

int main()
{
    FILE* f1 = fopen("rhs.txt", "r");
    if (f1 == NULL)
    {
        cout << "rhs.txt not found";
        return 0;
    }

    FILE* f2 = fopen("lhs.txt", "r");
    if (f2 == NULL)
    {
        cout << "lhs.txt not found";
        return 0;
    }

    FILE* fout = fopen("conclusions.txt", "w+");

//    generateRulesMatrixWithMaxSteps(128); return 0;

    uint32 sizeRHS = 0;
    uint32 lengthRHS = 16;
    ufast16 rowsRHS = 0;
    ufast16 lastValidRowsRHS = 0;

    bool* i = (bool*) malloc(sizeof(bool) * lengthRHS);
    for (char c = ' '; c != EOF; c = fgetc(f1))
    {
        if (c >= '0' && c <= '9')
        {
            i[sizeRHS++] = c - '0';
            lastValidRowsRHS = rowsRHS;
            if (sizeRHS >= lengthRHS)
            {
                lengthRHS <<= 2;
                i = (bool*) realloc(i, sizeof(*i) * lengthRHS);
            }
        }
        else if (c == '\n')
            rowsRHS++;
    }
    ++lastValidRowsRHS;
    fclose(f1);

    uint32 sizeLHS = 0;
    uint32 lengthLHS = 16;
    ufast16 rowsLHS = 0;
    ufast16 lastValidRowsLHS = 0;

    bool* j = (bool*)malloc(sizeof(bool) * lengthLHS);
    for (char c = ' '; c != EOF; c = fgetc(f2))
    {
        if (c >= '0' && c <= '9')
        {
            j[sizeLHS++] = c - '0';
            lastValidRowsLHS = rowsLHS;
            if (sizeLHS >= lengthLHS)
            {
                lengthLHS <<= 2;
                j = (bool*)realloc(j, sizeof(*j) * lengthLHS);
            }
        }
        else if (c == '\n')
            rowsLHS++;
    }
    ++lastValidRowsLHS;
    fclose(f2);

    assert(sizeRHS == sizeLHS);

    BoolMatrix* rhs = new BoolMatrix(lastValidRowsRHS, sizeRHS/lastValidRowsRHS, i);
    BoolMatrix* lhs = new BoolMatrix(lastValidRowsLHS, sizeLHS/lastValidRowsLHS, j);
    
#ifdef USE_CUDA
    cout << parseRules(i, j, lastValidRowsRHS, sizeRHS/lastValidRowsRHS) << endl;
#else
//    cout << parseRHS(*rhs) << endl;
//    cout << parseLHS(*lhs) << endl;
    cout << parseRules(*rhs, *lhs) << endl;
#endif

    free(i);
    free(j);

    ~(*rhs);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    FILE* const out = stdout;
//    computeAllSetsOfConclusions(rhs, lhs, out);
    computeLastSetOfConclusions(rhs, lhs, out);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

#ifdef USE_CUDA
    std::cout << "Time elapsed [CUDA] = "
#else
    std::cout << "Time elapsed [CPU] = " 
#endif
        << ((long double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())) / 1000000.0 << "[s]" << std::endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    delete lhs;
    delete rhs;

    fclose(fout);

    return 0;
}