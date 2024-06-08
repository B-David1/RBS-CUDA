#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <assert.h>

#include <chrono>
#include <thread>

using namespace std;

typedef uint_least32_t uint32;
typedef uint_fast16_t ufast16;

typedef uint_least64_t uint64;
typedef uint_fast32_t ufast32;

#define USE_CUDA true;

// Thread block size

// if you have below 1024 threads, use 16
// otherwise use 32

#define BLOCK_SIZE 32