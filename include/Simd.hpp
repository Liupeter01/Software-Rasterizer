#pragma once
#ifndef _SIMD_HPP_
#define _SIMD_HPP_
#include <cmath>
#include <iostream>

#if defined(__x86_64__) || defined(_WIN64)
#include <immintrin.h>
#include <xmmintrin.h >

#elif defined(__arm__) || defined(__aarch64__)
#include <arm/neon.h>
#include <sse2neon.h>

#else
#endif

#endif // _SIMD_HPP_
