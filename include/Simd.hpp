#pragma once
#ifndef _SIMD_HPP_
#define _SIMD_HPP_
#include <cmath>
#include <iostream>

#if defined(__x86_64__) || defined(_WIN64)
#include <x86/avx2.h>
#include <x86/fma.h>

#elif defined(__arm__) || defined(__aarch64__)
#include <arm/neon.h>

#else
#endif

#endif // _SIMD_HPP_
