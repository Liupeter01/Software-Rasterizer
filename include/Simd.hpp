#pragma once
#ifndef _SIMD_HPP_
#define  _SIMD_HPP_
#include <iostream>
#include <cmath>

#if defined(__x86_64__) || defined(_WIN64)
#include <x86/fma.h>
#include<x86/avx2.h>

#elif defined(__arm__) || defined(__aarch64__)
#include <arm/neon.h>

#else
#endif

#endif // _SIMD_HPP_