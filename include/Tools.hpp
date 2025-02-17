#pragma once
#ifndef _TOOLS_HPP_
#define _TOOLS_HPP_
#include <algorithm>
#include <glm/glm.hpp>
#include <hpc/Simd.hpp>

namespace SoftRasterizer {
#if defined(__x86_64__) || defined(_WIN64)
struct PointSIMD {
  __m256 x, y, z;
};

struct NormalSIMD {
  __m256 x, y, z;
  NormalSIMD() = default;
  NormalSIMD(const __m256 &_x, const __m256 &_y, const __m256 &_z);

  // Normalizing all the vector components
  NormalSIMD normalized();

  __m256 zero = _mm256_set1_ps(0.0f);
};

struct TexCoordSIMD {
  __m256 u, v;
};

struct ColorSIMD {
  ColorSIMD();
  ColorSIMD(const __m256 &_r, const __m256 &_g, const __m256 &_b);
  __m256 r, g, b;
  const __m256 zero = _mm256_set1_ps(0.f);
  const __m256 one = _mm256_set1_ps(1.f);
};

#elif defined(__arm__) || defined(__aarch64__)
#include <arm/neon.h>
struct PointSIMD {
  simde__m256 x, y, z;
};

struct NormalSIMD {
  simde__m256 x, y, z;
  NormalSIMD() = default;
  NormalSIMD(const simde__m256 &_x, const simde__m256 &_y,
             const simde__m256 &_z);

  // Normalizing all the vector components
  NormalSIMD normalized();

  simde__m256 zero = simde_mm256_set1_ps(0.0f);
};

struct TexCoordSIMD {
  simde__m256 u, v;
};

struct ColorSIMD {
  ColorSIMD();
  ColorSIMD(const simde__m256 &_r, const simde__m256 &_g,
            const simde__m256 &_b);
  simde__m256 r, g, b;
  const simde__m256 zero = simde_mm256_set1_ps(0.f);
  const simde__m256 one = simde_mm256_set1_ps(01.f);
};

#else
#endif

struct Triangle;

struct Tools {
  static constexpr float PI = 3.14159265358979323846f;
  static constexpr float PI_INV = 1.0f / PI;
  static constexpr float epsilon = 1e-5f;

  // Base case: min with two arguments
  template <typename T> static T min(T a, T b) { return (a < b) ? a : b; }

  // Recursive variadic template
  template <typename T, typename... Args> static T min(T first, Args... args) {
    return std::min(first, min(args...)); // Recursively compare
  }

  // Base case: max with two arguments
  template <typename T> static T max(T a, T b) { return (a > b) ? a : b; }

  // Recursive variadic template
  template <typename T, typename... Args> static T max(T first, Args... args) {
    return std::max(first, max(args...)); // Recursively compare
  }

  static glm::vec3 to_vec3(const glm::vec4 &vec);

  /**
   * @brief Converts normalized color values (range [0, 1]) to RGB values (range
   * [0, 255]).
   *
   * This function takes three float values representing normalized color
   * components (red, green, blue), clamps them to ensure they are within the
   * valid range of [0, 1], and scales them to the RGB range of [0, 255]. It
   * returns the resulting RGB values as an Eigen::Vector3i.
   *
   * @param red The normalized red component (range [0, 1]).
   * @param green The normalized green component (range [0, 1]).
   * @param blue The normalized blue component (range [0, 1]).
   *
   * @return Eigen::Vector3i A vector containing the RGB values in the range [0,
   * 255].
   */
  static glm::uvec3 normalizedToRGB(float red, float green, float blue);
  static glm::uvec3 normalizedToRGB(const glm::vec3 &color);

  static glm::vec3 interpolateNormal(float alpha, float beta, float gamma,
                                     const glm::vec3 &normal1,
                                     const glm::vec3 &normal2,
                                     const glm::vec3 &normal3);

#if defined(__x86_64__) || defined(_WIN64)
  static NormalSIMD interpolateNormal(const __m256 &alpha, const __m256 &beta,
                                      const __m256 &gamma,
                                      const glm::vec3 &normal1,
                                      const glm::vec3 &normal2,
                                      const glm::vec3 &normal3);

  static TexCoordSIMD
  interpolateTexCoord(const __m256 &alpha, const __m256 &beta,
                      const __m256 &gamma, const glm::vec2 &textCoord1,
                      const glm::vec2 &textCoord2, const glm::vec2 &textCoord3);

#elif defined(__arm__) || defined(__aarch64__)
  static NormalSIMD
  interpolateNormal(const simde__m256 &alpha, const simde__m256 &beta,
                    const simde__m256 &gamma, const glm::vec3 &normal1,
                    const glm::vec3 &normal2, const glm::vec3 &normal3);

  static TexCoordSIMD
  interpolateTexCoord(const simde__m256 &alpha, const simde__m256 &beta,
                      const simde__m256 &gamma, const glm::vec2 &textCoord1,
                      const glm::vec2 &textCoord2, const glm::vec2 &textCoord3);

#else
#endif

  static glm::vec2 interpolateTexCoord(float alpha, float beta, float gamma,
                                       const glm::vec2 &textCoord1,
                                       const glm::vec2 &textCoord2,
                                       const glm::vec2 &textCoord3);

  static glm::vec3 calculateNormalWithWeight(const glm::vec3 &pa,
                                             const glm::vec3 &pb,
                                             const glm::vec3 &pc);

  static float fresnel(const glm::vec3 &rayDirection, const glm::vec3 &normal,
                       const float &refractiveIndex);

  static const float random_generator();

  static void epsilonEqual(glm::vec3& transformedNormal);

  /*
   * Mathematical Transformation Principle
   * localRay = (x, y, z) => worldRay = xT+yB+zN
   */
  static glm::vec3 toWorld(const glm::vec3 &local, const glm::vec3 &N);

  static bool isfinite(const glm::vec3 &v);

  template <size_t Begin, size_t End, typename F> static void static_for(F f) {
    if constexpr (Begin < End) {
      std::integral_constant<size_t, Begin> compile_rt_int;
      f(compile_rt_int);
      static_for<Begin + 1, End, F>(f);
    }
  }

  template <typename SimdType, typename ElementType = float>
  constexpr static std::size_t num_elements_in_simd() {
    if constexpr (std::is_same_v<SimdType, __m128>) {
      return sizeof(__m128) / sizeof(ElementType);
    }
#if defined(__x86_64__) || defined(_WIN64)
    else if constexpr (std::is_same_v<SimdType, __m256>) {
      return sizeof(__m256) / sizeof(ElementType);
    }
#elif defined(__arm__) || defined(__aarch64__)
    else if constexpr (std::is_same_v<SimdType, simde__m256>) {
      return sizeof(simde__m256) / sizeof(ElementType);
    }
#else
#endif
    else {

      static_assert(
          "Unsupported SIMD type. Only __m128 and __m256 are supported.");
      return 0; // Unreachable due to static_assert, but required for
                // compilation.
    }
  }
};
} // namespace SoftRasterizer

#endif //_TOOLS_HPP_
