#pragma once
#ifndef _TOOLS_HPP_
#define _TOOLS_HPP_
#include <Eigen/Eigen>
#include <Simd.hpp>
#include <algorithm>

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
  const __m256 one = _mm256_set1_ps(01.f);
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

struct Tools {
  static constexpr float PI = 3.14159265358979323846f;

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

  // degree to radian
  static float degreeToRadian(float degree);

  static Eigen::Vector4f to_vec4(const Eigen::Vector3f &v3, float w = 1.0f);

  static Eigen::Vector3f to_vec3(const Eigen::Vector4f &v4);

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
  static Eigen::Vector3i normalizedToRGB(float red, float green, float blue);
  static Eigen::Vector3i normalizedToRGB(const Eigen::Vector3f &color);

  /**
   * @brief Interpolates between three RGB colors using alpha, beta, and gamma
   * weights.
   *
   * Given three RGB colors and interpolation parameters (alpha, beta, gamma),
   * this function computes the interpolated RGB color using a weighted average.
   *
   * @param alpha Weight for the first color (range [0, 1]).
   * @param beta Weight for the second color (range [0, 1]).
   * @param gamma Weight for the third color (range [0, 1]).
   * @param color1 The first RGB color (Eigen::Vector3i).
   * @param color2 The second RGB color (Eigen::Vector3i).
   * @param color3 The third RGB color (Eigen::Vector3i).
   *
   * @return Eigen::Vector3i The interpolated RGB color.
   */
  static Eigen::Vector3i interpolateRGB(float alpha, float beta, float gamma,
                                        const Eigen::Vector3i &color1,
                                        const Eigen::Vector3i &color2,
                                        const Eigen::Vector3i &color3);

  static Eigen::Vector3f interpolateNormal(float alpha, float beta, float gamma,
                                           const Eigen::Vector3f &normal1,
                                           const Eigen::Vector3f &normal2,
                                           const Eigen::Vector3f &normal3);

#if defined(__x86_64__) || defined(_WIN64)
  static NormalSIMD interpolateNormal(const __m256 &alpha, const __m256 &beta,
                                      const __m256 &gamma,
                                      const Eigen::Vector3f &normal1,
                                      const Eigen::Vector3f &normal2,
                                      const Eigen::Vector3f &normal3);

  static TexCoordSIMD interpolateTexCoord(const __m256 &alpha,
                                          const __m256 &beta,
                                          const __m256 &gamma,
                                          const Eigen::Vector2f &textCoord1,
                                          const Eigen::Vector2f &textCoord2,
                                          const Eigen::Vector2f &textCoord3);

#elif defined(__arm__) || defined(__aarch64__)
  static NormalSIMD interpolateNormal(const simde__m256 &alpha,
                                      const simde__m256 &beta,
                                      const simde__m256 &gamma,
                                      const Eigen::Vector3f &normal1,
                                      const Eigen::Vector3f &normal2,
                                      const Eigen::Vector3f &normal3);

  static TexCoordSIMD interpolateTexCoord(const simde__m256 &alpha,
                                          const simde__m256 &beta,
                                          const simde__m256 &gamma,
                                          const Eigen::Vector2f &textCoord1,
                                          const Eigen::Vector2f &textCoord2,
                                          const Eigen::Vector2f &textCoord3);

#else
#endif

  static Eigen::Vector2f interpolateTexCoord(float alpha, float beta,
                                             float gamma,
                                             const Eigen::Vector2f &textCoord1,
                                             const Eigen::Vector2f &textCoord2,
                                             const Eigen::Vector2f &textCoord3);

  static Eigen::Vector3f calculateNormalWithWeight(const Eigen::Vector3f &A,
                                                   const Eigen::Vector3f &B,
                                                   const Eigen::Vector3f &C);

  /**
   * @brief Calculates the rotation matrix using Rodrigues' rotation formula.
   *
   * This function computes a 4x4 rotation matrix that represents a rotation
   * of the given angle (in degrees) around the specified axis in 3D space.
   * It utilizes Rodrigues' rotation formula to generate the matrix.
   *
   * @param axis A 3D vector representing the axis of rotation.
   * @param angle The angle (in degrees) by which to rotate around the axis.
   *        A positive value rotates counterclockwise, and a negative value
   *        rotates clockwise around the axis.
   *
   * @return A 4x4 rotation matrix (Eigen::Matrix4f) representing the rotation.
   */
  static Eigen::Matrix4f calculateRotationMatrix(const Eigen::Vector3f &axis,
                                                 float angle);

  /**
   * @brief Calculates the model matrix for a 3D object.
   *
   * This function computes a 4x4 model matrix that represents the
   * transformation of a 3D object in world space. The model matrix is obtained
   * by combining the translation, rotation, and scaling transformations.
   *
   * @param translation A 3D vector representing the translation (position) of
   * the object.
   * @param rotation A 4x4 matrix representing the rotation of the object.
   * @param scale A 3D vector representing the scaling factors along the x, y,
   * and z axes.
   *
   * @return A 4x4 model matrix (Eigen::Matrix4f) representing the combined
   * transformations.
   */
  static Eigen::Matrix4f
  calculateModelMatrix(const Eigen::Vector3f &translation,
                       const Eigen::Matrix4f &rotation,
                       const Eigen::Vector3f &scale);

  /**
   * @brief Calculates the view matrix for a camera.
   *
   * This function computes a 4x4 view matrix that represents the transformation
   * from world coordinates to camera coordinates based on the camera's position
   * (eye), target point (center), and up direction (up).
   *
   * @param eye The position of the camera in world space.
   * @param center The point the camera is looking at in world space.
   * @param up The up direction vector for the camera, which defines the
   * orientation.
   *
   * @return A 4x4 view matrix (Eigen::Matrix4f) representing the camera's view
   * transformation.
   */
  static Eigen::Matrix4f calculateViewMatrix(const Eigen::Vector3f &eye,
                                             const Eigen::Vector3f &center,
                                             const Eigen::Vector3f &up);

  /**
   * @brief Calculates an orthographic projection matrix.
   *
   * This function computes a 4x4 orthographic projection matrix that maps 3D
   * world space coordinates to 2D normalized device coordinates. The
   * orthographic projection is defined by the left, right, bottom, top, zNear,
   * and zFar parameters.
   *
   * @param left The coordinate for the left clipping plane.
   * @param right The coordinate for the right clipping plane.
   * @param bottom The coordinate for the bottom clipping plane.
   * @param top The coordinate for the top clipping plane.
   * @param zNear The distance to the near clipping plane.
   * @param zFar The distance to the far clipping plane.
   *
   * @return A 4x4 orthographic projection matrix (Eigen::Matrix4f).
   */
  static Eigen::Matrix4f calculateOrthoMatrix(float left, float right,
                                              float bottom, float top,
                                              float zNear, float zFar);

  /**
   * @brief Calculates a perspective projection matrix.
   *
   * This function computes a 4x4 perspective projection matrix using the field
   * of view (fovy), aspect ratio, near clipping plane (zNear), and far clipping
   * plane (zFar). The perspective matrix is used to simulate a camera's field
   * of view, where objects closer to the camera appear larger.
   *
   * @param fovy The vertical field of view in degrees.
   * @param aspect The aspect ratio of the view (width / height).
   * @param zNear The distance to the near clipping plane.
   * @param zFar The distance to the far clipping plane.
   *
   * @return A 4x4 perspective projection matrix (Eigen::Matrix4f).
   */
  static Eigen::Matrix4f calculateProjectionMatrix(float fovy, float aspect,
                                                   float zNear, float zFar);

  template<size_t Begin, size_t End, typename F>
  static void static_for(F f) {
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
            else if constexpr (std::is_same_v<SimdType, __m256>) {
                      return sizeof(__m256) / sizeof(ElementType);
            }
            else {
                      static_assert(std::is_same_v<SimdType, __m128> || std::is_same_v<SimdType, __m256>,
                                "Unsupported SIMD type. Only __m128 and __m256 are supported.");
                      return 0;  // Unreachable due to static_assert, but required for compilation.
            }
  }
};
} // namespace SoftRasterizer

#endif //_TOOLS_HPP_
