#include <Tools.hpp>
#include <Triangle.hpp>
#include <spdlog/spdlog.h>

#if defined(__x86_64__) || defined(_WIN64)
SoftRasterizer::NormalSIMD::NormalSIMD(const __m256 &_x, const __m256 &_y,
                                       const __m256 &_z)
    : x(_x), y(_y), z(_z) {}

// Normalizing all the vector components
SoftRasterizer::NormalSIMD SoftRasterizer::NormalSIMD::normalized() {
  __m256 length = _mm256_sqrt_ps(
      _mm256_fmadd_ps(x, x, _mm256_fmadd_ps(y, y, _mm256_mul_ps(z, z))));

  // only filter lenth >0(GT) , then set mask to 1
  __m256 mask = _mm256_cmp_ps(length, zero, _CMP_GT_OQ);
  __m256 inverse = _mm256_blendv_ps(zero, _mm256_rcp_ps(length), mask);

  return NormalSIMD(_mm256_blendv_ps(zero, _mm256_mul_ps(x, inverse), mask),
                    _mm256_blendv_ps(zero, _mm256_mul_ps(y, inverse), mask),
                    _mm256_blendv_ps(zero, _mm256_mul_ps(z, inverse), mask));
}

SoftRasterizer::ColorSIMD::ColorSIMD()
    : r(_mm256_set1_ps(1.0f)), g(_mm256_set1_ps(1.0f)),
      b(_mm256_set1_ps(1.0f)) {}

SoftRasterizer::ColorSIMD::ColorSIMD(const __m256 &_r, const __m256 &_g,
                                     const __m256 &_b)
    : r(_r), g(_g), b(_b) {}

#elif defined(__arm__) || defined(__aarch64__)
#include <arm/neon.h>
SoftRasterizer::NormalSIMD::NormalSIMD(const simde__m256 &_x,
                                       const simde__m256 &_y,
                                       const simde__m256 &_z)
    : x(_x), y(_y), z(_z) {}

// Normalizing all the vector components
SoftRasterizer::NormalSIMD SoftRasterizer::NormalSIMD::normalized() {
  /*x^2 + y^2 + z^2*/
  auto squre = simde_mm256_mul_ps(x, x);
  auto squre_y = simde_mm256_mul_ps(y, y);
  auto squre_z = simde_mm256_mul_ps(z, z);

  squre_y = simde_mm256_add_ps(squre_y, squre_z);
  squre = simde_mm256_add_ps(squre, squre_y);

  simde__m256 length = simde_mm256_sqrt_ps(squre);

  // only filter lenth >0(GT) , then set mask to 1
  simde__m256 mask = simde_mm256_cmp_ps(length, zero, SIMDE_CMP_GT_OQ);

  return NormalSIMD(
      simde_mm256_blendv_ps(zero, simde_mm256_div_ps(x, length), mask),
      simde_mm256_blendv_ps(zero, simde_mm256_div_ps(y, length), mask),
      simde_mm256_blendv_ps(zero, simde_mm256_div_ps(z, length), mask));
}

SoftRasterizer::ColorSIMD::ColorSIMD()
    : r(simde_mm256_set1_ps(1.0f)), g(simde_mm256_set1_ps(1.0f)),
      b(simde_mm256_set1_ps(1.0f)) {}

SoftRasterizer::ColorSIMD::ColorSIMD(const simde__m256 &_r,
                                     const simde__m256 &_g,
                                     const simde__m256 &_b)
    : r(_r), g(_g), b(_b) {}

#else
#endif

// degree to radian
float SoftRasterizer::Tools::degreeToRadian(float degree) {
  return degree * (PI / 180.0f);
}

Eigen::Vector4f SoftRasterizer::Tools::to_vec4(const Eigen::Vector3f &v3,
                                               float w) {
  return Eigen::Vector4f(v3.x(), v3.y(), v3.z(), w);
}

Eigen::Vector3f SoftRasterizer::Tools::to_vec3(const Eigen::Vector4f &v4) {
  return Eigen::Vector3f(v4.x() / v4.w(), v4.y() / v4.w(), v4.z() / v4.w());
}

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
Eigen::Vector3i SoftRasterizer::Tools::normalizedToRGB(float red, float green,
                                                       float blue) {

  // Clamp values to ensure they're within the range [0, 1]
  red = std::clamp(red, 0.0f, 1.0f);
  green = std::clamp(green, 0.0f, 1.0f);
  blue = std::clamp(blue, 0.0f, 1.0f);

  // Scale to the range [0, 255] and convert to integers
  Eigen::Vector3i rgb;
  rgb << static_cast<int>(red * 255.0f), static_cast<int>(green * 255.0f),
      static_cast<int>(blue * 255.0f);

  return rgb;
}

Eigen::Vector3i
SoftRasterizer::Tools::normalizedToRGB(const Eigen::Vector3f &color) {
  return normalizedToRGB(color.x(), color.y(), color.z());
}

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
Eigen::Vector3i SoftRasterizer::Tools::interpolateRGB(
    float alpha, float beta, float gamma, const Eigen::Vector3i &color1,
    const Eigen::Vector3i &color2, const Eigen::Vector3i &color3) {
  // Ensure weights sum up to 1.0 (optional, but recommended for normalized
  // interpolation)
  float sum = alpha + beta + gamma;
  alpha /= sum;
  beta /= sum;
  gamma /= sum;

  // Perform the weighted average of the RGB components
  Eigen::Vector3i interpolatedColor;
  interpolatedColor =
      (alpha * color1.cast<float>() + beta * color2.cast<float>() +
       gamma * color3.cast<float>())
          .cast<int>();

  interpolatedColor = interpolatedColor.cwiseMin(255).cwiseMax(0);
  return interpolatedColor;
}

Eigen::Vector3f SoftRasterizer::Tools::interpolateNormal(
    float alpha, float beta, float gamma, const Eigen::Vector3f &normal1,
    const Eigen::Vector3f &normal2, const Eigen::Vector3f &normal3) {
  Eigen::Vector3f normal = alpha * normal1 + beta * normal2 + gamma * normal3;
  return normal.normalized();
}

#if defined(__x86_64__) || defined(_WIN64)
SoftRasterizer::NormalSIMD SoftRasterizer::Tools::interpolateNormal(
    const __m256 &alpha, const __m256 &beta, const __m256 &gamma,
    const Eigen::Vector3f &normal1, const Eigen::Vector3f &normal2,
    const Eigen::Vector3f &normal3) {
  return NormalSIMD(_mm256_fmadd_ps(
                        alpha, _mm256_set1_ps(normal1.x()),
                        _mm256_fmadd_ps(
                            beta, _mm256_set1_ps(normal2.x()),
                            _mm256_mul_ps(gamma, _mm256_set1_ps(normal3.x())))),
                    _mm256_fmadd_ps(
                        alpha, _mm256_set1_ps(normal1.y()),
                        _mm256_fmadd_ps(
                            beta, _mm256_set1_ps(normal2.y()),
                            _mm256_mul_ps(gamma, _mm256_set1_ps(normal3.y())))),
                    _mm256_fmadd_ps(
                        alpha, _mm256_set1_ps(normal1.z()),
                        _mm256_fmadd_ps(
                            beta, _mm256_set1_ps(normal2.z()),
                            _mm256_mul_ps(gamma, _mm256_set1_ps(normal3.z()))))

                        )
      .normalized();
}

SoftRasterizer::TexCoordSIMD SoftRasterizer::Tools::interpolateTexCoord(
    const __m256 &alpha, const __m256 &beta, const __m256 &gamma,
    const Eigen::Vector2f &textCoord1, const Eigen::Vector2f &textCoord2,
    const Eigen::Vector2f &textCoord3) {
  TexCoordSIMD result;

  result.u = _mm256_fmadd_ps(
      alpha, _mm256_set1_ps(textCoord1.x()),
      _mm256_fmadd_ps(beta, _mm256_set1_ps(textCoord2.x()),
                      _mm256_mul_ps(gamma, _mm256_set1_ps(textCoord3.x()))));
  result.v = _mm256_fmadd_ps(
      alpha, _mm256_set1_ps(textCoord1.y()),
      _mm256_fmadd_ps(beta, _mm256_set1_ps(textCoord2.y()),
                      _mm256_mul_ps(gamma, _mm256_set1_ps(textCoord3.y()))));

  // Return as a struct containing both components (x and y)
  return result;
}

#elif defined(__arm__) || defined(__aarch64__)
SoftRasterizer::NormalSIMD SoftRasterizer::Tools::interpolateNormal(
    const simde__m256 &alpha, const simde__m256 &beta, const simde__m256 &gamma,
    const Eigen::Vector3f &normal1, const Eigen::Vector3f &normal2,
    const Eigen::Vector3f &normal3) {

  NormalSIMD normal;
  auto a = simde_mm256_mul_ps(alpha, simde_mm256_set1_ps(normal1.x()));
  auto b = simde_mm256_mul_ps(beta, simde_mm256_set1_ps(normal2.x()));
  auto c = simde_mm256_mul_ps(gamma, simde_mm256_set1_ps(normal3.x()));

  a = simde_mm256_add_ps(a, b);
  normal.x = simde_mm256_add_ps(a, c);

  a = simde_mm256_mul_ps(alpha, simde_mm256_set1_ps(normal1.y()));
  b = simde_mm256_mul_ps(beta, simde_mm256_set1_ps(normal2.y()));
  c = simde_mm256_mul_ps(gamma, simde_mm256_set1_ps(normal3.y()));
  a = simde_mm256_add_ps(a, b);
  normal.y = simde_mm256_add_ps(a, c);

  a = simde_mm256_mul_ps(alpha, simde_mm256_set1_ps(normal1.z()));
  b = simde_mm256_mul_ps(beta, simde_mm256_set1_ps(normal2.z()));
  c = simde_mm256_mul_ps(gamma, simde_mm256_set1_ps(normal3.z()));
  a = simde_mm256_add_ps(a, b);
  normal.z = simde_mm256_add_ps(a, c);

  return normal.normalized();
}

SoftRasterizer::TexCoordSIMD SoftRasterizer::Tools::interpolateTexCoord(
    const simde__m256 &alpha, const simde__m256 &beta, const simde__m256 &gamma,
    const Eigen::Vector2f &textCoord1, const Eigen::Vector2f &textCoord2,
    const Eigen::Vector2f &textCoord3) {

  // Return as a struct containing both components (x and y)
  TexCoordSIMD result;
  auto a = simde_mm256_mul_ps(alpha, simde_mm256_set1_ps(textCoord1.x()));
  auto b = simde_mm256_mul_ps(beta, simde_mm256_set1_ps(textCoord2.x()));
  auto c = simde_mm256_mul_ps(gamma, simde_mm256_set1_ps(textCoord3.x()));
  a = simde_mm256_add_ps(a, b);
  result.u = simde_mm256_add_ps(a, c);

  a = simde_mm256_mul_ps(alpha, simde_mm256_set1_ps(textCoord1.y()));
  b = simde_mm256_mul_ps(beta, simde_mm256_set1_ps(textCoord2.y()));
  c = simde_mm256_mul_ps(gamma, simde_mm256_set1_ps(textCoord3.y()));
  a = simde_mm256_add_ps(a, b);
  result.v = simde_mm256_add_ps(a, c);

  return result;
}

#else
#endif

Eigen::Vector2f SoftRasterizer::Tools::interpolateTexCoord(
    float alpha, float beta, float gamma, const Eigen::Vector2f &textCoord1,
    const Eigen::Vector2f &textCoord2, const Eigen::Vector2f &textCoord3) {
  return alpha * textCoord1 + beta * textCoord2 + gamma * textCoord3;
}

Eigen::Vector3f
SoftRasterizer::Tools::calculateNormalWithWeight(const Eigen::Vector3f &A,
                                                 const Eigen::Vector3f &B,
                                                 const Eigen::Vector3f &C) {
  const Eigen::Vector3f AB = B - A;
  const Eigen::Vector3f AC = C - A;
  Eigen::Vector3f normal = AB.cross(AC);

  const float length = normal.norm();

  // arc_sin_degree
  const float arc_sin_degree = length / (AB.norm() * AC.norm());

  // climp value to [0.f, 1.0f]
  float weight = std::asin(arc_sin_degree) / length;

  if (-0.00000001f <= weight && weight <= 0.00000001f) {
    return normal;
  }

  Eigen::Vector3f weightedNormal = normal * weight;
  return normal * weight;
}

/**
 * @brief Calculates the rotation matrix using Rodrigues' rotation formula.
 *
 * This function computes a 4x4 rotation matrix that represents a rotation
 * of the given angle (in radian) around the specified axis in 3D space.
 * It utilizes Rodrigues' rotation formula to generate the matrix.
 *
 * @param axis A 3D vector representing the axis of rotation.
 * @param angle The angle (in radian) by which to rotate around the axis.
 *        A positive value rotates counterclockwise, and a negative value
 *        rotates clockwise around the axis.
 *
 * @return A 4x4 rotation matrix (Eigen::Matrix4f) representing the rotation.
 */
Eigen::Matrix4f
SoftRasterizer::Tools::calculateRotationMatrix(const Eigen::Vector3f &axis,
                                               float angle) {
  spdlog::debug(
      "Calculating rotation matrix for angle: {} and axis: ({}, {}, {})", angle,
      axis.x(), axis.y(), axis.z());
  Eigen::Matrix4f rotationMatrix = Eigen::Matrix4f::Identity();

  auto rad = degreeToRadian(angle);

  auto cos = std::cos(rad);
  auto sin = std::sin(rad);

  const Eigen::Vector3f origin =
      Eigen::Vector3f(0.f, 0.f, 0.f); // origin point(0, 0, 0)
  const Eigen::Vector3f normalize_axis = axis.normalized(); // normalize axis

  const Eigen::Vector3f move_back = axis - origin; // move from origin to axis
  const Eigen::Vector3f transform_to = -move_back; // move from axis to origin

  /*Transform Matrix*/
  Eigen::Matrix4f move_back_matrix;    // Transform from origin to axis
  Eigen::Matrix4f transform_to_matrix; // Transform from axis to origin

  /*Rotate Matrix*/
  Eigen::Matrix4f rotationMatrix4 = Eigen::Matrix4f::Identity();

  transform_to_matrix << 1, 0, 0, transform_to.x(), 0, 1, 0, transform_to.y(),
      0, 0, 1, transform_to.z(), 0, 0, 0, 1;

  move_back_matrix << 1, 0, 0, move_back.x(), 0, 1, 0, move_back.y(), 0, 0, 1,
      move_back.z(), 0, 0, 0, 1;

  /*Rodrigues Rotation*/
  Eigen::Matrix3f Rodrigues;
  Rodrigues << 0, -normalize_axis.z(), normalize_axis.y(), normalize_axis.z(),
      0, -normalize_axis.x(), -normalize_axis.y(), normalize_axis.x(), 0;

  // cos E + (1-cos)(n * n^T)+sin * Rodrigues
  Rodrigues = cos * Eigen::Matrix3f::Identity() +
              (1.0f - cos) * normalize_axis * normalize_axis.transpose() +
              sin * Rodrigues;

  rotationMatrix4.block<3, 3>(0, 0) = Rodrigues;

  return move_back_matrix * rotationMatrix4 * transform_to_matrix *
         rotationMatrix;
}

/**
 * @brief Calculates the model matrix for a 3D object.
 *
 * This function computes a 4x4 model matrix that represents the transformation
 * of a 3D object in world space. The model matrix is obtained by combining the
 * translation, rotation, and scaling transformations.
 *
 * @param translation A 3D vector representing the translation (position) of the
 * object.
 * @param rotation A 4x4 matrix representing the rotation of the object.
 * @param scale A 3D vector representing the scaling factors along the x, y, and
 * z axes.
 *
 * @return A 4x4 model matrix (Eigen::Matrix4f) representing the combined
 * transformations.
 */
Eigen::Matrix4f
SoftRasterizer::Tools::calculateModelMatrix(const Eigen::Vector3f &translation,
                                            const Eigen::Matrix4f &rotation,
                                            const Eigen::Vector3f &scale) {
  Eigen::Matrix4f modelMatrix = Eigen::Matrix4f::Identity();

  Eigen::Matrix4f T, S;
  T << 1, 0, 0, translation.x(), 0, 1, 0, translation.y(), 0, 0, 1,
      translation.z(), 0, 0, 0, 1;

  S << scale.x(), 0, 0, 0, 0, scale.y(), 0, 0, 0, 0, scale.z(), 0, 0, 0, 0, 1;

  return T * rotation * S;
}

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
Eigen::Matrix4f
SoftRasterizer::Tools::calculateViewMatrix(const Eigen::Vector3f &eye,
                                           const Eigen::Vector3f &center,
                                           const Eigen::Vector3f &up) {
  const Eigen::Vector3f up_normalize = up.normalized(); // up normalize
  const Eigen::Vector3f look_at =
      (eye - center).normalized(); // look at normalize
  auto right = look_at.cross(up_normalize).normalized();

  auto _up = up_normalize;
  auto _z = (-look_at);

  // Rotation Matrix On each Axis
  //  (And we could ignore matrix inverse)
  Eigen::Matrix4f R;
  R << right.x(), right.y(), right.z(), 0, _up.x(), _up.y(), _up.z(), 0, _z.x(),
      _z.y(), _z.z(), 0, 0, 0, 0, 1;

  // Transform Matrix
  Eigen::Matrix4f T;
  T << 1, 0, 0, -eye.x(), 0, 1, 0, -eye.y(), 0, 0, 1, -eye.z(), 0, 0, 0, 1;

  return R * T;
}

/**
 * @brief Calculates an orthographic projection matrix.
 *
 * This function computes a 4x4 orthographic projection matrix that maps 3D
 * world space coordinates to 2D normalized device coordinates. The orthographic
 * projection is defined by the left, right, bottom, top, zNear, and zFar
 * parameters.
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
Eigen::Matrix4f SoftRasterizer::Tools::calculateOrthoMatrix(
    float left, float right, float bottom, float top, float zNear, float zFar) {
  const Eigen::Vector3f center((left + right) / 2.0f, (top + bottom) / 2.0f,
                               (zNear + zFar) / 2.0f);

  Eigen::Matrix4f T;
  T << 1, 0, 0, -center.x(), 0, 1, 0, -center.y(), 0, 0, 1, -center.z(), 0, 0,
      0, 1;

  Eigen::Matrix4f S;
  S << 2.0f / (right - left), 0, 0, 0, 0, 2.0f / (top - bottom), 0, 0, 0, 0,
      2.0f / (zNear - zFar), 0, 0, 0, 0, 1;
  return S * T;
}

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
Eigen::Matrix4f SoftRasterizer::Tools::calculateProjectionMatrix(float fovy,
                                                                 float aspect,
                                                                 float zNear,
                                                                 float zFar) {
  Eigen::Matrix4f projectionMatrix = Eigen::Matrix4f::Identity();

  float top = std::tan(fovy / 2.0f) * std::abs(zNear);
  float bottom = -top;

  float right = top * aspect;
  float left = -right;

  Eigen::Matrix4f Ortho =
      calculateOrthoMatrix(left, right, bottom, top, zNear, zFar);

  /*convert projection matrix to ortho matrix*/
  Eigen::Matrix4f Persp2Ortho;
  Persp2Ortho << zNear, 0, 0, 0, 0, zNear, 0, 0, 0, 0, zNear + zFar,
      -zNear * zFar, 0, 0, 1, 0;

  return Ortho * Persp2Ortho;
}


/**
 * @brief Calculates the bounding box for a given triangle.
 *
 * This function determines the axis-aligned bounding box (AABB)
 * that encompasses the given triangle in 2D space. The bounding box
 * is represented as a pair of 2D integer vectors, indicating the
 * minimum and maximum corners of the box.
 *
 * @param triangle The triangle for which the bounding box is to be calculated.
 *                 The triangle is represented using the
 * `SoftRasterizer::Triangle` type.
 *
 * @return A pair of 2D integer vectors (Eigen::Vector2i), where:
 *         - The first vector represents the minimum corner of the bounding box
 * (bottom-left).
 *         - The second vector represents the maximum corner of the bounding box
 * (top-right).
 */
std::pair<Eigen::Vector2i, Eigen::Vector2i>
SoftRasterizer::Tools::calculateBoundingBox(
          const SoftRasterizer::Triangle& triangle) {
          auto A = triangle.a();
          auto B = triangle.b();
          auto C = triangle.c();

          auto min = Eigen::Vector2i{
              static_cast<int>(
                  std::floor(SoftRasterizer::Tools::min(A.x(), B.x(), C.x()))),
              static_cast<int>(
                  std::floor(SoftRasterizer::Tools::min(A.y(), B.y(), C.y()))) };

          auto max = Eigen::Vector2i{
              static_cast<int>(
                  std::ceil(SoftRasterizer::Tools::max(A.x(), B.x(), C.x()))),
              static_cast<int>(
                  std::ceil(SoftRasterizer::Tools::max(A.y(), B.y(), C.y()))) };

          return std::pair<Eigen::Vector2i, Eigen::Vector2i>(min, max);
}