#include <random>
#include <Tools.hpp>
#include <spdlog/spdlog.h>
#include <object/Triangle.hpp>

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

glm::vec3 SoftRasterizer::Tools::to_vec3(const glm::vec4 &vec) {
  return glm::vec3(vec.x / vec.w, vec.y / vec.w, vec.z / vec.w);
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
glm::uvec3 SoftRasterizer::Tools::normalizedToRGB(float red, float green,
                                                  float blue) {

  // Clamp values to ensure they're within the range [0, 1]
  red = std::clamp(red, 0.0f, 1.0f);
  green = std::clamp(green, 0.0f, 1.0f);
  blue = std::clamp(blue, 0.0f, 1.0f);

  // Scale to the range [0, 255] and convert to integers
  return glm::uvec3(red * 255.0f, green * 255.0f, blue * 255.0f);
}

glm::uvec3 SoftRasterizer::Tools::normalizedToRGB(const glm::vec3 &color) {
  return normalizedToRGB(color.x, color.y, color.z);
}

glm::vec3 SoftRasterizer::Tools::interpolateNormal(float alpha, float beta,
                                                   float gamma,
                                                   const glm::vec3 &normal1,
                                                   const glm::vec3 &normal2,
                                                   const glm::vec3 &normal3) {
  return glm::normalize(alpha * normal1 + beta * normal2 + gamma * normal3);
}

#if defined(__x86_64__) || defined(_WIN64)
SoftRasterizer::NormalSIMD SoftRasterizer::Tools::interpolateNormal(
    const __m256 &alpha, const __m256 &beta, const __m256 &gamma,
    const glm::vec3 &normal1, const glm::vec3 &normal2,
    const glm::vec3 &normal3) {

  auto x1 = _mm256_set1_ps(normal1.x);
  auto x2 = _mm256_set1_ps(normal2.x);
  auto x3 = _mm256_set1_ps(normal3.x);
  auto y1 = _mm256_set1_ps(normal1.y);
  auto y2 = _mm256_set1_ps(normal2.y);
  auto y3 = _mm256_set1_ps(normal3.y);
  auto z1 = _mm256_set1_ps(normal1.z);
  auto z2 = _mm256_set1_ps(normal2.z);
  auto z3 = _mm256_set1_ps(normal3.z);

  return NormalSIMD(_mm256_fmadd_ps(
                        alpha, x1,
                        _mm256_fmadd_ps(beta, x2, _mm256_mul_ps(gamma, x3))),
                    _mm256_fmadd_ps(
                        alpha, y1,
                        _mm256_fmadd_ps(beta, y2, _mm256_mul_ps(gamma, y3))),
                    _mm256_fmadd_ps(
                        alpha, z1,
                        _mm256_fmadd_ps(beta, z2, _mm256_mul_ps(gamma, z3))))
      .normalized();
}

SoftRasterizer::TexCoordSIMD SoftRasterizer::Tools::interpolateTexCoord(
    const __m256 &alpha, const __m256 &beta, const __m256 &gamma,
    const glm::vec2 &textCoord1, const glm::vec2 &textCoord2,
    const glm::vec2 &textCoord3) {
  TexCoordSIMD result;

  auto x1 = _mm256_set1_ps(textCoord1.x);
  auto x2 = _mm256_set1_ps(textCoord2.x);
  auto x3 = _mm256_set1_ps(textCoord3.x);

  auto y1 = _mm256_set1_ps(textCoord1.y);
  auto y2 = _mm256_set1_ps(textCoord2.y);
  auto y3 = _mm256_set1_ps(textCoord3.y);

  result.u = _mm256_fmadd_ps(
      alpha, x1, _mm256_fmadd_ps(beta, x2, _mm256_mul_ps(gamma, x3)));

  result.v = _mm256_fmadd_ps(
      alpha, y1, _mm256_fmadd_ps(beta, y2, _mm256_mul_ps(gamma, y3)));

  // Return as a struct containing both components (x and y)
  return result;
}

#elif defined(__arm__) || defined(__aarch64__)
SoftRasterizer::NormalSIMD SoftRasterizer::Tools::interpolateNormal(
    const simde__m256 &alpha, const simde__m256 &beta, const simde__m256 &gamma,
    const glm::vec3 &normal1, const glm::vec3 &normal2,
    const glm::vec3 &normal3) {

  auto x1 = simde_mm256_set1_ps(normal1.x);
  auto x2 = simde_mm256_set1_ps(normal2.x);
  auto x3 = simde_mm256_set1_ps(normal3.x);
  auto y1 = simde_mm256_set1_ps(normal1.y);
  auto y2 = simde_mm256_set1_ps(normal2.y);
  auto y3 = simde_mm256_set1_ps(normal3.y);
  auto z1 = simde_mm256_set1_ps(normal1.z);
  auto z2 = simde_mm256_set1_ps(normal2.z);
  auto z3 = simde_mm256_set1_ps(normal3.z);

  return NormalSIMD(
             simde_mm256_fmadd_ps(
                 alpha, x1,
                 simde_mm256_fmadd_ps(beta, x2, simde_mm256_mul_ps(gamma, x3))),
             simde_mm256_fmadd_ps(
                 alpha, y1,
                 simde_mm256_fmadd_ps(beta, y2, simde_mm256_mul_ps(gamma, y3))),
             simde_mm256_fmadd_ps(
                 alpha, z1,
                 simde_mm256_fmadd_ps(beta, z2, simde_mm256_mul_ps(gamma, z3))))
      .normalized();
}

SoftRasterizer::TexCoordSIMD SoftRasterizer::Tools::interpolateTexCoord(
    const simde__m256 &alpha, const simde__m256 &beta, const simde__m256 &gamma,
    const glm::vec2 &textCoord1, const glm::vec2 &textCoord2,
    const glm::vec2 &textCoord3) {

  // Return as a struct containing both components (x and y)
  TexCoordSIMD result;

  auto x1 = simde_mm256_set1_ps(textCoord1.x);
  auto x2 = simde_mm256_set1_ps(textCoord2.x);
  auto x3 = simde_mm256_set1_ps(textCoord3.x);

  auto y1 = simde_mm256_set1_ps(textCoord1.y);
  auto y2 = simde_mm256_set1_ps(textCoord2.y);
  auto y3 = simde_mm256_set1_ps(textCoord3.y);

  result.u = simde_mm256_fmadd_ps(
      alpha, x1, simde_mm256_fmadd_ps(beta, x2, simde_mm256_mul_ps(gamma, x3)));

  result.v = simde_mm256_fmadd_ps(
      alpha, y1, simde_mm256_fmadd_ps(beta, y2, simde_mm256_mul_ps(gamma, y3)));

  // Return as a struct containing both components (x and y)
  return result;
}

#else
#endif

glm::vec2 SoftRasterizer::Tools::interpolateTexCoord(
    float alpha, float beta, float gamma, const glm::vec2 &textCoord1,
    const glm::vec2 &textCoord2, const glm::vec2 &textCoord3) {
  return alpha * textCoord1 + beta * textCoord2 + gamma * textCoord3;
}

glm::vec3 SoftRasterizer::Tools::calculateNormalWithWeight(
    const glm::vec3 &pa, const glm::vec3 &pb, const glm::vec3 &pc) {
  const glm::vec3 AB = pb - pa;
  const glm::vec3 AC = pc - pa;
  glm::vec3 normal = glm::cross(AB, AC); // calculate normal vector

  const float length = glm::length(normal); // length of normal vector
  const float arc_sin_degree = length / (glm::length(AB) * glm::length(AC));

  /*length must not equal to zero*/
  if (!(-(1e-8) <= length && length <= 1e-8)) {
    normal = normal * (glm::asin(arc_sin_degree) / length);
  }
  return glm::normalize(normal);
}

float SoftRasterizer::Tools::fresnel(const glm::vec3 &rayDirection,
                                     const glm::vec3 &normal,
                                     const float &refractiveIndex) {

  // Compute the cosine of the angle between the incident ray and the normal
  float cosi = std::clamp(glm::dot(rayDirection, normal), -1.0f, 1.0f);
  float etai = 1, etat = refractiveIndex;
  if (cosi > 0) {
    std::swap(etai, etat);
  }
  // Compute sini using Snell's law
  float sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi));
  // Total internal reflection
  if (sint >= 1.0f) {
    return 1.0f;
  }
  float cost = sqrtf(std::max(0.f, 1 - sint * sint));
  cosi = fabsf(cosi);
  float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
  float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));

  return (Rs * Rs + Rp * Rp) / 2.f;
}

const float SoftRasterizer::Tools::random_generator() {
          /*Random Generator*/
          std::random_device rnd;
          std::mt19937 mt(rnd());
          std::uniform_real_distribution<float> unf(0.f, 1.0f);
          return unf(mt);
}