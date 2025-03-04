#pragma once
#ifndef _SHADER_HPP_
#define _SHADER_HPP_
#include <array>
#include <functional>
#include <hpc/Simd.hpp>
#include <light/Light.hpp>
#include <loader/TextureLoader.hpp>
#include <memory>
#include <tuple>
#include <vector>

namespace SoftRasterizer {

/*forward declartion*/
class Render;
class Shader;
class RayTracing;

struct PointSIMD;
struct NormalSIMD;
struct TexCoordSIMD;
struct ColorSIMD;

struct vertex_displacement {
  vertex_displacement(const glm::vec3 &pos, const glm::vec3 &normal)
      : new_position(pos), new_normal(normal) {}
  glm::vec3 new_position;
  glm::vec3 new_normal;
};

enum class SHADERS_TYPE : std::uint8_t {
  NORMAL = 0,
  TEXTURE,
  PHONG,
  DISPLACEMENT,
  BUMP
};

/*pixel shader*/
struct fragment_shader_payload {
  fragment_shader_payload(const glm::vec3 &_p, const glm::vec3 &_n,
                          const glm::vec2 &_texcoord,
                          const glm::vec3 &_color = glm::vec3(1.f));

  glm::vec3 position = glm::vec3(0.f);
  glm::vec3 normal = glm::vec3(0.f);
  glm::vec2 texCoords = glm::vec2(0.f);
  glm::vec3 color = glm::vec3(1.0f);
};

struct Shader {

  friend class RayTracing;

  using simd_shader = std::function<void(
      const glm::vec3 &, const std::vector<light_struct> &, const PointSIMD &,
      NormalSIMD &, TexCoordSIMD &, ColorSIMD &)>;

  using standard_shader = std::function<glm::vec3(
      const glm::vec3 &, const std::vector<light_struct> &,
      const fragment_shader_payload &)>;

  static glm::vec3 ka;
  static glm::vec3 ks;
  static float p, kh, kn;

public:
  Shader(const std::string &path);
  Shader(std::shared_ptr<TextureLoader> _loader);

public:
  std::shared_ptr<TextureLoader> &getTextureObject() { return texture; }

  bool setFragmentShader(SHADERS_TYPE type);

  /*User Vertex Shader*/
  vertex_displacement applyVertexShader(const glm::mat4 &Model,
                                        const glm::mat4 &View,
                                        const glm::mat4 &Projection,
                                        const fragment_shader_payload &payload);

  /*Use Fragment Shader*/
  void applyFragmentShader(const glm::vec3 &camera,
                           const std::vector<light_struct> &lights,
                           const PointSIMD &point, NormalSIMD &normal,
                           TexCoordSIMD &texcoord, ColorSIMD &colour);

  glm::vec3 applyFragmentShader(const glm::vec3 &camera,
                                const std::vector<light_struct> &lights,
                                const fragment_shader_payload &payload);

  // Static function to compute the Blinn-Phong reflection model
  static glm::vec3 BlinnPhong(const glm::vec3 &camera,
                              const fragment_shader_payload &shading_point,
                              const light_struct &light, const glm::vec3 &ka,
                              const glm::vec3 &kd, const glm::vec3 &ks,
                              const float p);

private:
  /*register multiple shader models*/
  void registerShaders();

  template <typename _simd>
  static std::tuple<_simd, _simd, _simd>
  BlinnPhong(NormalSIMD &normal, const _simd &ka_r, const _simd &ka_g,
             const _simd &ka_b, const _simd &kd_r, const _simd &kd_g,
             const _simd &kd_b, const _simd &ks_r, const _simd &ks_g,
             const _simd &ks_b, const _simd &camera_x, const _simd &camera_y,
             const _simd &camera_z, const _simd &light_pos_x,
             const _simd &light_pos_y, const _simd &light_pos_z,
             const _simd &light_intense_x, const _simd &light_intense_y,
             const _simd &light_intense_z, const _simd &shading_pointx,
             const _simd &shading_pointy, const _simd &shading_pointz,
             const _simd &_p) {

    _simd zero;
    _simd distribution_x, distribution_y, distribution_z;

    if constexpr (std::is_same_v<_simd, __m128>) {
    }
#if defined(__x86_64__) || defined(_WIN64)
    else if constexpr (std::is_same_v<_simd, __m256>) {
      zero = _mm256_setzero_ps();

      NormalSIMD light_dir(_mm256_sub_ps(light_pos_x, shading_pointx),
                           _mm256_sub_ps(light_pos_y, shading_pointy),
                           _mm256_sub_ps(light_pos_z, shading_pointz));

      // sqrt(x^2 + y^2)
      _simd distanceSquared = _mm256_rcp_ps(_mm256_sqrt_ps(_mm256_fmadd_ps(
          light_dir.x, light_dir.x, _mm256_mul_ps(light_dir.y, light_dir.y))));

      distribution_x = _mm256_mul_ps(light_intense_x, distanceSquared);
      distribution_y = _mm256_mul_ps(light_intense_y, distanceSquared);
      distribution_z = _mm256_mul_ps(light_intense_z, distanceSquared);

      // Specular reflection (Blinn-Phong) shadingpoint -> Camera(Your eye)
      // halfway vector: you have to do normalzied!!!!!
      NormalSIMD h =
          NormalSIMD(_mm256_add_ps(light_dir.x,
                                   _mm256_sub_ps(camera_x, shading_pointx)),
                     _mm256_add_ps(light_dir.y,
                                   _mm256_sub_ps(camera_y, shading_pointy)),
                     _mm256_add_ps(light_dir.z,
                                   _mm256_sub_ps(camera_z, shading_pointz)))
              .normalized();

#elif defined(__arm__) || defined(__aarch64__)
    else if constexpr (std::is_same_v<_simd, simde__m256>) {
      zero = simde_mm256_setzero_ps();

      NormalSIMD light_dir(simde_mm256_sub_ps(light_pos_x, shading_pointx),
                           simde_mm256_sub_ps(light_pos_y, shading_pointy),
                           simde_mm256_sub_ps(light_pos_z, shading_pointz));

      // sqrt(x^2 + y^2)
      _simd distanceSquared = simde_mm256_rcp_ps(simde_mm256_sqrt_ps(
          simde_mm256_fmadd_ps(light_dir.x, light_dir.x,
                               simde_mm256_mul_ps(light_dir.y, light_dir.y))));

      distribution_x = simde_mm256_mul_ps(light_intense_x, distanceSquared);
      distribution_y = simde_mm256_mul_ps(light_intense_y, distanceSquared);
      distribution_z = simde_mm256_mul_ps(light_intense_z, distanceSquared);

      // Specular reflection (Blinn-Phong) shadingpoint -> Camera(Your eye)
      // halfway vector: you have to do normalzied!!!!!
      NormalSIMD h =
          NormalSIMD(
              simde_mm256_add_ps(light_dir.x,
                                 simde_mm256_sub_ps(camera_x, shading_pointx)),
              simde_mm256_add_ps(light_dir.y,
                                 simde_mm256_sub_ps(camera_y, shading_pointy)),
              simde_mm256_add_ps(light_dir.z,
                                 simde_mm256_sub_ps(camera_z, shading_pointz)))
              .normalized();

#else
#endif

      // you have to do normalzied!!!!!
      NormalSIMD light_dir_normalized = light_dir.normalized();

#if defined(__x86_64__) || defined(_WIN64)
      // Diffuse reflection (Lambertian reflectance)  dot(light, normal) = x * x
      // + y * y + z * z
      _simd cosAlpha = _mm256_max_ps(
          zero,
          _mm256_fmadd_ps(light_dir_normalized.x, normal.x,
                          _mm256_fmadd_ps(light_dir_normalized.y, normal.y,
                                          _mm256_mul_ps(light_dir_normalized.z,
                                                        normal.z))));

      // std::pow(cosTheta, p);
      _simd cosTheta = _mm256_pow_ps(
          _mm256_max_ps(
              zero,
              _mm256_fmadd_ps(h.x, normal.x,
                              _mm256_fmadd_ps(h.y, normal.y,
                                              _mm256_mul_ps(h.z, normal.z)))),
          _p);

      // Combine all lighting components (La + Ld + Ls) * Kd

      _simd kd_dist_x = _mm256_mul_ps(distribution_x, kd_r);
      _simd ks_dist_x = _mm256_mul_ps(distribution_x, ks_r);
      _simd kd_dist_y = _mm256_mul_ps(distribution_y, kd_g);
      _simd ks_dist_y = _mm256_mul_ps(distribution_y, ks_g);
      _simd kd_dist_z = _mm256_mul_ps(distribution_z, kd_b);
      _simd ks_dist_z = _mm256_mul_ps(distribution_z, ks_b);

      return {
          _mm256_mul_ps(
              kd_r, _mm256_fmadd_ps(
                        ka_r, light_intense_x,
                        _mm256_fmadd_ps(kd_dist_x, cosAlpha,
                                        _mm256_mul_ps(ks_dist_x, cosTheta)))),

          _mm256_mul_ps(
              kd_g, _mm256_fmadd_ps(
                        ka_g, light_intense_y,
                        _mm256_fmadd_ps(kd_dist_y, cosAlpha,
                                        _mm256_mul_ps(ks_dist_y, cosTheta)))),

          _mm256_mul_ps(
              kd_b, _mm256_fmadd_ps(
                        ka_b, light_intense_z,
                        _mm256_fmadd_ps(kd_dist_z, cosAlpha,
                                        _mm256_mul_ps(ks_dist_z, cosTheta))))};

#elif defined(__arm__) || defined(__aarch64__)
      // Diffuse reflection (Lambertian reflectance)  dot(light, normal) = x * x
      // + y * y + z * z

      _simd cosAlpha = simde_mm256_max_ps(
          zero, simde_mm256_fmadd_ps(
                    light_dir_normalized.x, normal.x,
                    simde_mm256_fmadd_ps(
                        light_dir_normalized.y, normal.y,
                        simde_mm256_mul_ps(light_dir_normalized.z, normal.z))));

      // std::pow(cosTheta, p);
      _simd cosTheta = simde_mm256_pow_ps(
          simde_mm256_max_ps(
              zero, simde_mm256_fmadd_ps(
                        h.x, normal.x,
                        simde_mm256_fmadd_ps(
                            h.y, normal.y, simde_mm256_mul_ps(h.z, normal.z)))),
          _p);

      // Combine all lighting components (La + Ld + Ls) * Kd
      _simd kd_dist_x = simde_mm256_mul_ps(distribution_x, kd_r);
      _simd ks_dist_x = simde_mm256_mul_ps(distribution_x, ks_r);
      _simd kd_dist_y = simde_mm256_mul_ps(distribution_y, kd_g);
      _simd ks_dist_y = simde_mm256_mul_ps(distribution_y, ks_g);
      _simd kd_dist_z = simde_mm256_mul_ps(distribution_z, kd_b);
      _simd ks_dist_z = simde_mm256_mul_ps(distribution_z, ks_b);

      auto Constant_x = simde_mm256_mul_ps(ka_r, light_intense_x);
      auto Theta_x = simde_mm256_mul_ps(ks_dist_x, cosTheta);
      auto Alpha_x = simde_mm256_mul_ps(kd_dist_x, cosAlpha);

      auto Constant_y = simde_mm256_mul_ps(ka_g, light_intense_y);
      auto Theta_y = simde_mm256_mul_ps(ks_dist_y, cosTheta);
      auto Alpha_y = simde_mm256_mul_ps(kd_dist_y, cosAlpha);

      auto Constant_z = simde_mm256_mul_ps(ka_b, light_intense_z);
      auto Theta_z = simde_mm256_mul_ps(ks_dist_z, cosTheta);
      auto Alpha_z = simde_mm256_mul_ps(kd_dist_z, cosAlpha);

      return {simde_mm256_mul_ps(
                  kd_r, simde_mm256_add_ps(
                            Constant_x, simde_mm256_add_ps(Theta_x, Alpha_x))),

              simde_mm256_mul_ps(
                  kd_g, simde_mm256_add_ps(
                            Constant_y, simde_mm256_add_ps(Theta_y, Alpha_y))),

              simde_mm256_mul_ps(
                  kd_b, simde_mm256_add_ps(
                            Constant_z, simde_mm256_add_ps(Theta_z, Alpha_z)))};
#else
#endif
    }
  }

  void simd_normal_fragment_shader_impl(const glm::vec3 &camera,
                                        const std::vector<light_struct> &lights,
                                        const PointSIMD &point,
                                        NormalSIMD &normal,
                                        TexCoordSIMD &texcoord,
                                        ColorSIMD &colour);

  void
  simd_texture_fragment_shader_impl(const glm::vec3 &camera,
                                    const std::vector<light_struct> &lights,
                                    const PointSIMD &point, NormalSIMD &normal,
                                    TexCoordSIMD &texcoord, ColorSIMD &colour);

  void simd_phong_fragment_shader_impl(const glm::vec3 &camera,
                                       const std::vector<light_struct> &lights,
                                       const PointSIMD &point,
                                       NormalSIMD &normal,
                                       TexCoordSIMD &texcoord,
                                       ColorSIMD &colour);

  void simd_displacement_fragment_shader_impl(
      const glm::vec3 &camera, const std::vector<light_struct> &lights,
      const PointSIMD &point, NormalSIMD &normal, TexCoordSIMD &texcoord,
      ColorSIMD &colour);

  void simd_bump_fragment_shader_impl(const glm::vec3 &camera,
                                      const std::vector<light_struct> &lights,
                                      const PointSIMD &point,
                                      NormalSIMD &normal,
                                      TexCoordSIMD &texcoord,
                                      ColorSIMD &colour);

  /*Compute Bump Mapping*/
  glm::vec3 calcBumpMapping(const fragment_shader_payload &payload,
                            const float kh, const float kn);

  /*Compute Displacement Mapping*/
  vertex_displacement
  calcDisplacementMapping(const fragment_shader_payload &payload,
                          const float kh, const float kn);

  /*Visualizing normal directions or checking surface normal directions in some
   * debugging scenarios*/
  glm::vec3
  standard_normal_fragment_shader_impl(const glm::vec3 &camera,
                                       const std::vector<light_struct> &lights,
                                       const fragment_shader_payload &payload);

  glm::vec3
  standard_texture_fragment_shader_impl(const glm::vec3 &camera,
                                        const std::vector<light_struct> &lights,
                                        const fragment_shader_payload &payload);

  glm::vec3
  standard_phong_fragment_shader_impl(const glm::vec3 &camera,
                                      const std::vector<light_struct> &lights,
                                      const fragment_shader_payload &payload);

  glm::vec3 standard_displacement_fragment_shader_impl(
      const glm::vec3 &camera, const std::vector<light_struct> &lights,
      const fragment_shader_payload &payload);

  glm::vec3
  standard_bump_fragment_shader_impl(const glm::vec3 &camera,
                                     const std::vector<light_struct> &lights,
                                     const fragment_shader_payload &payload);

public:
  const __m128 zero_128 = _mm_set1_ps(0.0f);
  const __m128 one_128 = _mm_set1_ps(1.0f);
  const __m128 two_128 = _mm_set1_ps(2.0f);
  const __m128 point_five_128 = _mm_set1_ps(0.5f);
  __m128 width_128, height_128;

  // Preparing constants for transformation
#if defined(__x86_64__) || defined(_WIN64)
  const __m256 zero = _mm256_set1_ps(0.0f);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 two = _mm256_set1_ps(2.0f);
  const __m256 point_five = _mm256_set1_ps(0.5f);

  // change uv coordinates
  __m256 width_256, height_256;

#elif defined(__arm__) || defined(__aarch64__)
  const simde__m256 zero = simde_mm256_set1_ps(0.0f);
  const simde__m256 one = simde_mm256_set1_ps(1.0f);
  const simde__m256 two = simde_mm256_set1_ps(2.0f);
  const simde__m256 point_five = simde_mm256_set1_ps(0.5f);

  simde__m256 width_256, height_256;

#else
#endif

private:
  /*
   *   All Shader Type That You Could Choose
   *       NORMAL shader_type normal_fragment_shader;
   *       TEXTURE shader_type texture_fragment_shader;
   *       PHONG shader_type phong_fragment_shader;
   *       DISPLACEMENT shader_type displacement_fragment_shader;
   *       BUMP shader_type bump_fragment_shader;
   */
  std::array<standard_shader, 5> standard_shaders;
  std::array<simd_shader, 5> simd_shaders;

  /*activitied shading method*/
  standard_shader m_standard;
  simd_shader m_simd;

  /*texture loader*/
  std::shared_ptr<TextureLoader> texture;
};
} // namespace SoftRasterizer

#endif //_SHADER_HPP_
