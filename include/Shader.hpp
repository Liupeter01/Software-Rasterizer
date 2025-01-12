#pragma once
#ifndef _SHADER_HPP_
#define _SHADER_HPP_
#include <Eigen/Eigen>
#include <Simd.hpp>
#include <TextureLoader.hpp>
#include <array>
#include <functional>
#include <initializer_list>
#include <memory>

namespace SoftRasterizer {

/*forward declartion*/
class Render;
class Shader;

struct PointSIMD;
struct NormalSIMD;
struct TexCoordSIMD;
struct ColorSIMD;

struct light_struct {
  Eigen::Vector3f position;
  Eigen::Vector3f intensity;
};

struct vertex_displacement {
  vertex_displacement(const Eigen::Vector3f &pos, const Eigen::Vector3f &normal)
      : new_position(pos), new_normal(normal) {}
  Eigen::Vector3f new_position;
  Eigen::Vector3f new_normal;
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
  fragment_shader_payload(const Eigen::Vector3f &_p, const Eigen::Vector3f &_n,
                          const Eigen::Vector2f &_texcoord,
                          const Eigen::Vector3f &_color = Eigen::Vector3f(1.f,
                                                                          1.f,
                                                                          1.f));

  Eigen::Vector3f position = Eigen::Vector3f(0.f, 0.f, 0.f);
  Eigen::Vector3f normal = Eigen::Vector3f(0.f, 0.f, 0.f);
  Eigen::Vector2f texCoords = Eigen::Vector2f(0.f, 0.f);
  Eigen::Vector3f color = Eigen::Vector3f(1.f, 1.f, 1.f);
};

struct Shader {

  using simd_shader = std::function<void(
      const Eigen::Vector3f &, const std::initializer_list<light_struct> &,
      const PointSIMD &, NormalSIMD &, TexCoordSIMD &, ColorSIMD &)>;

  using standard_shader = std::function<Eigen::Vector3f(
      const Eigen::Vector3f &, const std::initializer_list<light_struct> &,
      const fragment_shader_payload &)>;

  static Eigen::Vector3f ka;
  static Eigen::Vector3f ks;
  static float p, kh, kn;

public:
  Shader(const std::string &path);
  Shader(std::shared_ptr<TextureLoader> _loader);

public:
  bool setFragmentShader(SHADERS_TYPE type);

  /*User Vertex Shader*/
  vertex_displacement applyVertexShader(const Eigen::Matrix4f &Model,
                                        const Eigen::Matrix4f &View,
                                        const Eigen::Matrix4f &Projection,
                                        const fragment_shader_payload &payload);

  /*Use Fragment Shader*/
  void applyFragmentShader(const Eigen::Vector3f &camera,
                           const std::initializer_list<light_struct> &lights,
                           const PointSIMD &point, NormalSIMD &normal,
                           TexCoordSIMD &texcoord, ColorSIMD &colour);

  Eigen::Vector3f
  applyFragmentShader(const Eigen::Vector3f &camera,
                      const std::initializer_list<light_struct> &lights,
                      const fragment_shader_payload &payload);

private:
  /*register multiple shader models*/
  void registerShaders();

  void simd_normal_fragment_shader_impl(
      const Eigen::Vector3f &camera,
      const std::initializer_list<light_struct> &lights, const PointSIMD &point,
      NormalSIMD &normal, TexCoordSIMD &texcoord, ColorSIMD &colour);

  void simd_texture_fragment_shader_impl(
      const Eigen::Vector3f &camera,
      const std::initializer_list<light_struct> &lights, const PointSIMD &point,
      NormalSIMD &normal,TexCoordSIMD &texcoord, ColorSIMD &colour);

  void simd_phong_fragment_shader_impl(
      const Eigen::Vector3f &camera,
      const std::initializer_list<light_struct> &lights, const PointSIMD &point,
      NormalSIMD &normal,TexCoordSIMD &texcoord, ColorSIMD &colour);

  void simd_displacement_fragment_shader_impl(
      const Eigen::Vector3f &camera,
      const std::initializer_list<light_struct> &lights, const PointSIMD &point,
      NormalSIMD &normal, TexCoordSIMD &texcoord, ColorSIMD &colour);

  void simd_bump_fragment_shader_impl(
      const Eigen::Vector3f &camera,
      const std::initializer_list<light_struct> &lights, const PointSIMD &point,
      NormalSIMD &normal, TexCoordSIMD &texcoord, ColorSIMD &colour);

  // Static function to compute the Blinn-Phong reflection model
  static Eigen::Vector3f
  BlinnPhong(const Eigen::Vector3f &camera,
             const fragment_shader_payload &shading_point,
             const light_struct &light, const Eigen::Vector3f &ka,
             const Eigen::Vector3f &kd, const Eigen::Vector3f &ks,
             const float p);

  /*Compute Bump Mapping*/
  Eigen::Vector3f calcBumpMapping(const fragment_shader_payload &payload,
                                  const float kh, const float kn);

  /*Compute Displacement Mapping*/
  vertex_displacement
  calcDisplacementMapping(const fragment_shader_payload &payload,
                          const float kh, const float kn);

  /*Visualizing normal directions or checking surface normal directions in some
   * debugging scenarios*/
  Eigen::Vector3f standard_normal_fragment_shader_impl(
      const Eigen::Vector3f &camera,
      const std::initializer_list<light_struct> &lights,
      const fragment_shader_payload &payload);

  Eigen::Vector3f standard_texture_fragment_shader_impl(
      const Eigen::Vector3f &camera,
      const std::initializer_list<light_struct> &lights,
      const fragment_shader_payload &payload);

  Eigen::Vector3f standard_phong_fragment_shader_impl(
      const Eigen::Vector3f &camera,
      const std::initializer_list<light_struct> &lights,
      const fragment_shader_payload &payload);

  Eigen::Vector3f standard_displacement_fragment_shader_impl(
      const Eigen::Vector3f &camera,
      const std::initializer_list<light_struct> &lights,
      const fragment_shader_payload &payload);

  Eigen::Vector3f standard_bump_fragment_shader_impl(
      const Eigen::Vector3f &camera,
      const std::initializer_list<light_struct> &lights,
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

  //change uv coordinates
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
