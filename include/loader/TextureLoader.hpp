#pragma once
#ifndef _TEXTURELOADER_HPP_
#define _TEXTURELOADER_HPP_
#include <Tools.hpp>
#include <algorithm> // Required for std::max and std::min
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

namespace SoftRasterizer {
class Shader;

class TextureLoader {
  friend class Shader;

public:
  TextureLoader(const std::string &path);
  virtual ~TextureLoader();

public:
  glm::vec3 getTextureColor(const glm::vec2 &uv);

  template <typename _simd>
  inline std::tuple<_simd, _simd, _simd> getTextureColor(const _simd &u,
                                                         const _simd &v) {
    constexpr int num_elements = Tools::num_elements_in_simd<_simd>();

    /*Read Value From __m256*/
    std::vector<int> x(num_elements);
    std::vector<int> y(num_elements);

    // Allocate memory for a block of 8 pixels (assuming 3 channels: R, G, B)
    std::vector<float> R_Block(num_elements);
    std::vector<float> G_Block(num_elements);
    std::vector<float> B_Block(num_elements);

    if constexpr (std::is_same_v<_simd, __m128>) {
      __m128i u_coord = _mm_cvtps_epi32(u);
      __m128i v_coord = _mm_cvtps_epi32(v);

      Tools::static_for<0, num_elements>(
          [this, &u_coord, &v_coord, &R_Block, &G_Block, &B_Block](auto i) {
            cv::Vec3b color =
                m_texture.at<cv::Vec3b>(_mm_extract_epi32(v_coord, i.value),
                                        _mm_extract_epi32(u_coord, i.value));
            R_Block[i.value] = static_cast<float>(color.val[0]); // Red;
            G_Block[i.value] = static_cast<float>(color.val[1]); // Green
            B_Block[i.value] = static_cast<float>(color.val[2]); // Blue
          });
    }
#if defined(__x86_64__) || defined(_WIN64)
    else if constexpr (std::is_same_v<_simd, __m256>) {
      __m256i u_coord = _mm256_cvtps_epi32(u);
      __m256i v_coord = _mm256_cvtps_epi32(v);

#elif defined(__arm__) || defined(__aarch64__)
    else if constexpr (std::is_same_v<_simd, simde__m256>) {
      simde__m256i u_coord = simde_mm256_cvtps_epi32(u);
      simde__m256i v_coord = simde_mm256_cvtps_epi32(v);
#else
#endif

      Tools::static_for<0, num_elements>(
          [this, &u_coord, &v_coord, &R_Block, &G_Block, &B_Block](auto i) {
#if defined(__x86_64__) || defined(_WIN64)
            cv::Vec3b color =
                m_texture.at<cv::Vec3b>(_mm256_extract_epi32(v_coord, i.value),
                                        _mm256_extract_epi32(u_coord, i.value));
            R_Block[i.value] = static_cast<float>(color.val[0]); // Red;
            G_Block[i.value] = static_cast<float>(color.val[1]); // Green
            B_Block[i.value] = static_cast<float>(color.val[2]); // Blue

#elif defined(__arm__) || defined(__aarch64__)
            cv::Vec3b color = m_texture.at<cv::Vec3b>(
                simde_mm256_extract_epi32(v_coord, i.value),
                simde_mm256_extract_epi32(u_coord, i.value));
            R_Block[i.value] = static_cast<float>(color.val[0]); // Red;
            G_Block[i.value] = static_cast<float>(color.val[1]); // Green
            B_Block[i.value] = static_cast<float>(color.val[2]); // Blue
#else
#endif
          });
    }

    _simd inverse;

    if constexpr (std::is_same_v<_simd, __m128>) {
      inverse = _mm_rcp_ps(_mm_set1_ps(255.0f));
      return {_mm_mul_ps(_mm_loadu_ps(R_Block.data()), inverse),
              _mm_mul_ps(_mm_loadu_ps(G_Block.data()), inverse),
              _mm_mul_ps(_mm_loadu_ps(B_Block.data()), inverse)};
    }

#if defined(__x86_64__) || defined(_WIN64)
    else if constexpr (std::is_same_v<_simd, __m256>) {
      inverse = _mm256_rcp_ps(_mm256_set1_ps(255.0f));
      return {_mm256_mul_ps(_mm256_loadu_ps(R_Block.data()), inverse),
              _mm256_mul_ps(_mm256_loadu_ps(G_Block.data()), inverse),
              _mm256_mul_ps(_mm256_loadu_ps(B_Block.data()), inverse)};

#elif defined(__arm__) || defined(__aarch64__)
    else if constexpr (std::is_same_v<_simd, simde__m256>) {
      inverse = simde_mm256_rcp_ps(simde_mm256_set1_ps(255.0f));
      return {
          simde_mm256_mul_ps(simde_mm256_loadu_ps(R_Block.data()), inverse),
          simde_mm256_mul_ps(simde_mm256_loadu_ps(G_Block.data()), inverse),
          simde_mm256_mul_ps(simde_mm256_loadu_ps(B_Block.data()), inverse)};
#else
#endif
    }

    /*Error Occurs*/
    return {};
  }

private:
  cv::Mat m_texture;
  std::string m_path;
  std::size_t m_width;
  std::size_t m_height;
};
} // namespace SoftRasterizer

#endif //_TEXTURELOADER_HPP_
