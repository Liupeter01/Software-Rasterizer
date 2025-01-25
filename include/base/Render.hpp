#pragma once
#include "oneapi/tbb/partitioner.h"
#ifndef _RENDER_HPP_
#define _RENDER_HPP_
#include <algorithm>
#include <hpc/Simd.hpp>
#include <loader/ObjLoader.hpp>
#include <object/Triangle.hpp>
#include <optional>
#include <scene/Scene.hpp>
#include <shader/Shader.hpp>
#include <tbb/parallel_for.h>
#include <tuple>
#include <unordered_map>

/*Use for unrolling calculation*/
#define ROUND_UP_TO_MULTIPLE_OF_4(x) (((x) + 3) & ~3)
#define ROUND_UP_TO_MULTIPLE_OF_8(x) (((x) + 7) & ~7)

#if defined(__x86_64__) || defined(_WIN64)
#define PREFETCH(address)                                                      \
  _mm_prefetch(reinterpret_cast<const char *>(address), _MM_HINT_T0)
// Macro to define the CPUID call and store results
#define GET_CPUID(info, function) __cpuid((info), (function))

// Macro to extract EBX from CPUID result
#define EBX_FROM_CPUID(info) (info[1])

// Macro to extract cache line size from EBX[15:8] and convert to bytes
#define CACHE_LINE_SIZE(ebx) ((((ebx) >> 8) & 0xFF) * 8)

// Macro to retrieve cache line size using CPUID function 1
#define GET_CACHE_LINE_SIZE()                                                  \
  ([]() -> unsigned {                                                          \
    int cpu_info[4] = {0};                                                     \
    GET_CPUID(cpu_info, 1);                                                    \
    unsigned ebx = EBX_FROM_CPUID(cpu_info);                                   \
    return CACHE_LINE_SIZE(ebx);                                               \
  }())

#elif defined(__arm__) || defined(__aarch64__)
#define PREFETCH(address)                                                      \
  __builtin_prefetch(reinterpret_cast<const char *>(address), 0, 1)

#define GET_CACHE_LINE_SIZE()                                                  \
  ([]() -> unsigned {                                                          \
    FILE *fp =                                                                 \
        popen("sysctl -a | grep 'cachelinesize' | awk '{print $2}'", "r");     \
    unsigned size = 0;                                                         \
    if (fp) {                                                                  \
      fscanf(fp, "%u", &size);                                                 \
      pclose(fp);                                                              \
    } else {                                                                   \
      std::cerr << "Error: Failed to run sysctl command\n";                    \
    }                                                                          \
    return size;                                                               \
  }())

#else
#define PREFETCH(address) // Prefetch not supported, fallback to no-op
#endif

namespace SoftRasterizer {
enum class Buffers { Color = 1, Depth = 2 };

inline Buffers operator|(Buffers a, Buffers b) {
  return Buffers((int)a | (int)b);
}

inline Buffers operator&(Buffers a, Buffers b) {
  return Buffers((int)a & (int)b);
}

enum class Primitive { LINES, TRIANGLES };

class RenderingPipeline {
public:
  RenderingPipeline();
  RenderingPipeline(const std::size_t width, const std::size_t height);
  virtual ~RenderingPipeline();

protected:
  /*draw graphics*/
  virtual void draw(Primitive type) = 0;
  void clearFrameBuffer();
  void clearZDepth();

public:
  void clear(SoftRasterizer::Buffers flags);

  /*display*/
  void display(Primitive type);
  bool addScene(std::shared_ptr<Scene> scene,
                std::optional<std::string> name = std::nullopt);

protected:
  template <typename _simd>
  inline void writePixel(const long long start_pos, const _simd &r,
                         const _simd &g, const _simd &b) {
#if defined(__x86_64__) || defined(_WIN64)
    if constexpr (std::is_same_v<_simd, __m256>) {
      _mm256_storeu_ps(m_channels[0].ptr<float>(0) + start_pos, r); // R
      _mm256_storeu_ps(m_channels[1].ptr<float>(0) + start_pos, g); // G
      _mm256_storeu_ps(m_channels[2].ptr<float>(0) + start_pos, b); // B

#elif defined(__arm__) || defined(__aarch64__)
    if constexpr (std::is_same_v<_simd, simde__m256>) {
      simde_mm256_storeu_ps(m_channels[0].ptr<float>(0) + start_pos, r); // R
      simde_mm256_storeu_ps(m_channels[1].ptr<float>(0) + start_pos, g); // G
      simde_mm256_storeu_ps(m_channels[2].ptr<float>(0) + start_pos, b); // B

#else
#endif
    } else if constexpr (std::is_same_v<_simd, __m128>) {
      _mm_storeu_ps(m_channels[0].ptr<float>(0) + start_pos, r); // R
      _mm_storeu_ps(m_channels[1].ptr<float>(0) + start_pos, g); // G
      _mm_storeu_ps(m_channels[2].ptr<float>(0) + start_pos, b); // B
    }
  }

inline void writePixel(
          const long long x, const long long y, const glm::vec3& color) {
          if (x >= 0 && x < m_width && y >= 0 && y < m_height) {
                    auto pos = x + y * m_width;

                    *(m_channels[0].ptr<float>(0) + pos) = color.x; // R
                    *(m_channels[1].ptr<float>(0) + pos) = color.y; // G
                    *(m_channels[2].ptr<float>(0) + pos) = color.z; // B
          }
}

inline void writePixel(
          const long long x, const long long y, const glm::uvec3& color) {
          writePixel(x, y, glm::vec3(color.x, color.y, color.z));
}

  inline void writePixel(const long long start_pos, const ColorSIMD &color) {
    writePixel(start_pos, color.r, color.g, color.b);
  }

  template <typename _simd>
  inline void writeZBuffer(const long long start_pos, const _simd &depth) {
#if defined(__x86_64__) || defined(_WIN64)
    if constexpr (std::is_same_v<_simd, __m256>) {
      _mm256_storeu_ps(reinterpret_cast<float *>(&m_zBuffer[start_pos]), depth);

#elif defined(__arm__) || defined(__aarch64__)
    if constexpr (std::is_same_v<_simd, simde__m256>) {
      simde_mm256_storeu_ps(reinterpret_cast<float *>(&m_zBuffer[start_pos]),
                            depth);
#else
#endif
    } else if constexpr (std::is_same_v<_simd, __m128>) {
      _mm_storeu_ps(reinterpret_cast<float *>(&m_zBuffer[start_pos]), depth);
    }
  }

  inline bool writeZBuffer(const long long x, const long long y,
                           const float depth);
  inline void writeZBuffer(const long long start_pos, const float depth);

  template <typename _simd>
  inline _simd readZBuffer(const long long start_pos) {
#if defined(__x86_64__) || defined(_WIN64)
    if constexpr (std::is_same_v<_simd, __m256>) {
      return _mm256_loadu_ps(reinterpret_cast<float *>(&m_zBuffer[start_pos]));

#elif defined(__arm__) || defined(__aarch64__)
    if constexpr (std::is_same_v<_simd, simde__m256>) {
      return simde_mm256_loadu_ps(
          reinterpret_cast<float *>(&m_zBuffer[start_pos]));
#else
#endif
    } else if constexpr (std::is_same_v<_simd, __m128>) {
      return _mm_loadu_ps(reinterpret_cast<float *>(&m_zBuffer[start_pos]));
    }
    return {};
  }

  inline const float readZBuffer(const long long x, const long long y);

  template <typename _simd>
  inline std::tuple<_simd, _simd, _simd> readPixel(const long long start_pos) {

#if defined(__x86_64__) || defined(_WIN64)
    if constexpr (std::is_same_v<_simd, __m256>) {
      return {
          _mm256_loadu_ps(m_channels[0].ptr<float>(0) + start_pos), // R
          _mm256_loadu_ps(m_channels[1].ptr<float>(0) + start_pos), // G
          _mm256_loadu_ps(m_channels[2].ptr<float>(0) + start_pos)  // B
      };

#elif defined(__arm__) || defined(__aarch64__)
    if constexpr (std::is_same_v<_simd, simde__m256>) {
      return {
          simde_mm256_loadu_ps(m_channels[0].ptr<float>(0) + start_pos), // R
          simde_mm256_loadu_ps(m_channels[1].ptr<float>(0) + start_pos), // G
          simde_mm256_loadu_ps(m_channels[2].ptr<float>(0) + start_pos)  // B
      };
#else
#endif
    } else if constexpr (std::is_same_v<_simd, __m128>) {
      return {
          _mm_loadu_ps(m_channels[0].ptr<float>(0) + start_pos), // R
          _mm_loadu_ps(m_channels[1].ptr<float>(0) + start_pos), // G
          _mm_loadu_ps(m_channels[2].ptr<float>(0) + start_pos)  // B
      };
    }
    return {};
  }

  /*Bresenham algorithm*/
  void drawLine(const glm::vec3 &p0, const glm::vec3 &p1,
                const glm::uvec3 &color);

protected:
  /*SIMD Support*/
  constexpr static std::size_t AVX512 = 16;
  constexpr static std::size_t AVX2 = 8;
  constexpr static std::size_t SSE = 4;
  constexpr static std::size_t SCALAR = 1;

  /*display resolution*/
  std::size_t m_width;
  std::size_t m_height;

#if defined(__x86_64__) || defined(_WIN64)
  const __m256 zero = _mm256_set1_ps(0.0f);
  const __m256 one = _mm256_set1_ps(1.0f);

  /*decribe inf distance in z buffer*/
  const __m256 inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());

#elif defined(__arm__) || defined(__aarch64__)
  const simde__m256 zero = simde_mm256_set1_ps(0.0f);
  const simde__m256 one = simde_mm256_set1_ps(1.0f);

  /*decribe inf distance in z buffer*/
  const simde__m256 inf =
      simde_mm256_set1_ps(std::numeric_limits<float>::infinity());

#else
#endif

  /*Scene Data*/
  std::unordered_map<std::string, std::shared_ptr<Scene>> m_scenes;

  /*RGB(3 channels)*/
  constexpr static std::size_t numbers = 3;
  std::vector<cv::Mat> m_channels;

  /*to store final frame*/
  cv::Mat m_frameBuffer;

  /*z buffer*/
  // SpinLock m_zBufferLock;
  alignas(64) std::vector<float> m_zBuffer;

  oneapi::tbb::affinity_partitioner ap;
};
} // namespace SoftRasterizer

#endif //_RENDER_HPP_
