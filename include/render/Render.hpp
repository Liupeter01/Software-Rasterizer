#pragma once
#ifndef _RENDER_HPP_
#define _RENDER_HPP_
#include <object/Triangle.hpp>
#include <algorithm>
#include <hpc/Simd.hpp>
#include <loader/ObjLoader.hpp>
#include <optional>
#include <scene/Scene.hpp>
#include <service/LockFree.hpp>
#include <shader/Shader.hpp>
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
  void draw(Primitive type);
  void clearFrameBuffer();
  void clearZDepth();

public:
  void clear(SoftRasterizer::Buffers flags);

  /*display*/
  void display(Primitive type);
  bool addScene(std::shared_ptr<Scene> scene,
                std::optional<std::string> name = std::nullopt);

private:
  /*Only Draw Line*/
  void rasterizeWireframe(const SoftRasterizer::Triangle &triangle);

  inline static bool insideTriangle(const std::size_t x_pos,
                                    const std::size_t y_pos,
                                    const SoftRasterizer::Triangle &triangle);

  static inline std::tuple<float, float, float>
  barycentric(const std::size_t x_pos, const std::size_t y_pos,
              const SoftRasterizer::Triangle &triangle);

  /**
   * @brief Calculates the barycentric coordinates (alpha, beta, gamma) for a
   * given point (x_pos, y_pos) with respect to a triangle. Also checks if the
   * point is inside the triangle using the `insideTriangle` function and
   * applies the result as a mask to ensure the coordinates are only valid for
   * points inside the triangle.
   *
   * @param x_pos SIMD register containing x positions of points.
   * @param y_pos SIMD register containing y positions of points.
   * @param triangle The triangle whose barycentric coordinates are to be
   * calculated.
   * @return A tuple of three simde__m256 values representing the barycentric
   * coordinates (alpha, beta, gamma) for the point (x_pos, y_pos). The
   * coordinates are zeroed out for points outside the triangle using a mask.
   */
#if defined(__x86_64__) || defined(_WIN64)
  static inline std::tuple<__m256, __m256, __m256>
  barycentric(const __m256 &x_pos, const __m256 &y_pos,
              const SoftRasterizer::Triangle &triangle);

#elif defined(__arm__) || defined(__aarch64__)
  static inline std::tuple<simde__m256, simde__m256, simde__m256>
  barycentric(const simde__m256 &x_pos, const simde__m256 &y_pos,
              const SoftRasterizer::Triangle &triangle);

#else
#endif

  /*Rasterize a triangle*/
  inline void
  rasterizeBatchAVX2(const int startx, const int endx, const int y,
                     const std::vector<SoftRasterizer::light_struct> &lists,
                     std::shared_ptr<SoftRasterizer::Shader> shader,
                     const SoftRasterizer::Triangle &packed,
                     const glm::vec3 &eye);

  template <typename _simd>
  inline void processFragByAVX2(
      const int x, const int y, const _simd &z0, const _simd &z1,
      const _simd &z2, const std::vector<SoftRasterizer::light_struct> &lists,
      std::shared_ptr<SoftRasterizer::Shader> shader,
      const SoftRasterizer::Triangle &packed, const glm::vec3 &eye);

  inline void
  rasterizeBatchScalar(const int startx, const int endx, const int y,
                       const std::vector<SoftRasterizer::light_struct> &lists,
                       std::shared_ptr<SoftRasterizer::Shader> shader,
                       const SoftRasterizer::Triangle &scalar,
                       const glm::vec3 &eye);

  inline void processFragByScalar(
      const int startx, const int x, const int y, const float old_z,
      const float z0, const float z1, const float z2, float *__restrict z,
      float *__restrict r, float *__restrict g, float *__restrict b,
      const std::vector<SoftRasterizer::light_struct> &lists,
      std::shared_ptr<SoftRasterizer::Shader> shader,
      const SoftRasterizer::Triangle &scalar, const glm::vec3 &eye);

  inline void rasterizeBatchSSE(const SoftRasterizer::Triangle &) = delete;

  /*My Computer Doesn't support AVX512*/
  inline void rasterizeBatchAVX512(const SoftRasterizer::Triangle &) = delete;

  template <typename _simd>
  inline void writePixel(const long long start_pos, const _simd &r,
                         const _simd &g, const _simd &b);
  inline void writePixel(const long long x, const long long y,
                         const glm::vec3 &color);
  inline void writePixel(const long long x, const long long y,
                         const glm::uvec3 &color);
  inline void writePixel(const long long start_pos, const ColorSIMD &color);

  template <typename _simd>
  inline void writeZBuffer(const long long start_pos, const _simd &depth);
  inline bool writeZBuffer(const long long x, const long long y,
                           const float depth);
  inline void writeZBuffer(const long long start_pos, const float depth);

  template <typename _simd> inline _simd readZBuffer(const long long start_pos);
  inline const float readZBuffer(const long long x, const long long y);

  template <typename _simd>
  inline std::tuple<_simd, _simd, _simd> readPixel(const long long start_pos);

  /*Bresenham algorithm*/
  void drawLine(const glm::vec3 &p0, const glm::vec3 &p1,
                const glm::uvec3 &color);

private:
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
  // SpinLock m_channelLock;
  constexpr static std::size_t numbers = 3;
  std::vector<cv::Mat> m_channels;

  /*to store final frame*/
  cv::Mat m_frameBuffer;

  /*z buffer*/
  // SpinLock m_zBufferLock;
  alignas(64) std::vector<float> m_zBuffer;
};
} // namespace SoftRasterizer

#endif //_RENDER_HPP_
