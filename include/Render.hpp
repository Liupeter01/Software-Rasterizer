#pragma once
#ifndef _RENDER_HPP_
#define _RENDER_HPP_
#include <ObjLoader.hpp>
#include <Shader.hpp>
#include <Triangle.hpp>
#include <algorithm>
#include <optional>
#include <tuple>
#include <unordered_map>

/*Use for unrolling calculation*/
#define ROUND_UP_TO_MULTIPLE_OF_4(x) (((x) + 3) & ~3)

#if defined(__x86_64__)
#include <intrin.h> // Required for __cpuid intrinsic
#include <xmmintrin.h>
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
  RenderingPipeline(
      const std::size_t width, const std::size_t height,
      const Eigen::Matrix4f &view = Eigen::Matrix4f::Identity(),
      const Eigen::Matrix4f &projection = Eigen::Matrix4f::Identity());

  virtual ~RenderingPipeline();

protected:
  /*draw graphics*/
  void draw(Primitive type);

  /*we don't want other user to deploy it directly, because we need to record
   * detailed arguments*/
  void setProjectionMatrix(const Eigen::Matrix4f &projection);
  void setViewMatrix(const Eigen::Matrix4f &view);

public:
  void clear(SoftRasterizer::Buffers flags);

  /*set MVP*/
  bool setModelMatrix(const std::string &meshName,
                      const Eigen::Matrix4f &model);

  void setViewMatrix(const Eigen::Vector3f &eye, const Eigen::Vector3f &center,
                     const Eigen::Vector3f &up);

  void setProjectionMatrix(float fovy, float zNear, float zFar);

  /*display*/
  void display(Primitive type);

  /*load ObjLoader object to load wavefront obj file*/
  bool addGraphicObj(const std::string &path, const std::string &meshName);

  bool addGraphicObj(
      const std::string &path, const std::string &meshName,
      const Eigen::Matrix4f &rotation,
      const Eigen::Vector3f &translation = Eigen::Vector3f(0.f, 0.f, 0.f),
      const Eigen::Vector3f &scale = Eigen::Matrix4f::Identity());

  bool addGraphicObj(
      const std::string &path, const std::string &meshName,
      const Eigen::Vector3f &axis, const float angle,
      const Eigen::Vector3f &translation = Eigen::Vector3f(0.f, 0.f, 0.f),
      const Eigen::Vector3f &scale = Eigen::Matrix4f::Identity());

  bool startLoadingMesh(const std::string &meshName);

  bool addShader(const std::string &shaderName, const std::string &texturePath,
                 SHADERS_TYPE type);

  bool addShader(const std::string &shaderName,
                 std::shared_ptr<TextureLoader> text, SHADERS_TYPE type);

  bool bindShader2Mesh(const std::string &meshName,
                       const std::string &shaderName);

private:
  std::vector<Eigen::Vector3f> &getFrameBuffer() { return m_frameBuffer; }

  /*------------------------------framebuffer-----------------------------------------*/
  /*|<----------------------------m_width------------------------------->|_________
     |*************************************************|      /\
     |*************************************************|       |
     |*************************************************| m_height
     |**************************************(x,y)*******|       |
     |*************************************************|      \/
     ____________________________________________________________*/

  /*Only Draw Line*/
  void rasterizeWireframe(const SoftRasterizer::Triangle &triangle);

  /**
   * @brief Calculates the bounding box for a given triangle.
   *
   * This function determines the axis-aligned bounding box (AABB)
   * that encompasses the given triangle in 2D space. The bounding box
   * is represented as a pair of 2D integer vectors, indicating the
   * minimum and maximum corners of the box.
   *
   * @param triangle The triangle for which the bounding box is to be
   * calculated. The triangle is represented using the
   * `SoftRasterizer::Triangle` type.
   *
   * @return A pair of 2D integer vectors (Eigen::Vector2i), where:
   *         - The first vector represents the minimum corner of the bounding
   * box (bottom-left).
   *         - The second vector represents the maximum corner of the bounding
   * box (top-right).
   */
  std::pair<Eigen::Vector2i, Eigen::Vector2i>
  calculateBoundingBox(const SoftRasterizer::Triangle &triangle);

  static bool insideTriangle(const std::size_t x_pos, const std::size_t y_pos,
                             const SoftRasterizer::Triangle &triangle);

  static std::optional<std::tuple<float, float, float>>
  linearBaryCentric(const std::size_t x_pos, const std::size_t y_pos,
                    const Eigen::Vector2i min, const Eigen::Vector2i max);

  static inline std::optional<std::tuple<float, float, float>>
  barycentric(const std::size_t x_pos, const std::size_t y_pos,
              const SoftRasterizer::Triangle &triangle);

  /*Rasterize a triangle*/
  void rasterizeTriangle(std::shared_ptr<SoftRasterizer::Shader> shader,
                         SoftRasterizer::Triangle &triangle);

  inline void writePixel(const long long x, const long long y,
                         const Eigen::Vector3f &color);
  inline void writePixel(const long long x, const long long y,
                         const Eigen::Vector3i &color);

  inline bool writeZBuffer(const long long x, const long long y,
                           const float depth);

  /*Bresenham algorithm*/
  void drawLine(const Eigen::Vector3f &p0, const Eigen::Vector3f &p1,
                const Eigen::Vector3i &color);

private:
  /*optimized*/
  unsigned cache_line_size = 0;

  std::size_t BLOCK_SIZE = 64;
  std::size_t UNROLLING_FACTOR;

  /*display resolution*/
  std::size_t m_width;
  std::size_t m_height;
  float m_aspectRatio;

  /*store all identified objs, waiting for loading*/
  std::unordered_map<std::string, std::unique_ptr<ObjLoader>> m_suspendObjs;

  /*store all loaded objs*/
  std::unordered_map<std::string, std::unique_ptr<Mesh>> m_loadedObjs;

  /*store all shaders*/
  std::unordered_map<std::string, std::shared_ptr<Shader>> m_shaders;

  /*Matrix View*/
  Eigen::Vector3f m_eye;
  Eigen::Vector3f m_center;
  Eigen::Vector3f m_up;
  Eigen::Matrix4f m_view;

  /*Matrix Projection*/
  // near and far clipping planes
  float m_fovy;
  float m_near = 0.1f;
  float m_far = 100.0f;
  Eigen::Matrix4f m_projection;

  /*Transform normalized coordinates into screen space coordinates*/
  Eigen::Matrix4f m_ndcToScreenMatrix;

  std::vector<Eigen::Vector3f> m_frameBuffer;

  /*z buffer*/
  std::vector<float> m_zBuffer;
};
} // namespace SoftRasterizer

#endif //_RENDER_HPP_
