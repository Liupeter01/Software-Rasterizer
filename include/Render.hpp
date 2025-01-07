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

public:
  void clear(SoftRasterizer::Buffers flags);

  /*set MVP*/
  bool setModelMatrix(const std::string &meshName,
                      const Eigen::Matrix4f &model);

  void setViewMatrix(const Eigen::Matrix4f &view);
  void setProjectionMatrix(const Eigen::Matrix4f &projection);

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
  barycentric(const std::size_t x_pos, const std::size_t y_pos,
              const SoftRasterizer::Triangle &triangle);

  /*Rasterize a triangle*/
  void rasterizeTriangle(const SoftRasterizer::Triangle &triangle);

  void writePixel(const Eigen::Vector3f &point, const Eigen::Vector3f &color);
  void writePixel(const Eigen::Vector3f &point, const Eigen::Vector3i &color);

  bool writeZBuffer(const Eigen::Vector3f &point, const float depth);

  /*Bresenham algorithm*/
  void drawLine(const Eigen::Vector3f &p0, const Eigen::Vector3f &p1,
                const Eigen::Vector3i &color);

private:
  // near and far clipping planes
  float m_near = 0.1f;
  float m_far = 100.0f;

  /*display resolution*/
  std::size_t m_width;
  std::size_t m_height;
  float m_aspectRatio;

  /*store all identified objs, waiting for loading*/
  std::unordered_map<std::string, std::unique_ptr<ObjLoader>> m_suspendObjs;

  /*store all loaded objs*/
  std::unordered_map<std::string, std::unique_ptr<Mesh>> m_loadedObjs;

  /*store all shaders*/
  std::unordered_map<std::string, std::shared_ptr<Shader>> m_texture;

  /*Matrix VP*/
  Eigen::Matrix4f m_view;
  Eigen::Matrix4f m_projection;

  /*Transform normalized coordinates into screen space coordinates*/
  Eigen::Matrix4f m_ndcToScreenMatrix;

  std::vector<Eigen::Vector3f> m_frameBuffer;

  /*z buffer*/
  std::vector<float> m_zBuffer;
};
} // namespace SoftRasterizer

#endif //_RENDER_HPP_
