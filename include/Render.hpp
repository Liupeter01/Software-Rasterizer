#pragma once
#ifndef _RENDER_HPP_
#define _RENDER_HPP_
#include <Eigen/Eigen>
#include <Triangle.hpp>
#include <algorithm>

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
      const Eigen::Matrix4f &model = Eigen::Matrix4f::Identity(),
      const Eigen::Matrix4f &view = Eigen::Matrix4f::Identity(),
      const Eigen::Matrix4f &projection = Eigen::Matrix4f::Identity());

  virtual ~RenderingPipeline();

protected:
  /*draw graphics*/
  void draw(Primitive type);

  const Eigen::Matrix4f &getScreenSpaceTransform() const {
    return m_screenSpaceTransform;
  }

public:
  void clear(SoftRasterizer::Buffers flags);

  /*load graphics vertices and faces*/
  void loadVertices(const std::vector<Eigen::Vector3f> &vertices) {
    m_vertices = vertices;
  }
  void loadIndices(const std::vector<Eigen::Vector3i> &indices) {
    m_faces = indices;
  }
  void loadColours(const std::vector<Eigen::Vector3f> &colours) {
    m_colours = colours;
  }

  /*set MVP*/
  void setModelMatrix(const Eigen::Matrix4f &model) { m_model = model; }
  void setViewMatrix(const Eigen::Matrix4f &view) { m_view = view; }
  void setProjectionMatrix(const Eigen::Matrix4f &projection) {
    m_projection = projection;
  }

  /*display*/
  void display(Primitive type);

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

  void rasterizeWireframe(const SoftRasterizer::Triangle &triangle);
  void writePixel(const Eigen::Vector3f &point, const Eigen::Vector3f &color);

  /*Bresenham algorithm*/
  void drawLine(const Eigen::Vector3f &p0, const Eigen::Vector3f &p1,
                const Eigen::Vector3f &color);

private:
  // near and far clipping planes
  float m_near = 0.1f;
  float m_far = 100.0f;

  /*display resolution*/
  std::size_t m_width;
  std::size_t m_height;
  float m_aspectRatio;

  /*user input vertices*/
  std::vector<Eigen::Vector3f> m_vertices;

  /*
   * vertices[x], vertices[y], vertices[z] which combined a face
   * so in faces, we have 3 vertices index to form a triangle
   */
  std::vector<Eigen::Vector3i> m_faces;

  /* color for each vertex*/
  std::vector<Eigen::Vector3f> m_colours;

  /*Matrix MVP*/
  Eigen::Matrix4f m_model;
  Eigen::Matrix4f m_view;
  Eigen::Matrix4f m_projection;

  /*Transform normalized coordinates into screen space coordinates*/
  Eigen::Matrix4f m_screenSpaceTransform;

  std::vector<Eigen::Vector3f> m_frameBuffer;

  /*z buffer*/
  std::vector<float> m_zBuffer;
};
} // namespace SoftRasterizer

#endif //_RENDER_HPP_
