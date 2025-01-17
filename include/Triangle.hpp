#pragma once
#ifndef _TRIANGLE_HPP_
#define _TRIANGLE_HPP_
#include <Eigen/Eigen>
#include <array>
#include <initializer_list>

namespace SoftRasterizer {
struct alignas(32) Triangle {
  Triangle();

  struct BoundingBox {
    long long startX;
    long long startY;

    long long endX;
    long long endY;
  };

  const Eigen::Vector3f &a() const { return m_vertex[0]; }
  const Eigen::Vector3f &b() const { return m_vertex[1]; }
  const Eigen::Vector3f &c() const { return m_vertex[2]; }

  void setVertex(std::initializer_list<Eigen::Vector3f> _vertex);
  void setNormal(std::initializer_list<Eigen::Vector3f> _normal);
  void setColor(std::initializer_list<Eigen::Vector3i> _color);
  void setTexCoord(std::initializer_list<Eigen::Vector2f> _texCoords);
  void calcBoundingBox(const std::size_t width, const std::size_t height);
  bool isOverlapping(const Triangle &box1, const Triangle &box2) const;

  /*the original coordinates of the triangle, v0, v1, v2 in counter clockwise
   * order*/
  std::array<Eigen::Vector3f, 3> m_vertex;

  // Color for each vertex
  std::array<Eigen::Vector3i, 3> m_color;

  // texture u,v coordinates for each vertex
  std::array<Eigen::Vector2f, 3> m_texCoords;

  // normal vector for each vertex
  std::array<Eigen::Vector3f, 3> m_normal;

  /*BoundBox Calculation!*/
  BoundingBox box;

protected:
  bool isOverlapping(const BoundingBox &box1, const BoundingBox &box2) const;
};
} // namespace SoftRasterizer

#endif //_TRIANGLE_HPP_
