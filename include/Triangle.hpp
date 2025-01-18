#pragma once
#ifndef _TRIANGLE_HPP_
#define _TRIANGLE_HPP_
#include <array>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <initializer_list>

namespace SoftRasterizer {
struct alignas(32) Triangle {
  Triangle();

  struct BoundingBox {
    BoundingBox() : startY(0), startX(0), endX(0), endY(0) {}

    long long startX;
    long long startY;

    long long endX;
    long long endY;
  };

  const glm::vec3 &a() const { return m_vertex[0]; }
  const glm::vec3 &b() const { return m_vertex[1]; }
  const glm::vec3 &c() const { return m_vertex[2]; }

  void setVertex(std::initializer_list<glm::vec3> _vertex);
  void setNormal(std::initializer_list<glm::vec3> _normal);
  void setColor(std::initializer_list<glm::vec3> _color);
  void setTexCoord(std::initializer_list<glm::vec2> _texCoords);

  void calcBoundingBox(const std::size_t width, const std::size_t height);
  bool isOverlapping(const Triangle &box1, const Triangle &box2) const;

  /*the original coordinates of the triangle, v0, v1, v2 in counter clockwise
   * order*/
  std::array<glm::vec3, 3> m_vertex;
  std::array<glm::vec3, 3> m_color; // Color for each vertex
  std::array<glm::vec2, 3>
      m_texCoords;                   // texture u,v coordinates for each vertex
  std::array<glm::vec3, 3> m_normal; // normal vector for each vertex

  /*BoundBox Calculation!*/
  BoundingBox box;

protected:
  bool isOverlapping(const BoundingBox &box1, const BoundingBox &box2) const;
};
} // namespace SoftRasterizer

#endif //_TRIANGLE_HPP_
