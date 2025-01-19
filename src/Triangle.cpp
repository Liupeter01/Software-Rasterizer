#include <Tools.hpp>
#include <object/Triangle.hpp>
#include <algorithm>

SoftRasterizer::Triangle::Triangle() : box() {
  for (std::size_t index = 0; index < 3; ++index) {
    m_vertex[index] = glm::vec3(0.f);
    m_color[index] = glm::vec3(0.f);
    m_texCoords[index] = glm::vec2(0.f);
    m_normal[index] = glm::vec3(0.f);
  }
}

void SoftRasterizer::Triangle::setVertex(
    std::initializer_list<glm::vec3> _vertex) {
  if (_vertex.size() != 3) {
    throw std::runtime_error("Invalid number of vertices");
  }
  std::copy(_vertex.begin(), _vertex.end(), m_vertex.begin());
}

void SoftRasterizer::Triangle::setNormal(
    std::initializer_list<glm::vec3> _normal) {
  if (_normal.size() != 3) {
    throw std::runtime_error("Invalid number of normals");
  }
  std::copy(_normal.begin(), _normal.end(), m_normal.begin());
}

void SoftRasterizer::Triangle::setColor(
    std::initializer_list<glm::vec3> _color) {
  if (_color.size() != 3) {
    throw std::runtime_error("Invalid number of colors");
  }
  auto it = m_color.begin();
  std::for_each(_color.begin(), _color.end(), [&it](const glm::vec3 &c) {
    if ((c[0] < 0) || (c[0] > 255) || (c[1] < 0) || (c[1] > 255) ||
        (c[2] < 0) || (c[2] > 255)) {
      throw std::runtime_error("Invalid color values");
    }
    (*it) = c;
    std::advance(it, 1);
  });
}

void SoftRasterizer::Triangle::setTexCoord(
    std::initializer_list<glm::vec2> _texCoords) {
  if (_texCoords.size() != 3) {
    throw std::runtime_error("Invalid number of texture coordinates");
  }
  std::copy(_texCoords.begin(), _texCoords.end(), m_texCoords.begin());
}

SoftRasterizer::Bounds3 SoftRasterizer::Triangle::getBounds(){
          return BoundsUnion(m_vertex[0], Bounds3(m_vertex[1], m_vertex[2]));
}

void SoftRasterizer::Triangle::calcBoundingBox(const std::size_t width,
                                               const std::size_t height) {
  box.startX = std::clamp(static_cast<long long>(std::min(
                              {m_vertex[0].x, m_vertex[1].x, m_vertex[2].x})),
                          0LL, static_cast<long long>(width - 1));
  box.startY = std::clamp(static_cast<long long>(std::min(
                              {m_vertex[0].y, m_vertex[1].y, m_vertex[2].y})),
                          0LL, static_cast<long long>(height - 1));
  box.endX = std::clamp(static_cast<long long>(std::max(
                            {m_vertex[0].x, m_vertex[1].x, m_vertex[2].x})),
                        0LL, static_cast<long long>(width - 1));
  box.endY = std::clamp(static_cast<long long>(std::max(
                            {m_vertex[0].y, m_vertex[1].y, m_vertex[2].y})),
                        0LL, static_cast<long long>(height - 1));
}