#include <Tools.hpp>
#include <Triangle.hpp>
#include <algorithm>

SoftRasterizer::Triangle::Triangle() {
  m_vertex[0] << 0.f, 0.f, 0.f;
  m_vertex[1] << 0.f, 0.f, 0.f;
  m_vertex[2] << 0.f, 0.f, 0.f;

  m_color[0] << 0.0f, 0.0f, 0.0f;
  m_color[1] << 0.0f, 0.0f, 0.0f;
  m_color[2] << 0.0f, 0.0f, 0.0f;

  m_texCoords[0] << 0.0f, 0.0f;
  m_texCoords[1] << 0.0f, 0.0f;
  m_texCoords[2] << 0.0f, 0.0f;
}

void SoftRasterizer::Triangle::setVertex(
    std::initializer_list<Eigen::Vector3f> _vertex) {
  if (_vertex.size() != 3) {
    throw std::runtime_error("Invalid number of vertices");
  }
  std::copy(_vertex.begin(), _vertex.end(), m_vertex.begin());
}

void SoftRasterizer::Triangle::setNormal(
    std::initializer_list<Eigen::Vector3f> _normal) {
  if (_normal.size() != 3) {
    throw std::runtime_error("Invalid number of normals");
  }
  std::copy(_normal.begin(), _normal.end(), m_normal.begin());
}

void SoftRasterizer::Triangle::setColor(
    std::initializer_list<Eigen::Vector3i> _color) {
  if (_color.size() != 3) {
    throw std::runtime_error("Invalid number of colors");
  }
  auto it = m_color.begin();
  std::for_each(_color.begin(), _color.end(), [&it](const Eigen::Vector3i &c) {
    if ((c[0] < 0) || (c[0] > 255) || (c[1] < 0) || (c[1] > 255) ||
        (c[2] < 0) || (c[2] > 255)) {
      throw std::runtime_error("Invalid color values");
    }
    (*it) = c;
    std::advance(it, 1);
  });
}

void SoftRasterizer::Triangle::setTexCoord(
    std::initializer_list<Eigen::Vector2f> _texCoords) {
  if (_texCoords.size() != 3) {
    throw std::runtime_error("Invalid number of texture coordinates");
  }
  std::copy(_texCoords.begin(), _texCoords.end(), m_texCoords.begin());
}

void SoftRasterizer::Triangle::calcBoundingBox(const std::size_t width, const std::size_t height) {
          box.startX = std::clamp(static_cast<long long>(std::min({ m_vertex[0].x(), m_vertex[1].x(), m_vertex[2].x() })), 0LL, static_cast<long long>(width - 1));
          box.startY = std::clamp(static_cast<long long>(std::min({ m_vertex[0].y(), m_vertex[1].y(), m_vertex[2].y() })), 0LL, static_cast<long long>(height - 1));
          box.endX = std::clamp(static_cast<long long>(std::max({ m_vertex[0].x(), m_vertex[1].x(), m_vertex[2].x() })), 0LL, static_cast<long long>(width - 1));
          box.endY = std::clamp(static_cast<long long>(std::max({ m_vertex[0].y(), m_vertex[1].y(), m_vertex[2].y() })), 0LL, static_cast<long long>(height - 1));
}


bool SoftRasterizer::Triangle::isOverlapping(const Triangle& box1, const Triangle& box2)  const {
          return isOverlapping(box1.box, box2.box);
}

bool SoftRasterizer::Triangle::isOverlapping(const BoundingBox& box1, const BoundingBox& box2) const {
          return !(box1.endX < box2.startX || box1.startX > box2.endX ||
                    box1.endY < box2.startY || box1.startY > box2.endY);
}