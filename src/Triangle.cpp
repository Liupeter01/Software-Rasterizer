#include <Triangle.hpp>
#include <algorithm>
#include <Tools.hpp>

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

void 
SoftRasterizer::Triangle::calcBoundingBox(const std::size_t width, const std::size_t height) {
          auto [min, max] = Tools::calculateBoundingBox(*this);

          box.startX = (min.x() >= 0 ? min.x() : 0);
          box.startY = (min.y() >= 0 ? min.y() : 0);

          box.endX = (max.x() < width ? max.x() : width);
          box.endY = (max.y() < height ? max.y() : height);
}