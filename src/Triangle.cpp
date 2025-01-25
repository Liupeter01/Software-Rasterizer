#include <Tools.hpp>
#include <algorithm>
#include <object/Triangle.hpp>

SoftRasterizer::Triangle::Triangle()
    : box(), m_material(std::make_shared<Material>()) {
  for (std::size_t index = 0; index < 3; ++index) {
    m_vertex[index] = glm::vec3(0.f);
    m_color[index] = glm::vec3(0.f);
    m_texCoords[index] = glm::vec2(0.f);
    m_normal[index] = glm::vec3(0.f);
  }
}

SoftRasterizer::Triangle::Triangle(
    std::shared_ptr<SoftRasterizer::Material> _material)
    : interpolatedNormal(0.f), geometryNormal(0.f), m_material(_material) {
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

  /*Calcuate Vertex Normal While set the vertex*/
  geometryNormal = glm::normalize(
      glm::cross(m_vertex[1] - m_vertex[0], m_vertex[2] - m_vertex[0]));

  interpolatedNormal =
      Tools::interpolateNormal(zero_point_3, zero_point_3, zero_point_3,
                               m_normal[0], m_normal[1], m_normal[2]);
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

SoftRasterizer::Bounds3 SoftRasterizer::Triangle::getBounds() {
  return BoundsUnion(m_vertex[0], Bounds3(m_vertex[1], m_vertex[2]));
}

bool SoftRasterizer::Triangle::intersect(const Ray &ray) { return true; }

bool SoftRasterizer::Triangle::intersect(const Ray &ray, float &tNear) {
  return false;
}

bool SoftRasterizer::Triangle::rayTriangleIntersect(
    const Ray &ray, const glm::vec3 &v0, const glm::vec3 &v1,
    const glm::vec3 &v2, float &tNear, float &u, float &v) {

  glm::vec3 e1 = v1 - v0;
  glm::vec3 e2 = v2 - v0;

  glm::vec3 pvec = glm::cross(ray.direction, e2);
  float det = glm::dot(e1, pvec);
  if (std::abs(det) < std::numeric_limits<float>::epsilon()) {
    return false;
  }

  float inv_det = 1.f / det;
  glm::vec3 tvec = ray.origin - v0;
  u = glm::dot(tvec, pvec) * inv_det;
  if (u < 0 || u > det) {
    return false;
  }

  glm::vec3 qvec = glm::cross(tvec, e1);
  v = glm::dot(ray.direction, qvec) * inv_det;
  if (v < 0 || u + v > det) {
    return false;
  }

  tNear = glm::dot(e2, qvec) * inv_det;
  u *= inv_det;
  v *= inv_det;

  return tNear > 0;
}

// Moller Trumbore Algorithm
SoftRasterizer::Intersection SoftRasterizer::Triangle::getIntersect(Ray &ray) {
  Intersection ret;
  glm::vec3 normal = getFaceNormal();

  // back face culling
  if (glm::dot(normal, ray.direction) > 0) {
    return ret;
  }

  // Caculate Edge Vectors
  glm::vec3 e1 = m_vertex[1] - m_vertex[0];
  glm::vec3 e2 = m_vertex[2] - m_vertex[0];

  // light and surface is parallel or not?
  glm::vec3 pvec = glm::cross(ray.direction, e2);
  float det = glm::dot(e1, pvec);
  if (std::abs(det) < std::numeric_limits<float>::epsilon()) {
    return ret;
  }

  // barycentric coordinates
  double det_inv = 1.f / det;
  glm::vec3 tvec = ray.origin - m_vertex[0];
  float u = glm::dot(tvec, pvec) * det_inv;
  if (u < 0 || u > 1) {
    return ret;
  }

  glm::vec3 qvec = glm::cross(tvec, e1);
  float v = glm::dot(ray.direction, qvec) * det_inv;
  if (v < 0 || u + v > 1) {
    return ret;
  }

  // calculate the intersect time
  float t0 = glm::dot(e2, qvec) * det_inv;
  if (t0 < 0) {
    return ret;
  }

  ret.obj = this;
  ret.intersect_time = t0;
  ret.coords = ray.direction * t0 + ray.origin;

  /*Normal of a sphere!*/
  ret.normal = normal;
  ret.material = getMaterial();

  // we could find a intersect time point
  ret.intersected = true;
  return ret;
}

const glm::vec3 &
SoftRasterizer::Triangle::getFaceNormal(FaceNormalType type) const {
  if (type == FaceNormalType::PerGeometry) {
    return geometryNormal;
  } else if (type == FaceNormalType::InterpolatedFace) {
    return interpolatedNormal;
  } else {
    throw std::runtime_error("Invalid Face Normal Type");
  }
}

std::shared_ptr<SoftRasterizer::Material>
SoftRasterizer::Triangle::getMaterial() {
  return std::shared_ptr<Material>(m_material.get(), [](Material *) {});
}

SoftRasterizer::Object::Properties
SoftRasterizer::Triangle::getSurfaceProperties(const std::size_t faceIndex,
                                               const glm::vec3 &Point,
                                               const glm::vec3 &viewDir,
                                               const glm::vec2 &uv) {
  Properties ret;
  ret.normal = getFaceNormal();
  return ret;
}

glm::vec3 SoftRasterizer::Triangle::getDiffuseColor(const glm::vec2 &uv) {
  return glm::vec3(1.0f);
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
