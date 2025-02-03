#include <Tools.hpp>
#include <algorithm>
#include <shader/Shader.hpp>
#include <object/Triangle.hpp>
#include <loader/TextureLoader.hpp>

SoftRasterizer::Triangle::Triangle()
    : box(), vert(3), Object(std::make_shared<Material>(), nullptr) {
  for (std::size_t index = 0; index < 3; ++index) {
    m_vertex[index] = glm::vec3(0.f);
    m_color[index] = glm::vec3(0.f);
    m_texCoords[index] = glm::vec2(0.f);
    m_normal[index] = glm::vec3(0.f);
  }
}

SoftRasterizer::Triangle::Triangle(
    std::shared_ptr<Material> _material, const glm::vec3 &VertexA,
    const glm::vec3 &VertexB, const glm::vec3 &VertexC,
    const glm::vec3 &NormalA, const glm::vec3 &NormalB,
    const glm::vec3 &NormalC, const glm::vec2 &texCoordA,
    const glm::vec2 &texCoordB, const glm::vec2 &texCoordC,
    const glm::vec3 &colorA, const glm::vec3 &colorB, const glm::vec3 &colorC)

    : box(), vert(3), Object(_material, nullptr){
  vert[0].position = m_vertex[0] = VertexA;
  vert[1].position = m_vertex[1] = VertexB;
  vert[2].position = m_vertex[2] = VertexC;

  vert[0].color = m_color[0] = colorA;
  vert[1].color = m_color[1] = colorB;
  vert[2].color = m_color[2] = colorC;

  vert[0].normal = m_normal[0] = NormalA;
  vert[1].normal = m_normal[1] = NormalB;
  vert[2].normal = m_normal[2] = NormalC;

  vert[0].texCoord = m_texCoords[0] = texCoordA;
  vert[1].texCoord = m_texCoords[1] = texCoordA;
  vert[2].texCoord = m_texCoords[2] = texCoordA;
}

void SoftRasterizer::Triangle::setVertex(
    std::initializer_list<glm::vec3> _vertex) {
  if (_vertex.size() != 3) {
    throw std::runtime_error("Invalid number of vertices");
  }
  std::copy(_vertex.begin(), _vertex.end(), m_vertex.begin());
  vert[0].position = m_vertex[0];
  vert[1].position = m_vertex[1];
  vert[2].position = m_vertex[2];
}

void SoftRasterizer::Triangle::setNormal(
    std::initializer_list<glm::vec3> _normal) {
  if (_normal.size() != 3) {
    throw std::runtime_error("Invalid number of normals");
  }
  std::copy(_normal.begin(), _normal.end(), m_normal.begin());
  vert[0].normal = m_normal[0];
  vert[1].normal = m_normal[1];
  vert[2].normal = m_normal[2];
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

  vert[0].color = m_color[0];
  vert[1].color = m_color[1];
  vert[2].color = m_color[2];
}

void SoftRasterizer::Triangle::setTexCoord(
    std::initializer_list<glm::vec2> _texCoords) {
  if (_texCoords.size() != 3) {
    throw std::runtime_error("Invalid number of texture coordinates");
  }
  std::copy(_texCoords.begin(), _texCoords.end(), m_texCoords.begin());
  vert[0].texCoord = m_texCoords[0];
  vert[1].texCoord = m_texCoords[1];
  vert[2].texCoord = m_texCoords[2];
}

SoftRasterizer::Bounds3 SoftRasterizer::Triangle::getBounds() {
  return BoundsUnion(vert[0].position,
                     Bounds3(vert[1].position, vert[2].position));
}

// Moller Trumbore Algorithm
SoftRasterizer::Intersection 
SoftRasterizer::Triangle::getIntersect(Ray &ray) {

  glm::vec3 normal = getFaceNormal();

  // back face culling
  if (glm::dot(normal, ray.direction) <= 0) {
            return {};
  }

  // Caculate Edge Vectors
  glm::vec3 e1 = vert[1].position - vert[0].position;
  glm::vec3 e2 = vert[2].position - vert[0].position;

  // light and surface is parallel or not?
  glm::vec3 pvec = glm::cross(ray.direction, e2);
  float det = glm::dot(e1, pvec);
  if (std::abs(det) < std::numeric_limits<float>::epsilon()) {
            return {};
  }

  // barycentric coordinates
  double det_inv = 1.f / det;
  glm::vec3 tvec = ray.origin - vert[0].position;
  float u = glm::dot(tvec, pvec) * det_inv;
  if (u < 0 || std::abs(u - 1.0f) < std::numeric_limits<float>::epsilon() ) {
            return {};
  }

  glm::vec3 qvec = glm::cross(tvec, e1);
  float v = glm::dot(ray.direction, qvec) * det_inv;
  if (v < 0 || u + v > 1) {
            return {};
  }

  // calculate the intersect time
  float t0 = glm::dot(e2, qvec) * det_inv;
  if (t0 < 0) {
            return {};
  }

  Intersection ret;
  ret.obj = this;
  ret.intersect_time = t0;
  ret.coords = ray.direction * ret.intersect_time + ray.origin;
  ret.uv = glm::vec2(u, v);

  // we could find a intersect time point
  ret.intersected = true;
  return ret;
}

glm::vec3 SoftRasterizer::Triangle::getFaceNormal(FaceNormalType type) const {
  if (type == FaceNormalType::PerGeometry) {
    return glm::normalize(glm::cross(vert[1].position - vert[0].position,
                                     vert[2].position - vert[0].position));
  } else if (type == FaceNormalType::InterpolatedFace) {
    return Tools::interpolateNormal(zero_point_3, zero_point_3, zero_point_3,
                                    vert[0].normal, vert[1].normal,
                                    vert[2].normal);
  } else {
    throw std::runtime_error("Invalid Face Normal Type");
  }
}

SoftRasterizer::Object::Properties
SoftRasterizer::Triangle::getSurfaceProperties(const std::size_t faceIndex,
                                               const glm::vec3 &Point,
                                               const glm::vec3 &viewDir,
                                               const glm::vec2 &uv) {
  Properties ret;
  //ret.normal = getFaceNormal();
  ret.normal = glm::normalize(
            (1 - uv.x - uv.y) * vert[0].normal +
            uv.x * vert[1].normal +
            uv.y * vert[2].normal
  );

  ret.uv = m_texCoords[0] * (1 - uv.x - uv.y) + m_texCoords[1] * uv.x + m_texCoords[2] * uv.y;

  /*get color of this point*/
  ret.color = getDiffuseColor(ret.uv);
  return ret;
}

glm::vec3 
SoftRasterizer::Triangle::getDiffuseColor(const glm::vec2 &uv) {
          //When m_shader is nullptr then skip this code block
          if (!m_shader) {
                    return glm::vec3(1.0f);
          }
         return m_shader->getTextureObject()->getTextureColor(uv);
}

void SoftRasterizer::Triangle::updatePosition(const glm::mat4x4 &NDC_MVP,
                                              const glm::mat4x4 &Normal_M) {

  vert[0].position = Tools::to_vec3(NDC_MVP * glm::vec4(m_vertex[0], 1.0f));
  vert[0].normal = Tools::to_vec3(Normal_M * glm::vec4(m_normal[0], 1.0f));
  vert[0].texCoord = m_texCoords[0];
  vert[0].color = m_color[0];

  vert[1].position = Tools::to_vec3(NDC_MVP * glm::vec4(m_vertex[1], 1.0f));
  vert[1].normal = Tools::to_vec3(Normal_M * glm::vec4(m_normal[1], 1.0f));
  vert[1].texCoord = m_texCoords[1];
  vert[1].color = m_color[1];

  vert[2].position = Tools::to_vec3(NDC_MVP * glm::vec4(m_vertex[2], 1.0f));
  vert[2].normal = Tools::to_vec3(Normal_M * glm::vec4(m_normal[2], 1.0f));
  vert[2].texCoord = m_texCoords[2];
  vert[2].color = m_color[2];
}

void SoftRasterizer::Triangle::bindShader2Mesh(std::shared_ptr<Shader> shader) {
          m_shader.reset();
          m_shader = shader;
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
