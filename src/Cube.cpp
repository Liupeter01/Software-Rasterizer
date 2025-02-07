#include <object/Cube.hpp>

SoftRasterizer::Cube::Cube() : Object(std::make_shared<Material>(), nullptr) {}

SoftRasterizer::Cube::~Cube() {}

SoftRasterizer::Bounds3 SoftRasterizer::Cube::getBounds() { return {}; }

bool SoftRasterizer::Cube::intersect(const Ray &ray) { return true; }

bool SoftRasterizer::Cube::intersect(const Ray &ray, float &tNear) {
  return true;
}

SoftRasterizer::Intersection SoftRasterizer::Cube::getIntersect(Ray &ray) {
  return {};
}

SoftRasterizer::Object::Properties SoftRasterizer::Cube::getSurfaceProperties(
    const std::size_t faceIndex, const glm::vec3 &Point,
    const glm::vec3 &viewDir, const glm::vec2 &uv) {
  return {};
}

glm::vec3 SoftRasterizer::Cube::getDiffuseColor(const glm::vec2 &uv) {
  return glm::vec3(0.5f);
}

std::tuple<SoftRasterizer::Intersection, float> 
SoftRasterizer::Cube::sample() {
          
          float pdf = { 1.f };
          SoftRasterizer::Intersection intersection;
          intersection.intersected = true;
          intersection.obj = this;
          intersection.emit = m_material->getEmission();

          return { intersection, 1.0f / pdf };
}

const float 
SoftRasterizer::Cube::getArea() {
          return {};
}

void SoftRasterizer::Cube::updatePosition(const glm::mat4x4 &NDC_MVP,
                                          const glm::mat4x4 &Normal_M) {}

void SoftRasterizer::Cube::bindShader2Mesh(std::shared_ptr<Shader> shader) {
  m_shader.reset();
  m_shader = shader;
}
