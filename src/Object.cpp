#include <glm/gtc/matrix_transform.hpp>
#include <object/Object.hpp>
#include <shader/Shader.hpp>

SoftRasterizer::Vertex::Vertex()
    : position(0.f), normal(0.f), texCoord(0.f), color(1.0f) {}

SoftRasterizer::Vertex::Vertex(const glm::vec3 &_pos, const glm::vec3 &_normal,
                               const glm::vec2 &_tex, const glm::vec3 &_color)
    : position(_pos), normal(_normal), texCoord(_tex), color(_color) {}

SoftRasterizer::Object::Object() : Object(nullptr, nullptr) {}

SoftRasterizer::Object::Object(std::shared_ptr<Material> material)
    : Object(material, nullptr) {}

SoftRasterizer::Object::Object(std::shared_ptr<Material> material,
                               std::shared_ptr<Shader> shader)
    : index(0), modelMatrix(1.0f), m_shader(shader), m_material(material) {}

SoftRasterizer::Object::~Object() {}

void SoftRasterizer::Object::updateModelMatrix(const glm::vec3 &axis,
                                               const float angle,
                                               const glm::vec3 &translation,
                                               const glm::vec3 &scale) {
  auto T = glm::translate(glm::mat4(1.0f), translation);
  auto R = glm::rotate(glm::mat4(1.0f), glm::radians(angle), axis);
  auto S = glm::scale(glm::mat4(1.0f), scale);
  modelMatrix = T * R * S;
}

const glm::mat4x4 &SoftRasterizer::Object::getModelMatrix() const {
  return modelMatrix;
}
