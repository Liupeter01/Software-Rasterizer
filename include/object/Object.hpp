#pragma once
#ifndef _OBJECT_HPP_
#define _OBJECT_HPP_
#include <bvh/Bounds3.hpp>
#include <memory>
#include <object/Material.hpp>
#include <ray/Intersection.hpp>
#include <ray/Ray.hpp> //ray def
#include <shader/Shader.hpp>
#include <tuple>

namespace SoftRasterizer {
/*forward declaration*/
class Shader;
class Scene;

struct Vertex {
  Vertex();
  Vertex(const glm::vec3 &_pos, const glm::vec3 &_normal, const glm::vec2 &_tex,
         const glm::vec3 &_color = glm::vec3(1.0f));

  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 texCoord;
  glm::vec3 color = glm::vec3(1.0f);

  bool operator==(const Vertex &other) const {
    return this->position == other.position && this->color == other.color &&
           this->normal == other.normal && this->texCoord == other.texCoord;
  }
};

struct Object {
  friend class Scene;

  struct Properties {
    glm::vec3 normal = glm::vec3(0.f);
    glm::vec2 uv = glm::vec2(0.f);
    glm::vec3 color = glm::vec3(0.f);
  };

  Object();
  Object(std::shared_ptr<Material> material);
  Object(std::shared_ptr<Material> material, std::shared_ptr<Shader> shader);
  virtual ~Object();
  virtual Bounds3 getBounds() = 0;
  virtual bool intersect(const Ray &) = 0;
  virtual bool intersect(const Ray &, float &) = 0;
  virtual Intersection getIntersect(Ray &) = 0;
  virtual glm::vec3 getDiffuseColor(const glm::vec2 &uv) = 0;
  virtual Properties getSurfaceProperties(const std::size_t faceIndex,
                                          const glm::vec3 &Point,
                                          const glm::vec3 &viewDir,
                                          const glm::vec2 &uv) = 0;

  virtual std::shared_ptr<Material> &getMaterial() = 0;
  virtual void setMaterial(std::shared_ptr<Material>) = 0;

  /*Compatible Consideration!*/
  virtual const std::vector<Vertex> &getVertices() const = 0;
  virtual const std::vector<glm::uvec3> &getFaces() const = 0;

  /*Perform (NDC) MVP Calculation*/
  virtual void updatePosition(const glm::mat4x4 &Model, const glm::mat4x4 &View,
                              const glm::mat4x4 &Projection,
                              const glm::mat4x4 &Ndc) = 0;

  virtual void bindShader2Mesh(std::shared_ptr<Shader> shader) = 0;

  /*Generate A Random Intersection Point on the Object*/
  virtual std::tuple<Intersection, float> sample() = 0;
  virtual const float getArea() = 0;

  /*Self Emissive object*/
  const bool isSelfEmissiveObject() { return m_material->hasEmission(); }

  void updateModelMatrix(const glm::vec3 &axis, const float angle,
                         const glm::vec3 &translation, const glm::vec3 &scale);

  const glm::mat4x4 &getModelMatrix() const;

public:
  std::size_t index = 0;
  glm::mat4x4 modelMatrix = glm::mat4x4(1.0f);

protected:
  std::shared_ptr<Material> m_material;

  // Shading structure
  std::shared_ptr<Shader> m_shader;
};
} // namespace SoftRasterizer

#endif //_OBJECT_HPP_
