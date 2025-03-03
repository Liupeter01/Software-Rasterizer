#pragma once
#ifndef _SPHERE_HPP_
#define _SPHERE_HPP_
#include <memory>
#include <object/Material.hpp>
#include <object/Object.hpp>

namespace SoftRasterizer {

class Material;

class Sphere : public Object {
public:
  Sphere();
  Sphere(const glm::vec3 &_center, const float _radius);
  virtual ~Sphere();

public:
  Bounds3 getBounds() override;
  float getSquare() const;
  [[nodiscard]] bool intersect(const Ray &ray) override;
  [[nodiscard]] bool intersect(const Ray &ray, float &tNear) override;
  [[nodiscard]] Intersection getIntersect(Ray &ray) override;
  [[nodiscard]] Properties getSurfaceProperties(const std::size_t faceIndex,
                                                const glm::vec3 &Point,
                                                const glm::vec3 &viewDir,
                                                const glm::vec2 &uv);

  [[nodiscard]] glm::vec3 getDiffuseColor(const glm::vec2 &uv) override {
    return glm::vec3(0.f);
  }
  [[nodiscard]] std::shared_ptr<Material> &getMaterial() override {
    return m_material;
  }

  /*Compatible Consideration!*/
  [[nodiscard]] const std::vector<Vertex> &getVertices() const override {
    return vert;
  }
  [[nodiscard]] const std::vector<glm::uvec3> &getFaces() const override {
    return faces;
  }

  /*Generate A Random Intersection Point on the Object*/
  [[nodiscard]] std::tuple<Intersection, float> sample() override;
  [[nodiscard]] const float getArea() override { return area; }

  void updatePosition(const glm::mat4x4 &Model, const glm::mat4x4 &View,
                      const glm::mat4x4 &Projection,
                      const glm::mat4x4 &Ndc) override;

  void bindShader2Mesh(std::shared_ptr<Shader> shader) override;

  void setMaterial(std::shared_ptr<Material> material) override;

  void calcArea();

private:
  float radius;
  float square;
  float area;

  /*This is the original center*/
  glm::vec3 center;

  /*This is the converted center after NDC_MVP*/
  std::vector<SoftRasterizer::Vertex> vert;
  std::vector<glm::uvec3> faces;
};
} // namespace SoftRasterizer

#endif //_SPHERE_HPP_
