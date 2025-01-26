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

  [[nodiscard]] glm::vec3 getDiffuseColor(const glm::vec2 &uv) override;
  [[nodiscard]] std::shared_ptr<Material>& getMaterial()override;

private:
  float radius;
  float square;
  glm::vec3 center;
  std::shared_ptr<Material> material;
};
} // namespace SoftRasterizer

#endif //_SPHERE_HPP_
