#pragma once
#ifndef _OBJECT_HPP_
#define _OBJECT_HPP_
#include <bvh/Bounds3.hpp>
#include <memory>
#include <optional>
#include <ray/Intersection.hpp>
#include <ray/Ray.hpp> //ray def

namespace SoftRasterizer {
struct Object {
  struct Properties {
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec3 color = glm::vec3(0.f);
  };

  std::size_t index = 0;

  Object() {}
  virtual ~Object() {}
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
};
} // namespace SoftRasterizer

#endif //_OBJECT_HPP_
