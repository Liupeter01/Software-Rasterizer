#pragma once
#ifndef _OBJECT_HPP_
#define _OBJECT_HPP_
#include <bvh/Bounds3.hpp>
#include <optional>
#include <ray/Intersection.hpp>
#include <ray/Ray.hpp> //ray def

namespace SoftRasterizer {
struct Object {
  Object() {}
  virtual ~Object() {}
  virtual Bounds3 getBounds() = 0;
  virtual bool intersect(const Ray &) = 0;
  virtual bool intersect(const Ray &, float &) = 0;
  virtual Intersection getIntersect(Ray &) = 0;
};
} // namespace SoftRasterizer

#endif //_OBJECT_HPP_
