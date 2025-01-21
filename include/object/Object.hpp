#pragma once
#ifndef _OBJECT_HPP_
#define _OBJECT_HPP_
#include <optional>
#include <ray/Ray.hpp>        //ray def
#include <bvh/Bounds3.hpp>
#include <ray/Intersection.hpp>

namespace SoftRasterizer {
struct Object {
  Object() {}
  virtual ~Object() {}
  virtual Bounds3 getBounds() = 0;
  virtual bool intersect(const Ray&) = 0;
  virtual bool intersect(const Ray&, float&) = 0;
  virtual Intersection getIntersect(Ray&) = 0;
};
} // namespace SoftRasterizer

#endif //_OBJECT_HPP_
