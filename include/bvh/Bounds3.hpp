#pragma once
#ifndef _BOUNDS3_HPP_
#define _BOUNDS3_HPP_
#include <Tools.hpp>
#include <cmath>
#include <glm/glm.hpp>
#include <limits>
#include <ray/Ray.hpp>

namespace SoftRasterizer {
struct Bounds3 {
  Bounds3();
  Bounds3(const glm::vec3 &p);
  Bounds3(const glm::vec3 &p1, const glm::vec3 &p2);

  glm::vec3 diagonal() const;
  glm::vec3 centroid() const;
  Bounds3 intersect(const Bounds3 &b);

  /*Calculate is there any intersects between BoundingBox and Ray*/
  bool intersect(const Ray &ray);

  bool overlaps(const Bounds3 &box1, const Bounds3 &box2);
  bool inside(const glm::vec3 &point);
  bool inside(const glm::vec3 &point, const Bounds3 &box);

  int maxExtent();

  double surfaceArea();

  // two points to specify the bounding box
  glm::vec3 min, max;
};

Bounds3 BoundsUnion(const Bounds3 &box1, const Bounds3 &box2);
Bounds3 BoundsUnion(const glm::vec3 &point, const Bounds3 &box);

} // namespace SoftRasterizer

#endif //_BOUNDS3_HPP_
