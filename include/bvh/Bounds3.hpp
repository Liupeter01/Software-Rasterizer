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
  Bounds3()
      : min(glm::vec3(std::numeric_limits<float>::infinity())),
        max(glm::vec3(-std::numeric_limits<float>::infinity())) {}

  Bounds3(const glm::vec3 &p) : min(p), max(p) {}
  Bounds3(const glm::vec3 &p1, const glm::vec3 &p2)
      : min(glm::vec3(std::fmin(p1.x, p2.x), std::fmin(p1.y, p2.y),
                      std::fmin(p1.z, p2.z))),
        max(glm::vec3(std::fmax(p1.x, p2.x), std::fmax(p1.y, p2.y),
                      std::fmax(p1.z, p2.z))) {}

  glm::vec3 diagonal() const { return max - min; }
  glm::vec3 centroid() const { return 0.5f * (min + max); }
  Bounds3 intersect(const Bounds3 &b) {
    return Bounds3(glm::vec3(std::fmax(this->min.x, b.min.x),
                             std::fmax(this->min.y, b.min.y),
                             std::fmax(this->min.z, b.min.z)),
                   glm::vec3(std::fmin(this->max.x, b.max.x),
                             std::fmin(this->max.y, b.max.y),
                             std::fmin(this->max.z, b.max.z)));
  }

  /*Calculate is there any intersects between BoundingBox and Ray*/
  bool intersect(const Ray &ray) {

    const glm::vec3 origin = ray.origin;
    const glm::vec3 dir = ray.direction;

    /*
     * Multiply is faster that Division
     * Use a large value to handle division by zero for rays parallel to an axis
     */
    const glm::vec3 inv_dir =
        glm::vec3(std::abs(ray.direction.x) > std::numeric_limits<float>::epsilon() ? 1.0f / ray.direction.x
                                       : std::numeric_limits<float>::max(),
                  std::abs(ray.direction.y) > std::numeric_limits<float>::epsilon() ? 1.0f / ray.direction.y
                                       : std::numeric_limits<float>::max(),
                  std::abs(ray.direction.z) > std::numeric_limits<float>::epsilon() ? 1.0f / ray.direction.z
                                       : std::numeric_limits<float>::max());

    auto x_min = min.x, x_max = max.x;
    auto y_min = min.y, y_max = max.y;
    auto z_min = min.z, z_max = max.z;

    /*
     * Origin + t * direction = Dst
     * t = (Dst - Origin) / direction =  (Dst - Origin) * inv_direction
     */
    auto x_min_t = (x_min - origin.x) * inv_dir.x,
         x_max_t = (x_max - origin.x) * inv_dir.x;
    auto y_min_t = (y_min - origin.y) * inv_dir.y,
         y_max_t = (y_max - origin.y) * inv_dir.y;
    auto z_min_t = (z_min - origin.z) * inv_dir.z,
         z_max_t = (z_max - origin.z) * inv_dir.z;

    // Swap the t_min and t_max values if necessary to ensure correct ordering
    if (x_min_t > x_max_t)
      std::swap(x_min_t, x_max_t);
    if (y_min_t > y_max_t)
      std::swap(y_min_t, y_max_t);
    if (z_min_t > z_max_t)
      std::swap(z_min_t, z_max_t);

    auto entering = Tools::max(x_min_t, y_min_t, z_min_t);
    auto leaving = Tools::min(x_max_t, y_max_t, z_max_t);

    // Use a small epsilon to account for floating - point precision errors
    return (entering <= leaving + std::numeric_limits<float>::epsilon() &&
            leaving >= std::numeric_limits<float>::epsilon());
  }

  bool overlaps(const Bounds3 &box1, const Bounds3 &box2) {
    return box1.max.x >= box2.min.x && box1.min.x <= box2.max.x &&
           box1.max.y >= box2.min.y && box1.min.y <= box2.max.y &&
           box1.max.z >= box2.min.z && box1.min.z <= box2.max.z;
  }

  bool inside(const glm::vec3 &point) { return inside(point, *this); }
  bool inside(const glm::vec3 &point, const Bounds3 &box) {
    return point.x >= box.min.x && point.x <= box.max.x &&
           point.y >= box.min.y && point.y <= box.max.y &&
           point.z >= box.min.z && point.z <= box.max.z;
  }

  int maxExtent() {
    auto d = diagonal();
    if (d.x > d.y && d.x > d.z)
      return 0; // x
    else if (d.y > d.z)
      return 1; // y
    else
      return 2; // z
  }

  // two points to specify the bounding box
  glm::vec3 min, max;
};

inline Bounds3 BoundsUnion(const Bounds3 &box1, const Bounds3 &box2) {
  Bounds3 ret;
  ret.min = glm::min(box1.min, box2.min);
  ret.max = glm::max(box1.max, box2.max);
  return ret;
}

inline Bounds3 BoundsUnion(const glm::vec3 &point, const Bounds3 &box) {
  Bounds3 ret;
  ret.min = glm::min(box.min, point);
  ret.max = glm::max(box.max, point);
  return ret;
}
} // namespace SoftRasterizer

#endif //_BOUNDS3_HPP_
