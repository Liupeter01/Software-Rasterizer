#pragma once
#ifndef _INTERSECTION_HPP_
#define _INTERSECTION_HPP_
#include <glm/glm.hpp>

namespace SoftRasterizer {

/*declaration*/
struct Object;

struct Intersection {
  Intersection()
      : intersected(false), coords(glm::vec3(0.f)), normal(glm::vec3(0.f)),
        intersect_time(std::numeric_limits<float>::max()), obj(nullptr) {}

  bool intersected;
  float intersect_time;
  glm::vec3 coords;
  glm::vec3 normal;
  Object *obj;
};
} // namespace SoftRasterizer

#endif //_INTERSECTION_HPP_
