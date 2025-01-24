#pragma once
#ifndef _INTERSECTION_HPP_
#define _INTERSECTION_HPP_
#include <glm/glm.hpp>

namespace SoftRasterizer {

/*declaration*/
struct Object;
struct Material;

struct Intersection {
          Intersection()
                    : intersected(false), coords(glm::vec3(0.f)), normal(glm::vec3(0.f)), index(0),
                    intersect_time(std::numeric_limits<float>::max()), obj(nullptr), material(nullptr) {
          }

  std::size_t index;
  bool intersected;
  float intersect_time;
  glm::vec3 coords;
  glm::vec3 normal;
  Object *obj;
  std::shared_ptr<Material> material;
};
} // namespace SoftRasterizer

#endif //_INTERSECTION_HPP_
