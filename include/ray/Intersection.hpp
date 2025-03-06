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
      : intersected(false), coords(glm::vec3(0.f)), normal(glm::vec3(0.f)),
        uv(glm::vec2(0.f)), index(0),
        intersect_time(std::numeric_limits<double>::max()), color(1.0f),
        emit(glm::vec3(0.f)), obj(nullptr), material(nullptr) {}

  std::size_t index;
  bool intersected;
  double intersect_time;
  glm::vec3 coords;
  glm::vec3 normal;
  glm::vec3 color;
  glm::vec2 uv;
  glm::vec3 emit;
  Object *obj;
  std::shared_ptr<Material> material;
};
} // namespace SoftRasterizer

#endif //_INTERSECTION_HPP_
