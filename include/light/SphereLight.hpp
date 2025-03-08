#pragma once
#ifndef _SPHERELIGHT_HPP_
#define _SPHERELIGHT_HPP_
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <object/Sphere.hpp>

namespace SoftRasterizer {
class SphereLight : public Sphere {
public:
  SphereLight();
  SphereLight(const glm::vec3 &_center,
              const glm::vec3 &intense = glm::vec3(1.f),
              const float _radius = 1.f);

  virtual ~SphereLight();

public:
  [[nodiscard]] const glm::vec3 &getIntensity() const;

protected:
  glm::vec3 intensity;
};
} // namespace SoftRasterizer

#endif //_SPHERELIGHT_HPP_
