#include "object/Sphere.hpp"
#include <light/SphereLight.hpp>

SoftRasterizer::SphereLight::SphereLight() : Sphere() {}

SoftRasterizer::SphereLight::SphereLight(const glm::vec3 &_center,
                                         const glm::vec3 &intense,
                                         const float _radius)
    : intensity(intense), Sphere(_center, _radius) {}

SoftRasterizer::SphereLight::~SphereLight() {}

const glm::vec3 &SoftRasterizer::SphereLight::getIntensity() const {
  return intensity;
}
