#include <Tools.hpp>
#include <light/AreaLight.hpp>

SoftRasterizer::AreaLight::AreaLight(const glm::vec3& pos,
                                                              const glm::vec3& intense)
          :normal(glm::vec3(0.f, -1.f, 0.f))
          , u(glm::vec3(1.f, 0.f, 0.f))
          , v(glm::vec3(0.f, 0.f, 1.f))
          , length(100.f)
          , light_struct(pos, intense) {
}

glm::vec3 SoftRasterizer::AreaLight::samplePoint() const {
          auto rand_u = Tools::random_generator();
          auto rand_v = Tools::random_generator();
          return position + rand_u * u + rand_v * v;
}