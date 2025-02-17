#pragma once
#ifndef _AREALIGHT_HPP_
#define _AREALIGHT_HPP_
#include <light/Light.hpp>

namespace SoftRasterizer {
          class AreaLight : public light_struct {
          public:
                    AreaLight(const glm::vec3& pos, const glm::vec3& intense);

          public:
                    glm::vec3 samplePoint() const;

          private:
                    float length;
                    glm::vec3 normal;
                    glm::vec3 u;
                    glm::vec3 v;
          };
}

#endif //_AREALIGHT_HPP_