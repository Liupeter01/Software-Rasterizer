#pragma once
#ifndef _LIGHT_HPP_
#define _LIGHT_HPP_
#include <glm/glm.hpp>

namespace SoftRasterizer {
          struct light_struct {
                    light_struct() :light_struct(glm::vec3(0.f), glm::vec3(0.f)) {}
                    light_struct(const glm::vec3& pos, const glm::vec3& intense)
                              : position(pos), intensity(intense)
                    {}

                    glm::vec3 position;
                    glm::vec3 intensity;
          };
}

#endif //_LIGHT_HPP_