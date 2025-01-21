#pragma once
#ifndef _RAY_HPP_
#define _RAY_HPP_
#include <glm/glm.hpp>
struct Ray {
          Ray(const glm::vec3& _origin, const glm::vec3& _direction, const double time = 0.0)
                    :origin(_origin),direction(_direction),transport_time(time)
          {
          }

          double transport_time;
          glm::vec3 origin;             //ray source
          glm::vec3 direction;          //ray direction
};

#endif  //_RAY_HPP_