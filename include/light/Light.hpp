#pragma once
#ifndef _LIGHT_HPP_
#define _LIGHT_HPP_
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace SoftRasterizer {
struct light_struct {
  light_struct() : light_struct(glm::vec3(0.f), glm::vec3(0.f), glm::vec3(0, 1, 0)) 
  {}

  light_struct(const glm::vec3& pos, const glm::vec3& intense)
            : light_struct(pos, intense, glm::vec3(0, 1, 0))
  {}

  light_struct(const glm::vec3& pos, const glm::vec3& intense, 
            const glm::vec3& _axis = glm::vec3(0, 1.0f, 0),
            const float _angle = 0.f,
            const glm::vec3& _translation = glm::vec3(0.0f),
            const glm::vec3& _scale = glm::vec3(1.0f))
            : position(pos), intensity(intense), axis(_axis), angle(_angle), translation(_translation), scale(_scale)
  {}

  glm::vec3 position;
  glm::vec3 intensity;

  void updateModelMatrix() {
            auto T = glm::translate(glm::mat4(1.0f), translation);
            auto R = glm::rotate(glm::mat4(1.0f), glm::radians(angle), axis);
            auto S = glm::scale(glm::mat4(1.0f), scale);
            m_model = T * R * S;
  }

  const glm::mat4x4& getModelMatrix() {
            updateModelMatrix();
            return m_model;
  }

protected:
          //Model Matrix Arguments
          glm::vec3 axis;
          float angle; 
          glm::vec3 translation;
          glm::vec3 scale;
          glm::mat4x4 m_model;
};
} // namespace SoftRasterizer

#endif //_LIGHT_HPP_
