#pragma once
#ifndef _OBJLOADER_HPP_
#define _OBJLOADER_HPP_
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <object/Mesh.hpp>
#include <optional>
#include <string>
#include <vector>

namespace SoftRasterizer {
class ObjLoader {
public:
  ObjLoader(const std::string &path, const std::string &meshName,
            const glm::mat4x4 &model = glm::mat4(1.0f));

  ObjLoader(const std::string &path, const std::string &meshName,
            const glm::vec3 &axis, const float angle,
            const glm::vec3 &translation, const glm::vec3 &scale);

  virtual ~ObjLoader();

public:
  void setObjFilePath(const std::string &path);
  void updateModelMatrix(const glm::vec3 &axis, const float angle,
                         const glm::vec3 &translation, const glm::vec3 &scale);

  const glm::mat4x4 &getModelMatrix();

  std::optional<std::unique_ptr<Mesh>>
  startLoadingFromFile(
            const glm::mat4x4& model,
            const glm::mat4x4& view,
            const glm::mat4x4& projection,
            const glm::mat4x4& ndc, 
            const std::string &objName = "",
            MaterialType _type = MaterialType::REFLECTION_AND_REFRACTION,
            const glm::vec3& _color = glm::vec3(1.0f),
            const glm::vec3& _Ka = glm::vec3(0.0f),
            const glm::vec3& _Kd = glm::vec3(0.0f),
            const glm::vec3& _Ks = glm::vec3(0.0f),
            const float _specularExponent = 0.0f,
            const float _ior = 0.f);

private:
  std::string m_path;
  std::string m_meshName;
  glm::mat4 m_model;
};
} // namespace SoftRasterizer
#endif // !_OBJLOADER_HPP_
