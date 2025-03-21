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
  startLoadingFromFile(const std::string &objName);

private:
  std::string m_path;
  std::string m_meshName;
  glm::mat4 m_model;
};
} // namespace SoftRasterizer
#endif // !_OBJLOADER_HPP_
