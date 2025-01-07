#pragma once
#ifndef _OBJLOADER_HPP_
#define _OBJLOADER_HPP_
#include <Mesh.hpp>
#include <optional>
#include <string>
#include <vector>

namespace SoftRasterizer {
class ObjLoader {
public:
  ObjLoader(const std::string &path, const std::string &meshName,
            Eigen::Matrix4f &&model = Eigen::Matrix4f::Identity());

  ObjLoader(const std::string &path, const std::string &meshName,
            const Eigen::Matrix4f &rotation,
            const Eigen::Vector3f &translation = Eigen::Vector3f(0.f, 0.f, 0.f),
            const Eigen::Vector3f &scale = Eigen::Matrix4f::Identity());

  ObjLoader(const std::string &path, const std::string &meshName,
            const Eigen::Vector3f &axis, const float angle,
            const Eigen::Vector3f &translation = Eigen::Vector3f(0.f, 0.f, 0.f),
            const Eigen::Vector3f &scale = Eigen::Matrix4f::Identity());

  virtual ~ObjLoader();

public:
  void setObjFilePath(const std::string &path);
  const Eigen::Matrix4f &getModelMatrix();

  std::optional<std::unique_ptr<Mesh>>
  startLoadingFromFile(const std::string &objName = "");

private:
  std::string m_path;
  std::string m_meshName;
  Eigen::Matrix4f m_model;
};
} // namespace SoftRasterizer
#endif // !_OBJLOADER_HPP_
