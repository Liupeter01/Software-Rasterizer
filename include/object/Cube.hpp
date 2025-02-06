#pragma once
#ifndef _CUBE_HPP_
#define _CUBE_HPP_
#include <object/Object.hpp>

namespace SoftRasterizer {
class Cube : public Object {
public:
  Cube();
  virtual ~Cube();

public:
  Bounds3 getBounds() override;
  [[nodiscard]] bool intersect(const Ray &ray) override;
  [[nodiscard]] bool intersect(const Ray &ray, float &tNear) override;
  [[nodiscard]] Intersection getIntersect(Ray &) override;
  [[nodiscard]] glm::vec3 getDiffuseColor(const glm::vec2 &uv) override;
  [[nodiscard]] Properties getSurfaceProperties(const std::size_t faceIndex,
                                                const glm::vec3 &Point,
                                                const glm::vec3 &viewDir,
                                                const glm::vec2 &uv) override;

  [[nodiscard]] std::shared_ptr<Material> &getMaterial() override {
    return m_material;
  }

  /*Compatible Consideration!*/
  [[nodiscard]] const std::vector<Vertex> &getVertices() const override {
    return vert;
  }
  [[nodiscard]] const std::vector<glm::uvec3> &getFaces() const override {
    return faces;
  }

  /*Perform (NDC) MVP Calculation*/
  [[nodiscard]] void updatePosition(const glm::mat4x4 &NDC_MVP,
                                    const glm::mat4x4 &Normal_M) override;

  void bindShader2Mesh(std::shared_ptr<Shader> shader) override;

  [[nodiscard]] std::tuple<Intersection, float> sample() override;
  [[nodiscard]] const float getArea() override;

private:
          float area = 0.f;
  std::vector<SoftRasterizer::Vertex> vert;
  std::vector<glm::uvec3> faces;
};
} // namespace SoftRasterizer

#endif //_CUBE_HPP_
