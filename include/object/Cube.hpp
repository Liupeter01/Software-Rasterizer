#pragma once
#ifndef _CUBE_HPP_
#define _CUBE_HPP_
#include <object/Object.hpp>

namespace  SoftRasterizer {
          class Cube :public Object {
          public:
                    Cube();
                    virtual ~Cube();

          public:
                    Bounds3 getBounds() override;
                    [[nodiscard]] bool intersect(const Ray& ray)override;
                    [[nodiscard]] bool intersect(const Ray& ray, float& tNear)override;
                    [[nodiscard]] Intersection getIntersect(Ray&) override;
                    [[nodiscard]] glm::vec3 getDiffuseColor(const glm::vec2& uv) override;
                    [[nodiscard]] Properties getSurfaceProperties(const std::size_t faceIndex,
                              const glm::vec3& Point,
                              const glm::vec3& viewDir,
                              const glm::vec2& uv) override;

                    [[nodiscard]] std::shared_ptr<Material>& getMaterial() override;

                    /*Compatible Consideration!*/
                    [[nodiscard]] const std::vector<Vertex>& getVertices() const override;
                    [[nodiscard]] const std::vector<glm::uvec3>& getFaces() const override;

                    /*Perform (NDC) MVP Calculation*/
                    [[nodiscard]] void updatePosition(const glm::mat4x4& NDC_MVP,
                              const glm::mat4x4& Normal_M) override;

          private:
                    std::shared_ptr<Material> material;
                    std::vector<SoftRasterizer::Vertex> vert;
                    std::vector<glm::uvec3> faces;
          };
}

#endif //_CUBE_HPP_