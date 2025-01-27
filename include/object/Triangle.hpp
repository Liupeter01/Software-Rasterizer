#pragma once
#ifndef _TRIANGLE_HPP_
#define _TRIANGLE_HPP_
#include <array>
#include <bvh/Bounds3.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <initializer_list>
#include <object/Material.hpp>
#include <object/Object.hpp>

namespace SoftRasterizer {

enum class FaceNormalType {
  PerGeometry,     // calculate by using cross product
  InterpolatedFace // calculate by using barycentric coordinates
};

struct alignas(32) Triangle : public Object {
  Triangle();
  Triangle(const glm::vec3 &VertexA, const glm::vec3& VertexB, const glm::vec3& VertexC,
                const glm::vec3& NormalA, const glm::vec3& NormalB, const glm::vec3& NormalC,
            const glm::vec2& texCoordA, const glm::vec2& texCoordB, const glm::vec2& texCoordC,
            const glm::vec3& colorA= glm::vec3(1.0f), const glm::vec3& colorB = glm::vec3(1.0f), const glm::vec3& colorC = glm::vec3(1.0f));

  constexpr static float zero_point_3 = 0.3333333f;

  const glm::vec3 &a() const { return m_vertex[0]; }
  const glm::vec3 &b() const { return m_vertex[1]; }
  const glm::vec3 &c() const { return m_vertex[2]; }

  void setVertex(std::initializer_list<glm::vec3> _vertex);
  void setNormal(std::initializer_list<glm::vec3> _normal);
  void setColor(std::initializer_list<glm::vec3> _color);
  void setTexCoord(std::initializer_list<glm::vec2> _texCoords);

  Bounds3 getBounds() override;
  [[nodiscard]] bool intersect(const Ray &ray) override { return true; }
  [[nodiscard]] bool intersect(const Ray &ray, float &tNear) override { return true; }

  // Moller Trumbore Algorithm
  [[nodiscard]] static bool
  rayTriangleIntersect(const Ray &ray, const glm::vec3 &v0, const glm::vec3 &v1,
                       const glm::vec3 &v2, float &tNear, float &u, float &v);

  // Moller Trumbore Algorithm
  [[nodiscard]] Intersection getIntersect(Ray &ray) override;
  [[nodiscard]] Properties getSurfaceProperties(const std::size_t faceIndex,
                                                const glm::vec3 &Point,
                                                const glm::vec3 &viewDir,
                                                const glm::vec2 &uv) override;

  [[nodiscard]] glm::vec3 
  getFaceNormal(FaceNormalType type = FaceNormalType::PerGeometry) const;
  [[nodiscard]] std::shared_ptr<Material>& getMaterial() override { return m_material; }
  [[nodiscard]] glm::vec3 getDiffuseColor(const glm::vec2 &uv) override;

  /*Compatible Consideration!*/
  [[nodiscard]] const std::vector<Vertex>& getVertices() const override { return vert; }
  [[nodiscard]] const std::vector<glm::uvec3>& getFaces() const override { return faces; }

  void updatePosition(const glm::mat4x4& NDC_MVP,
            const glm::mat4x4& Normal_M) override;

  void calcBoundingBox(const std::size_t width, const std::size_t height);

  /*
  * original coord of the triangle, v0, v1, v2 in counter clockwise order
  * Those are original arguments, they should not be changed!!!!!!
   */
  std::array<glm::vec3, 3> m_vertex;
  std::array<glm::vec3, 3> m_color; // Color for each vertex
  // texture u,v coordinates for each vertex
  std::array<glm::vec2, 3> m_texCoords;
  std::array<glm::vec3, 3> m_normal; // normal vector for each vertex

  /*All the calculations should be done using vert!!!*/
  std::vector<SoftRasterizer::Vertex> vert;
  std::vector<glm::uvec3> faces;

  std::shared_ptr<Material> m_material;

  /*BoundBox Calculation!*/
  struct BoundingBox {
    BoundingBox() : startY(0), startX(0), endX(0), endY(0) {}
    long long startX;
    long long startY;
    long long endX;
    long long endY;
  };
  BoundingBox box;
};
} // namespace SoftRasterizer

#endif //_TRIANGLE_HPP_
