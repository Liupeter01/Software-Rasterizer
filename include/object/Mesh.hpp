#pragma once
#ifndef _MESH_HPP_
#define _MESH_HPP_
#define GLM_ENABLE_EXPERIMENTAL // Enable experimental features
#include <bvh/Bounds3.hpp>
#include <functional> // For std::hash
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <memory>
#include <object/Material.hpp>
#include <object/Object.hpp>
#include <string>

namespace SoftRasterizer {

/*forward declaration*/
class Shader;

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 texCoord;
  glm::vec3 color = glm::vec3(1.0f);

  bool operator==(const Vertex &other) const {
    return this->position == other.position && this->color == other.color &&
           this->normal == other.normal && this->texCoord == other.texCoord;
  }
};

struct Mesh : public Object {
public:
  Mesh() : Mesh("") {}
  Mesh(const std::string &name) : meshname(name), m_shader(nullptr) {}
  Mesh(const std::string &name, const SoftRasterizer::Material &_material,
       const std::vector<Vertex> &_vertices,
       const std::vector<glm::uvec3> &_faces, const Bounds3 &box)
      : meshname(name), MeshMaterial(_material), vertices(_vertices),
        faces(_faces), bounding_box(box), m_shader(nullptr) {
    generateBVHAccel();
  }

  Mesh(const std::string &name, SoftRasterizer::Material &&_material,
       std::vector<Vertex> &&_vertices, std::vector<glm::uvec3> &&_faces,
       Bounds3 &&box)
      : meshname(name), MeshMaterial(std::move(_material)),
        vertices(std::move(_vertices)), faces(std::move(_faces)),
        bounding_box(std::move(box)), m_shader(nullptr) {
    generateBVHAccel();
  }

  void bindShader2Mesh(std::shared_ptr<Shader> shader) {
    /*bind shader2 mesh without dtor,  the life od this pointer is maintained by
     * render class*/
    m_shader.reset();
    m_shader = shader;
  }

public:
  void updateBounds(const Bounds3 &new_box) { bounding_box = new_box; }
  Bounds3 getBounds() { return bounding_box; }

private:
  /*Generating BVH Structure For First Time Use*/
  void generateBVHAccel() {}

public:
  // Mesh Name
  std::string meshname;
  std::vector<Vertex> vertices;
  std::vector<glm::uvec3> faces;

  // Material
  Material MeshMaterial;

  // Bounding Box
  Bounds3 bounding_box;

  // Shading structure
  std::shared_ptr<Shader> m_shader;
};
} // namespace SoftRasterizer

/*copy from boost*/
template <typename SizeT>
inline void hash_combine_impl(SizeT &seed, SizeT value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hash function for Vertex
namespace std {
template <> struct std::hash<SoftRasterizer::Vertex> {
  size_t operator()(const SoftRasterizer::Vertex &vertex) const {
    // Combine the hashes of position, color, normal, and texCoord
    size_t seed = 0;
    hash<glm::vec3> vec3Hasher;
    hash<glm::vec2> vec2Hasher;

    hash_combine_impl(seed, vec3Hasher(vertex.position));
    hash_combine_impl(seed, vec3Hasher(vertex.normal));
    hash_combine_impl(seed, vec3Hasher(vertex.color));
    hash_combine_impl(seed, vec2Hasher(vertex.texCoord));

    // Combine all hashes using bit manipulation
    return seed;
  }
};
} // namespace std

#endif // !_MESH_HPP_
