#pragma once
#ifndef _MESH_HPP_
#define _MESH_HPP_
#define GLM_ENABLE_EXPERIMENTAL // Enable experimental features
#include <functional>           // For std::hash
#include <glm/gtx/hash.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <memory>
#include <string>

namespace SoftRasterizer {

/*forward declaration*/
class Shader;

/*copy from boost*/
template <typename SizeT>
inline void hash_combine_impl(SizeT &seed, SizeT value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct Material {
  Material() {
    Ns = 0.0f;
    Ni = 0.0f;
    d = 0.0f;
    illum = 0;
  }

  std::string name;     // Material Name
  glm::vec3 Ka;         // Ambient Color
  glm::vec3 Kd;         // Diffuse Color
  glm::vec3 Ks;         // Specular Color
  float Ns;             // Specular Exponent
  float Ni;             // Optical Density
  float d;              // Dissolve
  int illum;            // Illumination
  std::string map_Ka;   // Ambient Texture Map
  std::string map_Kd;   // Diffuse Texture Map
  std::string map_Ks;   // Specular Texture Map
  std::string map_Ns;   // Specular Hightlight Map
  std::string map_d;    // Alpha Texture Map
  std::string map_bump; // Bump Map
};

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

struct Mesh {
  Mesh() : Mesh("") {}
  Mesh(const std::string &name) : meshname(name), m_shader(nullptr) {}
  Mesh(const std::string &name, const SoftRasterizer::Material &_material,
       const std::vector<Vertex> &_vertices,
       const std::vector<glm::uvec3> &_faces)
      : meshname(name), MeshMaterial(_material), vertices(_vertices),
        faces(_faces), m_shader(nullptr) {}

  Mesh(const std::string &name, SoftRasterizer::Material &&_material,
       std::vector<Vertex> &&_vertices, std::vector<glm::uvec3> &&_faces)
      : meshname(name), MeshMaterial(std::move(_material)),
        vertices(std::move(_vertices)), faces(std::move(_faces)),
        m_shader(nullptr) {}

  void bindShader2Mesh(std::shared_ptr<Shader> shader) {
    /*bind shader2 mesh without dtor,  the life od this pointer is maintained by
     * render class*/
    m_shader.reset();
    m_shader = shader;
  }

  // Mesh Name
  std::string meshname;
  std::vector<Vertex> vertices;
  std::vector<glm::uvec3> faces;

  // Material
  Material MeshMaterial;

  std::shared_ptr<Shader> m_shader;
};
} // namespace SoftRasterizer

// Hash function for Eigen types (Vector3f, Vector2f)
namespace std {
template <> struct std::hash<SoftRasterizer::Vertex> {
  size_t operator()(const SoftRasterizer::Vertex &vertex) const {
    // Combine the hashes of position, color, normal, and texCoord
    size_t seed = 0;
    hash<glm::vec3> vec3Hasher;
    hash<glm::vec2> vec2Hasher;

    SoftRasterizer::hash_combine_impl(seed, vec3Hasher(vertex.position));
    SoftRasterizer::hash_combine_impl(seed, vec3Hasher(vertex.normal));
    SoftRasterizer::hash_combine_impl(seed, vec3Hasher(vertex.color));
    SoftRasterizer::hash_combine_impl(seed, vec2Hasher(vertex.texCoord));

    // Combine all hashes using bit manipulation
    return seed;
  }
};
} // namespace std

#endif // !_MESH_HPP_
