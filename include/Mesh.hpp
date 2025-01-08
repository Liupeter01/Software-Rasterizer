
#pragma once
#ifndef _MESH_HPP_
#define _MESH_HPP_
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <functional> // For std::hash
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
  Eigen::Vector3f Ka;   // Ambient Color
  Eigen::Vector3f Kd;   // Diffuse Color
  Eigen::Vector3f Ks;   // Specular Color
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
  Eigen::Vector3f position;
  Eigen::Vector3f normal;
  Eigen::Vector2f texCoord;
  Eigen::Vector3f color = Eigen::Vector3f(1.0f, 1.0f, 1.0f);

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
       const std::vector<Eigen::Vector3i> &_faces)
      : meshname(name), MeshMaterial(_material), vertices(_vertices),
        faces(_faces), m_shader(nullptr) {}

  Mesh(const std::string &name, SoftRasterizer::Material &&_material,
       std::vector<Vertex> &&_vertices, std::vector<Eigen::Vector3i> &&_faces)
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
  std::vector<Eigen::Vector3i> faces;

  // Material
  Material MeshMaterial;

  std::shared_ptr<Shader> m_shader;
};
} // namespace SoftRasterizer

// Hash function for Eigen types (Vector3f, Vector2f)
namespace std {
template <> struct std::hash<Eigen::Vector3f> {
  size_t operator()(const Eigen::Vector3f &v) const {
    size_t seed = 0;
    SoftRasterizer::hash_combine_impl(
        seed, std::hash<float>()(v.x())); // Combine x component
    SoftRasterizer::hash_combine_impl(
        seed, std::hash<float>()(v.y())); // Combine y component
    SoftRasterizer::hash_combine_impl(
        seed, std::hash<float>()(v.z())); // Combine z component
    return seed;
  }
};

template <> struct std::hash<Eigen::Vector2f> {
  size_t operator()(const Eigen::Vector2f &v) const {
    size_t seed = 0;
    SoftRasterizer::hash_combine_impl(
        seed, std::hash<float>()(v.x())); // Combine x component
    SoftRasterizer::hash_combine_impl(
        seed, std::hash<float>()(v.y())); // Combine y component
    return seed;
  }
};

template <> struct std::hash<SoftRasterizer::Vertex> {
  size_t operator()(const SoftRasterizer::Vertex &vertex) const {
    // Combine the hashes of position, color, normal, and texCoord
    size_t h1 = std::hash<Eigen::Vector3f>()(vertex.position);
    size_t h2 = std::hash<Eigen::Vector3f>()(vertex.color);
    size_t h3 = std::hash<Eigen::Vector3f>()(vertex.normal);
    size_t h4 = std::hash<Eigen::Vector2f>()(vertex.texCoord);

    size_t seed = 0;
    SoftRasterizer::hash_combine_impl(seed, h1);
    SoftRasterizer::hash_combine_impl(seed, h2);
    SoftRasterizer::hash_combine_impl(seed, h3);
    SoftRasterizer::hash_combine_impl(seed, h4);

    // Combine all hashes using bit manipulation
    return seed;
  }
};
} // namespace std

#endif // !_MESH_HPP_
