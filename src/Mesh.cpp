#include <object/Mesh.hpp>

SoftRasterizer::Mesh::Mesh() : Mesh("") {}

SoftRasterizer::Mesh::Mesh(const std::string &name)
    : meshname(name), m_shader(nullptr) {}

SoftRasterizer::Mesh::Mesh(const std::string &name,
                           const SoftRasterizer::Material &_material,
                           const std::vector<Vertex> &_vertices,
                           const std::vector<glm::uvec3> &_faces,
                           const Bounds3 &box)
    : meshname(name), MeshMaterial(_material), vertices(_vertices),
      faces(_faces), bounding_box(box), m_shader(nullptr) {}

SoftRasterizer::Mesh::Mesh(const std::string &name,
                           SoftRasterizer::Material &&_material,
                           std::vector<Vertex> &&_vertices,
                           std::vector<glm::uvec3> &&_faces, Bounds3 &&box)
    : meshname(name), MeshMaterial(std::move(_material)),
      vertices(std::move(_vertices)), faces(std::move(_faces)),
      bounding_box(std::move(box)), m_shader(nullptr) {}

void SoftRasterizer::Mesh::bindShader2Mesh(std::shared_ptr<Shader> shader) {
  /*bind shader2 mesh without dtor,  the life od this pointer is maintained by
   * render class*/
  m_shader.reset();
  m_shader = shader;
}

bool SoftRasterizer::Mesh::intersect(const Ray &ray) { return true; }

bool SoftRasterizer::Mesh::intersect(const Ray &ray, float &tNear) {
  return false;
}

SoftRasterizer::Intersection SoftRasterizer::Mesh::getIntersect(Ray &ray) {
  return {};
}
