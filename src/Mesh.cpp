#include <Tools.hpp>
#include <object/Mesh.hpp>
#include <object/Triangle.hpp>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

SoftRasterizer::Mesh::Mesh(const std::string &name,
                           const SoftRasterizer::Material &_material,
                           const std::vector<Vertex> &_vertices,
                           const std::vector<glm::uvec3> &_faces,
                           const Bounds3 &box)
    : meshname(name), vertices(_vertices), faces(_faces), bounding_box(box),
      Object(std::make_shared<Material>(_material), nullptr) {

  /*Generating Triangles*/
  generateTriangles();

  /*Allocate BVH Structure*/
  preGenerateBVH();

  /*Generating BVH Structure*/
  buildBVHAccel();
}

SoftRasterizer::Mesh::Mesh(const std::string &name,
                           SoftRasterizer::Material &&_material,
                           std::vector<Vertex> &&_vertices,
                           std::vector<glm::uvec3> &&_faces, Bounds3 &&box)
    : meshname(name),
      Object(std::make_shared<Material>(std::move(_material)), nullptr),
      vertices(std::move(_vertices)), faces(std::move(_faces)),
      bounding_box(std::move(box)) {

  /*Generating Triangles*/
  generateTriangles();

  /*Allocate BVH Structure*/
  preGenerateBVH();

  /*Generating BVH Structure*/
  buildBVHAccel();
}

SoftRasterizer::Mesh::~Mesh() { m_bvh->clearBVHAccel(); }

SoftRasterizer::Intersection SoftRasterizer::Mesh::getIntersect(Ray &ray) {
  if (m_bvh == nullptr)
    return {};
  return m_bvh->getIntersection(ray);
}

std::tuple<SoftRasterizer::Intersection, float> SoftRasterizer::Mesh::sample() {
  if (m_bvh == nullptr)
    return {};
  return m_bvh->sample();
}

const float SoftRasterizer::Mesh::getArea() {
  return tbb::parallel_reduce(
      tbb::blocked_range<std::size_t>(0, m_triangles.size()), 0.f,
      [&](const tbb::blocked_range<std::size_t> &r, float sum) {
        for (std::size_t index = r.begin(); index < r.end(); ++index) {

          /*Update Triangle*/
          sum += m_triangles[index]->getArea();
        }
        return sum;
      },
      [](const float a, const float b) -> float { return a + b; });
}

void SoftRasterizer::Mesh::updatePosition(const glm::mat4x4 &NDC_MVP,
                                          const glm::mat4x4 &Normal_M) {

  tbb::parallel_for(tbb::blocked_range<long long>(0, m_triangles.size()),
                    [&](const tbb::blocked_range<long long> &r) {
                      for (long long index = r.begin(); index < r.end();
                           ++index) {

                        /*Update Triangle and Generate New Bounds3 Struct*/
                        m_triangles[index]->updatePosition(NDC_MVP, Normal_M);
                      }
                    });

  rebuildBVHAccel();
}

void SoftRasterizer::Mesh::bindShader2Mesh(std::shared_ptr<Shader> shader) {
  m_shader.reset();
  m_shader = shader;

  tbb::parallel_for(std::size_t(0), m_triangles.size(), [&](std::size_t i) {
    m_triangles[i]->bindShader2Mesh(m_shader);
  });
}

void SoftRasterizer::Mesh::setMaterial(std::shared_ptr<Material> material) {
  /*Change Mesh's Material*/
  m_material.reset();
  m_material = material;

  /*Change Triangles Material*/
  tbb::parallel_for(std::size_t(0), m_triangles.size(), [&](std::size_t i) {
    m_triangles[i]->setMaterial(material);
  });
}

/*Generating Triangles*/
void SoftRasterizer::Mesh::generateTriangles() {
  m_triangles.resize(faces.size());

  tbb::parallel_for(std::size_t(0), faces.size(), [&](std::size_t i) {
    // Polymorphism
    std::shared_ptr<Object> tri = std::make_shared<Triangle>(
                      m_material,
                      vertices[faces[i].x].position, vertices[faces[i].y].position, vertices[faces[i].z].position,
                      vertices[faces[i].x].normal, vertices[faces[i].y].normal, vertices[faces[i].z].normal,
                      vertices[faces[i].x].texCoord, vertices[faces[i].y].texCoord, vertices[faces[i].z].texCoord
                      /*, vertices[faces[i].x].color, vertices[faces[i].y].color, vertices[faces[i].z].color*/);

    /*Set triangle's index*/
    tri->index = i;
    m_triangles[i] = tri;
  });
}

void SoftRasterizer::Mesh::preGenerateBVH() {
  m_bvh.reset();
  m_bvh = std::make_unique<BVHAcceleration>(m_triangles);
}

/*Generating BVH Structure*/
void SoftRasterizer::Mesh::buildBVHAccel() {

  try {
    m_bvh->clearBVHAccel();
    m_bvh->startBuilding();
    bounding_box = m_bvh->getBoundingBox().value();
  } catch (const std::exception &e) {
    spdlog::error("BoundingBox of Mesh {} Error!", meshname);
  }
}

/*Rebuild BVH Structure*/
void SoftRasterizer::Mesh::rebuildBVHAccel() { buildBVHAccel(); }

std::optional<SoftRasterizer::Bounds3>
SoftRasterizer::Mesh::getBoundingBox() const {
  return m_bvh->getBoundingBox();
}
