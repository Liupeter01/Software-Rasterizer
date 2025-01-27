#include <object/Mesh.hpp>
#include <object/Triangle.hpp>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>
#include <Tools.hpp>

SoftRasterizer::Mesh::Mesh(const std::string &name,
                           const SoftRasterizer::Material &_material,
                           const std::vector<Vertex> &_vertices,
                           const std::vector<glm::uvec3> &_faces,
                           const Bounds3 &box)
    : meshname(name), MeshMaterial(std::make_shared<Material>(_material)),
      vertices(_vertices), faces(_faces), bounding_box(box) {

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
    : meshname(name), MeshMaterial(std::make_shared<Material>(_material)),
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

bool SoftRasterizer::Mesh::intersect(const Ray &ray, float &tNear) {
  bool status = false;
  float temp = std::numeric_limits<float>::infinity();
  std::for_each(faces.begin(), faces.end(), [&](auto &obj) {
    const glm::vec3 &v0 = vertices[obj.x].position;
    const glm::vec3 &v1 = vertices[obj.y].position;
    const glm::vec3 &v2 = vertices[obj.z].position;
    float t, u, v;
    if (Triangle::rayTriangleIntersect(ray, v0, v1, v2, t, u, v)) {
      if (t < temp) {
        temp = t;
        status = true;
      }
    }
  });

  tNear = temp;
  return tNear > 0 ? status : false;
}

SoftRasterizer::Intersection SoftRasterizer::Mesh::getIntersect(Ray &ray) {
  if (m_bvh == nullptr)
    return {};
  return m_bvh->getIntersection(ray);
}

SoftRasterizer::Object::Properties SoftRasterizer::Mesh::getSurfaceProperties(
    const std::size_t faceIndex, const glm::vec3 &Point,
    const glm::vec3 &viewDir, const glm::vec2 &uv) {

  Properties ret;

  /*Calculate Face Normal*/
  auto A = vertices[faces[faceIndex].x].position;
  auto B = vertices[faces[faceIndex].y].position;
  auto C = vertices[faces[faceIndex].z].position;
  ret.normal = glm::normalize(glm::cross(B - A, C - A));

  /*Calculate Texture Coord*/
  auto texA = vertices[faces[faceIndex].x].texCoord;
  auto texB = vertices[faces[faceIndex].y].texCoord;
  auto texC = vertices[faces[faceIndex].z].texCoord;
  ret.uv = texA * (1 - uv.x - uv.y) + texB * uv.x + texC * uv.y;
  return ret;
}

std::shared_ptr<SoftRasterizer::Material> &
SoftRasterizer::Mesh::getMaterial() {
          return MeshMaterial;
}

glm::vec3 SoftRasterizer::Mesh::getDiffuseColor(const glm::vec2 &uv) {
  return MeshMaterial->color;
}

const std::vector<SoftRasterizer::Vertex>& SoftRasterizer::Mesh::getVertices() const{
          return vertices;
}

const std::vector<glm::uvec3>& SoftRasterizer::Mesh::getFaces() const {
          return faces;
}

void SoftRasterizer::Mesh::updatePosition(const glm::mat4x4& NDC_MVP,
          const glm::mat4x4& Normal_M) {

          tbb::parallel_for(
                    tbb::blocked_range<long long>(0, m_triangles.size()),
                    [&](const tbb::blocked_range<long long>& r) {
                              for (long long index = r.begin(); index < r.end(); ++index) {

                                        /*Update Triangle and Generate New Bounds3 Struct*/
                                        m_triangles[index]->updatePosition(NDC_MVP, Normal_M);
                              }
                    });

          rebuildBVHAccel();

}

/*Generating Triangles*/
void SoftRasterizer::Mesh::generateTriangles() {
  m_triangles.resize(faces.size());

  tbb::parallel_for(std::size_t(0), faces.size(), [&](std::size_t i) {

            //Polymorphism
            std::shared_ptr<Object> tri = std::make_shared<Triangle>(
                      vertices[faces[i].x].position, vertices[faces[i].y].position, vertices[faces[i].z].position,
                      vertices[faces[i].x].normal, vertices[faces[i].y].normal, vertices[faces[i].z].normal,
                      vertices[faces[i].x].texCoord, vertices[faces[i].y].texCoord, vertices[faces[i].z].texCoord,
                      vertices[faces[i].x].color, vertices[faces[i].y].color, vertices[faces[i].z].color);

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
