#include <object/Mesh.hpp>
#include <object/Triangle.hpp>
#include <tbb/parallel_for.h>
#include <spdlog/spdlog.h>

SoftRasterizer::Mesh::Mesh() : Mesh("") {}

SoftRasterizer::Mesh::Mesh(const std::string &name)
    : meshname(name), m_shader(nullptr) , MeshMaterial(std::make_shared<Material>()){}

SoftRasterizer::Mesh::Mesh(const std::string &name,
                           const SoftRasterizer::Material &_material,
                           const std::vector<Vertex> &_vertices,
                           const std::vector<glm::uvec3> &_faces,
                           const Bounds3 &box)
    : meshname(name), MeshMaterial(std::make_shared<Material>(_material)), vertices(_vertices),
      faces(_faces), bounding_box(box), m_shader(nullptr),m_bvh(std::make_unique<BVHAcceleration>()) {

          /*Generating Triangles*/
          generateTriangles();

          /*Generating BVH Structure*/
          buildBVHAccel();
}

SoftRasterizer::Mesh::Mesh(const std::string &name,
                           SoftRasterizer::Material &&_material,
                           std::vector<Vertex> &&_vertices,
                           std::vector<glm::uvec3> &&_faces, Bounds3 &&box)
    : meshname(name), MeshMaterial(std::make_shared<Material>(_material)),
      vertices(std::move(_vertices)), faces(std::move(_faces)),
      bounding_box(std::move(box)), m_shader(nullptr), m_bvh(std::make_unique<BVHAcceleration>()) {

          /*Generating Triangles*/
          generateTriangles();

          /*Generating BVH Structure*/
          buildBVHAccel();
}

SoftRasterizer::Mesh::~Mesh(){
          m_bvh->clearBVHAccel();
}

void SoftRasterizer::Mesh::bindShader2Mesh(std::shared_ptr<Shader> shader) {
  /*bind shader2 mesh without dtor,  the life od this pointer is maintained by
   * render class*/
  m_shader.reset();
  m_shader = shader;
}

bool SoftRasterizer::Mesh::intersect(const Ray &ray) { return true; }

bool SoftRasterizer::Mesh::intersect(const Ray &ray, float &tNear) {
          bool status = false;
          float temp = std::numeric_limits<float>::infinity();
          std::for_each(faces.begin(), faces.end(), [&](auto& obj) {
                    const glm::vec3& v0 = vertices[obj.x].position;
                    const glm::vec3& v1 = vertices[obj.y].position;
                    const glm::vec3& v2 = vertices[obj.z].position;
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
          if (m_bvh == nullptr) return {};
          return m_bvh->getIntersection(ray);
}

SoftRasterizer::Object::Properties 
SoftRasterizer::Mesh::getSurfaceProperties(const std::size_t faceIndex, 
                                                                      const glm::vec3& Point, 
                                                                      const glm::vec3& viewDir, 
                                                                      const glm::vec2& uv) {

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
          ret.valid = true;
          return ret;
}

std::shared_ptr<SoftRasterizer::Material> SoftRasterizer::Mesh::getMaterial() {
          return std::shared_ptr<Material>(MeshMaterial.get(), [](Material*) {});
}

glm::vec3 SoftRasterizer::Mesh::getDiffuseColor(const glm::vec2& uv) {
          return MeshMaterial->color;
}

/*Generating Triangles*/
void SoftRasterizer::Mesh::generateTriangles(){
          m_triangles.resize(faces.size());
          m_converted.resize(faces.size());

          tbb::parallel_for(std::size_t(0), faces.size(), [&](std::size_t i) {
                    const glm::vec3& v0 = vertices[faces[i].x].position;
                    const glm::vec3& v1 = vertices[faces[i].y].position;
                    const glm::vec3& v2 = vertices[faces[i].z].position;

                    std::shared_ptr<Triangle> tri( std::make_shared<Triangle>());
                    tri->index = i;
                    tri->setVertex({ v0, v1, v2 });
                    tri->setNormal({ vertices[faces[i].x].normal, vertices[faces[i].y].normal, vertices[faces[i].z].normal });
                    tri->setColor({ vertices[faces[i].x].color, vertices[faces[i].y].color, vertices[faces[i].z].color });
                    tri->setTexCoord({ vertices[faces[i].x].texCoord, vertices[faces[i].y].texCoord, vertices[faces[i].z].texCoord });

                    m_triangles[i] = tri;
                    m_converted[i] = (*tri);
                    });
}

/*Generating BVH Structure*/
void SoftRasterizer::Mesh::buildBVHAccel(){

          try{
                    std::vector<Object*> objs(m_triangles.size());
                    std::transform(m_converted.begin(), m_converted.end(), objs.begin(), [](Triangle& tri) { return &tri; });
                    m_bvh->loadNewObjects(objs);
                    m_bvh->clearBVHAccel();
                    m_bvh->startBuilding();
                    bounding_box = m_bvh->getBoundingBox().value();
          }
          catch (const std::exception&e) {
                    spdlog::error("BoundingBox of Mesh {} Error!", meshname);
          }
}

/*Rebuild BVH Structure*/
void SoftRasterizer::Mesh::rebuildBVHAccel(){
          buildBVHAccel();
}

std::optional<SoftRasterizer::Bounds3> SoftRasterizer::Mesh::getBoundingBox() const {
          return m_bvh->getBoundingBox();
}