#include <object/Cube.hpp>

SoftRasterizer::Cube::Cube() 
{
}

SoftRasterizer::Cube::~Cube() {}

SoftRasterizer::Bounds3 SoftRasterizer::Cube::getBounds() {
          return {};
}

bool SoftRasterizer::Cube::intersect(const Ray& ray) {
          return true;
}

bool SoftRasterizer::Cube::intersect(const Ray& ray, float& tNear) {
          return true;
}

SoftRasterizer::Intersection SoftRasterizer::Cube::getIntersect(Ray& ray) {
          return {};
}

SoftRasterizer::Object::Properties
SoftRasterizer::Cube::getSurfaceProperties(const std::size_t faceIndex,
          const glm::vec3& Point,
          const glm::vec3& viewDir,
          const glm::vec2& uv) {
          return {};
}

std::shared_ptr<SoftRasterizer::Material>& SoftRasterizer::Cube::getMaterial() {
          return material;
}

/*Compatible Consideration!*/
const std::vector<SoftRasterizer::Vertex>& SoftRasterizer::Cube::getVertices() const {
          return vert;
}

const std::vector<glm::uvec3>& SoftRasterizer::Cube::getFaces() const  {
          return faces;
}

glm::vec3 SoftRasterizer::Cube::getDiffuseColor(const glm::vec2& uv) {
          return glm::vec3(0.5f);
}

void SoftRasterizer::Cube::updatePosition(const glm::mat4x4& NDC_MVP,
          const glm::mat4x4& Normal_M) {
}