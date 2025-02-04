#include <object/Sphere.hpp>

SoftRasterizer::Sphere::Sphere() : Sphere(glm::vec3(0.f), 1.f) {}

SoftRasterizer::Sphere::Sphere(const glm::vec3 &_center, const float _radius)
    : vert(1), center(_center), radius(_radius), square(radius * radius),
      Object(std::make_shared<Material>(), nullptr) {}

SoftRasterizer::Sphere::~Sphere() {}

void SoftRasterizer::Sphere::updatePosition(const glm::mat4x4 &NDC_MVP,
                                            const glm::mat4x4 &Normal_M) {
  vert[0].position = Tools::to_vec3(NDC_MVP * glm::vec4(center, 1.0f));
}

void SoftRasterizer::Sphere::bindShader2Mesh(std::shared_ptr<Shader> shader) {
  m_shader.reset();
  m_shader = shader;
}

SoftRasterizer::Bounds3 SoftRasterizer::Sphere::getBounds() {
  Bounds3 ret;
  ret.min = glm::vec3(vert[0].position.x - radius, vert[0].position.y - radius,
                      vert[0].position.z - radius);
  ret.max = glm::vec3(vert[0].position.x + radius, vert[0].position.y + radius,
                      vert[0].position.z + radius);
  return ret;
}

float SoftRasterizer::Sphere::getSquare() const { return square; }

bool SoftRasterizer::Sphere::intersect(const Ray &ray) {
  auto L = ray.origin - vert[0].position;
  auto a = glm::dot(ray.direction, ray.direction);
  auto b = 2.f * glm::dot(ray.direction, L);
  auto c = glm::dot(L, L) - square;

  auto res = b * b - 4.f * a * c;
  return res < 0 ? false : true;
}

bool SoftRasterizer::Sphere::intersect(const Ray &ray, float &tNear) {
  auto L = ray.origin - vert[0].position;
  auto a = glm::dot(ray.direction, ray.direction);
  auto b = 2.f * glm::dot(ray.direction, L);
  auto c = glm::dot(L, L) - square;

  auto res = b * b - 4.f * a * c;
  if (res < 0) {
    return false;
  }
  if (!res) { // res == 0
    tNear = -0.5f * b / a;
  } else {
    // Calculate q for better numerical stability
    float q = -0.5f * (b + std::copysign(std::sqrt(res), b));
    auto x0 = q / a;
    auto x1 = c / q;

    // Select the smallest positive root
    tNear = (x0 > 0 && x1 > 0) ? std::min(x0, x1) : (x0 > 0 ? x0 : x1);
    if (tNear <= 0) {
      return false; // No valid intersection
    }
  }
  return true;
}

SoftRasterizer::Intersection SoftRasterizer::Sphere::getIntersect(Ray &ray) {
  Intersection ret;
  auto L = ray.origin - vert[0].position;
  auto a = glm::dot(ray.direction, ray.direction);
  auto b = 2.f * glm::dot(ray.direction, L);
  auto c = glm::dot(L, L) - square;

  auto res = b * b - 4.f * a * c;

  float t0{}; // intersection time
  if (res < 0) {
    /*Invalid Intersection*/
    return {};
  } else if (!res) {
    t0 = -0.5f * b / a;
  } else {
    // Calculate q for better numerical stability
    float q = -0.5f * (b + std::copysign(std::sqrt(res), b));
    auto x0 = q / a;
    auto x1 = c / q;

    // Select the smallest positive root
    t0 = (x0 > 0 && x1 > 0) ? std::min(x0, x1) : (x0 > 0 ? x0 : x1);
    if (t0 <= 0) {
      return {}; // No valid intersection
    }
  }

  ret.obj = this;
  ret.intersect_time = t0;
  ret.material = m_material;
  ret.coords = ray.direction * t0 + ray.origin;

  /*Normal of a sphere!*/
  ret.normal = glm::normalize(ret.coords - center);

  // we could find a intersect time point
  ret.intersected = true;
  return ret;
}

SoftRasterizer::Object::Properties SoftRasterizer::Sphere::getSurfaceProperties(
    const std::size_t faceIndex, const glm::vec3 &Point,
    const glm::vec3 &viewDir, const glm::vec2 &uv) {
  Properties ret;
  ret.normal = glm::normalize(Point - vert[0].position);
  return ret;
}
