#include <object/Sphere.hpp>

SoftRasterizer::Sphere::Sphere() : Sphere(glm::vec3(0.f), 1.f) {}

SoftRasterizer::Sphere::Sphere(const glm::vec3 &_center, const float _radius)
    : center(_center), radius(_radius), square(radius * radius),
      material(std::make_shared<Material>()) {}

SoftRasterizer::Sphere::~Sphere() {}

SoftRasterizer::Bounds3 SoftRasterizer::Sphere::getBounds() {
  Bounds3 ret;
  ret.min = glm::vec3(center.x - radius, center.y - radius, center.z - radius);
  ret.max = glm::vec3(center.x + radius, center.y + radius, center.z + radius);
  return ret;
}

float SoftRasterizer::Sphere::getSquare() const { return square; }

bool SoftRasterizer::Sphere::intersect(const Ray &ray) {
  auto L = ray.direction - center;
  auto a = ray.direction.x * ray.direction.x +
           ray.direction.y * ray.direction.y +
           ray.direction.z * ray.direction.z;
  auto b = 2.f * ray.direction.x * L.x + ray.direction.y * L.y +
           ray.direction.z * L.z;
  auto c = L.x * L.x + L.y * L.y + L.z * L.z - square;

  auto res = b * b - 4.f * a * c;
  return res < 0 ? false : true;
}

bool SoftRasterizer::Sphere::intersect(const Ray &ray, float &tNear) {
  auto L = ray.direction - center;
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
    float q =
        (b > 0) ? -0.5 * (b + std::sqrt(res)) : -0.5 * (b - std::sqrt(res));
    auto x0 = q / a;
    auto x1 = c / q;

    if (x0 < x1 && x0 > 0) {
      tNear = x0;
    } else if (x1 < x0 && x1 > 0) {
      tNear = x1;
    } else {
      return false;
    }
  }
  return true;
}

SoftRasterizer::Intersection SoftRasterizer::Sphere::getIntersect(Ray &ray) {
  Intersection ret;
  auto L = ray.direction - center;
  auto a = glm::dot(ray.direction, ray.direction);
  auto b = 2.f * glm::dot(ray.direction, L);
  auto c = glm::dot(L, L) - square;

  auto res = b * b - 4.f * a * c;

  float t0{}; // intersection time
  if (res < 0) {
    /*Invalid Intersection*/
    return ret;
  } else if (!res) {
    t0 = -0.5f * b / a;
  } else {
    float q =
        (b > 0) ? -0.5 * (b + std::sqrt(res)) : -0.5 * (b - std::sqrt(res));
    auto x0 = q / a;
    auto x1 = c / q;

    if (x0 < x1 && x0 > 0) {
      t0 = x0;
    } else if (x1 < x0 && x1 > 0) {
      t0 = x1;
    } else {
      return ret;
    }
  }

  ret.obj = this;
  ret.intersect_time = t0;
  ret.coords = ray.direction * t0 + ray.origin;
  ret.material = getMaterial();

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
  ret.normal = glm::normalize(Point - center);
  return ret;
}

glm::vec3 SoftRasterizer::Sphere::getDiffuseColor(const glm::vec2 &uv) {
  return material->color;
}
