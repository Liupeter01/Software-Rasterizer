﻿#define GLM_ENABLE_EXPERIMENTAL
#include <Tools.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <object/Sphere.hpp>

SoftRasterizer::Sphere::Sphere() : Sphere(glm::vec3(0.f), 1.f) {}

SoftRasterizer::Sphere::Sphere(const glm::vec3 &_center, const float _radius)
    : vert(1), center(_center), radius(_radius), square(radius * radius),
      Object(std::make_shared<Material>(), nullptr) {

  // Might Changed Due to mvp;
  new_radius = radius;
  new_square = radius * radius;

  /*Calculate Area*/
  calcArea();
}

SoftRasterizer::Sphere::~Sphere() {}

void SoftRasterizer::Sphere::updatePosition(const glm::mat4x4 &Model,
                                            const glm::mat4x4 &View,
                                            const glm::mat4x4 &Projection,
                                            const glm::mat4x4 &Ndc) {

  glm::vec3 scale, translation, skew;
  glm::quat rotation;
  glm::vec4 perspective;
  glm::decompose(Model, scale, rotation, translation, skew, perspective);

  // update center points location
  vert[0].position =
      Tools::to_vec3(Projection * View * Model * glm::vec4(center, 1.0f));

  // update radius length
  new_radius = radius * glm::max(scale.x, glm::max(scale.y, scale.z));

  /*bad news, its wrong!!!!*/
  // new_radius = radius * glm::length(scale);
  new_square = new_radius * new_radius;
}

void SoftRasterizer::Sphere::bindShader2Mesh(std::shared_ptr<Shader> shader) {
  m_shader.reset();
  m_shader = shader;
}

void SoftRasterizer::Sphere::setMaterial(std::shared_ptr<Material> material) {
  m_material.reset();
  m_material = material;
}

void SoftRasterizer::Sphere::calcArea() { area = 4 * Tools::PI * new_square; }

SoftRasterizer::Bounds3 SoftRasterizer::Sphere::getBounds() {
  Bounds3 ret;
  ret.min = glm::vec3(vert[0].position.x - new_radius,
                      vert[0].position.y - new_radius,
                      vert[0].position.z - new_radius);
  ret.max = glm::vec3(vert[0].position.x + new_radius,
                      vert[0].position.y + new_radius,
                      vert[0].position.z + new_radius);
  return ret;
}

float SoftRasterizer::Sphere::getSquare() const { return square; }

bool SoftRasterizer::Sphere::intersect(const Ray &ray) {
  auto L = ray.origin - vert[0].position;
  auto a = glm::dot(ray.direction, ray.direction);
  auto b = 2.f * glm::dot(ray.direction, L);
  auto c = glm::dot(L, L) - new_square;

  auto res = b * b - 4.f * a * c;
  return res < 0 ? false : true;
}

bool SoftRasterizer::Sphere::intersect(const Ray &ray, float &tNear) {
  auto L = ray.origin - vert[0].position;
  auto a = glm::dot(ray.direction, ray.direction);
  auto b = 2.f * glm::dot(ray.direction, L);
  auto c = glm::dot(L, L) - new_square;

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
  auto c = glm::dot(L, L) - new_square;

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
  ret.normal = glm::normalize(ret.coords - vert[0].position);

  // we could find a intersect time point
  ret.intersected = true;
  ret.emit = m_material->getEmission();
  return ret;
}

SoftRasterizer::Object::Properties SoftRasterizer::Sphere::getSurfaceProperties(
    const std::size_t faceIndex, const glm::vec3 &Point,
    const glm::vec3 &viewDir, const glm::vec2 &uv) {
  Properties ret;
  ret.normal = glm::normalize(Point - vert[0].position);
  return ret;
}

std::tuple<SoftRasterizer::Intersection, float>
SoftRasterizer::Sphere::sample() {
  /*Generator 2D Random Sample Coordinates*/
  float theta =
      2.0f * Tools::PI * Tools::random_generator();  // azimuth angle [0, 2PI]
  float phi = Tools::PI * Tools::random_generator(); // polar angle [0, PI]

  /*
   * x=sinϕcosθ
   * y=sinϕsinθ
   * z=cosϕ
   */
  glm::vec3 dir(std::cos(phi), std::sin(phi) * std::cos(theta),
                std::sin(phi) * std::sin(theta));

  Intersection intersection;

  intersection.intersected = true;
  intersection.obj = this;
  intersection.coords = vert[0].position + new_radius * dir;
  intersection.normal = dir;
  intersection.emit = m_material->getEmission();

  calcArea();

  return {/*intersection = */ intersection,
          /*pdf = */ 1.0f / area};
}
