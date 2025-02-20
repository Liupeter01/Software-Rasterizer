#include <object/Material.hpp>

SoftRasterizer::Material::Material(MaterialType _type, const glm::vec3 &_Ka,
                                   const glm::vec3 &_Kd, const glm::vec3 &_Ks,
                                   const float _specularExponent,
                                   const glm::vec3 &_emission)

    : type(_type), Ka(_Ka), Kd(_Kd), Ks(_Ks), Ns(0.f), Ni(0.f), d(0.f),
      illum(0), ior(0.f), specularExponent(_specularExponent),
      emission(_emission), map_Ka(""), map_Kd(""), map_Ks(""), map_Ns(""),
      map_d(""), map_bump("") {}

// uniform sample on the hemisphere
glm::vec3 SoftRasterizer::Material::sample(const glm::vec3 &wi,
                                           const glm::vec3 &N) {
  if (type == MaterialType::DIFFUSE_AND_GLOSSY) {

    float u = std::max(Tools::random_generator(),
                       std::numeric_limits<float>::epsilon());
    float v = std::max(Tools::random_generator(),
                       std::numeric_limits<float>::epsilon());

    /* Generator 2D Random Sample Coordinates
     * x ^ 2 + y ^ 2 + z ^ 2 = 1 = r^2 + z^2 = 1 */
    float phi = 2.0f * Tools::PI * v; // angle [0, 2PI]
    float z = std::sqrt(1.0f - u);
    float r = std::sqrt(u); // R of sphere

    /*Generate a ray from (0, 0, 0) to (x, y, z)*/
    glm::vec3 local(r * std::cos(phi), r * std::sin(phi), z);

    /* Mathematical Transformation Principle
     * localRay = (x, y, z) => worldRay = xT+yB+zN*/
    return Tools::toWorld(local, N);
  }
  return glm::vec3(0.f);
}

/*
 * Given an incident direction, an outgoing direction, and a normal vector,
 * calculate the probability density of obtaining the outgoing direction using
 * the sampling method. uniform sample probability 1 / (2 * PI) = 0.5f * PI_INV
 */
const float SoftRasterizer::Material::pdf(const glm::vec3 &wi,
                                          const glm::vec3 &wo,
                                          const glm::vec3 &N) {
          return uniform_sampling_on_sphere;
}

/*
 * Given an incident direction, an outgoing direction, and a normal vector,
 *  calculate the contribution of this ray , which is Fr(P, wi, wo)
 */
glm::vec3 SoftRasterizer::Material::fr_contribution(const glm::vec3 &wi,
                                                    const glm::vec3 &wo,
                                                    const glm::vec3 &N) {

  auto angle = glm::dot(wi, N);

  if (type == MaterialType::DIFFUSE_AND_GLOSSY) {
    return Kd * Tools::PI_INV;
  }
  return glm::vec3(0.f);
}

const bool SoftRasterizer::Material::hasEmission() {
  /*The Emission Should not be zero!*/
  return glm::length(emission) > std::numeric_limits<float>::epsilon();
}
