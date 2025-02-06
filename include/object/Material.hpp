#pragma once
#ifndef _MATERIAL_HPP_
#define _MATERIAL_HPP_
#include <glm/glm.hpp>
#include <Tools.hpp>
#include <string>

namespace SoftRasterizer {

enum class MaterialType {
  DIFFUSE_AND_GLOSSY,
  REFLECTION_AND_REFRACTION,
  REFLECTION
};

struct Material {
  Material(MaterialType _type = MaterialType::REFLECTION_AND_REFRACTION,
           const glm::vec3 &_Ka = glm::vec3(0.0f),
           const glm::vec3 &_Kd = glm::vec3(0.0f),
           const glm::vec3 &_Ks = glm::vec3(0.0f),
           const float _specularExponent = 0.0f)

      : type(_type), Ka(_Ka), Kd(_Kd), Ks(_Ks), Ns(0.f), Ni(0.f), d(0.f),
        illum(0), ior(0.f), specularExponent(_specularExponent), map_Ka(""),
        map_Kd(""), map_Ks(""), map_Ns(""), map_d(""), map_bump("") {}

  MaterialType getMaterialType() const { return type; }

  // uniform sample on the hemisphere
  const glm::vec3& sample(const glm::vec3& wi, const glm::vec3& N) {
            if (type == MaterialType::DIFFUSE_AND_GLOSSY) {
                      /*
                      * Generator 2D Random Sample Coordinates
                      * x ^ 2 + y ^ 2 + z ^ 2 = 1 = r^2 + z^2 = 1
                      */
                      float z = std::abs(1.f - 2.f * Tools::random_generator());  //Generate Z |[-1, 1]|=>[0, 1]
                      float r = std::sqrt(1.f - z * z);     //on X and Y Axis
                      float phi = 2.0f * Tools::PI * Tools::random_generator(); // angle [0, 2PI]

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
  * calculate the probability density of obtaining the outgoing direction using the sampling method.
  * uniform sample probability 1 / (2 * PI) = 0.5f * PI_INV
  */
  const float pdf(const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& N) {
            return glm::dot(wi, N) > 0 ? uniform_sampling_on_sphere : 0.f;
  }

  /* Sample a ray by material properties */

  /* 
   * Given an incident direction, an outgoing direction, and a normal vector, 
   *  calculate the contribution of this ray , which is Fr(P, wi, wo)
   */
  const glm::vec3& fr_contribution(const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& N) {
            if (type == MaterialType::DIFFUSE_AND_GLOSSY) {
                      return glm::dot(wi, N) > 0 ? Kd * Tools::PI_INV : glm::vec3(0.f);
            }
            return glm::vec3(0.f);
  }

  std::string name;       // Material Name
  MaterialType type;      // Material Type
  glm::vec3 Ka;           // Ambient Color
  glm::vec3 Kd;           // Diffuse Color
  glm::vec3 Ks;           // Specular Color
  float Ns;               // Specular Exponent
  float Ni;               // Optical Density
  float d;                // Dissolve
  int illum;              // Illumination
  float ior;              // Index of Refraction
  float specularExponent; // Specular Exponent
  std::string map_Ka;     // Ambient Texture Map
  std::string map_Kd;     // Diffuse Texture Map
  std::string map_Ks;     // Specular Texture Map
  std::string map_Ns;     // Specular Hightlight Map
  std::string map_d;      // Alpha Texture Map
  std::string map_bump;   // Bump Map

  //Unit Sphere's radius
  static constexpr float radius = 1.0f;       

  //Uniform Random Sphere Sampling Variable
  static constexpr float uniform_sampling_on_sphere = 0.5f * Tools::PI_INV;
};
} // namespace SoftRasterizer

#endif //_MATERIAL_HPP_
