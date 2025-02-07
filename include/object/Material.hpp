#pragma once
#ifndef _MATERIAL_HPP_
#define _MATERIAL_HPP_
#include <Tools.hpp>
#include <glm/glm.hpp>
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
           const float _specularExponent = 0.0f,
           const glm::vec3 &_emission = glm::vec3(0.f));

  MaterialType getMaterialType() const { return type; }

  // uniform sample on the hemisphere
glm::vec3 sample(const glm::vec3 &wi, const glm::vec3 &N);

  /*
   * Given an incident direction, an outgoing direction, and a normal vector,
   * calculate the probability density of obtaining the outgoing direction using
   * the sampling method. uniform sample probability 1 / (2 * PI) = 0.5f *
   * PI_INV
   */
  const float pdf(const glm::vec3 &wi, const glm::vec3 &wo, const glm::vec3 &N);

  /*
   * Given an incident direction, an outgoing direction, and a normal vector,
   *  calculate the contribution of this ray , which is Fr(P, wi, wo)
   */
  glm::vec3 fr_contribution(const glm::vec3 &wi, const glm::vec3 &wo,
                                   const glm::vec3 &N);

  const glm::vec3 &getEmission() const { return emission; }
  [[nodiscard]] const bool hasEmission();

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

  // Self Emissive object
  glm::vec3 emission;

  // Unit Sphere's radius
  static constexpr float radius = 1.0f;

  // Uniform Random Sphere Sampling Variable
  static constexpr float uniform_sampling_on_sphere = 0.5f * Tools::PI_INV;
};
} // namespace SoftRasterizer

#endif //_MATERIAL_HPP_
