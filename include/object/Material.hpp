#pragma once
#ifndef _MATERIAL_HPP_
#define _MATERIAL_HPP_
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
           const glm::vec3 &_color = glm::vec3(1.0f),
           const glm::vec3 &_Ka = glm::vec3(0.0f),
           const glm::vec3 &_Kd = glm::vec3(0.0f),
           const glm::vec3 &_Ks = glm::vec3(0.0f),
           const float _specularExponent = 0.0f)

      : type(_type), Ka(_Ka), Kd(_Kd), Ks(_Ks), color(_color), Ns(0.f), Ni(0.f),
        d(0.f), illum(0), ior(0.f), specularExponent(_specularExponent),
        map_Ka(""), map_Kd(""), map_Ks(""), map_Ns(""), map_d(""),
        map_bump("") {}

  MaterialType getMaterialType() const { return type; }

  std::string name;       // Material Name
  MaterialType type;      // Material Type
  glm::vec3 Ka;           // Ambient Color
  glm::vec3 Kd;           // Diffuse Color
  glm::vec3 Ks;           // Specular Color
  glm::vec3 color;        // Color
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
};
} // namespace SoftRasterizer

#endif //_MATERIAL_HPP_
