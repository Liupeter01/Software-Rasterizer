#pragma once
#include <hpc/Sampler.hpp>
#include <loader/TextureLoader.hpp>
#include <memory>

namespace SoftRasterizer {
enum class MaterialType {
  Diffuse,
  Mirror,
  Dielectric,
  // Source-compatible aliases. Diffuse is Lambertian in the path tracer.
  DIFFUSE_AND_GLOSSY = Diffuse,
  REFLECTION = Mirror,
  REFLECTION_AND_REFRACTION = Dielectric
};

struct BsdfSample {
  glm::vec3 direction{0}, weight{0};
  bool delta = false, transmitted = false;
};

struct Material {
  MaterialType type = MaterialType::DIFFUSE_AND_GLOSSY;
  glm::vec3 Kd{0.7f}, emission{0}, Ks{0};
  float ior = 1.5f, specularExponent = 32;
  std::shared_ptr<TextureLoader> texture;

  explicit Material(MaterialType t = MaterialType::DIFFUSE_AND_GLOSSY)
      : type(t) {}

  glm::vec3 albedo(const glm::vec2 &uv) const {
    return glm::clamp(
        Kd * (texture ? texture->getTextureColor(uv) : glm::vec3(1)),
        glm::vec3(0), glm::vec3(1));
  }

  bool hasEmission() const {
    return maxComponent(emission) > 0;
  }

  bool isDelta() const {
    return type != MaterialType::DIFFUSE_AND_GLOSSY;
  }

  BsdfSample sample(const glm::vec3 &incoming, const glm::vec3 &normal,
                    bool frontFace, const glm::vec2 &uv, Sampler &rng) const;
};
} // namespace SoftRasterizer
