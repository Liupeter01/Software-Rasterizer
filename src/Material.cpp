#include <object/Material.hpp>

SoftRasterizer::BsdfSample
SoftRasterizer::Material::sample(const glm::vec3 &incoming,
                                 const glm::vec3 &normal, bool frontFace,
                                 const glm::vec2 &uv, Sampler &rng) const {
  const auto tint = albedo(uv);
  if (type == MaterialType::DIFFUSE_AND_GLOSSY) {
    return {rng.cosineHemisphere(normal), tint, false, false};
  }
  if (type == MaterialType::REFLECTION) {
    return {unit(glm::reflect(incoming, normal)), tint, true, false};
  }
  if (!std::isfinite(ior) || ior <= 0) {
    throw std::invalid_argument("invalid dielectric IOR");
  }
  const float incidentIOR = frontFace ? 1.f : ior,
              transmittedIOR = frontFace ? ior : 1.f,
              eta = incidentIOR / transmittedIOR;
  const float fresnel = dielectricFresnel(-glm::dot(incoming, normal),
                                          incidentIOR, transmittedIOR);
  if (rng.next() < fresnel) {
    return {unit(glm::reflect(incoming, normal)), tint, true, false};
  }
  auto direction = glm::refract(incoming, normal, eta);
  if (glm::dot(direction, direction) < 1e-12f) {
    return {unit(glm::reflect(incoming, normal)), tint, true, false};
  }
  // Radiance transport across a dielectric interface includes eta^2.
  return {unit(direction), tint * (eta * eta), true, true};
}
