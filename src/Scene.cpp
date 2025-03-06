#include "glm/fwd.hpp"
#include <Tools.hpp>
#include <base/Render.hpp>
#include <ctime>
#include <glm/geometric.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtx/norm.hpp>
#include <light/Light.hpp>
#include <light/SphereLight.hpp>
#include <limits>
#include <memory>
#include <numeric> // For std::accumulate
#include <scene/Scene.hpp>
#include <shader/Shader.hpp>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_reduce.h>

SoftRasterizer::Scene::Scene(const std::string &sceneName, const glm::vec3 &eye,
                             const glm::vec3 &center, const glm::vec3 &up,
                             glm::vec3 backgroundColor,
                             const std::size_t maxdepth, const float rr)
    : m_width(0), m_height(0), m_sceneName(sceneName), m_maxDepth(maxdepth),
      m_backgroundColor(backgroundColor), m_eye(eye), m_center(center),
      p_rr(rr), m_up(up), m_fovy(45.0f), m_aspectRatio(0.0f), scale(0.0f),
      offset(0.0f), m_cameraLight(nullptr),
      m_bvh(std::make_unique<BVHAcceleration>()) {
  try {
    setViewMatrix(eye, center, up);
  } catch (const std::exception &e) {
    spdlog::error("Scene Constructor Error! Reason: {}", e.what());
  }
}

SoftRasterizer::Scene::~Scene() { clearBVHAccel(); }

bool SoftRasterizer::Scene::addGraphicObj(
    const std::string &path, const std::string &meshName, const glm::vec3 &axis,
    const float angle, const glm::vec3 &translation, const glm::vec3 &scale) {
  /*This Object has already been identified!*/
  if (m_loadedObjs.find(meshName) != m_loadedObjs.end()) {
    spdlog::error(
        "Add Graphic Obj Error! This Object has already been identified");
    return false;
  }

  try {
    m_loadedObjs[meshName].loader = std::make_unique<ObjLoader>(
        path, meshName, axis, angle, translation, scale);
  } catch (const std::exception &e) {
    spdlog::error("Add Graphic Obj Error! Reason: {}", e.what());
    return false;
  }
  return true;
}

bool SoftRasterizer::Scene::addGraphicObj(const std::string &path,
                                          const std::string &meshName) {

  /*This Object has already been identified!*/
  if (m_loadedObjs.find(meshName) != m_loadedObjs.end()) {
    spdlog::error("This Object has already been identified");
    return false;
  }

  try {
    m_loadedObjs[meshName].loader = std::make_unique<ObjLoader>(path, meshName);
  } catch (const std::exception &e) {
    spdlog::error("Add Graphic Obj Error! Reason: {}", e.what());
    return false;
  }
  return true;
}

bool SoftRasterizer::Scene::addGraphicObj(std::unique_ptr<Object> object,
                                          const std::string &objectName) {
  /*This Object has already been identified!*/
  if (m_loadedObjs.find(objectName) != m_loadedObjs.end()) {
    spdlog::error("This Object has already been identified");
    return false;
  }

  try {
    m_loadedObjs[objectName].loader = std::nullopt;
    m_loadedObjs[objectName].mesh = std::move(object);
  } catch (const std::exception &e) {
    spdlog::error("Add Graphic Obj Error! Reason: {}", e.what());
    return false;
  }
  return true;
}

bool SoftRasterizer::Scene::startLoadingMesh(const std::string &meshName) {

  /*This Object has already been identified!*/
  if (m_loadedObjs.find(meshName) == m_loadedObjs.end()) {
    spdlog::error("Start Loading Mesh Failed! Because There is nothing found "
                  "in m_loadedObjs");
    return false;
  }

  if (m_loadedObjs[meshName].mesh != nullptr) {
    spdlog::error("Start Loading Mesh Failed! Because {} Has Already Loaded "
                  "into m_loadedObjs",
                  meshName);
    return false;
  }

  try {

    std::optional<std::unique_ptr<Mesh>> mesh_op =
        m_loadedObjs[meshName].loader.value()->startLoadingFromFile(meshName);

    if (!mesh_op.has_value()) {
      spdlog::error(
          "Start Loading Mesh Failed! Because Loading Internel Error!");
      return false;
    }

    m_loadedObjs[meshName].mesh = std::move(mesh_op.value());

  } catch (const std::exception &e) {
    spdlog::error("Start Loading Mesh Failed! Reason: {}", e.what());
    return false;
  }
  return true;
}

std::optional<std::shared_ptr<SoftRasterizer::Object>>
SoftRasterizer::Scene::getMeshObj(const std::string &meshName) {
  /*This Object has already been identified!*/
  if (m_loadedObjs.find(meshName) == m_loadedObjs.end()) {
    spdlog::error("Get Mesh Failed! Because There is nothing found "
                  "in m_loadedObjs");
    return std::nullopt;
  }

  if (m_loadedObjs[meshName].mesh == nullptr) {
    spdlog::error("You Have to get Mesh Object After Deploy startLoadingMesh",
                  meshName);
    return std::nullopt;
  }

  return std::shared_ptr<SoftRasterizer::Object>(
      m_loadedObjs[meshName].mesh.get(), [](Object *) {});
}

bool SoftRasterizer::Scene::addShader(const std::string &shaderName,
                                      const std::string &texturePath,
                                      SHADERS_TYPE type) {
  if (m_shaders.find(shaderName) != m_shaders.end()) {
    spdlog::error("Add Shader Failed! Because Shader {} Already Exist!",
                  shaderName);
    return false;
  }
  try {
    m_shaders[shaderName] = std::make_shared<Shader>(texturePath);
    m_shaders[shaderName]->setFragmentShader(type);
  } catch (const std::exception &e) {
    spdlog::error("Add Shader Failed! Reason: {}", e.what());
    return false;
  }
  return true;
}

bool SoftRasterizer::Scene::addShader(const std::string &shaderName,
                                      std::shared_ptr<TextureLoader> text,
                                      SHADERS_TYPE type) {
  if (m_shaders.find(shaderName) != m_shaders.end()) {
    spdlog::error("Add Shader Failed! Because Shader {} Already Exist!",
                  shaderName);
    return false;
  }
  try {
    m_shaders[shaderName] = std::make_shared<Shader>(text);
    m_shaders[shaderName]->setFragmentShader(type);
  } catch (const std::exception &e) {
    spdlog::error("Add Shader Failed! Reason: {}", e.what());
    return false;
  }
  return true;
}

bool SoftRasterizer::Scene::bindShader2Mesh(const std::string &meshName,
                                            const std::string &shaderName) {

  if (m_loadedObjs.find(meshName) == m_loadedObjs.end()) {
    spdlog::error(
        "Bind Shader To Mesh Failed! Because Loaded Mesh {} Not found!",
        meshName);
    return false;
  }

  if (m_shaders.find(shaderName) == m_shaders.end()) {
    spdlog::error("Bind Shader To Mesh Failed! Because Shader {} Not found!",
                  shaderName);
    return false;
  }

  try {
    m_loadedObjs[meshName].mesh->bindShader2Mesh(m_shaders[shaderName]);

  } catch (const std::exception &e) {
    spdlog::error("Bind Shader To Mesh Failed! Reason: {}", e.what());
    return false;
  }
  return true;
}

void SoftRasterizer::Scene::addLight(std::string name,
                                     std::shared_ptr<light_struct> light) {
  if (m_lights.find(name) != m_lights.end()) {
    spdlog::warn("Add Light Success! Because Light {} Already Been Added!",
                 name);
    return;
  }

  try {
    m_lights[name] = light;
  } catch (const std::exception &e) {
    spdlog::error("Add Light Failed! Reason: {}", e.what());
  }
}

void SoftRasterizer::Scene::addLights(
    std::vector<std::pair<std::string, std::shared_ptr<light_struct>>> lights) {
  for (auto &[name, shared] : lights) {
    addLight(name, shared);
  }
}

void SoftRasterizer::Scene::cameraLight(bool status) {
  // if status is false, then set intensity to zero
  if (!status) {
    m_lights["sys_camera"]->intensity = glm::vec3(0.f);
    return;
  }
  cameraLight(glm::vec3(1.f));
}

void SoftRasterizer::Scene::cameraLight(const glm::vec3 &intensity) {
  m_lights["sys_camera"]->intensity = intensity;
}

/*set MVP*/
bool SoftRasterizer::Scene::setModelMatrix(const std::string &meshName,
                                           const glm::vec3 &axis,
                                           const float angle,
                                           const glm::vec3 &translation,
                                           const glm::vec3 &scale) {
  if (m_loadedObjs.find(meshName) == m_loadedObjs.end()) {
    spdlog::error("Editing Model Matrix Failed! Because {} Not Found",
                  meshName);
    return false;
  }

  m_loadedObjs[meshName].mesh->updateModelMatrix(axis, angle, translation,
                                                 scale);
  return true;
}

void SoftRasterizer::Scene::setViewMatrix(const glm::vec3 &eye,
                                          const glm::vec3 &center,
                                          const glm::vec3 &up) {
  m_eye = eye;
  m_center = center;
  m_up = up;

  m_view = glm::lookAtLH(eye, center, up);
}

void SoftRasterizer::Scene::setProjectionMatrix(float fovy, float zNear,
                                                float zFar) {
  m_fovy = fovy;
  m_near = zNear;
  m_far = zFar;

  scale = (m_far - m_near) / 2.0f;
  offset = (m_far + m_near) / 2.0f;

#if defined(__x86_64__) || defined(_WIN64)
  scale_simd = _mm256_set1_ps(scale);
  offset_simd = _mm256_set1_ps(offset);

#elif defined(__arm__) || defined(__aarch64__)
  scale_simd = simde_mm256_set1_ps(scale);
  offset_simd = simde_mm256_set1_ps(offset);

#else
#endif

  m_projection = glm::perspectiveLH_NO(fovy, m_aspectRatio, zNear, zFar);
}

std::vector<SoftRasterizer::light_struct> SoftRasterizer::Scene::loadLights() {
  std::vector<SoftRasterizer::light_struct> res(m_lights.size());
  for (std::size_t index = 0; index < m_exportedObjs.size(); ++index) {
    if (auto lightptr =
            dynamic_cast<SphereLight *>(m_exportedObjs[index].get())) {
      if (lightptr->isSelfEmissiveObject()) {
        SoftRasterizer::light_struct ls;

        ls.intensity = lightptr->getIntensity();
        ls.position = lightptr->getCenter();

        res.push_back(ls);
      }
    }
  }
  return res;
}

void SoftRasterizer::Scene::setNDCMatrix(const std::size_t width,
                                         const std::size_t height) {

  m_width = width;
  m_height = height;

  /*calculate ratio*/
  if (!m_height) {
    throw std::runtime_error("Height cannot be zero!");
  }

  m_aspectRatio = static_cast<float>(m_width) / static_cast<float>(m_height);

  glm::mat4 matrix(1.0f);

  matrix[0][0] = width / 2.0f * m_aspectRatio; // x scaling
  matrix[1][1] = height / 2.0f;                // y scaling (flipping y)
  matrix[3][0] = width / 2.0f;                 // x translation
  matrix[3][1] = height / 2.0f;                // y translation

  m_ndcToScreenMatrix = matrix;
}

/* Generate Pointers to Triangles and load it to BVH Structure*/
void SoftRasterizer::Scene::preGenerateBVH() {
  m_exportedObjs.clear();
  m_exportedObjs.resize(m_loadedObjs.size());

  std::transform(m_loadedObjs.begin(), m_loadedObjs.end(),
                 m_exportedObjs.begin(), [](const auto &obj) {
                   return std::shared_ptr<Object>(obj.second.mesh.get(),
                                                  [](Object *) {});
                 });
}

SoftRasterizer::Intersection SoftRasterizer::Scene::traceScene(Ray &ray) {

  /*retrieve arguments from atomic variables*/
  Intersection ret = tbb::parallel_reduce(
      tbb::blocked_range<std::size_t>(0, m_exportedObjs.size()), Intersection{},
      [&](const tbb::blocked_range<std::size_t> &range, Intersection init) {
        for (auto i = range.begin(); i != range.end(); ++i) {
          Intersection intersect = m_exportedObjs[i]->getIntersect(ray);
          if (!intersect.intersected) {
            continue;
          }
          // Check if the current intersection is better (i.e., closer)
          if (intersect.intersect_time < init.intersect_time) {
            init = intersect;
          }
        }
        return init;
      },
      [](const Intersection &a, const Intersection &b) {
        return (a.intersect_time < b.intersect_time) ? a : b;
      });

  /*Invalid Intersection*/
  if (!ret.obj || ret.intersect_time < 0) {
    return {};
  }

  /*
   * Valid Intersection is here! We Are going to get properties by using
   * obj->getSurfaceProperties getSurfaceProperties method could belong to
   * Sphere, Mesh(Triangle), Cube Classes Every object inhertied classes should
   * implement getSurfaceProperties method!!!
   */
  auto properites =
      ret.obj->getSurfaceProperties(ret.index, ret.coords, ray.direction,
                                    ret.uv // Its Barycentric coordinates
      );

  /*interpolated Normal!*/
  ret.normal = properites.normal;
  ret.uv = properites.uv;
  ret.color = properites.color;

  // Debug Color Mode
  // ret.color = (glm::normalize(ret.normal) + glm::vec3(1.0f)) / 2.0f;
  ret.intersected = true;
  return ret;
}

[[nodiscard]] std::tuple<glm::dvec3, double>
SoftRasterizer::Scene::sampleLightOnCenter(const glm::vec3 &shadingPoint) {
  // Collect all emissive objects and approximate their bounding spheres**
  std::vector<std::pair<glm::dvec3, double>> lightSpheres;
  for (const auto &obj : m_exportedObjs) {
    if (obj->isSelfEmissiveObject()) {
      Bounds3 bbox = obj->getBounds();
      glm::dvec3 center = (glm::dvec3(bbox.min) + glm::dvec3(bbox.max)) * 0.5;
      double radius = glm::length(glm::dvec3(bbox.diagonal())) * 0.5;
      lightSpheres.emplace_back(center, radius);
    }
  }

  if (lightSpheres.empty()) {
    spdlog::warn("No emissive objects found in the scene!");
    return {glm::dvec3(0.0), 0.0};
  }

  // Randomly select a light source
  int randomIndex =
      static_cast<int>(Tools::random_generator() * lightSpheres.size());
  glm::dvec3 sphereCenter = lightSpheres[randomIndex].first;
  double sphereRadius = lightSpheres[randomIndex].second;

  glm::dvec3 lightDir = glm::normalize(sphereCenter - glm::dvec3(shadingPoint));

  // Compute probability density function (PDF)
  double pdf = 0.5 * Tools::PI_INV;
  return {lightDir, pdf};
}

std::tuple<glm::dvec3, double>
SoftRasterizer::Scene::sampleLight(const glm::vec3 &shadingPoint) {
  // Collect all emissive objects and approximate their bounding spheres**
  std::vector<std::pair<glm::dvec3, double>> lightSpheres;
  for (const auto &obj : m_exportedObjs) {
    if (obj->isSelfEmissiveObject()) {
      Bounds3 bbox = obj->getBounds();
      glm::dvec3 center = (glm::dvec3(bbox.min) + glm::dvec3(bbox.max)) * 0.5;
      double radius = glm::length(glm::dvec3(bbox.diagonal())) * 0.5;
      lightSpheres.emplace_back(center, radius);
    }
  }

  if (lightSpheres.empty()) {
    spdlog::warn("No emissive objects found in the scene!");
    return {glm::dvec3(0.0), 0.0};
  }

  // Randomly select a light source
  int randomIndex =
      static_cast<int>(Tools::random_generator() * lightSpheres.size());
  glm::dvec3 sphereCenter = lightSpheres[randomIndex].first;
  double sphereRadius = lightSpheres[randomIndex].second;

  glm::dvec3 baselineDir =
      glm::normalize(sphereCenter - glm::dvec3(shadingPoint));

  // Sample a random direction on the light source sphere
  glm::dvec3 sampleDir = glm::sphericalRand(1.0);
  if (glm::dot(sampleDir, baselineDir) < 0.0) {
    sampleDir = -sampleDir;
  }

  // Apply random perturbation for anti-aliasing and soft shadow
  double perturbationStrength = 1e-6;
  glm::dvec3 randomPerturbation = glm::sphericalRand(perturbationStrength);
  sampleDir = glm::normalize(sampleDir + randomPerturbation);

  glm::dvec3 samplePos = sphereCenter + sampleDir * sphereRadius;

  // Compute direction from shading point to light source
  glm::dvec3 lightDir = glm::normalize(samplePos - glm::dvec3(shadingPoint));

  // Compute probability density function (PDF)
  double cosTheta = glm::dot(lightDir, baselineDir);
  double pdf = 0.5 * Tools::PI_INV * cosTheta;
  return {lightDir, pdf};
}

glm::vec3 SoftRasterizer::Scene::whittedRayTracing(Ray &ray, int depth,
                                                   const std::size_t sample) {

  glm::vec3 final_color = this->m_backgroundColor;

  /*DON NOT FORGET TO NORMALIZE THE NORMAL*/
  auto rayDirection = glm::normalize(ray.direction);

  if (depth > m_maxDepth) {
    // Return black if the ray has reached the maximum depth
    return glm::vec3(0.f);
    // return this->m_backgroundColor;
  }

  Intersection intersection = traceScene(ray);
  if (!intersection.intersected) {
    // Return black if the ray does not intersect with any object
    spdlog::debug("Ray Intersection Not Found On Depth {}", depth);
    return this->m_backgroundColor;
  }

  // Get the hit point
  auto hitPoint = intersection.coords;
  auto hitNormal = glm::normalize(glm::dvec3(intersection.normal));

  const float ior = intersection.material->ior;
  const glm::vec3 I = rayDirection;
  const glm::vec3 N =
      hitNormal; // We consider it as the surface normal by default

  float kr = std::clamp(Tools::fresnel(I, N, ior), 0.f, 1.f);

  /*Phong illumation model*/
  if (intersection.material->getMaterialType() ==
      MaterialType::DIFFUSE_AND_GLOSSY) {

    final_color = tbb::parallel_reduce(
        tbb::blocked_range<std::size_t>(0, sample),
        glm::vec3(0.0f), // Initial value
        [&](const tbb::blocked_range<std::size_t> &range,
            glm::vec3 local_sum) -> glm::vec3 {
          for (std::size_t i = range.begin(); i < range.end(); ++i) {

            auto [shading2LightDir, lightAreaPdf] =
                sampleLightOnCenter(hitPoint);

            Ray lightSampleRay(hitPoint, shading2LightDir);
            Intersection lightSampleIntersection = traceScene(lightSampleRay);
            if (!lightSampleIntersection.intersected ||
                (lightSampleIntersection.intersected &&
                 glm::length(lightSampleIntersection.emit) < m_epsilon)) {
              return glm::vec3(0.f);
            }

            // Diffuse reflection (Lambertian)
            double diff =
                std::max(0., glm::dot(glm::dvec3(N), shading2LightDir));

            // Specular reflection (Blinn-Phong)
            glm::dvec3 reflectDir = glm::normalize(
                glm::reflect(-shading2LightDir, glm::dvec3(hitNormal)));
            double spec = std::pow(
                std::max(0., -glm::dot(glm::dvec3(ray.direction), reflectDir)),
                intersection.material->specularExponent);

            double distanceSquare =
                glm::length2(hitPoint - lightSampleIntersection.coords);
            double timeSquare = lightSampleIntersection.intersect_time *
                                lightSampleIntersection.intersect_time;
            bool is_shadow = std::abs(timeSquare - distanceSquare) > 1e-6f;

            spdlog::debug(
                "timeSquare={}, distanceSquare={},delta = {}, is_shadow={}",
                timeSquare, distanceSquare,
                std::abs(distanceSquare - timeSquare), is_shadow);

            // Compute light contribution
            glm::vec3 ambient =
                !is_shadow ? lightSampleIntersection.emit : glm::vec3(0.f);
            glm::vec3 diffuse =
                !is_shadow ? glm::vec3(diff) * lightSampleIntersection.emit
                           : glm::vec3(0.f);
            glm::vec3 specular =
                spec * glm::dvec3(lightSampleIntersection.emit);

            // Accumulate to local sum
            local_sum = (ambient * intersection.material->Ka) +
                        (intersection.color * diffuse) +
                        (specular * intersection.material->Ks);
          }
          return local_sum;
        },
        [](const glm::vec3 &a, const glm::vec3 &b) {
          return a + b;
        } // Combine partial results
    );

    final_color /= sample;
  }

  else if (intersection.material->getMaterialType() ==
           MaterialType::REFLECTION_AND_REFRACTION) {

    glm::vec3 reflectPath = glm::vec3(0.0f), refractPath = glm::vec3(0.0f);
    glm::vec3 reflectedColor = glm::vec3(0.0f),
              refractedColor = glm::vec3(0.0f);

    // Calculate the dot product of I and N (to determine the angle between
    // them)
    const float dot = std::clamp(glm::dot(I, N), -1.f, 1.f);

    // Check if the ray origin is inside the object
    const bool isInside = intersection.obj->getBounds().inside(ray.origin);

    // Determine if the light is coming from inside to outside or outside to
    // inside
    const bool insideToOutside =
        isInside && dot >= 0; // Inside to outside, normal direction
    const bool outsideToInside =
        !isInside && dot < 0; // Outside to inside, normal direction

    // Light is coming from outside to inside
    if (outsideToInside) {
      reflectPath = glm::normalize(glm::reflect(I, N));
      refractPath = glm::refract(I, N, 1.0f / ior);
    } else if (insideToOutside) {
      reflectPath = glm::normalize(
          glm::reflect(I, -N)); // Reflecting against the opposite normal
      refractPath = glm::normalize(
          glm::refract(I, -N, ior)); // Adjust refraction for material to air
    }

    // calculate offset
    auto offset = glm::dot(I, -N) < 0 ? -N * m_epsilon : N * m_epsilon;

    // prevent relfection and refraction from happening at the same time
    auto reflectCoord = hitPoint + offset;
    auto refractCoord = hitPoint - offset;

    /* Total Internal Reflection, TIR */
    if (glm::length(refractPath) < 1e-6f || std::abs(kr - 1.0f) < 1e-6f) {
      // prevent relfection and refraction from happening at the same time
      Ray reflectedRay(reflectCoord, reflectPath);
      reflectedColor = whittedRayTracing(reflectedRay, depth + 1, sample);
      kr = 1.0f;
    } else {
      Ray reflectedRay(reflectCoord, reflectPath);
      Ray refractedRay(refractCoord, refractPath);

      reflectedColor = whittedRayTracing(reflectedRay, depth + 1, sample);
      refractedColor = whittedRayTracing(refractedRay, depth + 1, sample);
    }

    final_color = reflectedColor * kr + refractedColor * (1.0f - kr);

  }

  else if (intersection.material->getMaterialType() ==
           MaterialType::REFLECTION) {

    glm::vec3 reflectPath = glm::vec3(0.0f);
    glm::vec3 reflectedColor = glm::vec3(0.0f);

    // Calculate the dot product of I and N (to determine the angle between
    // them)
    const float dot = std::clamp(glm::dot(I, N), -1.f, 1.f);

    // Check if the ray origin is inside the object
    const bool isInside = intersection.obj->getBounds().inside(ray.origin);

    // Determine if the light is coming from inside to outside or outside to
    // inside
    const bool insideToOutside =
        isInside && dot >= 0; // Inside to outside, normal direction
    const bool outsideToInside =
        !isInside && dot < 0; // Outside to inside, normal direction

    // Light is coming from outside to inside
    if (outsideToInside) {
      reflectPath = glm::normalize(glm::reflect(I, N));
    } else if (insideToOutside) {
      reflectPath = glm::normalize(
          glm::reflect(I, -N)); // Reflecting against the opposite normal
    }

    // calculate offset
    auto offset = glm::dot(I, -N) < 0 ? -N * m_epsilon : N * m_epsilon;

    auto reflectCoord = hitPoint + offset;

    Ray reflectedRay(reflectCoord, reflectPath);

    reflectedColor = whittedRayTracing(reflectedRay, depth + 1, sample);

    final_color = reflectedColor * kr;
  }

  return final_color;
}

// Uniformly sample the light
std::tuple<SoftRasterizer::Intersection, float>
SoftRasterizer::Scene::sampleLight() {

  /*
   * Generate A Random Sampling Area Value
   * Generate a random area value and traverse the objects until the cumulative
   * area exceeds that value
   */
  float random_area_size =
      Tools::random_generator() *

      /*
       * Calculate Self - illuminating Total Area Size
       * Compute Total Area: Sum the areas of all self-emissive objects.
       */
      tbb::parallel_reduce(
          tbb::blocked_range<std::size_t>(0, m_exportedObjs.size()), 0.f,
          [&](const tbb::blocked_range<std::size_t> &range, float init) {
            for (auto i = range.begin(); i != range.end(); ++i) {
              /*Self self-illuminating object*/
              if (m_exportedObjs[i]->isSelfEmissiveObject()) {
                init += m_exportedObjs[i]->getArea();
              }
            }
            return init;
          },
          [](const float a, const float b) { return a + b; });

  float area_sum = 0.f;
  Intersection intersection{};
  float pdf = 0.f;

  // Find a self-emission object according to a random area value
  for (const auto &obj : m_exportedObjs) {
    if (obj->isSelfEmissiveObject()) {
      area_sum += obj->getArea();
      if (random_area_size <= area_sum) {

        /* Sample A Point From An object:
         * Call the object's sample() method to get the intersection and pdf */
        auto [obj_intersect, obj_pdf] = obj->sample();
        intersection = obj_intersect;
        pdf = obj_pdf;
        break;
      }
    }
  }

  return {intersection, pdf};
}

glm::vec3 SoftRasterizer::Scene::pathTracingDirectLight(
    const Intersection &shadeObjIntersection, const glm::vec3 &wo) {

  const glm::dvec3 N = glm::normalize(shadeObjIntersection.normal);

  /*Maybe this Ray Could hit the self-illuminateion Object directly*/
  if (glm::length(shadeObjIntersection.emit) > m_epsilon) {
    return shadeObjIntersection.color;
  }

  /*  Sampling The Light*/
  auto [shading2Light, lightAreaPdf] = sampleLight(shadeObjIntersection.coords);
  if (std::isnan(lightAreaPdf) || lightAreaPdf < m_epsilon) {
    spdlog::debug("Warning: Light area PDF is too small!");
    return glm::vec3(0.0f);
  }

  glm::dvec3 perturbation = glm::dvec3(shadeObjIntersection.coords) + 1e-6 * N;
  Ray lightSampleRay(perturbation, shading2Light);

  Intersection lightSampleIntersection = traceScene(lightSampleRay);
  if (!lightSampleIntersection.intersected ||
      (lightSampleIntersection.intersected &&
       glm::length(lightSampleIntersection.emit) < m_epsilon)) {
    return glm::vec3(0.f);
  }

  double distanceSquare = glm::length2(shadeObjIntersection.coords -
                                       lightSampleIntersection.coords);
  double timeSquare = lightSampleIntersection.intersect_time *
                      lightSampleIntersection.intersect_time;
  bool is_shadow = std::abs(timeSquare - distanceSquare) > 1e-4f;
  if (is_shadow) {
    return glm::vec3(0.f);
  }

  auto object_theta = std::max(0.0, glm::dot(N, shading2Light));
  auto light_theta =
      std::max(0.0, glm::dot(glm::dvec3(lightSampleIntersection.normal),
                             -shading2Light));

  auto Fr = shadeObjIntersection.obj->getMaterial()->fr_contribution(
      shading2Light, wo, N);

  return glm::dvec3(lightSampleIntersection.emit * Fr) * object_theta *
         light_theta / lightAreaPdf / distanceSquare;
}

// Calculate Points Direct light
// glm::vec3 SoftRasterizer::Scene::pathTracingDirectLight(
//    const Intersection &shadeObjIntersection, const glm::vec3 &wo) {
//
//  const glm::vec3 N = glm::normalize(shadeObjIntersection.normal);
//
//  // If the shading point itself is emissive, return its color
//  if (glm::length(shadeObjIntersection.emit) > m_epsilon) {
//    return shadeObjIntersection.color;
//  }
//
//  // Sample a light source
//  auto [lightSample, lightAreaPdf] = sampleLight();
//  if (std::isnan(lightAreaPdf) || lightAreaPdf < m_epsilon) {
//    spdlog::warn("Warning: Light area PDF is too small!");
//    return glm::vec3(0.0f);
//  }
//
//  // Compute shading-to-light vector with high precision
//  glm::dvec3 delta =
//      glm::dvec3(shadeObjIntersection.coords) -
//      glm::dvec3(lightSample.coords);
//  glm::vec3 light2ShadingPointDir = glm::normalize(delta);
//
//  // Shift the shadow ray origin slightly to avoid self-shadowing
//  Ray light2ShadingPoint(lightSample.coords + light2ShadingPointDir * 1e-4f,
//                         light2ShadingPointDir);
//
//  // Trace the shadow ray
//  auto intersection_status = traceScene(light2ShadingPoint);
//
//  // Compute distances with double precision for accuracy
//  double distToIntersection = glm::length(
//      glm::dvec3(lightSample.coords) -
//      glm::dvec3(intersection_status.coords));
//  double distToLight = glm::length(glm::dvec3(lightSample.coords) -
//                                   glm::dvec3(shadeObjIntersection.coords));
//
//  if (!intersection_status.intersected) {
//    return glm::vec3(0.f);
//  }
//
//  // Use squared distance for better numerical stability
//  double distanceSquared = distToIntersection * distToIntersection;
//  if (distanceSquared < static_cast<double>(m_epsilon)) {
//    return glm::vec3(0.f);
//  }
//
//  if (std::abs(distToIntersection - distToLight) < static_cast<double>(1e-4f))
//  {
//    // Compute angles
//    float object_theta = std::max(0.f, glm::dot(N, -light2ShadingPointDir));
//    float light_theta =
//        std::max(0.f, glm::dot(lightSample.normal, light2ShadingPointDir));
//
//    // Check if the angles are valid
//    if (object_theta < m_epsilon || light_theta < m_epsilon) {
//      return glm::vec3(0.f);
//    }
//
//    auto Li = lightSample.emit;
//    auto Fr = shadeObjIntersection.obj->getMaterial()->fr_contribution(
//        -light2ShadingPointDir, wo, N); // BRDF
//
//    return Li * Fr * object_theta * light_theta /
//           static_cast<float>(lightAreaPdf * distanceSquared);
//  }
//  return glm::vec3(0.f);
//}

// Calculate Point From Indirect Light
glm::vec3 SoftRasterizer::Scene::pathTracingIndirectLight(
    const Intersection &shadeObjIntersection, const glm::vec3 &wo,
    const std::size_t maxRecursionDepth, std::size_t currentDepth) {

  const glm::dvec3 N = glm::normalize(shadeObjIntersection.normal);

  /*Russian Roulette with probability RussianRoulette
   * And also, This Object should not be a illumination source*/
  if (Tools::random_generator() > p_rr)
    return glm::vec3(0.f);

  glm::dvec3 wi =
      glm::normalize(shadeObjIntersection.obj->getMaterial()->sample(wo, N));

  // prevent relfection and refraction from happening at the same time
  glm::dvec3 perturbation = glm::dvec3(shadeObjIntersection.coords) + 1e-6 * N;
  Ray newray(perturbation, wi);
  Intersection nextObj = traceScene(newray);
  if (!nextObj.intersected) {
    return glm::vec3(0.f);
  }

  // if nextobj is a self-illumination object
  if (glm::length(nextObj.emit) > m_epsilon) {
    return glm::vec3(0.0f);
  }

  auto Fr = shadeObjIntersection.obj->getMaterial()->fr_contribution(wi, wo,
                                                                     N); // BRDF
  auto pdf = shadeObjIntersection.obj->getMaterial()->pdf(wi, wo, N);    // PDF
  auto object_theta = std::max(0.0, glm::dot(wi, N));

  if (std::isnan(pdf) || pdf < m_epsilon) {
    spdlog::debug("Warning: Light area PDF is too small!");
    return glm::vec3(0.0f);
  }

  glm::dvec3 indirectLight =
      pathTracingShading(nextObj, -wi, maxRecursionDepth, currentDepth + 1);
  return indirectLight * glm::dvec3(Fr) * object_theta /
         static_cast<double>(pdf * p_rr);
}

glm::vec3 SoftRasterizer::Scene::pathTracingShading(
    const Intersection &shadeObjIntersection, const glm::vec3 &wo,
    int maxRecursionDepth, int currentDepth) {

  glm::vec3 direct{0.f}, indirect{0.f};

  if (currentDepth <
      maxRecursionDepth / 2) { // Parallelize only at early recursion levels
    tbb::task_group tg;
    tg.run([&] { direct = pathTracingDirectLight(shadeObjIntersection, wo); });
    tg.run([&] {
      indirect = pathTracingIndirectLight(shadeObjIntersection, wo,
                                          maxRecursionDepth, currentDepth + 1);
    });
    tg.wait();

  } else {
    direct = pathTracingDirectLight(shadeObjIntersection, wo);
    indirect = pathTracingIndirectLight(shadeObjIntersection, wo,
                                        maxRecursionDepth, currentDepth + 1);
  }
  return direct + indirect;
}

glm::vec3 SoftRasterizer::Scene::pathTracing(Ray &ray) {

  /*Camera emits a ray, find a shading point in the scene*/
  Intersection shadeObjIntersection = traceScene(ray);
  if (!shadeObjIntersection.intersected) {
    return this->m_backgroundColor;
  }

  return pathTracingShading(shadeObjIntersection, -ray.direction);
}

void SoftRasterizer::Scene::buildBVHAccel() {
  try {
    m_bvh->loadNewObjects(m_exportedObjs);
    m_bvh->startBuilding();
    m_boundingBox = m_bvh->getBoundingBox().value();
  } catch (const std::exception &e) {
    spdlog::error("BoundingBox Processing Error in Scene {}! Error is: {}",
                  m_sceneName, e.what());
  }
}

/*Remove BVH Structure*/
void SoftRasterizer::Scene::clearBVHAccel() { m_bvh->clearBVHAccel(); }

void SoftRasterizer::Scene::updatePosition() {

  tbb::parallel_for(
      oneapi::tbb::blocked_range<std::size_t>(0, m_exportedObjs.size()),
      [&](const oneapi::tbb::blocked_range<std::size_t> &range) {
        for (auto i = range.begin(); i != range.end(); ++i) {
          const auto &modelMatrix = m_exportedObjs[i]->getModelMatrix();

          m_exportedObjs[i]->updatePosition(modelMatrix, m_view, m_projection,
                                            m_ndcToScreenMatrix);
        }
      },
      ap);

  /*Delete Existing BVH structure*/
  clearBVHAccel();

  /*Rebuild BVH*/
  buildBVHAccel();
}

tbb::concurrent_vector<SoftRasterizer::Scene::ObjTuple>
SoftRasterizer::Scene::loadTriangleStream() {

  // Use tbb::concurrent_vector to collect results safely in parallel
  tbb::concurrent_vector<ObjTuple> stream;

  // Estimate and reserve the total size for triangle_stream to avoid
  // reallocations
  stream.reserve(std::accumulate(
      m_loadedObjs.begin(), m_loadedObjs.end(), static_cast<std::size_t>(0),
      [](std::size_t sum, const auto &objPair) {
        return sum + objPair.second.mesh->getFaces().size();
      }));

  for (const auto &[meshName, objData] : m_loadedObjs) {
    const auto &mesh = objData.mesh;
    const auto &shader = mesh->m_shader;
    const auto &modelMatrix = objData.mesh->getModelMatrix();

    auto NDC_MVP = m_ndcToScreenMatrix * m_projection * m_view * modelMatrix;
    auto Normal_M = glm::transpose(glm::inverse(modelMatrix));

    tbb::concurrent_vector<SoftRasterizer::Triangle> ret;

    tbb::parallel_for(
        tbb::blocked_range<long long>(0, mesh->getFaces().size()),
        [&](const tbb::blocked_range<long long> &r) {
          for (long long face_index = r.begin(); face_index < r.end();
               ++face_index) {
            const auto &face = mesh->getFaces()[face_index];
            SoftRasterizer::Vertex A = mesh->getVertices()[face.x];
            SoftRasterizer::Vertex B = mesh->getVertices()[face.y];
            SoftRasterizer::Vertex C = mesh->getVertices()[face.z];

            A.position = Tools::to_vec3(NDC_MVP * glm::vec4(A.position, 1.0f));
            A.position.z = A.position.z * scale + offset; // Z-Depth
            A.normal = Tools::to_vec3(Normal_M * glm::vec4(A.normal, 1.0f));

            B.position = Tools::to_vec3(NDC_MVP * glm::vec4(B.position, 1.0f));
            B.position.z = B.position.z * scale + offset; // Z-Depth
            B.normal = Tools::to_vec3(Normal_M * glm::vec4(B.normal, 1.0f));

            C.position = Tools::to_vec3(NDC_MVP * glm::vec4(C.position, 1.0f));
            C.position.z = C.position.z * scale + offset; // Z-Depth
            C.normal = Tools::to_vec3(Normal_M * glm::vec4(C.normal, 1.0f));

            SoftRasterizer::Triangle T;
            T.setVertex({A.position, B.position, C.position});
            T.setNormal({A.normal, B.normal, C.normal});
            T.setTexCoord({A.texCoord, B.texCoord, C.texCoord});
            T.calcBoundingBox(m_width, m_height);

            // Thread-safe insertion into concurrent_vector
            ret.emplace_back(std::move(T));
          }
        });

    stream.push_back({shader, ret});
  }

  return stream;
}
