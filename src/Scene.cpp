#include <Tools.hpp>
#include <base/Render.hpp>
#include <glm/geometric.hpp>
#include <glm/gtx/norm.hpp>
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
    initCameraLight();
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
  m_projection[1][1] *= -1;
}

std::vector<SoftRasterizer::light_struct> SoftRasterizer::Scene::loadLights() {
  std::vector<SoftRasterizer::light_struct> res(m_lights.size());
  std::transform(m_lights.begin(), m_lights.end(), res.begin(),
                 [](const decltype(m_lights)::value_type &light) {
                   return *light.second;
                 });
  return res;
}

void SoftRasterizer::Scene::initCameraLight() {
  m_cameraLight.reset();
  m_cameraLight = std::make_shared<light_struct>(m_eye, glm::vec3(0.f));

  addLight("sys_camera", m_cameraLight);
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

glm::vec3 SoftRasterizer::Scene::whittedRayTracing(
    Ray &ray, int depth,
    const std::vector<SoftRasterizer::light_struct> &lights) {

  glm::vec3 final_color = this->m_backgroundColor;

  if (depth > m_maxDepth) {
    // Return black if the ray has reached the maximum depth
    return glm::vec3(0.f);
    // return this->m_backgroundColor;
  }

  Intersection intersection = traceScene(ray);
  if (!intersection.intersected || !intersection.obj) {
    // Return black if the ray does not intersect with any object
    spdlog::debug("Ray Intersection Not Found On Depth {}", depth);
    return this->m_backgroundColor;
  }

  // Get the hit point
  auto hitPoint = intersection.coords;

  /*DON NOT FORGET TO NORMALIZE THE NORMAL*/
  auto rayDirection = glm::normalize(ray.direction);
  auto hitNormal = glm::normalize(intersection.normal);

  const float ior = intersection.material->ior;
  const glm::vec3 I = rayDirection;
  const glm::vec3 N =
      hitNormal; // We consider it as the surface normal by default

  float kr = std::clamp(Tools::fresnel(I, N, ior), 0.f, 1.f);

  /*Phong illumation model*/
  if (intersection.material->getMaterialType() ==
      MaterialType::DIFFUSE_AND_GLOSSY) {

    /*Self-Intersection Problem Avoidance*/
    glm::vec3 shadowCoord = glm::dot(I, N) < 0 ? hitPoint - N * m_epsilon
                                               : hitPoint + N * m_epsilon;

    final_color = tbb::parallel_reduce(
        tbb::blocked_range<std::size_t>(0, lights.size()),
        glm::vec3(0.0f), // Initial value
        [&](const tbb::blocked_range<std::size_t> &range,
            glm::vec3 local_sum) -> glm::vec3 {
          for (std::size_t i = range.begin(); i < range.end(); ++i) {
            glm::vec3 lightDir = lights[i].position - hitPoint;
            float distance = glm::dot(lightDir, lightDir);
            lightDir = glm::normalize(lightDir);

            // Diffuse reflection (Lambertian)
            float diff = std::max(0.f, glm::dot(N, lightDir));

            // Specular reflection (Blinn-Phong)
            glm::vec3 reflectDir =
                glm::normalize(glm::reflect(-lightDir, hitNormal));
            float spec =
                std::pow(std::max(0.f, -glm::dot(ray.direction, reflectDir)),
                         intersection.material->specularExponent);

            // Shadow test
            Ray shadow_ray(shadowCoord, lightDir);
            Intersection shadow_result = traceScene(shadow_ray);
            bool is_shadow =
                shadow_result.intersected &&
                (std::pow(shadow_result.intersect_time, 2.f) < distance);

            // Compute light contribution
            glm::vec3 ambient =
                !is_shadow ? lights[i].intensity : glm::vec3(0.f);
            glm::vec3 diffuse = !is_shadow
                                    ? glm::vec3(diff) * lights[i].intensity
                                    : glm::vec3(0.f);
            glm::vec3 specular = spec * lights[i].intensity;

            // Accumulate to local sum
            local_sum += (ambient * intersection.material->Ka) +
                         (intersection.color * diffuse) +
                         (specular * intersection.material->Ks);
          }
          return local_sum;
        },
        [](const glm::vec3 &a, const glm::vec3 &b) {
          return a + b;
        } // Combine partial results
    );
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
      refractPath =
          glm::refract(I, -N, ior); // Adjust refraction for material to air
    }

    // calculate offset
    auto offset = glm::dot(I, N) < 0 ? -N * m_epsilon : N * m_epsilon;

    // prevent relfection and refraction from happening at the same time
    auto reflectCoord = hitPoint + offset;
    auto refractCoord = hitPoint + offset;

    /* Total Internal Reflection, TIR */
    if (glm::length(refractPath) < 1e-6f || std::abs(kr - 1.0f) < 1e-6f) {
      // prevent relfection and refraction from happening at the same time
      Ray reflectedRay(reflectCoord, reflectPath);
      reflectedColor = whittedRayTracing(reflectedRay, depth + 1, lights);
      kr = 1.0f;
    } else {
      Ray reflectedRay(reflectCoord, reflectPath);
      Ray refractedRay(refractCoord, refractPath);

      reflectedColor = whittedRayTracing(reflectedRay, depth + 1, lights);
      refractedColor = whittedRayTracing(refractedRay, depth + 1, lights);
    }

    final_color = glm::clamp(reflectedColor * kr + refractedColor * (1.f - kr),
                             glm::vec3(0.0f), glm::vec3(1.0f));

  }

  else if (intersection.material->getMaterialType() ==
           MaterialType::REFLECTION) {

    glm::vec3 reflectPath = glm::vec3(0.0f);
    glm::vec3 reflectedColor = glm::vec3(0.0f);

    reflectPath = glm::normalize(glm::reflect(I, N));

    // calculate offset
    auto offset = glm::dot(I, N) < 0 ? -N * m_epsilon : N * m_epsilon;

    auto reflectCoord = hitPoint + offset;

    Ray reflectedRay(reflectCoord, reflectPath);

    reflectedColor = whittedRayTracing(reflectedRay, depth + 1, lights);

    final_color =
        glm::clamp(reflectedColor * kr, glm::vec3(0.0f), glm::vec3(1.0f));
  }

  return final_color;
}

// Calculate Points Direct light
glm::vec3 SoftRasterizer::Scene::pathTracingDirectLight(
    const Intersection &shadeObjIntersection, Ray &ray) {

  const glm::vec3 I = glm::normalize(ray.direction);
  const glm::vec3 N = glm::normalize(shadeObjIntersection.normal);
  const glm::vec3 wi = -I;

  /*  Sampling The Light, Finding The Intersection point on the light, and its
   * Pdf*/
  auto [lightSample, lightAreaPdf] = sampleLight();

  glm::vec3 light2ShadingPointDir =
      glm::normalize(shadeObjIntersection.coords - lightSample.coords);

  Ray light2ShadingPoint(lightSample.coords, light2ShadingPointDir);

  auto intersection_status = traceScene(light2ShadingPoint);
  if (!intersection_status.intersected) {
    return glm::vec3(0.f);
  }

  // Shadow Detection: If the ray is not blocked in the middle
  // And the intersection point is NOT a self-illuminate light source
  float distToIntersection =
      glm::length(lightSample.coords - intersection_status.coords);
  float distToLight =
      glm::length(lightSample.coords - shadeObjIntersection.coords);
  if (std::abs(distToIntersection - distToLight) > m_epsilon &&
      glm::length(intersection_status.emit) < m_epsilon) {
    return glm::vec3(0.f);
  }

  auto distanceSquare = std::max(0.f, pow(distToLight, 2.f));

  /*Radiant Radiance (L)*/
  glm::vec3 ObjectNormal =
      glm::faceforward(N, light2ShadingPointDir, -N); // Correct normal facing
  glm::vec3 LightNormal =
      glm::faceforward(lightSample.normal, light2ShadingPointDir,
                       -lightSample.normal); // Light normal

  auto object_theta =
      std::max(0.f, glm::dot(ObjectNormal, light2ShadingPointDir));
  auto light_theta =
      std::max(0.f, glm::dot(LightNormal, light2ShadingPointDir));

  auto Li = lightSample.emit;
  auto Fr = shadeObjIntersection.obj->getMaterial()->fr_contribution(
      wi, light2ShadingPointDir, ObjectNormal); /*BRDF*/

  return Li * Fr * object_theta * light_theta / (lightAreaPdf * distanceSquare);
}

// Calculate Point From Indirect Light
glm::vec3 SoftRasterizer::Scene::pathTracingIndirectLight(
    const Intersection &shadeObjIntersection, Ray &ray,
    const std::size_t maxRecursionDepth, std::size_t currentDepth) {

  const glm::vec3 I = glm::normalize(ray.direction);
  const glm::vec3 N = glm::normalize(shadeObjIntersection.normal);
  const glm::vec3 wi = -I;

  /*Russian Roulette with probability RussianRoulette
   * And also, This Object should not be a illumination source*/
  auto p = Tools::random_generator();
  if (p > p_rr) {
    return glm::vec3(0.f);
  }

  auto ObjectNormal = glm::faceforward(N, wi, -N);
  glm::vec3 wo = glm::normalize(
      shadeObjIntersection.obj->getMaterial()->sample(wi, ObjectNormal));

  // prevent relfection and refraction from happening at the same time
  Ray newray(shadeObjIntersection.coords, wo);

  auto Fr = shadeObjIntersection.obj->getMaterial()->fr_contribution(
      wi, wo, ObjectNormal); // BRDF
  auto pdf = std::max(
      shadeObjIntersection.obj->getMaterial()->pdf(wi, wo, ObjectNormal),
      m_epsilon); // PDF
  auto object_theta = std::max(0.f, glm::dot(wi, ObjectNormal));

  // Skip Recursive Function
  if (object_theta < m_epsilon || pdf < m_epsilon || Fr == glm::vec3(0.f)) {
    return glm::vec3(0.f);
  }

  return Fr * object_theta / (pdf * p) * pathTracingShading(newray, maxRecursionDepth, currentDepth + 1);
}

glm::vec3 SoftRasterizer::Scene::pathTracingShading(Ray &ray,
                                                    int maxRecursionDepth,
                                                    int currentDepth) {

  /*Camera emits a ray, find a shading point in the scene*/
  Intersection shadeObjIntersection = traceScene(ray);
  if (!shadeObjIntersection.intersected) {
    return glm::vec3(0.f);
  }

  glm::vec3 direct = pathTracingDirectLight(shadeObjIntersection, ray);

  /*indirect light should not hit a light source!*/
  glm::vec3 indirect = glm::vec3(0.f);
  if (glm::length(shadeObjIntersection.emit) < m_epsilon) {

            if (currentDepth < maxRecursionDepth) {
                      tbb::task_group tg;
                      tg.run([&]() {

                                indirect = pathTracingIndirectLight(shadeObjIntersection, ray,
                                          maxRecursionDepth, currentDepth + 1);
                                });
                      tg.wait();
            }
            else {
                      indirect = pathTracingIndirectLight(shadeObjIntersection, ray,
                                maxRecursionDepth, currentDepth + 1);
            }
  }
  return direct + indirect;
}

glm::vec3 SoftRasterizer::Scene::pathTracing(Ray &ray) {

  /*Camera emits a ray, find a shading point in the scene*/
  Intersection shadeObjIntersection = traceScene(ray);
  if (!shadeObjIntersection.intersected) {
    return this->m_backgroundColor;
  }

  /*Maybe this Ray Could hit the self-illuminateion Object directly*/
  if (glm::length(shadeObjIntersection.emit) > m_epsilon) {
    return shadeObjIntersection.color;
  }

  return glm::clamp(pathTracingShading(ray), glm::vec3(0.f), glm::vec3(1.f));
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

          auto NDC_MVP =
              /*m_ndcToScreenMatrix **/ m_projection * m_view * modelMatrix;
          auto Normal_M =
              glm::mat4(glm::transpose(glm::inverse(glm::mat3(modelMatrix))));

          m_exportedObjs[i]->updatePosition(NDC_MVP, Normal_M);
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
