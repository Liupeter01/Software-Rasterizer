#include <Tools.hpp>
#include <base/Render.hpp>
#include <numeric> // For std::accumulate
#include <scene/Scene.hpp>
#include <shader/Shader.hpp>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>

SoftRasterizer::Scene::Scene(const std::string &sceneName, const glm::vec3 &eye,
                             const glm::vec3 &center, const glm::vec3 &up,
                             glm::vec3 backgroundColor,
                             const std::size_t maxdepth)
    : m_width(0), m_height(0), m_sceneName(sceneName), m_maxDepth(maxdepth),
      m_backgroundColor(backgroundColor), m_eye(eye), m_center(center),
      m_up(up), m_fovy(45.0f), m_aspectRatio(0.0f), scale(0.0f), offset(0.0f),
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

const glm::vec3 &SoftRasterizer::Scene::loadEyeVec() const { return m_eye; }

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

// emit ray from eye to pixel and trace the scene to find the nearest object
// intersected by the ray
std::optional<std::shared_ptr<SoftRasterizer::Object>>
SoftRasterizer::Scene::traceScene(const Ray &ray, float &tNear) {
  Object *nearestObj = nullptr;
  std::for_each(m_bvh->objs.begin(), m_bvh->objs.end(),
                [&nearestObj, &tNear, &ray](Object *obj) {
                  float temp = std::numeric_limits<float>::infinity();
                  if (obj->intersect(ray, temp)) {
                    nearestObj = temp < tNear ? obj : nearestObj;
                    tNear = temp < tNear ? temp : tNear;
                  }
                });

  if (nearestObj == nullptr || tNear < 0) {
    return std::nullopt;
  }
  return std::shared_ptr<SoftRasterizer::Object>(nearestObj, [](Object *) {});
}

SoftRasterizer::Intersection SoftRasterizer::Scene::traceScene(Ray &ray) {
  Intersection ret;
  float tNear = std::numeric_limits<float>::infinity();
  std::for_each(
      m_bvh->objs.begin(), m_bvh->objs.end(), [&ret, &tNear, &ray](auto &obj) {
        Intersection intersect = obj->getIntersect(ray);
        if (intersect.intersected && intersect.intersect_time < tNear) {
          tNear = intersect.intersect_time;
          ret.obj = intersect.obj;
          ret.intersect_time = intersect.intersect_time;
        }
      });

  /*Invalid Intersection*/
  if (!ret.obj || tNear < 0) {
    return {};
  }

  /*
   * Valid Intersection is here! We Are going to get properties by using
   * obj->getSurfaceProperties getSurfaceProperties method could belong to
   * Sphere, Mesh(Triangle), Cube Classes Every object inhertied classes should
   * implement getSurfaceProperties method!!!
   */
  ret.index = ret.obj->index;
  ret.coords = ray.origin + ray.direction * ret.intersect_time;
  ret.material = ret.obj->getMaterial();

  /*If it is a mesh, then the object is triangle*/
  ret.normal = ret.obj
                   ->getSurfaceProperties(ret.index, ret.coords, ray.direction,
                                          glm::vec2(0.f))
                   .normal;
  ret.intersected = true;
  return ret;
}

glm::vec3 SoftRasterizer::Scene::whittedRayTracing(
    Ray &ray, int depth,
    const std::vector<SoftRasterizer::light_struct> &lights) {
  glm::vec3 final_color = this->m_backgroundColor;
  if (depth > m_maxDepth) {
    // Return black if the ray has reached the maximum depth
    return glm::vec3(0.f);
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

  /*Phong illumation model*/
  if (intersection.material->getMaterialType() ==
      MaterialType::DIFFUSE_AND_GLOSSY) {

    /*Self-Intersection Problem Avoidance*/
    glm::vec3 shadowCoord =
        glm::dot(rayDirection, hitNormal) < 0
            ? hitPoint - hitNormal * std::numeric_limits<float>::epsilon()
            : hitPoint + hitNormal * std::numeric_limits<float>::epsilon();

    glm::vec3 ambient(0.f);
    glm::vec3 diffuse(0.f);
    glm::vec3 specular(0.f);
    glm::vec3 result(0.f);

    for (const auto &light : lights) {
      glm::vec3 lightDir = light.position - hitPoint;

      /*the direction of intersected coordinate to light position*/
       float distance = glm::dot(lightDir, lightDir);
      // glm::vec3 distribution = light.intensity / distance;

      lightDir = glm::normalize(lightDir);

      // Diffuse reflection (Lambertian reflectance)
      float diff = std::max(0.f, glm::dot(hitNormal, lightDir));

      // Specular reflection(Blinn - Phong)
      glm::vec3 reflectDir = glm::normalize(
          glm::reflect(-lightDir, hitNormal)); // reflection direction

      float spec = std::pow(std::max(0.f, -glm::dot(ray.direction, reflectDir)),
                            intersection.material->specularExponent);

      /*
       * Ambient lighting
       * Emit A ray to from point to light, to see is there any obstacles
       * When intersected = true, it means there is an obstacle between the
       * point and the light source
       */
      Ray ray(shadowCoord, lightDir);
      Intersection shadow_result = traceScene(ray);

      // is the point in shadow, and is the nearest occluding object closer to the object than the light itself?
      bool is_shadow = shadow_result.intersected && (std::pow(shadow_result.intersect_time, 2.f) < distance);

      ambient += !is_shadow ?  light.intensity : glm::vec3(0.f);
      diffuse += !is_shadow ? glm::vec3(diff)  : glm::vec3(0.f);
      specular += spec * light.intensity;
    }

    final_color = 
              ambient * intersection.material->Ka
              + diffuse * intersection.material->Kd * intersection.obj->getDiffuseColor(glm::vec2(0.f))
              + specular * intersection.material->Ks;
  } 
  
  else if (intersection.material->getMaterialType() ==
             MaterialType::REFLECTION_AND_REFRACTION) {

    /*Safety Consideration*/
    auto reflectPath = glm::normalize(glm::reflect(rayDirection, hitNormal));
    auto refractPath = glm::normalize(
        glm::refract(rayDirection, hitNormal, intersection.material->ior));

    // prevent relfection and refraction from happening at the same time
    auto reflectCoord =
        glm::dot(reflectPath, hitNormal) < 0
            ? hitPoint - hitNormal * std::numeric_limits<float>::epsilon()
            : hitPoint + hitNormal * std::numeric_limits<float>::epsilon();

    auto refractCoord =
        glm::dot(refractPath, hitNormal) < 0
            ? hitPoint - hitNormal * std::numeric_limits<float>::epsilon()
            : hitPoint + hitNormal * std::numeric_limits<float>::epsilon();

    Ray reflectedRay(reflectCoord, reflectPath);
    Ray refractedRay(refractCoord, refractPath);

    glm::vec3 reflectedColor =
        whittedRayTracing(reflectedRay, depth + 1, lights);
    glm::vec3 refractedColor =
        whittedRayTracing(refractedRay, depth + 1, lights);

    float kr =
        Tools::fresnel(rayDirection, hitNormal, intersection.material->ior);
    final_color = (reflectedColor * kr + refractedColor * (1 - kr));
  } else if (intersection.material->getMaterialType() ==
             MaterialType::REFLECTION) {
    /*Safety Consideration*/
    auto reflectPath = glm::normalize(glm::reflect(rayDirection, hitNormal));

    // prevent relfection and refraction from happening at the same time
    auto reflectCoord =
        glm::dot(reflectPath, hitNormal) < 0
            ? hitPoint - hitNormal * std::numeric_limits<float>::epsilon()
            : hitPoint + hitNormal * std::numeric_limits<float>::epsilon();

    Ray reflectedRay(reflectCoord, reflectPath);
    glm::vec3 reflectedColor =
        whittedRayTracing(reflectedRay, depth + 1, lights);
    float kr =
        Tools::fresnel(rayDirection, hitNormal, intersection.material->ior);
    final_color += reflectedColor * kr;
  }

  return final_color;
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

  for (const auto &[meshName, objData] : m_loadedObjs) {
    const auto &modelMatrix = objData.mesh->getModelMatrix();

    auto NDC_MVP =
        /*m_ndcToScreenMatrix **/ m_projection * m_view * modelMatrix;
    auto Normal_M = glm::transpose(glm::inverse(modelMatrix));

    objData.mesh->updatePosition(NDC_MVP, Normal_M);
  }

  ///*Start to generate pointers to triangles*/
  // preGenerateBVH();

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
