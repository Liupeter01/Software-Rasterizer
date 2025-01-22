#include <Tools.hpp>
#include <base/Render.hpp>
#include <numeric> // For std::accumulate
#include <scene/Scene.hpp>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>

SoftRasterizer::Scene::Scene(const std::string& sceneName, const glm::vec3& eye,
          const glm::vec3& center, const glm::vec3& up, glm::vec3 backgroundColor, const std::size_t maxdepth)
          : m_width(0), m_height(0), m_sceneName(sceneName), m_maxDepth(maxdepth), m_backgroundColor(backgroundColor),
          m_eye(eye), m_center(center), m_up(up), m_fovy(45.0f), m_aspectRatio(0.0f), scale(0.0f), offset(0.0f),
      m_bvh(std::make_unique<BVHAcceleration>()) {
  try {
    setViewMatrix(eye, center, up);
  } catch (const std::exception &e) {
    spdlog::error("Scene Constructor Error! Reason: {}", e.what());
  }
}

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

bool SoftRasterizer::Scene::startLoadingMesh(const std::string &meshName, MaterialType _type,
          const glm::vec3& _color,
          const glm::vec3& _Ka ,
          const glm::vec3& _Kd,
          const glm::vec3& _Ks,
          const float _specularExponent,
          const float _ior) {

  /*This Object has already been identified!*/
  if (m_loadedObjs.find(meshName) == m_loadedObjs.end()) {
    spdlog::error("Start Loading Mesh Failed! Because There is nothing found "
                  "in m_suspendObjs");
    return false;
  }

  if (m_loadedObjs[meshName].mesh != nullptr) {
    spdlog::error("Start Loading Mesh Failed! Because {} Has Already Loaded "
                  "into m_loadedObjs",
                  meshName);
    return false;
  }

  std::optional<std::unique_ptr<Mesh>> mesh_op =
            m_loadedObjs[meshName].loader->startLoadingFromFile(meshName, _type, _color, _Ka, _Kd, _Ks, _specularExponent, _ior);
  if (!mesh_op.has_value()) {
    spdlog::error("Start Loading Mesh Failed! Because Loading Internel Error!");
    return false;
  }

  try {
    m_loadedObjs[meshName].mesh = std::move(mesh_op.value());
    m_loadedObjs[meshName].mesh->meshname = meshName;
  } catch (const std::exception &e) {
    spdlog::error("Start Loading Mesh Failed! Reason: {}", e.what());
    return false;
  }
  return true;
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

  m_loadedObjs[meshName].loader->updateModelMatrix(axis, angle, translation,
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

std::vector<SoftRasterizer::Object *> SoftRasterizer::Scene::getLoadedObjs() {
  std::vector<SoftRasterizer::Object *> ret(m_loadedObjs.size());
  std::transform(m_loadedObjs.begin(), m_loadedObjs.end(), ret.begin(),
                 [](const auto &obj) { return obj.second.mesh.get(); });
  return ret;
}

//emit ray from eye to pixel and trace the scene to find the nearest object intersected by the ray
std::optional<std::shared_ptr<SoftRasterizer::Object>> 
SoftRasterizer::Scene::traceScene(const Ray& ray, float& tNear){
          Object* nearestObj = nullptr;
          std::for_each(m_bvh->objs.begin(), m_bvh->objs.end(), [&nearestObj, &tNear, &ray](Object* obj) {
                    float temp = std::numeric_limits<float>::infinity();
                    if (obj->intersect(ray, temp)) {
                              nearestObj = temp < tNear ? obj : nearestObj;
                              tNear = temp < tNear ? temp : tNear;
                    }
          });

          if (nearestObj == nullptr || tNear < 0) {
                    return std::nullopt;
          }
          return std::shared_ptr<SoftRasterizer::Object>(nearestObj, [](Object*) {});
}

SoftRasterizer::Intersection
SoftRasterizer::Scene::traceScene(Ray& ray){
          Intersection ret;
          float tNear = std::numeric_limits<float>::infinity();
          std::for_each(m_bvh->objs.begin(), m_bvh->objs.end(), [&ret, &tNear, &ray](auto& obj) {
                    float temp = std::numeric_limits<float>::infinity();
                    if (obj->intersect(ray, temp)) {
                              if (temp < tNear) {
                                        tNear = temp;
                                        ret.obj = obj;
                                        ret.intersect_time = tNear;
                              }
                    }
           });
          if(ret.obj == nullptr || tNear < 0) {
                    ret.intersected = false;
                    return ret;
          }

          ret.coords = ray.origin + ray.direction * ret.intersect_time;
          ret.material = ret.obj->getMaterial();
          ret.normal = ret.obj->getSurfaceProperties(0, ret.coords, ray.direction, glm::vec2(0.f)).normal;
          ret.intersected = true;
          return ret;
}

glm::vec3 SoftRasterizer::Scene::whittedRayTracing(Ray& ray, int depth, const  std::vector<SoftRasterizer::light_struct>& lights){
          if (depth > m_maxDepth) {
                    //Return black if the ray has reached the maximum depth
                    return glm::vec3(0.f);
          }

          Intersection intersection = traceScene(ray);
          if (!intersection.intersected || !intersection.obj) {
                    //Return black if the ray does not intersect with any object
                    spdlog::debug("Ray Intersection Not Found On Depth {}", depth);
                    return  this->m_backgroundColor;
          }

          glm::vec3 final_color = this->m_backgroundColor;

          //Get the hit point
          auto hitPoint = intersection.coords;

          /*DON NOT FORGET TO NORMALIZE THE NORMAL*/
          auto rayDirection = glm::normalize(ray.direction);
          auto hitNormal = glm::normalize(intersection.normal);

          /*Phong illumation model*/
          if (intersection.material->getMaterialType() == MaterialType::DIFFUSE_AND_GLOSSY) {

                    /*Self-Intersection Problem Avoidance*/
                    glm::vec3 shadowCoord = glm::dot(rayDirection, hitNormal) < 0 ?
                              hitPoint - hitNormal * std::numeric_limits<float>::epsilon() :
                              hitPoint + hitNormal * std::numeric_limits<float>::epsilon();

                    glm::vec3 ambient(glm::vec3(0.f)), specular(glm::abs(glm::vec3(0.f))), diffuse(glm::vec3(0.f));

                    for (const auto& light : lights) {
                              /*the direction of intersected coordinate to light position*/
                              float distance = std::sqrt(std::pow(light.position.x - hitPoint.x, 2) + std::pow(light.position.y - hitPoint.y, 2));
                              glm::vec3 lightDir = glm::normalize(light.position - hitPoint);
                              glm::vec3 reflectDir = glm::reflect(-lightDir, hitNormal);            //reflection direction
                              glm::vec3 intensity = light.intensity / distance;

                              //Diffuse reflection (Lambertian reflectance)
                              float diff = std::max(0.f, glm::dot(hitNormal, lightDir));

                              //Specular reflection(Blinn - Phong)
                              float spec = std::pow(std::max(0.f, glm::dot(hitNormal, reflectDir)), intersection.material->specularExponent);

                              /*
                              * Ambient lighting
                              * Emit A ray to from point to light, to see is there any obstacles
                              * When intersected = true, it means there is an obstacle between the point and the light source
                              */
                              ambient += traceScene(Ray(shadowCoord, lightDir)).intersected ? glm::vec3(0.f) : light.intensity;
                              diffuse += intensity * diff;
                              specular += intensity * spec;
                    }

                    final_color = ambient * intersection.material->Ka + diffuse * intersection.material->Kd + specular * intersection.material->Ks;
          }
          else if (intersection.material->getMaterialType() == MaterialType::REFLECTION_AND_REFRACTION) {
                    /*Safety Consideration*/
                    auto reflectPath = glm::normalize(glm::reflect(rayDirection, hitNormal));
                    auto refractPath = glm::normalize(glm::refract(rayDirection, hitNormal, intersection.material->ior));

                    //prevent relfection and refraction from happening at the same time
                    auto reflectCoord = glm::dot(reflectPath, hitNormal) < 0 ? 
                              hitPoint - hitNormal * std::numeric_limits<float>::epsilon() : 
                              hitPoint + hitNormal * std::numeric_limits<float>::epsilon();

                    auto refractCoord = glm::dot(refractPath, hitNormal) < 0 ?
                              hitPoint - hitNormal * std::numeric_limits<float>::epsilon() :
                              hitPoint + hitNormal * std::numeric_limits<float>::epsilon();

                    Ray reflectedRay(reflectCoord, reflectPath);
                    Ray refractedRay(refractCoord, refractPath);

                    glm::vec3 reflectedColor = whittedRayTracing(reflectedRay, depth + 1, lights);
                    glm::vec3 refractedColor = whittedRayTracing(refractedRay, depth + 1, lights);

                    float kr = Tools::fresnel(rayDirection, hitNormal, intersection.material->ior);
                    final_color = (reflectedColor * kr + refractedColor * (1 - kr));    
          }
          else if (intersection.material->getMaterialType() == MaterialType::REFLECTION) {
                    /*Safety Consideration*/
                    auto reflectPath = glm::normalize(glm::reflect(rayDirection, hitNormal));

                    //prevent relfection and refraction from happening at the same time
                    auto reflectCoord = glm::dot(reflectPath, hitNormal) < 0 ?
                              hitPoint - hitNormal * std::numeric_limits<float>::epsilon() :
                              hitPoint + hitNormal * std::numeric_limits<float>::epsilon();

                    Ray reflectedRay(reflectCoord, reflectPath);
                    glm::vec3 reflectedColor = whittedRayTracing(reflectedRay, depth + 1, lights);
                    float kr = Tools::fresnel(rayDirection, hitNormal, intersection.material->ior);
                    final_color += reflectedColor * kr;
          }

          return final_color;
}

void SoftRasterizer::Scene::buildBVHAccel() {
  m_bvh->loadNewObjects(getLoadedObjs());
  m_bvh->startBuilding();
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
        return sum + objPair.second.mesh->faces.size();
      }));

  for (const auto &[meshName, objData] : m_loadedObjs) {
    const auto &mesh = objData.mesh;
    const auto &shader = mesh->m_shader;
    const auto &modelMatrix = objData.loader->getModelMatrix();

    auto NDC_MVP = m_ndcToScreenMatrix * m_projection * m_view * modelMatrix;
    auto Normal_M = glm::transpose(glm::inverse(modelMatrix));

    tbb::concurrent_vector<SoftRasterizer::Triangle> ret;

    tbb::parallel_for(
        tbb::blocked_range<long long>(0, mesh->faces.size()),
        [&](const tbb::blocked_range<long long> &r) {
          for (long long face_index = r.begin(); face_index < r.end();
               ++face_index) {
            const auto &face = mesh->faces[face_index];
            SoftRasterizer::Vertex A = mesh->vertices[face.x];
            SoftRasterizer::Vertex B = mesh->vertices[face.y];
            SoftRasterizer::Vertex C = mesh->vertices[face.z];

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
