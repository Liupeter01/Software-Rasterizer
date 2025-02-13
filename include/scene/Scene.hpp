#pragma once
#ifndef _SCENE_HPP_
#define _SCENE_HPP_
#include <tuple>
#include <future>
#include <atomic>
#include <optional>
#include <random>   //generate random number
#include <unordered_map>
#include <hpc/Simd.hpp>
#include <shader/Shader.hpp>
#include <loader/ObjLoader.hpp>
#include <tbb/concurrent_vector.h>
#include <bvh/BVHAcceleration.hpp>

namespace SoftRasterizer {
class Triangle;
class RenderingPipeline;
class TraditionalRasterizer;
class RayTracing;
class PathTracing;

class Scene {
  friend class TraditionalRasterizer;
  friend class RayTracing;
  friend class RenderingPipeline;
  friend class PathTracing;

public:
  using ObjTuple = std::tuple<std::shared_ptr<Shader>,
                              tbb::concurrent_vector<SoftRasterizer::Triangle>>;
  using ObjFuture = std::future<ObjTuple>;

public:
  Scene(const std::string &sceneName, const glm::vec3 &eye,
        const glm::vec3 &center, const glm::vec3 &up,
        glm::vec3 m_backgroundColor = glm::vec3(0.f),
        const std::size_t maxdepth = 5, const float rr = 0.8f);

  virtual ~Scene();

public:
  const glm::vec3 &loadEyeVec() const { return m_eye; }

  /*set MVP*/
  bool setModelMatrix(const std::string &meshName, const glm::vec3 &axis,
                      const float angle, const glm::vec3 &translation,
                      const glm::vec3 &scale);

  void setViewMatrix(const glm::vec3 &eye, const glm::vec3 &center,
                     const glm::vec3 &up);

  void setProjectionMatrix(float fovy, float zNear, float zFar);

  /*load ObjLoader object to load wavefront obj file*/
  bool addGraphicObj(const std::string &path, const std::string &meshName);
  bool addGraphicObj(const std::string &path, const std::string &meshName,
                     const glm::vec3 &axis, const float angle,
                     const glm::vec3 &translation, const glm::vec3 &scale);

  bool addGraphicObj(std::unique_ptr<Object> object,
                     const std::string &objectName);

  bool startLoadingMesh(const std::string &meshName);

  std::optional<std::shared_ptr<Object>>
  getMeshObj(const std::string &meshName);

  bool addShader(const std::string &shaderName, const std::string &texturePath,
                 SHADERS_TYPE type);

  bool addShader(const std::string &shaderName,
                 std::shared_ptr<TextureLoader> text, SHADERS_TYPE type);

  bool bindShader2Mesh(const std::string &meshName,
                       const std::string &shaderName);

  void addLight(std::string name, std::shared_ptr<light_struct> light);
  void
  addLights(std::vector<std::pair<std::string, std::shared_ptr<light_struct>>>
                lights);

  void cameraLight(bool status);
  void cameraLight(const glm::vec3 &intensity);

  /*Generating BVH Structure*/
  void buildBVHAccel();

  /*Remove BVH Structure*/
  void clearBVHAccel();

  //Path Tracing
  glm::vec3 pathTracing(Ray& ray);

protected:
  void updatePosition();

  /*For Rasterizer, Not Ray Tracing*/
  tbb::concurrent_vector<SoftRasterizer::Scene::ObjTuple> loadTriangleStream();
  std::vector<SoftRasterizer::light_struct> loadLights();

private:
  /*The intensity is 0 by default*/
  void initCameraLight();

  /*NDC Matrix Function is prepare for renderpipeline class!*/
  void setNDCMatrix(const std::size_t width, const std::size_t height);

  /* Generate Pointers to Triangles and load it to BVH Structure*/
  void preGenerateBVH();

  // emit ray from eye to pixel and trace the scene to find the nearest object
  // intersected by the ray
  Intersection traceScene(Ray &ray);

  // Uniformly sample the light
  [[nodiscard]] std::tuple<Intersection, float> sampleLight();

  //Whitted Style Ray Tracing
  glm::vec3
  whittedRayTracing(Ray &ray, int depth,
                    const std::vector<SoftRasterizer::light_struct> &lights);

glm::vec3 pathTracingShading(Ray& ray, int maxRecursionDepth = 5,int currentDepth = 0);

//Calculate Points Direct light
glm::vec3
pathTracingDirectLight(const Intersection& shadeObjIntersection, Ray& ray);

//Calculate Point From Indirect Light
glm::vec3
pathTracingIndirectLight(const Intersection& shadeObjIntersection, 
                                        Ray& ray, 
                                        const std::size_t maxRecursionDepth = std::thread::hardware_concurrency() / 2,
                                        std::size_t currentDepth = 0);

private:
          /*Russian Roulette*/
          float p_rr;

  /*Scene Configuration*/
  std::string m_sceneName;
  const std::size_t m_maxDepth;
  glm::vec3 m_backgroundColor;
  Bounds3 m_boundingBox;

  /*
   * Instead of using numberic_limits, we increase it appropriately
   * To avoid floating-point precision errors
   * Small offset to prevent self-intersection artifacts
   * Ensures that reflection and refraction rays start slightly away from the
   * surface to avoid numerical precision issues when tracing subsequent rays.
   */
  const float m_epsilon = 1e-5f;

  /*display resolution*/
  std::size_t m_width, m_height;
  float m_aspectRatio;

  /*Camera Light Status*/
  std::shared_ptr<light_struct> m_cameraLight;

  /*Matrix View*/
  glm::vec3 m_eye, m_center, m_up;
  glm::mat4 m_view;

  /*Matrix Projection*/
  // near and far clipping planes
  float m_fovy, m_near = 0.1f, m_far = 100.0f;

  // controls the stretching/compression of the  & shifts the range
  float scale, offset;

#if defined(__x86_64__) || defined(_WIN64)
  __m256 scale_simd;
  __m256 offset_simd;

  const __m256 zero = _mm256_set1_ps(0.0f);
  const __m256 one = _mm256_set1_ps(1.0f);

  /*decribe inf distance in z buffer*/
  const __m256 inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());

#elif defined(__arm__) || defined(__aarch64__)
  simde__m256 scale_simd;
  simde__m256 offset_simd;

  const simde__m256 zero = simde_mm256_set1_ps(0.0f);
  const simde__m256 one = simde_mm256_set1_ps(1.0f);

  /*decribe inf distance in z buffer*/
  const simde__m256 inf =
      simde_mm256_set1_ps(std::numeric_limits<float>::infinity());

#else
#endif

  glm::mat4 m_projection;

  /*Transform normalized coordinates into screen space coordinates*/
  glm::mat4 m_ndcToScreenMatrix;

  /*BVH Acceleration Structure*/
  std::unique_ptr<BVHAcceleration> m_bvh;

  /*We Prepare this for loading fragment_shader_payload parallelly!*/
  std::vector<ObjFuture> m_future;

  /*store all shaders in current scene*/
  std::unordered_map<std::string, std::shared_ptr<Shader>> m_shaders;

  // creating the scene (adding objects and lights)
  std::unordered_map<std::string, std::shared_ptr<light_struct>> m_lights;

  struct ObjInfo {
    std::optional<std::unique_ptr<ObjLoader>> loader;
    std::unique_ptr<Object> mesh;
  };

  /*All Loaded Objects, including Triangle, Sphere, Mesh, Cube*/
  std::unordered_map<std::string, ObjInfo> m_loadedObjs;

  /*Pointers of All Loaded Objects, Used for Creating BVH*/
  tbb::concurrent_vector<std::shared_ptr<Object>> m_exportedObjs;

  oneapi::tbb::affinity_partitioner ap;
};
} // namespace SoftRasterizer

#endif //_SCENE_HPP_
