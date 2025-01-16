#pragma once
#ifndef _SCENE_HPP_
#define _SCENE_HPP_
#include <tuple>
#include <atomic>
#include <future>
#include <hpc/Simd.hpp>
#include <unordered_map>
#include <shader/Shader.hpp>
#include <loader/ObjLoader.hpp>

namespace  SoftRasterizer {
          class Triangle;
          class RenderingPipeline;

          class Scene {
                    friend class RenderingPipeline;

          public:
                    using ObjTuple = std::tuple<std::shared_ptr<Shader>, std::vector< SoftRasterizer::Triangle>>;
                    using ObjFuture = std::future< ObjTuple >;

          public:
                    struct ObjInfo {
                              std::unique_ptr<ObjLoader> loader;
                              std::unique_ptr<Mesh> mesh;
                    };

                    Scene(const std::string& sceneName,
                              const Eigen::Matrix4f& view = Eigen::Matrix4f::Identity(),
                              const Eigen::Matrix4f& projection = Eigen::Matrix4f::Identity());

                    virtual ~Scene();

          public:
                    const Eigen::Vector3f& loadEyeVec() const;

                    /*set MVP*/
                    bool setModelMatrix(const std::string& meshName,
                              const Eigen::Matrix4f& model);

                    void setViewMatrix(const Eigen::Vector3f& eye, const Eigen::Vector3f& center,
                              const Eigen::Vector3f& up);

                    void setProjectionMatrix(float fovy, float zNear, float zFar);

                    /*load ObjLoader object to load wavefront obj file*/
                    bool addGraphicObj(const std::string& path, const std::string& meshName);

                    bool addGraphicObj(
                              const std::string& path, const std::string& meshName,
                              const Eigen::Matrix4f& rotation,
                              const Eigen::Vector3f& translation = Eigen::Vector3f(0.f, 0.f, 0.f),
                              const Eigen::Vector3f& scale = Eigen::Matrix4f::Identity());

                    bool addGraphicObj(
                              const std::string& path, const std::string& meshName,
                              const Eigen::Vector3f& axis, const float angle,
                              const Eigen::Vector3f& translation = Eigen::Vector3f(0.f, 0.f, 0.f),
                              const Eigen::Vector3f& scale = Eigen::Matrix4f::Identity());

                    bool startLoadingMesh(const std::string& meshName);

                    bool addShader(const std::string& shaderName, const std::string& texturePath,
                              SHADERS_TYPE type);

                    bool addShader(const std::string& shaderName,
                              std::shared_ptr<TextureLoader> text, SHADERS_TYPE type);

                    bool bindShader2Mesh(const std::string& meshName,
                              const std::string& shaderName);

                    void addLight(std::string name, std::shared_ptr<light_struct> light);
                    void addLights(std::vector<std::pair<std::string, std::shared_ptr<light_struct>>>lights);

          protected:
                    /*we don't want other user to deploy it directly
                    * because we need to record detailed arguments*/
                    void setProjectionMatrix(const Eigen::Matrix4f& projection);
                    void setViewMatrix(const Eigen::Matrix4f& view);

                    std::vector<ObjTuple>  loadTriangleStream();
                    std::vector<SoftRasterizer::light_struct> loadLights();

          private:
                    /*NDC Matrix Function is prepare for renderpipeline class!*/
                    void setNDCMatrix(const std::size_t width, const std::size_t height);

          private:
                    std::string m_sceneName;

                    /*display resolution*/
                    std::size_t m_width, m_height;
                    float m_aspectRatio;

                    /*Matrix View*/
                    Eigen::Vector3f m_eye, m_center, m_up;
                    Eigen::Matrix4f m_view;

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

                    Eigen::Matrix4f m_projection;

                    /*Transform normalized coordinates into screen space coordinates*/
                    Eigen::Matrix4f m_ndcToScreenMatrix;

                    /*We Prepare this for loading fragment_shader_payload parallelly!*/
                    std::vector< ObjFuture> m_future;

                    /*store all shaders in current scene*/
                    std::unordered_map<std::string, std::shared_ptr<Shader>> m_shaders;

                    // creating the scene (adding objects and lights)
                    std::unordered_map<std::string, std::shared_ptr<light_struct>> m_lights;
                    std::unordered_map<std::string, ObjInfo> m_loadedObjs;
          };
}

#endif //_SCENE_HPP_