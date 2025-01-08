#pragma once
#ifndef _SHADER_HPP_
#define _SHADER_HPP_
#include <array>
#include <memory>
#include <functional>
#include <Eigen/Eigen>
#include <initializer_list>
#include <TextureLoader.hpp>

namespace SoftRasterizer {

          /*forward declartion*/
          class Render;
          class Shader;

          struct light_struct {
                    Eigen::Vector3f position;
                    Eigen::Vector3f intensity;
          };

          struct vertex_displacement {
                    vertex_displacement(const Eigen::Vector3f& pos, const Eigen::Vector3f& normal)
                              :new_position(pos),new_normal(normal)
                    { }
                    Eigen::Vector3f new_position;
                    Eigen::Vector3f new_normal;
          };

          enum class SHADERS_TYPE : std::uint8_t{
                    NORMAL = 0,
                    TEXTURE,
                    PHONG,
                    DISPLACEMENT,
                    BUMP
          };

          /*pixel shader*/
          struct fragment_shader_payload {
                    fragment_shader_payload(const Eigen::Vector3f& _p,
                                                  const Eigen::Vector3f& _n,
                                                  const Eigen::Vector2f& _texcoord,
                                                  const Eigen::Vector3f& _color = Eigen::Vector3f(1.f, 1.f, 1.f));

                    Eigen::Vector3f position = Eigen::Vector3f(0.f, 0.f, 0.f);
                    Eigen::Vector3f normal = Eigen::Vector3f(0.f, 0.f, 0.f);
                    Eigen::Vector2f texCoords = Eigen::Vector2f(0.f, 0.f);
                    Eigen::Vector3f color = Eigen::Vector3f(1.f, 1.f, 1.f);
          };

          struct Shader {
                    using shader_type =  std::function< Eigen::Vector3f(const Eigen::Vector3f&, 
                                                                                const std::initializer_list<light_struct>&,
                                                                                const fragment_shader_payload&)>;

                    static Eigen::Vector3f ka;
                    static Eigen::Vector3f ks;
                    static float p, kh, kn;

          public:
                    Shader(const std::string& path);
                    Shader(std::shared_ptr<TextureLoader> _loader);

          public:
                    // Static function to compute the Blinn-Phong reflection model
                    static Eigen::Vector3f BlinnPhong(const Eigen::Vector3f& camera,
                              const fragment_shader_payload& shading_point,
                              const light_struct &light,
                              const Eigen::Vector3f& ka,
                              const Eigen::Vector3f& kd,
                              const Eigen::Vector3f& ks,
                              const float p
                    );

                    bool setFragmentShader(SHADERS_TYPE type);
                    
                    /*User Vertex Shader*/
                    vertex_displacement applyVertexShader(const Eigen::Matrix4f& Model,
                                                                                const Eigen::Matrix4f& View,
                                                                                const Eigen::Matrix4f& Projection,
                                                                                const fragment_shader_payload& payload);

                    /*Use Fragment Shader*/
                    Eigen::Vector3f applyFragmentShader(const Eigen::Vector3f& camera,
                                                                                   const std::initializer_list<light_struct>& lights,
                                                                                    const fragment_shader_payload& payload);

          private:
                    /*register multiple shader models*/
                    void registerShaders();

                    /*Visualizing normal directions or checking surface normal directions in some debugging scenarios*/
                    Eigen::Vector3f normal_fragment_shader_impl(const Eigen::Vector3f& camera,
                                                                                                 const std::initializer_list<light_struct>& lights,
                                                                                                 const fragment_shader_payload& payload);

                    Eigen::Vector3f texture_fragment_shader_impl(const Eigen::Vector3f& camera, 
                                                                                                 const std::initializer_list<light_struct>& lights,
                                                                                                  const fragment_shader_payload& payload);

                    Eigen::Vector3f phong_fragment_shader_impl(const Eigen::Vector3f& camera, 
                                                                                                const std::initializer_list<light_struct>& lights,
                                                                                                 const fragment_shader_payload& payload);

                    Eigen::Vector3f displacement_fragment_shader_impl(const Eigen::Vector3f& camera, 
                                                                                                           const std::initializer_list<light_struct>& lights,
                                                                                                           const fragment_shader_payload& payload);
                 
                    Eigen::Vector3f bump_fragment_shader_impl(const Eigen::Vector3f& camera, 
                                                                                               const std::initializer_list<light_struct>& lights,
                                                                                               const fragment_shader_payload& payload);
                    /*Compute Bump Mapping*/
                    Eigen::Vector3f calcBumpMapping(const fragment_shader_payload& payload,
                                                                              const float kh, 
                                                                              const float kn);

                    /*Compute Displacement Mapping*/
                    vertex_displacement calcDisplacementMapping(const fragment_shader_payload& payload,
                                                                                          const float kh,
                                                                                          const float kn);
                    
          private:
                    /*All Shader Type That You Could Choose*/
                    std::array<shader_type, 5> shaders{
                                 //NORMAL shader_type normal_fragment_shader;
                                 //TEXTURE shader_type texture_fragment_shader;
                                 //PHONG shader_type phong_fragment_shader;
                                 //DISPLACEMENT shader_type displacement_fragment_shader;
                                 //BUMP shader_type bump_fragment_shader;
                    };

                    /*activitied shading method*/
                    shader_type m_activiedMethod;

                    /*texture loader*/
                    std::shared_ptr<TextureLoader> texture;
          };
}

#endif //_SHADER_HPP_