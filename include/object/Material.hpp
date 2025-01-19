#pragma once
#ifndef _MATERIAL_HPP_
#define _MATERIAL_HPP_
#include <string>
#include <glm/glm.hpp>

namespace  SoftRasterizer {
          struct Material {
                    Material() {
                              Ns = 0.0f;
                              Ni = 0.0f;
                              d = 0.0f;
                              illum = 0;
                    }

                    std::string name;     // Material Name
                    glm::vec3 Ka;         // Ambient Color
                    glm::vec3 Kd;         // Diffuse Color
                    glm::vec3 Ks;         // Specular Color
                    float Ns;             // Specular Exponent
                    float Ni;             // Optical Density
                    float d;              // Dissolve
                    int illum;            // Illumination
                    std::string map_Ka;   // Ambient Texture Map
                    std::string map_Kd;   // Diffuse Texture Map
                    std::string map_Ks;   // Specular Texture Map
                    std::string map_Ns;   // Specular Hightlight Map
                    std::string map_d;    // Alpha Texture Map
                    std::string map_bump; // Bump Map
          };
}

#endif //_MATERIAL_HPP_