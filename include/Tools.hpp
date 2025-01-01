#pragma once
#ifndef _TOOLS_HPP_
#define _TOOLS_HPP_
#include <Eigen/Eigen>

namespace SoftRasterizer {
          struct Tools{
                    static Eigen::Vector4f to_vec4(const Eigen::Vector3f& v3, float w = 1.0f){
                              return Eigen::Vector4f(v3.x(), v3.y(), v3.z(), w);
                    }

                    static Eigen::Vector3f to_vec3(const Eigen::Vector4f& v4) {
                              return Eigen::Vector3f(v4.x() / v4.w(), v4.y() / v4.w(), v4.z() / v4.w());
                    }
          };
}

#endif //_TOOLS_HPP_