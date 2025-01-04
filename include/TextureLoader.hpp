#pragma once
#ifndef _TEXTURELOADER_HPP_
#define _TEXTURELOADER_HPP_
#include<string>
#include<Eigen/Eigen>
#include<opencv2/opencv.hpp>

namespace SoftRasterizer {
          class TextureLoader {
          public:
                    TextureLoader(const std::string& path);
                    virtual ~TextureLoader();

          public:
                    Eigen::Vector3f getTextureColor(const Eigen::Vector2f& uv);

          private:
                    cv::Mat m_texture;
                    std::string m_path;
                    std::size_t m_width;
                    std::size_t m_height;
          };
}

#endif //_TEXTURELOADER_HPP_