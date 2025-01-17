#include <loader/TextureLoader.hpp>

SoftRasterizer::TextureLoader::TextureLoader(const std::string &path)
    : m_path(path), m_texture(cv::imread(path)), m_width(0), m_height(0) {
  if (m_texture.empty()) {
    throw std::runtime_error("Cannot open file: " + path);
  }

  // cv::cvtColor(m_texture, m_texture, cv::COLOR_RGB2BGR);
  m_width = m_texture.cols;
  m_height = m_texture.rows;
}

Eigen::Vector3f
SoftRasterizer::TextureLoader::getTextureColor(const Eigen::Vector2f &uv) {
  auto x = static_cast<int>(uv.x() * m_width);
  auto y = static_cast<int>(uv.y() * m_height);

  if (x < 0 || x > m_width || y < 0 || y > m_height) {
    return Eigen::Vector3f(0.f, 0.f, 0.f);
  }

  auto color = m_texture.at<cv::Vec3b>(y, x);
  return Eigen::Vector3f(color[0] / 255.0f, color[1] / 255.0f,
                         color[2] / 255.0f);
}

SoftRasterizer::TextureLoader::~TextureLoader() {}
