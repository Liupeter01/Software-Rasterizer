#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"
#include <Tools.hpp>
#include <opencv2/opencv.hpp>
#include <base/Render.hpp>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>
#include <type_traits>

SoftRasterizer::RenderingPipeline::RenderingPipeline()
    : RenderingPipeline(800, 600) {}

SoftRasterizer::RenderingPipeline::RenderingPipeline(const std::size_t width,
                                                     const std::size_t height)
    : m_width(width), m_height(height), m_channels(numbers) /*set to three*/
      ,
      m_frameBuffer(m_height, m_width, CV_32FC3) {

  /*set channel ammount to three!*/
  m_channels.resize(numbers);

  /*resize std::vector of z-Buffer*/
  m_zBuffer.resize(width * height);

  /*init framebuffer*/
  clear(SoftRasterizer::Buffers::Color | SoftRasterizer::Buffers::Depth);
}

SoftRasterizer::RenderingPipeline::~RenderingPipeline() {}

void SoftRasterizer::RenderingPipeline::clearFrameBuffer() {
  // #pragma omp parallel for
  for (long long i = 0; i < numbers; ++i) {
    m_channels[i] = cv::Mat::zeros(m_height, m_width, CV_32FC1);
  }

  m_frameBuffer = cv::Mat::zeros(m_height, m_width, CV_32FC3);
}

void SoftRasterizer::RenderingPipeline::clearZDepth() {
  std::for_each(m_zBuffer.begin(), m_zBuffer.end(), [](float &depth) {
    depth = std::numeric_limits<float>::infinity();
  });
}

void SoftRasterizer::RenderingPipeline::clear(SoftRasterizer::Buffers flags) {
  if ((flags & SoftRasterizer::Buffers::Color) ==
      SoftRasterizer::Buffers::Color) {
    clearFrameBuffer();
  }
  if ((flags & SoftRasterizer::Buffers::Depth) ==
      SoftRasterizer::Buffers::Depth) {
    clearZDepth();
  }
}

void SoftRasterizer::RenderingPipeline::display(Primitive type) {
  /*draw pictures according to the specific type*/
  draw(type);

  cv::merge(m_channels, m_frameBuffer);
  m_frameBuffer.convertTo(m_frameBuffer, CV_8UC3, 1.0f);
  cv::imshow("SoftRasterizer", m_frameBuffer);
}

bool SoftRasterizer::RenderingPipeline::addScene(
    std::shared_ptr<Scene> scene, std::optional<std::string> name) {
  try {
    if (scene == nullptr) {
      return false;
    }
    if (name.has_value()) {
      scene->m_sceneName = name.value();
    }

    /*Set Render's width and height info to scene*/
    scene->setNDCMatrix(m_width, m_height);

    if (m_scenes.find(scene->m_sceneName) != m_scenes.end()) {
      spdlog::error("Add Scene Failed! Scene Already Exist");
      return false;
    }

    m_scenes[scene->m_sceneName] = scene;
  } catch (const std::exception &e) {
    spdlog::error("Add Scene Failed! Reason: {}", e.what());
    return false;
  }
  return true;
}

inline void SoftRasterizer::RenderingPipeline::writePixel(
    const long long x, const long long y, const glm::vec3 &color) {
  if (x >= 0 && x < m_width && y >= 0 && y < m_height) {
    auto pos = x + y * m_width;

    *(m_channels[0].ptr<float>(0) + pos) = color.x; // R
    *(m_channels[1].ptr<float>(0) + pos) = color.y; // G
    *(m_channels[2].ptr<float>(0) + pos) = color.z; // B
  }
}

inline void SoftRasterizer::RenderingPipeline::writePixel(
    const long long x, const long long y, const glm::uvec3 &color) {
  writePixel(x, y, glm::vec3(color.x, color.y, color.z));
}

inline void
SoftRasterizer::RenderingPipeline::writeZBuffer(const long long start_pos,
                                                const float depth) {
  m_zBuffer[start_pos] = depth;
}

inline const float
SoftRasterizer::RenderingPipeline::readZBuffer(const long long x,
                                               const long long y) {
  return m_zBuffer[x + y * m_width];
}

/* Bresenham algorithm*/
void SoftRasterizer::RenderingPipeline::drawLine(const glm::vec3 &p0,
                                                 const glm::vec3 &p1,
                                                 const glm::uvec3 &color) {

  auto x1 = p0.x;
  auto y1 = p0.y;
  auto x2 = p1.x;
  auto y2 = p1.y;

  int x, y, dx, dy, dx1, dy1, px, py, xe, ye, i;

  dx = x2 - x1;
  dy = y2 - y1;
  dx1 = fabs(dx);
  dy1 = fabs(dy);
  px = 2 * dy1 - dx1;
  py = 2 * dx1 - dy1;

  if (dy1 <= dx1) {
    if (dx >= 0) {
      x = x1;
      y = y1;
      xe = x2;
    } else {
      x = x2;
      y = y2;
      xe = x1;
    }

    writePixel(x, y, color);

    for (i = 0; x < xe; i++) {
      x = x + 1;
      if (px < 0) {
        px = px + 2 * dy1;
      } else {
        if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)) {
          y = y + 1;
        } else {
          y = y - 1;
        }
        px = px + 2 * (dy1 - dx1);
      }
      writePixel(x, y, color);
    }
  } else {
    if (dy >= 0) {
      x = x1;
      y = y1;
      ye = y2;
    } else {
      x = x2;
      y = y2;
      ye = y1;
    }

    writePixel(x, y, color);

    for (i = 0; y < ye; i++) {
      y = y + 1;
      if (py <= 0) {
        py = py + 2 * dx1;
      } else {
        if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)) {
          x = x + 1;
        } else {
          x = x - 1;
        }
        py = py + 2 * (dx1 - dy1);
      }

      writePixel(x, y, color);
    }
  }
}
