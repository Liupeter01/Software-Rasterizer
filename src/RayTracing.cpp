#include <algorithm> // Add this include for std::clamp
#include <base/Render.hpp>
#include <render/RayTracing.hpp>
#include <spdlog/spdlog.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

void SoftRasterizer::RayTracing::draw(Primitive type) {
  if ((type != SoftRasterizer::Primitive::LINES) &&
      (type != SoftRasterizer::Primitive::TRIANGLES)) {
    spdlog::error("Primitive Type is not supported!");
    throw std::runtime_error("Primitive Type is not supported!");
  }

  float aspect_ratio = m_width / static_cast<float>(m_height);

  for (auto &[SceneName, SceneObj] : m_scenes) {
    /*
     * Update Triangle Position Because Of NDC_MVP Change
     * We only need to update each triangle by using shared_ptr pointers
     */
    SceneObj->updatePosition();

    const glm::vec3 eye = SceneObj->loadEyeVec();

    float scale = std::tan(glm::radians(SceneObj->m_fovy * 0.5));

    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range2d<std::size_t>(0, m_height, 16, 0, m_width,
                                                  16), // 16x16 Block
        [&](const oneapi::tbb::blocked_range2d<std::size_t> &range) {
          for (std::size_t ry = range.rows().begin(); ry < range.rows().end();
               ++ry) {
            for (std::size_t rx = range.cols().begin(); rx < range.cols().end();
                 ++rx) {

              float x = (2 * (rx + 0.5f) / static_cast<float>(m_width) - 1) *
                        aspect_ratio * scale;
              float y = (1.f - 2 * (ry + 0.5f) / static_cast<float>(m_height)) *
                        scale;

              try {
                Ray ray(eye, glm::normalize(glm::vec3(x, y, 0) - eye));
                glm::vec3 color = SceneObj->whittedRayTracing(ray, 0);
                writePixel(rx, ry, Tools::normalizedToRGB(color));
              } catch (const std::exception &e) {
                spdlog::error("RayTracing System Error! Message: {}", e.what());
              }
            }
          }
        },
        ap);
  }
}
