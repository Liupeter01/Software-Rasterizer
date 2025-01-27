#include <algorithm> // Add this include for std::clamp
#include <base/Render.hpp>
#include <render/RayTracing.hpp>
#include <spdlog/spdlog.h>

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

    std::vector<SoftRasterizer::light_struct> lights = SceneObj->loadLights();
    const glm::vec3 eye = SceneObj->loadEyeVec();

    float scale = std::tan(glm::radians(SceneObj->m_fovy * 0.5));

    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<std::size_t>(0, m_height, 16),
        [&](const oneapi::tbb::blocked_range<std::size_t> &yrange) {
          for (auto ry = yrange.begin(); ry != yrange.end(); ++ry) {

            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range<std::size_t>(0, m_width, 16),
                [&](const oneapi::tbb::blocked_range<std::size_t> &xrange) {
                  for (auto rx = xrange.begin(); rx != xrange.end(); ++rx) {

                    float x =
                        (2 * (rx + 0.5f) / static_cast<float>(m_width) - 1) *
                        aspect_ratio * scale;
                    float y =
                        (2 * (ry + 0.5f) / static_cast<float>(m_height) - 1) *
                        scale;

                    /*Don't forgot to normalize the direction argument*/
                    Ray ray(eye, glm::normalize(glm::vec3(x, y, 0) - eye));

                    /*Get the Nearest Intersection Point*/
                    writePixel(
                        rx, ry,
                        Tools::normalizedToRGB(
                            SceneObj->whittedRayTracing(ray, 0, lights)));
                  }
                },
                ap);
          }
        },
        ap);
  }
}
