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
              oneapi::tbb::blocked_range<std::size_t>(0, m_height * m_width),
              [&](const oneapi::tbb::blocked_range<std::size_t>& range) {
                        for (auto idx = range.begin(); idx != range.end(); ++idx) {
                                  int rx = idx % m_width, ry = idx / m_width;

                                  float x = (2 * (rx + 0.5f) / static_cast<float>(m_width) - 1) * aspect_ratio * scale;
                                  float y = (2 * (ry + 0.5f) / static_cast<float>(m_height) - 1) * scale;

                                  Ray ray(eye, glm::normalize(glm::vec3(x, y, 0) - eye));

                                  writePixel(rx, ry,
                                            Tools::normalizedToRGB(SceneObj->whittedRayTracing(ray, 0, lights))
                                  );
                        }
              },
              ap);
  }
}
