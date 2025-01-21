#pragma once
#ifndef _RASTERIZER_HPP_
#define _RASTERIZER_HPP_
#include <base/Render.hpp>

namespace SoftRasterizer {
          // Traditional Rasterizer
class TraditionalRasterizer :public RenderingPipeline{
public:
          TraditionalRasterizer() :RenderingPipeline() { }
          TraditionalRasterizer(const std::size_t width, const std::size_t height)
                    :RenderingPipeline(width, height) {}

public:
          void draw(Primitive type) override;

private:
          /*Only Draw Line*/
          void rasterizeWireframe(const SoftRasterizer::Triangle& triangle);

          inline static bool insideTriangle(const std::size_t x_pos,
                    const std::size_t y_pos,
                    const SoftRasterizer::Triangle& triangle);

          static inline std::tuple<float, float, float>
                    barycentric(const std::size_t x_pos, const std::size_t y_pos,
                              const SoftRasterizer::Triangle& triangle);

          /**
           * @brief Calculates the barycentric coordinates (alpha, beta, gamma) for a
           * given point (x_pos, y_pos) with respect to a triangle. Also checks if the
           * point is inside the triangle using the `insideTriangle` function and
           * applies the result as a mask to ensure the coordinates are only valid for
           * points inside the triangle.
           *
           * @param x_pos SIMD register containing x positions of points.
           * @param y_pos SIMD register containing y positions of points.
           * @param triangle The triangle whose barycentric coordinates are to be
           * calculated.
           * @return A tuple of three simde__m256 values representing the barycentric
           * coordinates (alpha, beta, gamma) for the point (x_pos, y_pos). The
           * coordinates are zeroed out for points outside the triangle using a mask.
           */
#if defined(__x86_64__) || defined(_WIN64)
          static inline std::tuple<__m256, __m256, __m256>
                    barycentric(const __m256& x_pos, const __m256& y_pos,
                              const SoftRasterizer::Triangle& triangle);

#elif defined(__arm__) || defined(__aarch64__)
          static inline std::tuple<simde__m256, simde__m256, simde__m256>
                    barycentric(const simde__m256& x_pos, const simde__m256& y_pos,
                              const SoftRasterizer::Triangle& triangle);

#else
#endif
           /*Rasterize a triangle*/
          inline void
                    rasterizeBatchAVX2(const int startx, const int endx, const int y,
                              const std::vector<SoftRasterizer::light_struct>& lists,
                              std::shared_ptr<SoftRasterizer::Shader> shader,
                              const SoftRasterizer::Triangle& packed,
                              const glm::vec3& eye);

          template <typename _simd>
          inline void processFragByAVX2(
                    const int x, const int y, const _simd& z0, const _simd& z1,
                    const _simd& z2, const std::vector<SoftRasterizer::light_struct>& lists,
                    std::shared_ptr<SoftRasterizer::Shader> shader,
                    const SoftRasterizer::Triangle& packed, const glm::vec3& eye);

          inline void
                    rasterizeBatchScalar(const int startx, const int endx, const int y,
                              const std::vector<SoftRasterizer::light_struct>& lists,
                              std::shared_ptr<SoftRasterizer::Shader> shader,
                              const SoftRasterizer::Triangle& scalar,
                              const glm::vec3& eye);

          inline void processFragByScalar(
                    const int startx, const int x, const int y, const float old_z,
                    const float z0, const float z1, const float z2, float* __restrict z,
                    float* __restrict r, float* __restrict g, float* __restrict b,
                    const std::vector<SoftRasterizer::light_struct>& lists,
                    std::shared_ptr<SoftRasterizer::Shader> shader,
                    const SoftRasterizer::Triangle& scalar, const glm::vec3& eye);
};
} // namespace SoftRasterizer

#endif //_RASTERIZER_HPP_