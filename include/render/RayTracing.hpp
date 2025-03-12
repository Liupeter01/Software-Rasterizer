#pragma once
#ifndef _RAYTRACING_HPP_
#define _RAYTRACING_HPP_
#include <base/Render.hpp>

namespace SoftRasterizer {
class RayTracing : public RenderingPipeline {
public:
  RayTracing();
  RayTracing(const std::size_t width, const std::size_t height);
  RayTracing(const std::size_t width, const std::size_t height,
             const std::size_t spp = 16);

  // Sample Per Pixel
  void setSPP(const std::size_t spp);

private:
  void draw(Primitive type) override;

private:
  /*Path Tracing Sample Variable*/
  std::size_t sample = 16;
};
} // namespace SoftRasterizer

#endif //_RAYTRACING_HPP_
