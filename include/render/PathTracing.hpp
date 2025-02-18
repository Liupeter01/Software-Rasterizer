#pragma once
#ifndef _PATHTRACING_HPP_
#define _PATHTRACING_HPP_
#include <base/Render.hpp>

namespace SoftRasterizer {
class PathTracing : public RenderingPipeline {
public:
  PathTracing() : RenderingPipeline() {}
  PathTracing(const std::size_t width, const std::size_t height)
      : RenderingPipeline(width, height) {}

  PathTracing(const std::size_t width, const std::size_t height,
              const std::size_t spp = 16);

  // Sample Per Pixel
  void setSPP(const std::size_t spp);

private:
  void draw(Primitive type) override;

  /*Path Tracing Sample Variable*/
  std::size_t sample = 16;
};
} // namespace SoftRasterizer

#endif
