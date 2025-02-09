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

private:
  void draw(Primitive type) override;
};
} // namespace SoftRasterizer

#endif
