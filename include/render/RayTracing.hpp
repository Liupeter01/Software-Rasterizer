#pragma once
#ifndef _RAYTRACING_HPP_
#define _RAYTRACING_HPP_
#include <base/Render.hpp>

namespace SoftRasterizer {
          class RayTracing :public RenderingPipeline {
          public:
                    RayTracing() :RenderingPipeline() {}
                    RayTracing(const std::size_t width, const std::size_t height)
                              : RenderingPipeline(width, height) {
                    }

          private:
                    void draw(Primitive type) override;
          };
}

#endif  //_RAYTRACING_HPP_