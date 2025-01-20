#pragma once
#ifndef _OBJECT_HPP_
#define _OBJECT_HPP_
#include <bvh/Bounds3.hpp>

namespace SoftRasterizer {
class Object {
public:
  Object() {}
  virtual ~Object() {}
  virtual Bounds3 getBounds() = 0;
};
} // namespace SoftRasterizer

#endif //_OBJECT_HPP_
