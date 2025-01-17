#pragma once
#ifndef _NONCOPYABLE_HPP_
#define _NONCOPYABLE_HPP_

namespace SoftRasterizer {
class Noncopyable {
protected:
  Noncopyable() {};
  ~Noncopyable() {};

private:
  Noncopyable(const Noncopyable &);
  const Noncopyable &operator=(const Noncopyable &);
};
} // namespace SoftRasterizer

#endif //_NONCOPYABLE_HPP_
