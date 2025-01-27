#pragma once
#ifndef _BVH_HPP_
#define _BVH_HPP_
#include <bvh/Bounds3.hpp>
#include <memory>
#include <object/Object.hpp>
#include <object/Triangle.hpp>
#include <optional>
#include <tbb/concurrent_vector.h>

namespace SoftRasterizer {
class Scene;

struct BVHBuildNode {
  BVHBuildNode() : left(nullptr), right(nullptr), box(), obj(nullptr) {}

  Bounds3 box;
  std::unique_ptr<BVHBuildNode> left;
  std::unique_ptr<BVHBuildNode> right;
  std::shared_ptr<Object> obj;
};

class BVHAcceleration {
  friend class Scene;

public:
  BVHAcceleration();
  BVHAcceleration(const tbb::concurrent_vector<std::shared_ptr<Object>>& stream);
  virtual ~BVHAcceleration();

public:
  void loadNewObjects(const tbb::concurrent_vector<std::shared_ptr<Object>>& stream);
  void startBuilding();
  void rebuildBVHAccel();
  Intersection getIntersection(Ray &ray) const;
  void clearBVHAccel();
  std::optional<Bounds3> getBoundingBox() const;

protected:
  void clearBVHAccel(std::unique_ptr<BVHBuildNode> &node);
  void buildBVH();
  [[nodiscard]] std::unique_ptr<BVHBuildNode>
  recursive(tbb::concurrent_vector<Object *> objs);
  [[nodiscard]] Intersection intersection(BVHBuildNode *node, Ray &ray) const;

private:
  /*BVH Head Node*/
  std::unique_ptr<BVHBuildNode> root;
  tbb::concurrent_vector<Object *> objs;
};
} // namespace SoftRasterizer

#endif //_BVH_HPP_
