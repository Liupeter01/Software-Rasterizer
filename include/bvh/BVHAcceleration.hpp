#pragma once
#ifndef _BVH_HPP_
#define _BVH_HPP_
#include <bvh/Bounds3.hpp>
#include <memory>
#include <optional>
#include <object/Object.hpp>
#include <object/Triangle.hpp>
#include <vector>

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
  BVHAcceleration(const std::vector<Object *> &stream);
  BVHAcceleration(std::vector<Object *> &&stream);
  virtual ~BVHAcceleration();

public:
  void loadNewObjects(const std::vector<Object *> &stream);
  void startBuilding();
  void rebuildBVHAccel();
  Intersection getIntersection(Ray &ray) const;
  void clearBVHAccel();
  std::optional<Bounds3> getBoundingBox() const;

protected:
  void clearBVHAccel(std::unique_ptr< BVHBuildNode> &node);
  void buildBVH();
  [[nodiscard]] std::unique_ptr<BVHBuildNode> recursive(std::vector<Object *> objs);
  [[nodiscard]] Intersection intersection(BVHBuildNode *node, Ray &ray) const;

private:
  /*BVH Head Node*/
  std::unique_ptr<BVHBuildNode> root;
  std::vector<Object *> objs;
};
} // namespace SoftRasterizer

#endif //_BVH_HPP_
