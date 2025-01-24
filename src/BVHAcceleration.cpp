#include <algorithm>
#include <bvh/BVHAcceleration.hpp>
#include <chrono>
#include <spdlog/spdlog.h>

SoftRasterizer::BVHAcceleration::BVHAcceleration() : root(nullptr), objs(0) {}

SoftRasterizer::BVHAcceleration::BVHAcceleration(
    const std::vector<SoftRasterizer::Object *> &stream)
    : root(nullptr), objs(stream) {

  buildBVH();
}

SoftRasterizer::BVHAcceleration::BVHAcceleration(
    std::vector<SoftRasterizer::Object *> &&stream)
    : root(nullptr), objs(std::move(stream)) {

  buildBVH();
}

SoftRasterizer::BVHAcceleration::~BVHAcceleration() {
          clearBVHAccel(root);
}

void SoftRasterizer::BVHAcceleration::loadNewObjects(
    const std::vector<Object *> &stream) {
  objs.erase(objs.begin(), objs.end());
  objs.resize(stream.size());
  std::copy(stream.begin(), stream.end(), objs.begin());
}

void SoftRasterizer::BVHAcceleration::clearBVHAccel() {
          clearBVHAccel(root);
}

void SoftRasterizer::BVHAcceleration::clearBVHAccel(std::unique_ptr< BVHBuildNode>& node) {
          if (node == nullptr) {
                    return;
          }

          if (node->left != nullptr) {
                    clearBVHAccel(node->left);
          }

          if (node->right != nullptr) {
                    clearBVHAccel(node->right);
          }

          node.reset();
}

void SoftRasterizer::BVHAcceleration::rebuildBVHAccel(){
          clearBVHAccel(root);
          startBuilding();
}

void SoftRasterizer::BVHAcceleration::startBuilding() { buildBVH(); }

void SoftRasterizer::BVHAcceleration::buildBVH() {

  if (objs.empty()) {
    spdlog::info("Build BVH List Error, No Objects Found!");
    return;
  }

  /*Start Time Point*/
  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();

  /*Start Building BVH Structure*/
  root = recursive(objs);

  /*End Time Point*/
  std::chrono::high_resolution_clock::time_point end =
      std::chrono::high_resolution_clock::now();

  spdlog::info("Start BVH Generation complete: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}

std::optional<SoftRasterizer::Bounds3>
SoftRasterizer::BVHAcceleration::getBoundingBox() const {
          if (root == nullptr) {
                    return std::nullopt;
          }
          return root->box;
}

SoftRasterizer::Intersection
SoftRasterizer::BVHAcceleration::getIntersection(Ray &ray) const {
          Intersection ret;
  if (root != nullptr) {
            ret = intersection(root.get(), ray);
  }

  if (ret.intersected) {
            ret.material = ret.obj->getMaterial();
            ret.index = ret.obj->index;
  }
  return ret;
}

SoftRasterizer::Intersection
SoftRasterizer::BVHAcceleration::intersection(BVHBuildNode *node,
                                              Ray &ray) const {
  if (node == nullptr) {
    return {};
  }

  if (node->left == nullptr && node->right == nullptr && node->obj != nullptr) {
            Intersection temp = node->obj->getIntersect(ray);
            if (temp.intersected) {
                      return temp;
            }
  }

  // Check left and right child nodes recursively
  Intersection left = intersection(node->left.get(), ray);
  Intersection right = intersection(node->right.get(), ray);

  // Determine which intersection is closer
  if (left.intersected && right.intersected) {
    return left.intersect_time < right.intersect_time ? left : right;
  } else if (left.intersected || !right.intersected) {
    return left;
  } else if (right.intersected || !left.intersected) {
    return right;
  }

  /*Safty Consideration And Handling All None Intersected situation*/
  return {};
}

std::unique_ptr<SoftRasterizer::BVHBuildNode>
SoftRasterizer::BVHAcceleration::recursive(
    std::vector<SoftRasterizer::Object *> objs) {
  auto node = std::make_unique<SoftRasterizer::BVHBuildNode>();

  Bounds3 box;
  for (const auto &obj : objs) {
    box = BoundsUnion(box, obj->getBounds());
  }

  /*I'm the Leaf Node*/
  if (objs.size() == 1) {
    node->left = nullptr;
    node->right = nullptr;
    node->box = (*objs.begin())->getBounds();
    node->obj = std::shared_ptr<Object>(*objs.begin(), [](auto T) {});
    return node;
  }
  /*I am The Root Node*/
  else if (objs.size() == 2) {
    node->left = recursive(std::vector{objs[0]});
    node->right = recursive(std::vector{objs[1]});
    node->box = BoundsUnion(node->left->box, node->right->box);
    return node;
  }
  /*Other Condition*/
  else {
    Bounds3 centric;
    for (const auto &obj : objs) {
      centric = BoundsUnion(centric, obj->getBounds().centroid());
    }

    std::sort(objs.begin(), objs.end(),
              [dim = centric.maxExtent()](auto f1, auto f2) {
                switch (dim) {
                case 0:
                  return f1->getBounds().centroid().x <
                         f2->getBounds().centroid().x; // X
                  break;
                case 1:
                  return f1->getBounds().centroid().y <
                         f2->getBounds().centroid().y; // Y
                  break;
                case 2:
                  return f1->getBounds().centroid().z <
                         f2->getBounds().centroid().z; // Z
                  break;
                }
                return true;
              });

    /*Seperate The vector in half, by using the longest axis*/
    auto middle = objs.size() / 2;
    node->left = recursive(std::vector<SoftRasterizer::Object *>(
        objs.begin(), objs.begin() + middle));
    node->right = recursive(std::vector<SoftRasterizer::Object *>(
        objs.begin() + middle, objs.end()));
    node->box = BoundsUnion(node->left->box, node->right->box);
  }
  return node;
}
