#include <algorithm>
#include <bvh/BVHAcceleration.hpp>
#include <chrono>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>

SoftRasterizer::BVHAcceleration::BVHAcceleration() : root(nullptr), objs(0) {}

SoftRasterizer::BVHAcceleration::BVHAcceleration(
    const tbb::concurrent_vector<std::shared_ptr<Object>> &stream) {
  loadNewObjects(stream);
}

SoftRasterizer::BVHAcceleration::~BVHAcceleration() { clearBVHAccel(root); }

void SoftRasterizer::BVHAcceleration::loadNewObjects(
    const tbb::concurrent_vector<std::shared_ptr<Object>> &stream) {
  objs.clear();
  objs.resize(stream.size());

  tbb::parallel_for(tbb::blocked_range<long long>(0, stream.size()),
                    [&](const tbb::blocked_range<long long> &r) {
                      for (long long index = r.begin(); index < r.end();
                           ++index) {
                        objs[index] = stream[index].get();
                      }
                    });
}

void SoftRasterizer::BVHAcceleration::clearBVHAccel() { clearBVHAccel(root); }

void SoftRasterizer::BVHAcceleration::clearBVHAccel(
    std::unique_ptr<BVHBuildNode> &node) {
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

void SoftRasterizer::BVHAcceleration::rebuildBVHAccel() {
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

  spdlog::debug(
      "Start BVH Generation complete: {}ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count());
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
  if (root == nullptr) {
    return {};
  }
  return intersection(root.get(), ray);
}

SoftRasterizer::Intersection
SoftRasterizer::BVHAcceleration::intersection(BVHBuildNode *node,
                                              Ray &ray) const {
  if (!node)
    return {};

  /*BoundingBox Test, Optimize Calculation*/
  if (!node->box.intersect(ray)) {
    return {};
  }

  /*Every Obj is on leaf node!*/
  if (node->left == nullptr && node->right == nullptr) {
    if (node->obj) {

      // Return intersection if object exists
      return node->obj->getIntersect(ray);
    }
    return {}; // Return empty intersection if no object in leaf node
  }

  // Check left and right child nodes recursively
  Intersection left = intersection(node->left.get(), ray);
  Intersection right = intersection(node->right.get(), ray);

  // Determine which intersection is closer
  if (left.intersected && right.intersected) {
    return left.intersect_time < right.intersect_time ? left : right;
  }
  // If one of them is not intersected, return the one that is
  else if (left.intersected && !right.intersected) {
    return left;
  } else if (!left.intersected && right.intersected) {
    return right;
  }
  // No Intersect At ALL
  return {};
}

std::unique_ptr<SoftRasterizer::BVHBuildNode>
SoftRasterizer::BVHAcceleration::recursive(
    tbb::concurrent_vector<SoftRasterizer::Object *> objs) {
  auto node = std::make_unique<SoftRasterizer::BVHBuildNode>();

  Bounds3 box;
  for (const auto &obj : objs) {
    box = BoundsUnion(box, obj->getBounds());
  }

  /*I'm the Leaf Node*/
  if (objs.size() == 1) {
    auto obj = (*objs.begin());
    node->left = nullptr;
    node->right = nullptr;
    node->box = obj->getBounds();
    node->obj = std::shared_ptr<Object>(obj, [](auto T) {});
    node->area = obj->getArea();
    return node;
  }
  /*I am The Root Node*/
  else if (objs.size() == 2) {
    node->left = std::make_unique<BVHBuildNode>();
    node->right = std::make_unique<BVHBuildNode>();
    node->left->obj = std::shared_ptr<Object>(objs[0], [](auto) {});
    node->right->obj = std::shared_ptr<Object>(objs[1], [](auto) {});
    node->left->box = objs[0]->getBounds();
    node->right->box = objs[1]->getBounds();
    node->left->area = objs[0]->getArea();
    node->right->area = objs[1]->getArea();
  }
  /*Other Condition*/
  else {

    // Calculate centroids and partition objects along the longest axis
    Bounds3 centric;
    for (const auto &obj : objs) {
      centric = BoundsUnion(centric, obj->getBounds().centroid());
    }

    std::sort(objs.begin(), objs.end(),
              [dim = centric.maxExtent()](auto f1, auto f2) {
                return f1->getBounds().centroid()[dim] <
                       f2->getBounds().centroid()[dim];
              });

    /*Seperate The vector in half, by using the longest axis*/
    auto middle = objs.size() / 2;
    node->left = recursive(
        tbb::concurrent_vector<Object *>(objs.begin(), objs.begin() + middle));
    node->right = recursive(
        tbb::concurrent_vector<Object *>(objs.begin() + middle, objs.end()));
  }
  node->box = BoundsUnion(node->left->box, node->right->box);
  node->area = node->left->area + node->right->area;
  return node;
}

void SoftRasterizer::BVHAcceleration::sample(BVHBuildNode *node,
                                             const float area,
                                             Intersection &intersect,
                                             float &pdf) {
  if (!node)
    return;

  /*Every Obj is on leaf node!*/
  if (node->left == nullptr && node->right == nullptr) {
    auto [obj_intersection, obj_pdf] = node->obj->sample();
    intersect = obj_intersection;
    pdf = obj_pdf * node->area;
    return;
  }
  if (area < node->left->area) {
    if (node->left != nullptr)
      sample(node->left.get(), area, intersect, pdf);
  } else {
    if (node->right != nullptr)
      sample(node->right.get(), area - node->left->area, intersect, pdf);
  }
}

/*Read Parameters from the object of sample*/
std::tuple<SoftRasterizer::Intersection, float>
SoftRasterizer::BVHAcceleration::sample() {
  Intersection intersect{};
  float pdf = 0.f;

  /*Use Total Area Value and a ratio to do sample*/
  const float area =
      std::sqrt(std::max(Tools::random_generator(),
                         std::numeric_limits<float>::epsilon())) *
      root->area;

  sample(root.get(), area, intersect, pdf);
  return {intersect, pdf /= root->area};
}
