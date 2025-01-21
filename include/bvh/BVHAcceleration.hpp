#pragma once
#ifndef _BVH_HPP_
#define _BVH_HPP_
#include <vector>
#include <memory>
#include <bvh/Bounds3.hpp>
#include <object/Object.hpp>
#include <object/Triangle.hpp>

namespace SoftRasterizer {
          struct BVHBuildNode {
                    BVHBuildNode()
                              :left(nullptr), right(nullptr), box(), obj(nullptr)
                    {
                    }

                    Bounds3 box;
                    std::unique_ptr< BVHBuildNode> left;
                    std::unique_ptr< BVHBuildNode> right;
                    std::shared_ptr<Object> obj;
          };

          class BVHAcceleration {
          public:
                    BVHAcceleration();
                    BVHAcceleration(const std::vector<Object*>& stream);
                    BVHAcceleration(std::vector<Object*>&& stream);

          public:
                    void loadNewObjects(const std::vector<Object*>& stream);
                    void startBuilding();
                    Intersection getIntersection(Ray& ray) const;

          protected:
                    void buildBVH();
                    [[nodiscard]] std::unique_ptr<BVHBuildNode> recursive(std::vector<Object*> objs);
                    [[nodiscard]] Intersection intersection(BVHBuildNode* node, Ray& ray) const;

          private:
                    /*BVH Head Node*/
                    std::unique_ptr<BVHBuildNode> root;
                    std::vector<Object*> objs;
          };
}

#endif //_BVH_HPP_