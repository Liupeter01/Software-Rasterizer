#pragma once
#ifndef _SPHERE_HPP_
#define _SPHERE_HPP_
#include <object/Object.hpp>
#include <object/Material.hpp>

namespace  SoftRasterizer {
          class Sphere :public Object {
          public:
                    Sphere();
                    Sphere(const glm::vec3& _center, const float _radius);
                    virtual ~Sphere();

          public:
                    Bounds3 getBounds() override;
                    float getSquare() const;
                    [[nodiscard]] bool intersect(const Ray& ray) override;
                    [[nodiscard]] bool intersect(const Ray& ray, float& tNear) override;
                    [[nodiscard]] Intersection getIntersect(Ray& ray) override;
                   
          private:
                    glm::vec3 center;
                    float radius;
                    float square;
          };
}


#endif //_SPHERE_HPP_