#include <TestHelpers.hpp>

using namespace SoftRasterizer;
using namespace SoftRasterizer::TestHelpers;

namespace {

TEST(PathTracing, MirrorSampleReportsDiscreteCoefficient) {
  auto material = mat({.8f, .6f, .3f});
  material->type = MaterialType::Mirror;
  Sampler sampler(11);
  const auto sample = material->sample({.6f, 0, -.8f}, {0, 0, 1}, true,
                                       {}, sampler);
  EXPECT_TRUE(sample.delta);
  EXPECT_FALSE(sample.transmitted);
  expectNear(sample.direction, {.6f, 0, .8f});
  expectNear(sample.fr, glm::vec3(0));
  expectNear(sample.pdf, 1);
  expectNear(sample.deltaCoefficient, material->Kd);
  expectNear(sample.deltaCoefficient / sample.pdf, material->Kd);
}

} // namespace
