#include <object/Sphere.hpp>
#include <opencv2/opencv.hpp>
#include <render/PathTracing.hpp>
#include <render/Rasterizer.hpp>
#include <render/RayTracing.hpp>
#include <scene/Scene.hpp>

int main() {
  int key = 0;
  float degree = 0.f;

  // Create Path Tracing Main Class
  auto render =
      std::make_shared<SoftRasterizer::PathTracing>(1024, 1024, /*SPP=*/16);

  // Create A Scene
  auto scene = std::make_shared<SoftRasterizer::Scene>(
      "TestScene",
      /*eye=*/glm::vec3(0.0f, 0.3f, -0.9f),
      /*center=*/glm::vec3(0.0f, 0.0f, 0.0f),
      /*up=*/glm::vec3(0.0f, 1.0f, 0.0f),
      /*background color*/ glm::vec3(0.f));

  std::shared_ptr<SoftRasterizer::Material> red =
      std::make_shared<SoftRasterizer::Material>();
  std::shared_ptr<SoftRasterizer::Material> green =
      std::make_shared<SoftRasterizer::Material>();
  std::shared_ptr<SoftRasterizer::Material> white =
      std::make_shared<SoftRasterizer::Material>();
  std::shared_ptr<SoftRasterizer::Material> light =
      std::make_shared<SoftRasterizer::Material>();

  red->Kd = glm::vec3(0.f, 0.f, 1.0f);
  green->Kd = glm::vec3(0.f, 1.0f, 0.f);
  white->Kd = glm::vec3(0.68f, 0.71f, 0.725f);
  light->Kd = glm::vec3(1.0f);
  light->emission = glm::vec3(31.0808f, 38.5664f, 47.8848f);

  scene->addGraphicObj(
      CONFIG_HOME "examples/models/cornellbox/cornellbox_parts/floor.obj",
      "floor", glm::vec3(0, 1, 0), degree, glm::vec3(0.f), glm::vec3(1.f));

  scene->addGraphicObj(
      CONFIG_HOME "examples/models/cornellbox/cornellbox_parts/back.obj",
      "back", glm::vec3(0, 1, 0), degree, glm::vec3(0.f), glm::vec3(1.f));

  scene->addGraphicObj(
      CONFIG_HOME "examples/models/cornellbox/cornellbox_parts/top.obj", "top",
      glm::vec3(0, 1, 0), degree, glm::vec3(0.f), glm::vec3(1.f));

  scene->addGraphicObj(
      CONFIG_HOME "examples/models/cornellbox/cornellbox_parts/left.obj",
      "left", glm::vec3(0, 1, 0), degree, glm::vec3(0.f), glm::vec3(1.f));

  scene->addGraphicObj(
      CONFIG_HOME "examples/models/cornellbox/cornellbox_parts/right.obj",
      "right", glm::vec3(0, 1, 0), degree, glm::vec3(0.f), glm::vec3(1.f));

  scene->addGraphicObj(
      CONFIG_HOME "examples/models/cornellbox/cornellbox_parts/light.obj",
      "light", glm::vec3(0, 1, 0), degree, glm::vec3(0.f), glm::vec3(1.f));

  scene->addGraphicObj(
      CONFIG_HOME "examples/models/cornellbox/cornellbox_parts/small.obj",
      "shortbox", glm::vec3(0, 1, 0), degree, glm::vec3(0.f), glm::vec3(1.f));

  scene->addGraphicObj(
      CONFIG_HOME "examples/models/cornellbox/cornellbox_parts/large.obj",
      "tallbox", glm::vec3(0, 1, 0), degree, glm::vec3(0.f), glm::vec3(1.f));

  scene->startLoadingMesh("floor");
  scene->startLoadingMesh("back");
  scene->startLoadingMesh("top");
  scene->startLoadingMesh("left");
  scene->startLoadingMesh("right");
  scene->startLoadingMesh("light");
  scene->startLoadingMesh("shortbox");
  scene->startLoadingMesh("tallbox");

  if (auto lightOpt = scene->getMeshObj("light"); lightOpt) {
    (*lightOpt)->setMaterial(light);
  }
  if (auto leftOpt = scene->getMeshObj("left"); leftOpt) {
    (*leftOpt)->setMaterial(red);
  }
  if (auto rightOpt = scene->getMeshObj("right"); rightOpt) {
    (*rightOpt)->setMaterial(green);
  }
  if (auto floorOpt = scene->getMeshObj("floor"); floorOpt) {
    (*floorOpt)->setMaterial(white);
  }
  if (auto topOpt = scene->getMeshObj("top"); topOpt) {
    (*topOpt)->setMaterial(white);
  }
  if (auto backOpt = scene->getMeshObj("back"); backOpt) {
    (*backOpt)->setMaterial(white);
  }
  if (auto shortboxOpt = scene->getMeshObj("shortbox"); shortboxOpt) {
    (*shortboxOpt)->setMaterial(white);
  }
  if (auto tallboxOpt = scene->getMeshObj("tallbox"); tallboxOpt) {
    (*tallboxOpt)->setMaterial(white);
  }

  /*Register Scene To Render Main Frame*/
  render->addScene(scene);

  while (key != 27) {

    /*clear both shading and depth!*/
    render->clear(SoftRasterizer::Buffers::Color |
                  SoftRasterizer::Buffers::Depth);

    scene->setModelMatrix("floor", glm::vec3(0, 1, 0), degree, glm::vec3(0.f),
                          glm::vec3(0.25f));
    scene->setModelMatrix("back", glm::vec3(0, 1, 0), degree, glm::vec3(0.f),
                          glm::vec3(0.25f));
    scene->setModelMatrix("top", glm::vec3(0, 1, 0), degree, glm::vec3(0.f),
                          glm::vec3(0.25f));
    scene->setModelMatrix("left", glm::vec3(0, 1, 0), degree, glm::vec3(0.f),
                          glm::vec3(0.25f));
    scene->setModelMatrix("right", glm::vec3(0, 1, 0), degree, glm::vec3(0.f),
                          glm::vec3(0.25f));
    scene->setModelMatrix("light", glm::vec3(0, 1, 0), degree, glm::vec3(0.f),
                          glm::vec3(0.25f));
    scene->setModelMatrix("shortbox", glm::vec3(0, 1, 0), degree,
                          glm::vec3(0.f), glm::vec3(0.25f));
    scene->setModelMatrix("tallbox", glm::vec3(0, 1, 0), degree, glm::vec3(0.f),
                          glm::vec3(0.25f));

    /*View Matrix*/
    scene->setViewMatrix(
        /*eye=*/glm::vec3(0.0f, 0.0f, -0.9f),
        /*center=*/glm::vec3(0.0f, 0.0f, 0.0f),
        /*up=*/glm::vec3(0.0f, 1.0f, 0.0f));

    /*Projection Matrix*/
    scene->setProjectionMatrix(
        /*fov=*/45.0f,
        /*near=*/0.1f,
        /*far=*/100.0f);

    render->display(SoftRasterizer::Primitive::TRIANGLES);

    key = cv::waitKey(0);
    if (key == 'a' || key == 'A') {
      degree += 10.0f;
    } else if (key == 'd' || key == 'D') {
      degree -= 10.0f;
    }

    /*reset the degree*/
    auto delta = degree - 360.f;
    if (delta >= -0.00000001f && delta <= 0.00000001f) {
      degree = 0.0f;
    }
  }
  return 0;
}
