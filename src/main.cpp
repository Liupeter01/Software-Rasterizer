#include <Eigen/Eigen>
#include <Tools.hpp>
#include <Triangle.hpp>
#include <loader/ObjLoader.hpp>
#include <opencv2/opencv.hpp>
#include <render/Render.hpp>
#include <shader/Shader.hpp>
#include <spdlog/spdlog.h>

int main() {
  int key = 0;
  float degree = 0.0f;

  auto render = std::make_shared<SoftRasterizer::RenderingPipeline>(
      1024, 1024); // Create Render Main Class
  auto scene =
      std::make_shared<SoftRasterizer::Scene>("TestScene"); // Create A Scene
  scene->addGraphicObj(
      CONFIG_HOME "examples/models/spot/spot_triangulated_good.obj", "spot",
      Eigen::Vector3f(0.f, 1.f, 0.f), degree,
      Eigen::Vector3f(-0.35f, 0.0f, 0.0f), Eigen::Vector3f(0.3f, 0.3f, 0.3f));

  scene->addGraphicObj(CONFIG_HOME "examples/models/Crate/Crate1.obj", "Crate",
                       Eigen::Vector3f(0.f, 1.f, 0.f), degree,
                       Eigen::Vector3f(0.3f, 0.0f, 0.0f),
                       Eigen::Vector3f(0.2f, 0.2f, 0.2f));

  scene->addShader("spot_shader",
                   CONFIG_HOME "examples/models/spot/spot_texture.png",
                   SoftRasterizer::SHADERS_TYPE::TEXTURE);

  scene->addShader("crate_shader",
                   CONFIG_HOME "examples/models/Crate/crate1.png",
                   SoftRasterizer::SHADERS_TYPE::TEXTURE);

  scene->startLoadingMesh("spot");
  scene->startLoadingMesh("Crate");
  scene->bindShader2Mesh("spot", "spot_shader");
  scene->bindShader2Mesh("Crate", "crate_shader");

  /*Add Light To Scene*/
  auto light1 = std::make_shared<SoftRasterizer::light_struct>();
  light1->position = Eigen::Vector3f{0.9, 0.9, -0.9f};
  light1->intensity = Eigen::Vector3f{100, 100, 100};

  auto light2 = std::make_shared<SoftRasterizer::light_struct>();
  light2->position = Eigen::Vector3f{0.f, 0.8f, 0.9f};
  light2->intensity = Eigen::Vector3f{50, 50, 50};

  scene->addLight("Light1", light1);
  scene->addLight("Light2", light2);

  /*Register Scene To Render Main Frame*/
  render->addScene(scene);

  while (key != 27) {

    /*clear both shading and depth!*/
    render->clear(SoftRasterizer::Buffers::Color |
                  SoftRasterizer::Buffers::Depth);

    /*Model Matrix*/
    scene->setModelMatrix(
        "spot",
        SoftRasterizer::Tools::calculateModelMatrix(
            /*transform=*/Eigen::Vector3f(0.f, 0.f, 0.f),
            /*rotations=*/
            SoftRasterizer::Tools::calculateRotationMatrix(
                /*axis=*/Eigen::Vector3f(0.f, 1.f, 0.f),
                /*degree=+ for Counterclockwise;- for Clockwise*/ degree),
            /*scale=*/Eigen::Vector3f(0.3f, 0.3f, 0.3f)));

    /*View Matrix*/
    scene->setViewMatrix(
        /*eye=*/Eigen::Vector3f(0.0f, 0.0f, 0.9f),
        /*center=*/Eigen::Vector3f(0.0f, 0.0f, 0.0f),
        /*up=*/Eigen::Vector3f(0.0f, 1.0f, 0.0f));

    /*Projection Matrix*/
    scene->setProjectionMatrix(
        /*fov=*/45.0f,
        /*near=*/0.1f,
        /*far=*/100.0f);

    render->display(SoftRasterizer::Primitive::TRIANGLES);

    key = cv::waitKey(0);
    if (key == 'a' || key == 'A') {
      degree += 1.0f;
    } else if (key == 'd' || key == 'D') {
      degree -= 1.0f;
    }

    /*reset the degree*/
    auto delta = degree - 360.f;
    if (delta >= -0.00000001f && delta <= 0.00000001f) {
      degree = 0.0f;
    }
  }
  return 0;
}
