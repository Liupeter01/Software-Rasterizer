#include <shader/Shader.hpp>
#include <Eigen/Eigen>
#include <loader/ObjLoader.hpp>
#include <render/Render.hpp>
#include <Tools.hpp>
#include <Triangle.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

int main() {
  int key = 0;
  float degree = 0.0f;

  SoftRasterizer::RenderingPipeline render(1024, 1024);

  render.addShader("shader",
                   CONFIG_HOME "examples/models/spot/spot_texture.png",
                   SoftRasterizer::SHADERS_TYPE::TEXTURE);

  render.addGraphicObj(
      CONFIG_HOME "examples/models/spot/spot_triangulated_good.obj", "spot",
      Eigen::Vector3f(0.f, 1.f, 0.f), degree, Eigen::Vector3f(0.f, 0.0f, 0.0f),
      Eigen::Vector3f(0.3f, 0.3f, 0.3f));

  render.startLoadingMesh("spot");
  render.bindShader2Mesh("spot", "shader");

  while (key != 27) {

    /*clear both shading and depth!*/
    render.clear(SoftRasterizer::Buffers::Color |
                 SoftRasterizer::Buffers::Depth);

    /*Model Matrix*/
    render.setModelMatrix(
        "spot",
        SoftRasterizer::Tools::calculateModelMatrix(
            /*transform=*/Eigen::Vector3f(0.f, 0.0f, 0.0f),
            /*rotations=*/
            SoftRasterizer::Tools::calculateRotationMatrix(
                /*axis=*/Eigen::Vector3f(0.f, 1.f, 0.f),
                /*degree=+ for Counterclockwise;- for Clockwise*/ degree),
            /*scale=*/Eigen::Vector3f(0.3f, 0.3f, 0.3f)));

    /*View Matrix*/
    render.setViewMatrix(
        /*eye=*/Eigen::Vector3f(0.0f, 0.0f, 0.9f),
        /*center=*/Eigen::Vector3f(0.0f, 0.0f, 0.0f),
        /*up=*/Eigen::Vector3f(0.0f, 1.0f, 0.0f));

    /*Projection Matrix*/
    render.setProjectionMatrix(
        /*fov=*/45.0f,
        /*near=*/0.1f,
        /*far=*/100.0f);

    render.display(SoftRasterizer::Primitive::TRIANGLES);

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
