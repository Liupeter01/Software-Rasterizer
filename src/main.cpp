
#include <iostream>
#include <Tools.hpp>
#include <Render.hpp>
#include <Eigen/Eigen>
#include <Triangle.hpp>
#include <ObjLoader.hpp>
#include <opencv2/opencv.hpp>

int main() {
  int key = 0;
  int frame_count = 0;
  float degree = 0.0f;

  SoftRasterizer::RenderingPipeline render(1000, 1000);

  render.addGraphicObj(CONFIG_HOME"examples/models/bunny/bunny.obj", "bunny",
            Eigen::Vector3f(0.f, 1.f, 0.f), 90.f,
            Eigen::Vector3f(0.f, -0.2f, 0.5f),
            Eigen::Vector3f(2.0f, 2.0f, 2.0f)
  );

  //render.addGraphicObj(CONFIG_HOME"examples/models/spot/spot_triangulated_good.obj", "spot",
  //          Eigen::Vector3f(0.f, 1.f, 0.f), 45.f,
  //          Eigen::Vector3f(0.f, 0.0f, 0.0f),
  //          Eigen::Vector3f(0.3f, 0.3f, 0.3f)
  //          );

  //render.startLoadingMesh("spot");
  render.startLoadingMesh("bunny");

  while (key != 27) {
    /*clear both shading and depth!*/
    render.clear(SoftRasterizer::Buffers::Color |
                 SoftRasterizer::Buffers::Depth);

    /*Rotations*/
    auto rotations = SoftRasterizer::Tools::calculateRotationMatrix(
        /*axis=*/Eigen::Vector3f(0, 0, 1),
        /*degree=+ for Counterclockwise;- for Clockwise*/ degree);

    /*Model Matrix*/
    //auto model = SoftRasterizer::Tools::calculateModelMatrix(
    //    /*transform=*/Eigen::Vector3f(0.0f, 0.0f, 0.f),
    //    /*rotations=*/rotations,
    //    /*scale=*/Eigen::Vector3f(0.3f, 0.3f, 0.3f));

    /*View Matrix*/
    auto view = SoftRasterizer::Tools::calculateViewMatrix(
        /*eye=*/Eigen::Vector3f(0.0f, 0.0f, 0.9f),
        /*center=*/Eigen::Vector3f(0.0f, 0.0f, 0.0f),
        /*up=*/Eigen::Vector3f(0.0f, 1.0f, 0.0f));

     auto projection = SoftRasterizer::Tools::calculateProjectionMatrix(
               /*fov=*/45.0f,
               /*aspect=*/1.0f,
               /*near=*/0.1f,
               /*far=*/100.0f
    );

    /*draw lines*/
    //render.setModelMatrix(model);
    render.setViewMatrix(view);
    render.setProjectionMatrix(projection);

    render.display(SoftRasterizer::Primitive::LINES);

    key = cv::waitKey(1);
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
