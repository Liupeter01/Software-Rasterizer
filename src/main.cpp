
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

  /*set up all vertices*/
  std::vector<Eigen::Vector3f> pos{{0.f, 0.5f, -0.1f},   {-0.5f, -0.5f, -0.1f},
                                   {0.5f, -0.5f, -0.1f}, {0.f, 0.5f, 0.1f},
                                   {0.7f, 0.5f, 0.1f},   {0.7f, -0.3f, 0.1f}};

  /*set up all shading param(colours)*/
  std::vector<Eigen::Vector3f> color{{1.0f, 0.f, 0.f}, {0.f, 1.0f, 0.f},
                                     {0.f, 0.f, 1.0f}, {1.0f, 1.f, 1.f},
                                     {1.f, 1.0f, 1.f}, {1.f, 1.f, 1.0f}};

  /*set up all indices(faces)*/
  std::vector<Eigen::Vector3i> ind{{0, 1, 2}/*, {3, 4, 1}*/};
  render.loadVertices(pos);
  render.loadColours(color);
  render.loadIndices(ind);

  while (key != 27) {
    /*clear both shading and depth!*/
    render.clear(SoftRasterizer::Buffers::Color |
                 SoftRasterizer::Buffers::Depth);

    /*Rotations*/
    auto rotations = SoftRasterizer::Tools::calculateRotationMatrix(
        /*axis=*/Eigen::Vector3f(0, 0, 1),
        /*degree=+ for Counterclockwise;- for Clockwise*/ degree);

    /*Model Matrix*/
    auto model = SoftRasterizer::Tools::calculateModelMatrix(
        /*transform=*/Eigen::Vector3f(0.0f, 0.0f, 0.f),
        /*rotations=*/rotations,
        /*scale=*/Eigen::Vector3f(0.3f, 0.3f, 0.3f));

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
    render.setModelMatrix(model);
    render.setViewMatrix(view);
    render.setProjectionMatrix(projection);

    render.display(SoftRasterizer::Primitive::TRIANGLES);

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
