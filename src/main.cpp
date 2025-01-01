#include <Triangle.hpp>
#include <Render.hpp>
#include <Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
          SoftRasterizer::RenderingPipeline render(800, 600);

          /*set up all vertices*/
          std::vector<Eigen::Vector3f> pos{
                  {0.f, 0.5f, 0.f},
                  {-0.5f, -0.5f, 0.f},
                  {0.5f, -0.5f, 0.f}
          };
          render.loadVertices(pos);

          /*set up all shading param(colours)*/
          std::vector<Eigen::Vector3f> color{
                  {1.0f, 0.f, 0.f},
                  {0.f, 1.0f, 0.f},
                  {0.f, 0.f, 1.0f}
          };
          render.loadColours(color);

          /*set up all indices(faces)*/
          std::vector<Eigen::Vector3i> ind {{0, 1, 2}};
          render.loadIndices(ind);

          int key = 0;
          int frame_count = 0;
          while (key != 27) {
                    /*clear both shading and depth!*/
                    render.clear(SoftRasterizer::Buffers::Color | SoftRasterizer::Buffers::Depth);

                    /*draw lines*/
                    render.display(SoftRasterizer::Primitive::LINES);

                    key = cv::waitKey(10);
          }
          return 0;
}
