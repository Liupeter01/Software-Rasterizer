#include <Tools.hpp>
#include <Render.hpp>
#include <opencv2/opencv.hpp>

SoftRasterizer::RenderingPipeline::RenderingPipeline()
          : RenderingPipeline(800, 600)
{
}

SoftRasterizer::RenderingPipeline::RenderingPipeline(const std::size_t width,
                                                                                          const std::size_t height,
                                                                                          const Eigen::Matrix4f& model,
                                                                                          const Eigen::Matrix4f& view,
                                                                                          const Eigen::Matrix4f& projection)

          : m_width(width)
          , m_height(height)
{
          /*calculate ratio*/
          if (!height) {
                    throw std::runtime_error("Height cannot be zero!");
          }

          m_aspectRatio = static_cast<float>(m_width) / static_cast<float>(m_height);

          /*init MVP*/
          m_model = model;
          m_view = view;
          m_projection = projection;

          /*resize std::vector of framebuffer and z-Buffer*/
          m_frameBuffer.resize(width * height);
          m_zBuffer.resize(width * height);
}

SoftRasterizer::RenderingPipeline::~RenderingPipeline()
{
}

void SoftRasterizer::RenderingPipeline::draw(SoftRasterizer::Primitive type) {
          /*only draw lines*/
          if (type == SoftRasterizer::Primitive::LINES) {

                    for (const auto& face : m_faces) {

                              /*create a triangle class*/
                              SoftRasterizer::Triangle triangle;

                              Eigen::Vector3f A = m_vertices[face[0]];
                              Eigen::Vector3f B = m_vertices[face[1]];
                              Eigen::Vector3f C = m_vertices[face[2]];
 
                              //X,Y
                             A[0] = (A[0] + 1.0f) * m_width / 2.0f;    //X
                             A[1] = (A[1] + 1.0f) * m_height / 2.0f * m_aspectRatio;   //Y

                              //X,Y
                             B[0] = (B[0] + 1.0f) * m_width / 2.0f;    //X
                             B[1] = (B[1] + 1.0f) * m_height / 2.0f * m_aspectRatio;   //Y

                              //X,Y
                            C[0] = (C[0] + 1.0f) * m_width / 2.0f;    //X
                              C[1] = (C[1] + 1.0f) * m_height / 2.0f * m_aspectRatio;   //Y
                              A[2]  = B[2] = C[2] =1; //Z-Depth

                              triangle.setVertex({A, B, C});

                              triangle.setColor({ 
                                        m_colours[face[0]], 
                                        m_colours[face[1]], 
                                        m_colours[face[2]] }
                              );

                              rasterizeWireframe(triangle);
                    }

          }
          else if (type == SoftRasterizer::Primitive::TRIANGLES) {

                    // controls the stretching/compression of the range
                    float scale = (m_far - m_near) / 2.0f;

                    //  shifts the range
                    float offset = (m_far + m_near) / 2.0f;

                    /*MVP Matrix*/
                    Eigen::Matrix4f mvp = m_projection * m_view * m_model;

                    for (const auto& face : m_faces) {
                              /*create a triangle class*/
                              SoftRasterizer::Triangle triangle;

                              /*Vertex(4) MVP Transform to Vec(3)*/
                              Eigen::Vector4f vec4_a = mvp * SoftRasterizer::Tools::to_vec4(m_vertices[face[0]]);
                              Eigen::Vector4f vec4_b = mvp * SoftRasterizer::Tools::to_vec4(m_vertices[face[1]]);
                              Eigen::Vector4f vec4_c = mvp * SoftRasterizer::Tools::to_vec4(m_vertices[face[2]]);

                              Eigen::Vector3f  mvp_a = SoftRasterizer::Tools::to_vec3(vec4_a);
                              Eigen::Vector3f  mvp_b = SoftRasterizer::Tools::to_vec3(vec4_b);
                              Eigen::Vector3f  mvp_c = SoftRasterizer::Tools::to_vec3(vec4_c);
                     
                              //X,Y, Z-Depth
                              mvp_a[0] = (mvp_a[0] + 1.0f) * m_width / 2.0f;    //X
                              mvp_a[1] = (mvp_a[1] + 1.0f) * m_height / 2.0f * m_aspectRatio;   //Y
                              mvp_a[2] = mvp_a[2] * scale + offset; //Z-Depth

                              //X,Y, Z-Depth
                              mvp_b[0] = (mvp_b[0] + 1.0f) * m_width / 2.0f;    //X
                              mvp_b[1] = (mvp_b[1] + 1.0f) * m_height / 2.0f * m_aspectRatio;   //Y
                              mvp_b[2] = mvp_b[2] * scale + offset; //Z-Depth

                              //X,Y, Z-Depth
                              mvp_c[0] = (mvp_c[0] + 1.0f) * m_width / 2.0f;    //X
                              mvp_c[1] = (mvp_c[1] + 1.0f) * m_height / 2.0f * m_aspectRatio;   //Y
                              mvp_c[2] = mvp_c[2] * scale + offset; //Z-Depth

                              triangle.setVertex({ mvp_a , mvp_b, mvp_c });
                              triangle.setColor({ m_colours[face[0]], m_colours[face[1]], m_colours[face[2]] });

                              /*draw lines first*/
                              rasterizeWireframe(triangle);
                    }
          }
          else {
                    throw std::runtime_error("Drawing primitives other than triangle and line is not implemented yet!");
          }
}

void SoftRasterizer::RenderingPipeline::display(Primitive type){
          /*draw pictures according to the specific type*/
          draw(type);
          
          cv::Mat image(m_height, m_width, CV_32FC3, getFrameBuffer().data());
          image.convertTo(image, CV_8UC3, 1.0f);
          cv::imshow("SoftRasterizer", image);
}

void SoftRasterizer::RenderingPipeline::writePixel(const Eigen::Vector3f& point,
                                                                                     const Eigen::Vector3f& color){
          if (point.x() >= 0 && point.x() < m_width && point.y() >= 0 && point.y() < m_height) {
                    m_frameBuffer[static_cast<int>(point.x()) + static_cast<int>(point.y()) * m_width] = color;
          }
}

void SoftRasterizer::RenderingPipeline::rasterizeWireframe(const  SoftRasterizer::Triangle& triangle){
          drawLine(triangle.c(), triangle.a(), triangle.m_color[0]);
          drawLine(triangle.c(), triangle.b(), triangle.m_color[1]);
          drawLine(triangle.b(), triangle.a(), triangle.m_color[2]);
}

/* Bresenham algorithm*/
void SoftRasterizer::RenderingPipeline::drawLine(const Eigen::Vector3f& p0, 
                                                                                  const Eigen::Vector3f& p1, 
                                                                                  const Eigen::Vector3f& color)
{
          Eigen::Vector3f line_color = { 255, 255, 255 };

          /*----------------------------------*
          *  -   |
                 p0
                  | \
                  |   \
             dy |     \
                  |       \
                  |         p1
              -   |<-dx->|
          *-----------------------------------*/

          auto x1 = p0.x();
          auto y1 = p0.y();
          auto x2 = p1.x();
          auto y2 = p1.x();

          int x, y, dx, dy, dx1, dy1, px, py, xe, ye, i;

          dx = x2 - x1;
          dy = y2 - y1;
          dx1 = fabs(dx);
          dy1 = fabs(dy);
          px = 2 * dy1 - dx1;
          py = 2 * dx1 - dy1;

          if (dy1 <= dx1)
          {
                    if (dx >= 0)
                    {
                              x = x1;
                              y = y1;
                              xe = x2;
                    }
                    else
                    {
                              x = x2;
                              y = y2;
                              xe = x1;
                    }

                    writePixel(Eigen::Vector3f(x, y, 1.0f), line_color);

                    for (i = 0; x < xe; i++)
                    {
                              x = x + 1;
                              if (px < 0)
                              {
                                        px = px + 2 * dy1;
                              }
                              else
                              {
                                        if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)){
                                                  y = y + 1;
                                        }
                                        else{
                                                  y = y - 1;
                                        }
                                        px = px + 2 * (dy1 - dx1);
                              }
                              writePixel(Eigen::Vector3f(x, y, 1.0f), line_color);
                    }
          }
          else
          {
                    if (dy >= 0)
                    {
                              x = x1;
                              y = y1;
                              ye = y2;
                    }
                    else
                    {
                              x = x2;
                              y = y2;
                              ye = y1;
                    }

                    writePixel(Eigen::Vector3f(x, y, 1.0f), line_color);

                    for (i = 0; y < ye; i++)
                    {
                              y = y + 1;
                              if (py <= 0)
                              {
                                        py = py + 2 * dx1;
                              }
                              else
                              {
                                        if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0))
                                        {
                                                  x = x + 1;
                                        }
                                        else
                                        {
                                                  x = x - 1;
                                        }
                                        py = py + 2 * (dx1 - dy1);
                              }

                              writePixel(Eigen::Vector3f(x, y, 1.0f), line_color);
                    }
          }
}

void SoftRasterizer::RenderingPipeline::clear(SoftRasterizer::Buffers flags){
          if ((flags & SoftRasterizer::Buffers::Color) == SoftRasterizer::Buffers::Color){
                    std::for_each(m_frameBuffer.begin(), m_frameBuffer.end(), [](Eigen::Vector3f& color) {
                              color = Eigen::Vector3f{ 0, 0, 0 };
                     });
          }
          if ((flags & SoftRasterizer::Buffers::Depth) == SoftRasterizer::Buffers::Depth){
                    std::for_each(m_zBuffer.begin(), m_zBuffer.end(), [](float& depth) {
                              depth = std::numeric_limits<float>::infinity();
                     });
          }
}