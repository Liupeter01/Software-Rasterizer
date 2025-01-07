#include <Render.hpp>
#include <Tools.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

SoftRasterizer::RenderingPipeline::RenderingPipeline()
    : RenderingPipeline(800, 600) {}

SoftRasterizer::RenderingPipeline::RenderingPipeline(
    const std::size_t width, const std::size_t height,
    const Eigen::Matrix4f &view, const Eigen::Matrix4f &projection)
    : m_width(width), m_height(height) {

  /*calculate ratio*/
  if (!height) {
    throw std::runtime_error("Height cannot be zero!");
  }

  m_aspectRatio = static_cast<float>(m_width) / static_cast<float>(m_height);

  /*init MVP*/
  setViewMatrix(view);
  setProjectionMatrix(projection);

  /*Transform normalized coordinates into screen space coordinates*/
  Eigen::Matrix4f translate, scale, aspect, flipy;
  translate << 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1;
  scale << m_width / 2, 0, 0, 0, 0, m_height / 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
  flipy << -1, 0, 0, m_width, 0, -1, 0, m_height, 0, 0, 1, 0, 0, 0, 0, 1;

  if (-0.0000001f <= m_aspectRatio - 1.0f &&
      m_aspectRatio - 1.0f <= 0.0000001f) {
    aspect = Eigen::Matrix4f::Identity();
  } else {
    /*width maybe more/less than height*/
    aspect << 1, 0, 0, 0, 0, m_aspectRatio, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
  }

  m_ndcToScreenMatrix = flipy * aspect * scale * translate;

  /*resize std::vector of framebuffer and z-Buffer*/
  m_frameBuffer.resize(width * height);
  m_zBuffer.resize(width * height);
}

SoftRasterizer::RenderingPipeline::~RenderingPipeline() {}

void SoftRasterizer::RenderingPipeline::clear(SoftRasterizer::Buffers flags) {
  if ((flags & SoftRasterizer::Buffers::Color) ==
      SoftRasterizer::Buffers::Color) {
    std::for_each(
        m_frameBuffer.begin(), m_frameBuffer.end(),
        [](Eigen::Vector3f &color) { color = Eigen::Vector3f{0, 0, 0}; });
  }
  if ((flags & SoftRasterizer::Buffers::Depth) ==
      SoftRasterizer::Buffers::Depth) {
    std::for_each(m_zBuffer.begin(), m_zBuffer.end(), [](float &depth) {
      depth = std::numeric_limits<float>::infinity();
    });
  }
}

bool SoftRasterizer::RenderingPipeline::addGraphicObj(
    const std::string &path, const std::string &meshName,
    const Eigen::Matrix4f &rotation, const Eigen::Vector3f &translation,
    const Eigen::Vector3f &scale) {
  /*This Object has already been identified!*/
  if (m_suspendObjs.find(meshName) != m_suspendObjs.end()) {
    spdlog::error("This Object has already been identified");
    return false;
  }

  try {
    m_suspendObjs[meshName] = std::make_unique<ObjLoader>(
        path, meshName, rotation, translation, scale);
  } catch (const std::exception &e) {
    spdlog::info("Add Graphic Obj Error! Reason: {}", e.what());
    return false;
  }
  return true;
}

bool SoftRasterizer::RenderingPipeline::addGraphicObj(
    const std::string &path, const std::string &meshName,
    const Eigen::Vector3f &axis, const float angle,
    const Eigen::Vector3f &translation, const Eigen::Vector3f &scale) {
  /*This Object has already been identified!*/
  if (m_suspendObjs.find(meshName) != m_suspendObjs.end()) {
    spdlog::error(
        "Add Graphic Obj Error! This Object has already been identified");
    return false;
  }

  try {
    m_suspendObjs[meshName] = std::make_unique<ObjLoader>(
        path, meshName, axis, angle, translation, scale);
  } catch (const std::exception &e) {
    spdlog::error("Add Graphic Obj Error! Reason: {}", e.what());
    return false;
  }
  return true;
}

bool SoftRasterizer::RenderingPipeline::addGraphicObj(
    const std::string &path, const std::string &meshName) {

  /*This Object has already been identified!*/
  if (m_suspendObjs.find(meshName) != m_suspendObjs.end()) {
    spdlog::error("This Object has already been identified");
    return false;
  }

  try {
    m_suspendObjs[meshName] = std::make_unique<ObjLoader>(path, meshName);
  } catch (const std::exception &e) {
    spdlog::error("Add Graphic Obj Error! Reason: {}", e.what());
    return false;
  }
  return true;
}

bool SoftRasterizer::RenderingPipeline::startLoadingMesh(
    const std::string &meshName) {

  if (m_loadedObjs.find(meshName) != m_loadedObjs.end()) {
    spdlog::error("Start Loading Mesh Failed! Because {} Has Already Loaded "
                  "into m_loadedObjs",
                  meshName);
    return false;
  }

  /*This Object has already been identified!*/
  if (m_suspendObjs.find(meshName) == m_suspendObjs.end()) {
    spdlog::error("Start Loading Mesh Failed! Because There is nothing found "
                  "in m_suspendObjs");
    return false;
  }

  std::optional<std::unique_ptr<Mesh>> mesh_op =
      m_suspendObjs[meshName]->startLoadingFromFile(meshName);
  if (!mesh_op.has_value()) {
    spdlog::error("Start Loading Mesh Failed! Because Loading Internel Error!");
    return false;
  }

  try {
    m_loadedObjs[meshName] = std::move(mesh_op.value());
    m_loadedObjs[meshName]->meshname = meshName;
  } catch (const std::exception &e) {
    spdlog::error("Start Loading Mesh Failed! Reason: {}", e.what());
    return false;
  }
  return true;
}

/*set MVP*/
bool SoftRasterizer::RenderingPipeline::setModelMatrix(
    const std::string &meshName, const Eigen::Matrix4f &model) {
  if (m_suspendObjs.find(meshName) == m_suspendObjs.end()) {
    spdlog::error("Editing Model Matrix Failed! Because {} Not Found",
                  meshName);
    return false;
  }

  m_suspendObjs[meshName]->updateModelMatrix(model);
  return true;
}

void SoftRasterizer::RenderingPipeline::setViewMatrix(
    const Eigen::Matrix4f &view) {
  m_view = view;
}
void SoftRasterizer::RenderingPipeline::setProjectionMatrix(
    const Eigen::Matrix4f &projection) {
  m_projection = projection;
}

void SoftRasterizer::RenderingPipeline::draw(SoftRasterizer::Primitive type) {
  if ((type != SoftRasterizer::Primitive::LINES) &&
      (type != SoftRasterizer::Primitive::TRIANGLES)) {
    spdlog::error("Primitive Type is not supported!");
    throw std::runtime_error("Primitive Type is not supported!");
  }

  // controls the stretching/compression of the range
  float scale = (m_far - m_near) / 2.0f;

  //  shifts the range
  float offset = (m_far + m_near) / 2.0f;

  std::for_each(
      m_loadedObjs.begin(), m_loadedObjs.end(),
      [this, scale, offset,
       type](const decltype(m_loadedObjs)::value_type &objPair) {
        auto faces = objPair.second->faces;
        auto vertices = objPair.second->vertices;

        /*MVP Matrix*/
        Eigen::Matrix4f mvp =
            m_projection * m_view *
            m_suspendObjs[objPair.second->meshname]->getModelMatrix();

        for (long long face_index = 0; face_index < faces.size();
             ++face_index) {
          /*create a triangle class*/
          SoftRasterizer::Triangle triangle;

          SoftRasterizer::Vertex A = vertices[faces[face_index].x()];
          SoftRasterizer::Vertex B = vertices[faces[face_index].y()];
          SoftRasterizer::Vertex C = vertices[faces[face_index].z()];

          // vertices[faces[face_index].x()]

          /*triangle v*/
          std::array<Eigen::Vector3f, 3> vertex = {A.position, B.position,
                                                   C.position};

          /*texcoord vt*/
          std::array<Eigen::Vector2f, 3> texCoords = {A.texCoord, B.texCoord,
                                                      C.texCoord};

          /*normal coordinates vn*/
          std::array<Eigen::Vector3f, 3> normals = {A.normal, B.normal,
                                                    C.normal};

          /*Vertex(4) NDC Transform to Vec(3)*/
          // A[0] = (A[0] + 1.0f) * m_width / 2.0f; // X
          // A[1] = (m_height - (A[1] + 1.0f) * m_height / 2.0f) * (1.0f /
          // m_aspectRatio); // Y B[0] = (B[0] + 1.0f) * m_width / 2.0f; // X
          // B[1] = (m_height - (B[1] + 1.0f) * m_height / 2.0f) *(1.0f /
          // m_aspectRatio);
          // // Y C[0] = (C[0] + 1.0f) * m_width / 2.0f; // X C[1] = (m_height -
          // (C[1] + 1.0f) * m_height / 2.0f) *(1.0f / m_aspectRatio); // Y
          vertex[0] = Tools::to_vec3(m_ndcToScreenMatrix * mvp *
                                     Tools::to_vec4(vertex[0], 1.0f));
          vertex[1] = Tools::to_vec3(m_ndcToScreenMatrix * mvp *
                                     Tools::to_vec4(vertex[1], 1.0f));
          vertex[2] = Tools::to_vec3(m_ndcToScreenMatrix * mvp *
                                     Tools::to_vec4(vertex[2], 1.0f));

          vertex[0].z() = vertex[0].z() * scale + offset; // Z-Depth
          vertex[1].z() = vertex[1].z() * scale + offset; // Z-Depth
          vertex[2].z() = vertex[2].z() * scale + offset; // Z-Depth

          spdlog::debug("A(x, y)=({}, {}), B(x, y)=({}, {}), C(x, y)=({}, {})",
                        vertex[0].x(), vertex[0].y(), vertex[1].x(),
                        vertex[1].y(), vertex[2].x(), vertex[2].y());

          Eigen::Vector3f a(1.0f, 1.0f, 1.0f);

          auto white = Tools::normalizedToRGB(a.x(), a.y(), a.z());

          triangle.setColor({white, white, white});

          //   triangle.setColor({
          //   Tools::normalizedToRGB(m_colours[face[0]].x(),
          //                       m_colours[face[0]].y(),
          //                       m_colours[face[0]].z()),
          // Tools::normalizedToRGB(m_colours[face[1]].x(),
          //                       m_colours[face[1]].y(),
          //                       m_colours[face[1]].z()),
          // Tools::normalizedToRGB(m_colours[face[2]].x(),
          //                       m_colours[face[2]].y(),
          //                       m_colours[face[2]].z()) });

          /*set Vertex position*/
          triangle.setVertex({vertex[0], vertex[1], vertex[2]});

          /*draw line*/
          if (type == SoftRasterizer::Primitive::LINES) {
            rasterizeWireframe(triangle);
          }
          /*draw triangle*/
          else if (type == SoftRasterizer::Primitive::TRIANGLES) {
            rasterizeTriangle(triangle);
          }
        }
      });
}

void SoftRasterizer::RenderingPipeline::display(Primitive type) {
  /*draw pictures according to the specific type*/
  draw(type);

  cv::Mat image(m_height, m_width, CV_32FC3, getFrameBuffer().data());
  image.convertTo(image, CV_8UC3, 1.0f);
  cv::imshow("SoftRasterizer", image);
}

void SoftRasterizer::RenderingPipeline::writePixel(
    const Eigen::Vector3f &point, const Eigen::Vector3f &color) {
  if (point.x() >= 0 && point.x() < m_width && point.y() >= 0 &&
      point.y() < m_height) {
    m_frameBuffer[static_cast<int>(point.x()) +
                  static_cast<int>(point.y()) * m_width] = color;
  }
}

void SoftRasterizer::RenderingPipeline::writePixel(
    const Eigen::Vector3f &point, const Eigen::Vector3i &color) {
  writePixel(point, Eigen::Vector3f(static_cast<float>(color.x()),
                                    static_cast<float>(color.y()),
                                    static_cast<float>(color.z())));
}

bool SoftRasterizer::RenderingPipeline::writeZBuffer(
    const Eigen::Vector3f &point, const float depth) {
  if (point.x() >= 0 && point.x() < m_width && point.y() >= 0 &&
      point.y() < m_height) {

    auto cur_depth = m_zBuffer[static_cast<int>(point.x()) +
                               static_cast<int>(point.y()) * m_width];
    if (depth < cur_depth) {
      m_zBuffer[static_cast<int>(point.x()) +
                static_cast<int>(point.y()) * m_width] = depth;
      return true;
    }
  }
  return false;
}

void SoftRasterizer::RenderingPipeline::rasterizeWireframe(
    const SoftRasterizer::Triangle &triangle) {
  drawLine(triangle.b(), triangle.a(), triangle.m_color[0]);
  drawLine(triangle.b(), triangle.c(), triangle.m_color[1]);
  drawLine(triangle.a(), triangle.c(), triangle.m_color[2]);
}

/**
 * @brief Calculates the bounding box for a given triangle.
 *
 * This function determines the axis-aligned bounding box (AABB)
 * that encompasses the given triangle in 2D space. The bounding box
 * is represented as a pair of 2D integer vectors, indicating the
 * minimum and maximum corners of the box.
 *
 * @param triangle The triangle for which the bounding box is to be calculated.
 *                 The triangle is represented using the
 * `SoftRasterizer::Triangle` type.
 *
 * @return A pair of 2D integer vectors (Eigen::Vector2i), where:
 *         - The first vector represents the minimum corner of the bounding box
 * (bottom-left).
 *         - The second vector represents the maximum corner of the bounding box
 * (top-right).
 */
std::pair<Eigen::Vector2i, Eigen::Vector2i>
SoftRasterizer::RenderingPipeline::calculateBoundingBox(
    const SoftRasterizer::Triangle &triangle) {
  auto A = triangle.a();
  auto B = triangle.b();
  auto C = triangle.c();

  auto min = Eigen::Vector2i{
      static_cast<int>(
          std::floor(SoftRasterizer::Tools::min(A.x(), B.x(), C.x()))),
      static_cast<int>(
          std::floor(SoftRasterizer::Tools::min(A.y(), B.y(), C.y())))};

  auto max = Eigen::Vector2i{
      static_cast<int>(
          std::ceil(SoftRasterizer::Tools::max(A.x(), B.x(), C.x()))),
      static_cast<int>(
          std::ceil(SoftRasterizer::Tools::max(A.y(), B.y(), C.y())))};

  return std::pair<Eigen::Vector2i, Eigen::Vector2i>(min, max);
}

bool SoftRasterizer::RenderingPipeline::insideTriangle(
    const std::size_t x_pos, const std::size_t y_pos,
    const SoftRasterizer::Triangle &triangle) {
  const Eigen::Vector3f P = {static_cast<float>(x_pos),
                             static_cast<float>(y_pos), 1.0f};

  Eigen::Vector3f A = triangle.a();
  Eigen::Vector3f B = triangle.b();
  Eigen::Vector3f C = triangle.c();

  // Vectors representing the edges of the triangle
  Eigen::Vector3f AB = B - A;
  Eigen::Vector3f BC = C - B;
  Eigen::Vector3f CA = A - C;

  // Vectors from the point to each vertex
  Eigen::Vector3f AP = P - A;
  Eigen::Vector3f BP = P - B;
  Eigen::Vector3f CP = P - C;

  // Cross products to determine the relative orientation of the point with
  // respect to each edge
  Eigen::Vector3f crossABP = AB.cross(AP);
  Eigen::Vector3f crossBCP = BC.cross(BP);
  Eigen::Vector3f crossCAP = CA.cross(CP);

  // Check if all cross products have the same sign
  // If all cross products have the same sign, the point is inside the triangle
  return crossABP.z() * crossBCP.z() > 0 && crossBCP.z() * crossCAP.z() > 0 &&
         crossCAP.z() * crossABP.z() > 0;
}

std::optional<std::tuple<float, float, float>>
SoftRasterizer::RenderingPipeline::barycentric(
    const std::size_t x_pos, const std::size_t y_pos,
    const SoftRasterizer::Triangle &triangle) {
  if (!insideTriangle(x_pos, y_pos, triangle)) {
    return std::nullopt;
  }

  const Eigen::Vector3f P = {static_cast<float>(x_pos),
                             static_cast<float>(y_pos), 1.0f};

  Eigen::Vector3f A = triangle.a();
  Eigen::Vector3f B = triangle.b();
  Eigen::Vector3f C = triangle.c();

  /*For Triangle Sabc*/
  auto AC = C - A;
  auto AB = B - A;
  auto SquareOfTriangle = AC.cross(AB).norm() / 2.0f;

  /*For Small Triangle Spbc*/
  auto PB = B - P;
  auto PC = C - P;
  auto SquareOfSmallTrianglePBC = PC.cross(PB).norm() / 2.0f;

  /*For Small Triangle Sapc*/
  auto PA = A - P;
  auto SquareOfSmallTrianglePAC = PA.cross(PC).norm() / 2.0f;

  float alpha = SquareOfSmallTrianglePBC / SquareOfTriangle;
  float beta = SquareOfSmallTrianglePAC / SquareOfTriangle;

  return std::tuple<float, float, float>(alpha, beta, 1.0f - alpha - beta);
}

void SoftRasterizer::RenderingPipeline::rasterizeTriangle(
    const SoftRasterizer::Triangle &triangle) {
  /*Get All Points Inside Triangle*/
  auto A = triangle.a();
  auto B = triangle.b();
  auto C = triangle.c();

  auto normalA = triangle.m_normal[0];
  auto normalB = triangle.m_normal[1];
  auto normalC = triangle.m_normal[2];

  auto colorA = triangle.m_color[0];
  auto colorB = triangle.m_color[1];
  auto colorC = triangle.m_color[2];

  /*min and max point cood*/
  auto [min, max] = calculateBoundingBox(triangle);

  spdlog::debug("Bounding Box: min(x,y)=({},{}), max(x,y)=({},{})", min.x(),
                min.y(), max.x(), max.y());

#pragma omp parallel for collapse(2)
  for (long long y = (min.y() >= 0 ? min.y() : 0);
       y < (max.y() > m_height ? m_height : max.y()); y++) {
    for (long long x = (min.x() >= 0 ? min.x() : 0);
         x < (max.x() > m_width ? m_width : max.x()); x++) {
      /*is this Point(x,y) inside triangle*/
      if (insideTriangle(x + 0.5f, y + 0.5f, triangle)) [[likely]] {
        auto [alpha, beta, gamma] = barycentric(x, y, triangle).value();

        /*for Z-buffer interpolated*/

        float w_reciprocal = 1.0f / (alpha + beta + gamma);
        float z_interpolated = alpha * A.z() + beta * B.z() + gamma * C.z();
        z_interpolated *= w_reciprocal;

        /* test and write z-buffer
         * if depth is smaller than the current depth, update the z-buffer
         * meanwhile, write the color to the frame buffer
         */
        if (writeZBuffer(Eigen::Vector3f(x, y, 1.0f), z_interpolated)) {

          /*for color interpolated*/
          auto RGB_i =
              Tools::interpolateRGB(alpha, beta, gamma, colorA, colorB, colorC);

          writePixel(Eigen::Vector3f(x, y, 1.0f), RGB_i);
        }
      }
    }
  }
}

/* Bresenham algorithm*/
void SoftRasterizer::RenderingPipeline::drawLine(const Eigen::Vector3f &p0,
                                                 const Eigen::Vector3f &p1,
                                                 const Eigen::Vector3i &color) {

  auto x1 = p0.x();
  auto y1 = p0.y();
  auto x2 = p1.x();
  auto y2 = p1.y();

  int x, y, dx, dy, dx1, dy1, px, py, xe, ye, i;

  dx = x2 - x1;
  dy = y2 - y1;
  dx1 = fabs(dx);
  dy1 = fabs(dy);
  px = 2 * dy1 - dx1;
  py = 2 * dx1 - dy1;

  if (dy1 <= dx1) {
    if (dx >= 0) {
      x = x1;
      y = y1;
      xe = x2;
    } else {
      x = x2;
      y = y2;
      xe = x1;
    }

    writePixel(Eigen::Vector3f(x, y, 1.0f), color);

    for (i = 0; x < xe; i++) {
      x = x + 1;
      if (px < 0) {
        px = px + 2 * dy1;
      } else {
        if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)) {
          y = y + 1;
        } else {
          y = y - 1;
        }
        px = px + 2 * (dy1 - dx1);
      }
      writePixel(Eigen::Vector3f(x, y, 1.0f), color);
    }
  } else {
    if (dy >= 0) {
      x = x1;
      y = y1;
      ye = y2;
    } else {
      x = x2;
      y = y2;
      ye = y1;
    }

    writePixel(Eigen::Vector3f(x, y, 1.0f), color);

    for (i = 0; y < ye; i++) {
      y = y + 1;
      if (py <= 0) {
        py = py + 2 * dx1;
      } else {
        if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)) {
          x = x + 1;
        } else {
          x = x - 1;
        }
        py = py + 2 * (dx1 - dy1);
      }

      writePixel(Eigen::Vector3f(x, y, 1.0f), color);
    }
  }
}
