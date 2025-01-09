#include <Render.hpp>
#include <Tools.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

SoftRasterizer::RenderingPipeline::RenderingPipeline()
    : RenderingPipeline(800, 600) {}

SoftRasterizer::RenderingPipeline::RenderingPipeline(
    const std::size_t width, const std::size_t height,
    const Eigen::Matrix4f &view, const Eigen::Matrix4f &projection)
    : m_width(width), m_height(height), UNROLLING_FACTOR(1){

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

  cache_line_size = GET_CACHE_LINE_SIZE();
  UNROLLING_FACTOR = ROUND_UP_TO_MULTIPLE_OF_4(cache_line_size / (sizeof(Eigen::Vector3f) + sizeof(float)));
  spdlog::info("Current Arch Support {}B Cache Line! "
            "UNROLLING_FACTOR Was Set To {}", cache_line_size, UNROLLING_FACTOR);
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

bool SoftRasterizer::RenderingPipeline::addShader(
    const std::string &shaderName, const std::string &texturePath,
    SHADERS_TYPE type) {
  if (m_shaders.find(shaderName) != m_shaders.end()) {
    spdlog::error("Add Shader Failed! Because Shader {} Already Exist!",
                  shaderName);
    return false;
  }
  try {
    m_shaders[shaderName] = std::make_shared<Shader>(texturePath);
    m_shaders[shaderName]->setFragmentShader(type);
  } catch (const std::exception &e) {
    spdlog::error("Add Shader Failed! Reason: {}", e.what());
    return false;
  }
  return true;
}

bool SoftRasterizer::RenderingPipeline::addShader(
    const std::string &shaderName, std::shared_ptr<TextureLoader> text,
    SHADERS_TYPE type) {
  if (m_shaders.find(shaderName) != m_shaders.end()) {
    spdlog::error("Add Shader Failed! Because Shader {} Already Exist!",
                  shaderName);
    return false;
  }
  try {
    m_shaders[shaderName] = std::make_shared<Shader>(text);
    m_shaders[shaderName]->setFragmentShader(type);
  } catch (const std::exception &e) {
    spdlog::error("Add Shader Failed! Reason: {}", e.what());
    return false;
  }
  return true;
}

bool SoftRasterizer::RenderingPipeline::bindShader2Mesh(
    const std::string &meshName, const std::string &shaderName) {

  if (m_loadedObjs.find(meshName) == m_loadedObjs.end()) {
    spdlog::error(
        "Bind Shader To Mesh Failed! Because Loaded Mesh {} Not found!",
        meshName);
    return false;
  }

  if (m_shaders.find(shaderName) == m_shaders.end()) {
    spdlog::error("Bind Shader To Mesh Failed! Because Shader {} Not found!",
                  shaderName);
    return false;
  }

  try {
    m_loadedObjs[meshName]->bindShader2Mesh(m_shaders[shaderName]);
  } catch (const std::exception &e) {
    spdlog::error("Bind Shader To Mesh Failed! Reason: {}", e.what());
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

void SoftRasterizer::RenderingPipeline::setViewMatrix(
    const Eigen::Vector3f &eye, const Eigen::Vector3f &center,
    const Eigen::Vector3f &up) {
  m_eye = eye;
  m_center = center;
  m_up = up;
  setViewMatrix(Tools::calculateViewMatrix(
      /*eye=*/m_eye,
      /*center=*/m_center,
      /*up=*/m_up));
}

void SoftRasterizer::RenderingPipeline::setProjectionMatrix(float fovy,
                                                            float zNear,
                                                            float zFar) {
  m_fovy = fovy;
  m_near = zNear;
  m_far = zFar;
  setProjectionMatrix(Tools::calculateProjectionMatrix(
      /*fov=*/m_fovy,
      /*aspect=*/m_aspectRatio,
      /*near=*/m_near,
      /*far=*/m_far));
}

void SoftRasterizer::RenderingPipeline::display(Primitive type) {
  /*draw pictures according to the specific type*/
  draw(type);

  cv::Mat image(m_height, m_width, CV_32FC3, getFrameBuffer().data());
  image.convertTo(image, CV_8UC3, 1.0f);
  cv::imshow("SoftRasterizer", image);
}

inline
void 
SoftRasterizer::RenderingPipeline::writePixel(
          const long long x, const long long y, const Eigen::Vector3f &color) {
          if (x >= 0 && x < m_width && y >= 0 &&
                    y < m_height) {
    m_frameBuffer[x +
                  y* m_width] = color;
  }
}

inline 
void 
SoftRasterizer::RenderingPipeline::writePixel(
          const long long x, const long long y, const Eigen::Vector3i &color) {
          if (x >= 0 && x < m_width && y >= 0 &&
                    y < m_height) {
                    m_frameBuffer[x +
                              y * m_width] = Eigen::Vector3f(color.x(), color.y(), color.z());
          }
}

inline bool SoftRasterizer::RenderingPipeline::writeZBuffer(
          const long long x, const long long y, const float depth) {
  if (x >= 0 && x < m_width && y >= 0 &&
      y < m_height) {

    auto cur_depth = m_zBuffer[x+y* m_width];
    if (depth <= cur_depth) {
              m_zBuffer[x + y * m_width] = depth;
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

  A.z() = B.z() = C.z() = 1.0f;

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
SoftRasterizer::RenderingPipeline::linearBaryCentric(
    const std::size_t x_pos, const std::size_t y_pos, const Eigen::Vector2i min,
    const Eigen::Vector2i max) {
  // Bounds check: ensure x_pos and y_pos are inside the rectangle defined by
  // min and max
  if (x_pos < min.x() || x_pos >= max.x() || y_pos < min.y() ||
      y_pos >= max.y()) {
    return std::nullopt; // Point is outside the bounds
  }
  // Linear interpolation of alpha based on x position
  float alpha = static_cast<float>(x_pos - min.x()) /
                static_cast<float>(max.x() - min.x());

  // Linear interpolation of beta based on y position
  float beta = static_cast<float>(y_pos - min.y()) /
               static_cast<float>(max.y() - min.y());

  // Calculate gamma (the remainder to sum to 1)
  float gamma = 1.0f - alpha - beta;
  return std::tuple<float, float, float>(alpha, beta, gamma);
}

inline std::optional<std::tuple<float, float, float>>
SoftRasterizer::RenderingPipeline::barycentric(
    const std::size_t x_pos, const std::size_t y_pos,
    const SoftRasterizer::Triangle &triangle) {

  if (!insideTriangle(x_pos + 0.5f, y_pos + 0.5f, triangle)) {
    return std::nullopt;
  }

  const Eigen::Vector3f P = {static_cast<float>(x_pos),
                             static_cast<float>(y_pos), 1.0f};

  Eigen::Vector3f A = triangle.a();
  Eigen::Vector3f B = triangle.b();
  Eigen::Vector3f C = triangle.c();

  A.z() = B.z() = C.z() = 1.0f;

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

void SoftRasterizer::RenderingPipeline::draw(SoftRasterizer::Primitive type) {
  if ((type != SoftRasterizer::Primitive::LINES) &&
      (type != SoftRasterizer::Primitive::TRIANGLES)) {
    spdlog::error("Primitive Type is not supported!");
    throw std::runtime_error("Primitive Type is not supported!");
  }

  std::for_each(
      m_loadedObjs.begin(), m_loadedObjs.end(),
      [this, type](const decltype(m_loadedObjs)::value_type &objPair) {
        auto meshName = objPair.first;
        auto faces = objPair.second->faces;
        auto vertices = objPair.second->vertices;

        /*get shader object for this mesh obj*/
        std::shared_ptr<SoftRasterizer::Shader> shader =
            m_loadedObjs[meshName]->m_shader;

        /*MVP Matrix*/
        auto Model = m_suspendObjs[meshName]->getModelMatrix();

        for (long long face_index = 0; face_index < faces.size();
             ++face_index) {

          const auto &face = faces[face_index];

          /*create a triangle class*/
          SoftRasterizer::Triangle triangle;

          SoftRasterizer::Vertex A = vertices[face.x()];
          SoftRasterizer::Vertex B = vertices[face.y()];
          SoftRasterizer::Vertex C = vertices[face.z()];

          /*triangle v, texcoord vt, normal coordinates vn*/
          fragment_shader_payload payloads[] = {
              {A.position, A.normal, A.texCoord},
              {B.position, B.normal, B.texCoord},
              {C.position, C.normal, C.texCoord}};

          vertex_displacement newVertices[] = {
              shader->applyVertexShader(Model, m_view, m_projection,
                                        payloads[0]),
              shader->applyVertexShader(Model, m_view, m_projection,
                                        payloads[1]),
              shader->applyVertexShader(Model, m_view, m_projection,
                                        payloads[2])};

          payloads[0].position = newVertices[0].new_position;
          payloads[0].normal = newVertices[0].new_normal;
          payloads[1].position = newVertices[1].new_position;
          payloads[1].normal = newVertices[1].new_normal;
          payloads[2].position = newVertices[2].new_position;
          payloads[2].normal = newVertices[2].new_normal;

          /*set Vertex position*/
          triangle.setVertex({payloads[0].position, payloads[1].position,
                              payloads[2].position});

          triangle.setTexCoord({payloads[0].texCoords, payloads[1].texCoords,
                                payloads[2].texCoords});
          triangle.setNormal(
              {payloads[0].normal, payloads[1].normal, payloads[2].normal});

          /*draw line*/
          if (type == SoftRasterizer::Primitive::LINES) {
            rasterizeWireframe(triangle);
          }
          /*draw triangle*/
          else if (type == SoftRasterizer::Primitive::TRIANGLES) {
            rasterizeTriangle(shader, triangle);
          }
        }
      });
}

void SoftRasterizer::RenderingPipeline::rasterizeTriangle(
    std::shared_ptr<SoftRasterizer::Shader> shader,
    SoftRasterizer::Triangle &triangle) {

  // controls the stretching/compression of the range
  float scale = (m_far - m_near) / 2.0f;

  //  shifts the range
  float offset = (m_far + m_near) / 2.0f;

  std::initializer_list<light_struct> lights = {
      {m_eye, Eigen::Vector3f{80, 80, 80}},
      {Eigen::Vector3f{0.9, 0.9, -0.9f}, Eigen::Vector3f{80, 80, 80}}};

  fragment_shader_payload payloads[] = {
      {triangle.m_vertex[0], triangle.m_normal[0],
       triangle.m_texCoords[0]}, // A
      {triangle.m_vertex[1], triangle.m_normal[1],
       triangle.m_texCoords[1]},                                            // B
      {triangle.m_vertex[2], triangle.m_normal[2], triangle.m_texCoords[2]} // C
  };

  /*Vertex(4) NDC Transform to Vec(3)*/
  payloads[0].position = Tools::to_vec3(
      m_ndcToScreenMatrix * Tools::to_vec4(payloads[0].position, 1.0f));
  payloads[1].position = Tools::to_vec3(
      m_ndcToScreenMatrix * Tools::to_vec4(payloads[1].position, 1.0f));
  payloads[2].position = Tools::to_vec3(
      m_ndcToScreenMatrix * Tools::to_vec4(payloads[2].position, 1.0f));

  payloads[0].position.z() =
      payloads[0].position.z() * scale + offset; // Z-Depth
  payloads[1].position.z() =
      payloads[1].position.z() * scale + offset; // Z-Depth
  payloads[2].position.z() =
      payloads[2].position.z() * scale + offset; // Z-Depth

  /*update triangle position!*/
  auto A = triangle.m_vertex[0] = payloads[0].position;
  auto B = triangle.m_vertex[1] = payloads[1].position;
  auto C = triangle.m_vertex[2] = payloads[2].position;

  /*min and max point cood*/
  auto [min, max] = calculateBoundingBox(triangle);

  long long startX = (min.x() >= 0 ? min.x() : 0);
  long long startY = (min.y() >= 0 ? min.y() : 0);

  long long endX = (max.x() > m_width ? m_width  : max.x());
  long long endY = (max.y() > m_height ? m_height : max.y());

  auto prefetch_value = startY * m_width + startX;

  /*zBuffer each item size is 3 float*/
  PREFETCH(reinterpret_cast<char *>(&m_frameBuffer[prefetch_value]));

  /*zBuffer each item size is a float*/
  PREFETCH(reinterpret_cast<char *>(&m_zBuffer[prefetch_value]));

#pragma omp parallel for collapse(2)
  for (auto y = startY; y < endY; y ++) {
            for (auto xbase = startX; xbase < endX; xbase += UNROLLING_FACTOR) { // Loop unrolled by UNROLLING_FACTOR in x

                      // Process points  in a (UNROLLING_FACTOR, UNROLLING_FACTOR) block
                      for (auto x = xbase; x < xbase + UNROLLING_FACTOR && x < endX; ++x) {

                                // Check if the point (currentX, currentY) is inside the triangle
                                auto res = barycentric(x, y, triangle);
                                if (!res.has_value()) {
                                          continue;
                                }

                                auto [alpha, beta, gamma] = res.value();

                                // For Z-buffer interpolation
                                float w_reciprocal = 1.0f / (alpha + beta + gamma);
                                float z_interpolated = alpha * A.z() + beta * B.z() + gamma * C.z();
                                z_interpolated *= w_reciprocal;

                                // Test and write z-buffer (check depth before writing)
                                if (writeZBuffer( x,y ,
                                          z_interpolated)) {

                                          /*interpolate normal*/
                                          auto interpolation_normal = Tools::interpolateNormal(
                                                    alpha, beta, gamma, payloads[0].normal, payloads[1].normal,
                                                    payloads[2].normal);

                                          /*interpolate uv*/
                                          auto interpolation_texCoord = Tools::interpolateTexCoord(
                                                    alpha, beta, gamma, payloads[0].texCoords,
                                                    payloads[1].texCoords, payloads[2].texCoords);

                                          fragment_shader_payload shading_on_xy(
                                                    Eigen::Vector3f(x, y, z_interpolated),
                                                    interpolation_normal, interpolation_texCoord);

                                          auto color = Tools::normalizedToRGB(
                                                    shader->applyFragmentShader(m_eye, lights, shading_on_xy));

                                          // Write pixel to the frame buffer
                                          writePixel( x, y , color);
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

    writePixel(x, y, color);

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
      writePixel( x, y , color);
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

    writePixel(x, y , color);

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

      writePixel( x, y , color);
    }
  }
}
