#include <Render.hpp>
#include <Tools.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

SoftRasterizer::RenderingPipeline::RenderingPipeline()
    : RenderingPipeline(800, 600) {}

SoftRasterizer::RenderingPipeline::RenderingPipeline(
          const std::size_t width, const std::size_t height,
          const Eigen::Matrix4f& view, const Eigen::Matrix4f& projection)
          : m_width(width), m_height(height)
          , m_channels(numbers)        /*set to three*/
          , m_frameBuffer(m_height, m_width, CV_32FC3)
          , inf(simde_mm256_set1_ps(std::numeric_limits<float>::infinity()))         /*SIMD inf*/
          , one(simde_mm256_set1_ps(1.0f))                                                               /*SIMD one*/
          , zero(simde_mm256_set1_ps(0.0f))
          , UNROLLING_FACTOR(8){
          /*set channel ammount to three!*/
          m_channels.resize(numbers);

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
          }
          else {
                    /*width maybe more/less than height*/
                    aspect << 1, 0, 0, 0, 0, m_aspectRatio, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
          }

          m_ndcToScreenMatrix = flipy * aspect * scale * translate;

          /*resize std::vector of z-Buffer*/
          m_zBuffer.resize(width * height);

          /*init framebuffer*/
          clear(SoftRasterizer::Buffers::Color |
                    SoftRasterizer::Buffers::Depth);

          UNROLLING_FACTOR = 8;
}

SoftRasterizer::RenderingPipeline::~RenderingPipeline() {}

void  
SoftRasterizer::RenderingPipeline::clearFrameBuffer(){
//#pragma omp parallel for
          for(long long i = 0 ; i < numbers ; ++i){
                    m_channels[i] = cv::Mat::zeros(m_height, m_width, CV_32FC1);
          }

          m_frameBuffer = cv::Mat::zeros(m_height, m_width, CV_32FC3);
}

void 
SoftRasterizer::RenderingPipeline::clearZDepth(){
          std::for_each(m_zBuffer.begin(), m_zBuffer.end(), [](float& depth) {
                    depth = std::numeric_limits<float>::infinity();
                    });
}

void SoftRasterizer::RenderingPipeline::clear(SoftRasterizer::Buffers flags) {
  if ((flags & SoftRasterizer::Buffers::Color) ==
      SoftRasterizer::Buffers::Color) {
            clearFrameBuffer();
  }
  if ((flags & SoftRasterizer::Buffers::Depth) ==
      SoftRasterizer::Buffers::Depth) {
            clearZDepth();
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

  scale = (m_far - m_near) / 2.0f;
  offset = (m_far + m_near) / 2.0f;

  scale_simd = simde_mm256_set1_ps(scale);
  offset_simd = simde_mm256_set1_ps(offset);

  setProjectionMatrix(Tools::calculateProjectionMatrix(
      /*fov=*/m_fovy,
      /*aspect=*/m_aspectRatio,
      /*near=*/m_near,
      /*far=*/m_far));
}

void SoftRasterizer::RenderingPipeline::display(Primitive type) {
  /*draw pictures according to the specific type*/
  draw(type);

  cv::merge(m_channels, m_frameBuffer);
  m_frameBuffer.convertTo(m_frameBuffer, CV_8UC3, 1.0f);
  cv::imshow("SoftRasterizer", m_frameBuffer);
}

inline void 
SoftRasterizer::RenderingPipeline::writePixel(
    const long long x, const long long y, const Eigen::Vector3f &color) {
  if (x >= 0 && x < m_width && y >= 0 && y < m_height) {
            auto pos = x + y * m_width;

            *(m_channels[0].ptr<float>(0) + pos) = color.x(); //R
            *(m_channels[1].ptr<float>(0) + pos) = color.y(); //G
            *(m_channels[2].ptr<float>(0) + pos) = color.z(); //B
  }
}

inline void 
SoftRasterizer::RenderingPipeline::writePixel(
          const long long x, const long long y, const Eigen::Vector3i& color) {
          writePixel(x, y, Eigen::Vector3f(color.x(), color.y(), color.z()));
}

inline 
void 
SoftRasterizer::RenderingPipeline::writePixel(
          const long long start_pos, const ColorSIMD& color){
          writePixel(start_pos, color.r, color.g, color.b);
}

inline void SoftRasterizer::RenderingPipeline::writePixel(const long long start_pos,
          const simde__m256& r, const simde__m256& g, const simde__m256& b)
{
          simde_mm256_storeu_ps(m_channels[0].ptr<float>(0) + start_pos, r);//R
          simde_mm256_storeu_ps(m_channels[1].ptr<float>(0) + start_pos, g);//G
          simde_mm256_storeu_ps(m_channels[2].ptr<float>(0) + start_pos, b);//B
}

inline bool SoftRasterizer::RenderingPipeline::writeZBuffer(const long long x,
                                                            const long long y,
                                                            const float depth) {
  if (x >= 0 && x < m_width && y >= 0 && y < m_height) {

    auto cur_depth = m_zBuffer[x + y * m_width];
    if (depth < cur_depth) {
      m_zBuffer[x + y * m_width] = depth;
      return true;
    }
  }
  return false;
}

inline void
SoftRasterizer::RenderingPipeline::writeZBuffer(const long long start_pos,
          const simde__m256& depth){
          simde_mm256_storeu_ps(reinterpret_cast<float*>(&m_zBuffer[start_pos]), depth);
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

  // Cross product results (we only need the z-components)
  const float crossABP_z = AB.x() * AP.y() - AB.y() * AP.x();
  const float crossBCP_z = BC.x() * BP.y() - BC.y() * BP.x();
  const float crossCAP_z = CA.x() * CP.y() - CA.y() * CP.x();

  // Check if all cross products have the same sign
  return (crossABP_z > 0 && crossBCP_z > 0 && crossCAP_z > 0) ||
            (crossABP_z < 0 && crossBCP_z < 0 && crossCAP_z < 0);
}

simde__m256 
SoftRasterizer::RenderingPipeline::insideTriangle(const simde__m256& x, const simde__m256& y, 
          const SoftRasterizer::Triangle& triangle) {

          Eigen::Vector3f A = triangle.a();
          Eigen::Vector3f B = triangle.b();
          Eigen::Vector3f C = triangle.c();

          A.z() = B.z() = C.z() = 1.0f;

          // Load triangle vertex positions into SIMD registers
          simde__m256 ax = simde_mm256_set1_ps(A.x());
          simde__m256 ay = simde_mm256_set1_ps(A.y());
          simde__m256 bx = simde_mm256_set1_ps(B.x());
          simde__m256 by = simde_mm256_set1_ps(B.y());
          simde__m256 cx = simde_mm256_set1_ps(C.x());
          simde__m256 cy = simde_mm256_set1_ps(C.y());

          // Vectors from point P (x_pos, y_pos) to each vertex
          simde__m256 px = simde_mm256_add_ps(x, simde_mm256_set1_ps(0.5f));
          simde__m256 py = simde_mm256_add_ps(y, simde_mm256_set1_ps(0.5f));

          // Cross products (z-component only)
          simde__m256 crossABP = simde_mm256_sub_ps(
                    simde_mm256_mul_ps(simde_mm256_sub_ps(bx, ax), simde_mm256_sub_ps(py, ay)),
                    simde_mm256_mul_ps(simde_mm256_sub_ps(by, ay), simde_mm256_sub_ps(px, ax))
          );

          simde__m256 crossBCP = simde_mm256_sub_ps(
                    simde_mm256_mul_ps(simde_mm256_sub_ps(cx, bx), simde_mm256_sub_ps(py, by)),
                    simde_mm256_mul_ps(simde_mm256_sub_ps(cy, by), simde_mm256_sub_ps(px, bx))
          );

          simde__m256 crossCAP = simde_mm256_sub_ps(
                    simde_mm256_mul_ps(simde_mm256_sub_ps(ax, cx), simde_mm256_sub_ps(py, cy)),
                    simde_mm256_mul_ps(simde_mm256_sub_ps(ay, cy), simde_mm256_sub_ps(px, cx))
          );

          // Check if all cross products have the same sign (positive or negative)
          simde__m256 zero = simde_mm256_set1_ps(0.0f);
          simde__m256 signABP = simde_mm256_cmp_ps(crossABP, zero, SIMDE_CMP_GT_OQ); // > 0
          simde__m256 signBCP = simde_mm256_cmp_ps(crossBCP, zero, SIMDE_CMP_GT_OQ); // > 0
          simde__m256 signCAP = simde_mm256_cmp_ps(crossCAP, zero, SIMDE_CMP_GT_OQ); // > 0

          // Combine the signs: all positive or all negative
          simde__m256 allPositive = simde_mm256_and_ps(simde_mm256_and_ps(signABP, signBCP), signCAP);
          simde__m256 allNegative = simde_mm256_and_ps(
                    simde_mm256_and_ps(simde_mm256_cmp_ps(crossABP, zero, SIMDE_CMP_LT_OQ),
                              simde_mm256_cmp_ps(crossBCP, zero, SIMDE_CMP_LT_OQ)),
                    simde_mm256_cmp_ps(crossCAP, zero, SIMDE_CMP_LT_OQ)
          );

          return simde_mm256_or_ps(allPositive, allNegative);
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

inline std::tuple<float, float, float>
SoftRasterizer::RenderingPipeline::barycentric(
    const std::size_t x_pos, const std::size_t y_pos,
    const SoftRasterizer::Triangle &triangle) {

  Eigen::Vector3f A = triangle.a();
  Eigen::Vector3f B = triangle.b();
  Eigen::Vector3f C = triangle.c();

    // Compute edges
  const float ABx = B.x() - A.x(), ABy = B.y() - A.y();
  const float ACx = C.x() - A.x(), ACy = C.y() - A.y();
  const float PAx =  A.x() - x_pos, PAy =  A.y() - y_pos;
  const float BCx = C.x() - B.x(), BCy = C.y() - B.y();
  const float PBx =  B.x() - x_pos, PBy = B.y() - y_pos;
  const float PCx = C.x() - x_pos, PCy =  C.y() - y_pos;

  // Compute areas directly using the 2D cross product (determinant)
  const float areaABC = ABx * ACy - ABy * ACx;  // Area of triangle ABC
  const float areaPBC = PBx * PCy - PBy * PCx;  // Area of triangle PBC
  const float areaPCA = PCx * PAy - PCy * PAx;  // Area of triangle PCA

  // Calculate barycentric coordinates
  const float alpha = areaPBC / areaABC;
  const float beta = areaPCA / areaABC;

  return { alpha, beta, 1.0f - alpha - beta };
}

/**
 * @brief Calculates the barycentric coordinates (alpha, beta, gamma) for a given point
 *        (x_pos, y_pos) with respect to a triangle. Also checks if the point is inside
 *        the triangle using the `insideTriangle` function and applies the result as a mask
 *        to ensure the coordinates are only valid for points inside the triangle.
 *
 * @param x_pos SIMD register containing x positions of points.
 * @param y_pos SIMD register containing y positions of points.
 * @param triangle The triangle whose barycentric coordinates are to be calculated.
 * @return A tuple of three simde__m256 values representing the barycentric coordinates
 *         (alpha, beta, gamma) for the point (x_pos, y_pos).
 *         The coordinates are zeroed out for points outside the triangle using a mask.
 */
inline std::tuple<simde__m256, simde__m256, simde__m256>
SoftRasterizer::RenderingPipeline::barycentric(
          const simde__m256& x_pos, const simde__m256& y_pos,
          const SoftRasterizer::Triangle& triangle) {

          const Eigen::Vector3f A = triangle.a();
          const Eigen::Vector3f B = triangle.b();
          const Eigen::Vector3f C = triangle.c();

          simde__m256 ax = simde_mm256_set1_ps(A.x()), ay = simde_mm256_set1_ps(A.y());
          simde__m256 bx = simde_mm256_set1_ps(B.x()), by = simde_mm256_set1_ps(B.y());
          simde__m256 cx = simde_mm256_set1_ps(C.x()), cy = simde_mm256_set1_ps(C.y());

          // Edges
          simde__m256 ABx = simde_mm256_sub_ps(bx, ax), ABy = simde_mm256_sub_ps(by, ay);
          simde__m256 ACx = simde_mm256_sub_ps(cx, ax), ACy = simde_mm256_sub_ps(cy, ay);
          simde__m256 PBx = simde_mm256_sub_ps(bx, x_pos), PBy = simde_mm256_sub_ps(by, y_pos);
          simde__m256 PCx = simde_mm256_sub_ps(cx, x_pos), PCy = simde_mm256_sub_ps(cy, y_pos);
          simde__m256 PAx = simde_mm256_sub_ps(ax, x_pos), PAy = simde_mm256_sub_ps(ay, y_pos);

          // Compute area of triangle ABC (cross product of AB ¡Á AC)
          simde__m256 areaABC = simde_mm256_fmsub_ps(ABx, ACy, simde_mm256_mul_ps(ACx, ABy));  // AB x AC
          simde__m256 inverse = simde_mm256_rcp_ps(areaABC);

          // Compute area of triangle PBC (cross product of PB ¡Á PC)
          simde__m256 areaPBC = simde_mm256_fmsub_ps(PBx, PCy, simde_mm256_mul_ps(PCx, PBy)); // PB ¡Á PC

          // Compute area of triangle PCA (cross product of PC ¡Á PA)
          simde__m256 areaPCA = simde_mm256_fmsub_ps(PCx, PAy, simde_mm256_mul_ps(PAx, PCy));  // PC ¡Á PA

          // Barycentric coordinates
          simde__m256 alpha = simde_mm256_mul_ps(areaPBC, inverse);
          simde__m256 beta = simde_mm256_mul_ps(areaPCA, inverse);

          return  std::tuple<simde__m256, simde__m256, simde__m256>(
                    alpha, 
                    beta, 
                    simde_mm256_sub_ps(simde_mm256_set1_ps(1.0f), simde_mm256_add_ps(alpha, beta))
          );
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

void  
SoftRasterizer::RenderingPipeline::rasterizeTriangle(std::shared_ptr<SoftRasterizer::Shader> shader,
          SoftRasterizer::Triangle& triangle)
{
          std::initializer_list<light_struct> lights = {
    {m_eye, Eigen::Vector3f{80, 80, 80}},
    {Eigen::Vector3f{0.9, 0.9, -0.9f}, Eigen::Vector3f{80, 80, 80}} };

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
          auto A_Point = triangle.m_vertex[0] = payloads[0].position;
          auto B_Point = triangle.m_vertex[1] = payloads[1].position;
          auto C_Point = triangle.m_vertex[2] = payloads[2].position;

          // Assuming payloads[0..2] are stored in simde__m256 types
          simde__m256 z0 = simde_mm256_set1_ps(A_Point.z());
          simde__m256 z1 = simde_mm256_set1_ps(B_Point.z());
          simde__m256 z2 = simde_mm256_set1_ps(C_Point.z());

          /*min and max point cood*/
          auto [min, max] = calculateBoundingBox(triangle);

          long long startX = (min.x() >= 0 ? min.x() : 0);
          long long startY = (min.y() >= 0 ? min.y() : 0);

          long long endX = (max.x() > m_width ? m_width : max.x());
          long long endY = (max.y() > m_height ? m_height : max.y());

          //auto prefetch_value = startY * m_width + endX;
          //PREFETCH(&m_zBuffer[prefetch_value]);
          //PREFETCH(m_channels[0].ptr<float>(0) + prefetch_value);
          //PREFETCH(m_channels[1].ptr<float>(0) + prefetch_value);
          //PREFETCH(m_channels[2].ptr<float>(0) + prefetch_value);

#pragma omp parallel for collapse(2)
          for (auto y = startY; y < endY; y++) {

                    long long x = startX;
                    PointSIMD point;
                    point.y = simde_mm256_set1_ps(static_cast<float>(y));

                    for (x = startX; x + UNROLLING_FACTOR < endX; x += UNROLLING_FACTOR) { // Loop unrolled by UNROLLING_FACTOR in x
                              auto start_pos = y * m_width + x;

                              PREFETCH(&m_zBuffer[start_pos + UNROLLING_FACTOR]);
                              PREFETCH(m_channels[0].ptr<float>(0) + start_pos + UNROLLING_FACTOR);
                              PREFETCH(m_channels[1].ptr<float>(0) + start_pos + UNROLLING_FACTOR);
                              PREFETCH(m_channels[2].ptr<float>(0) + start_pos + UNROLLING_FACTOR);

                              simde__m256 Original_Z = simde_mm256_loadu_ps(reinterpret_cast<float*>(&m_zBuffer[start_pos]));
                              simde__m256 Original_Blue = simde_mm256_loadu_ps(m_channels[2].ptr<float>(0) + start_pos);
                              simde__m256 Original_Green = simde_mm256_loadu_ps(m_channels[1].ptr<float>(0) + start_pos);
                              simde__m256 Original_Red = simde_mm256_loadu_ps(m_channels[0].ptr<float>(0) + start_pos);

                              // Initial x values for 8 points
                              point.x = simde_mm256_set_ps(x + 7.f, x + 6.f, x + 5.f, x + 4.f, x + 3.f, x + 2.f, x + 1.f, x + 0.f);
                              //point.x = simde_mm256_set_ps(x + 0.f, x + 1.f, x + 2.f, x + 3.f, x + 4.f, x + 5.f, x + 6.f, x + 7.f);

                             /*
                              * Calculates the barycentric coordinates(alpha, beta, gamma) for each point
                              * Checks if the point(x_pos, y_pos) is inside the triangle using the
                              * `insideTriangle` function.A mask is generated based on this check.
                              * (x_pos, y_pos) with respect to the triangle.The coordinates are calculated
                              * based on the edge vectors and point vectors.
                              * 
                              * The coordinates are then masked to zero out any invalid values(those outside the triangle).
                              */
                              auto [alpha, beta, gamma] = barycentric(point.x, point.y, triangle);

                              //if sum is zero, then it means the point is not inside the triangle!!!
                              simde__m256 alpha_valid = simde_mm256_and_ps(simde_mm256_cmp_ps(alpha, zero, SIMDE_CMP_GT_OQ), simde_mm256_cmp_ps(alpha, one, _CMP_LT_OQ));
                              simde__m256 beta_valid = simde_mm256_and_ps(simde_mm256_cmp_ps(beta, zero, SIMDE_CMP_GT_OQ), simde_mm256_cmp_ps(beta, one, _CMP_LT_OQ));
                              simde__m256 gamma_valid = simde_mm256_and_ps(simde_mm256_cmp_ps(gamma, zero, SIMDE_CMP_GT_OQ), simde_mm256_cmp_ps(gamma, one, _CMP_LT_OQ));
                              simde__m256 inside_mask = simde_mm256_and_ps(simde_mm256_and_ps(alpha_valid, beta_valid), gamma_valid);

                              /*when all points are not inside triangle! continue to next loop*/
                              if (simde_mm256_testz_ps(inside_mask, inside_mask)) {
                                        continue;
                              }

                              // Compute the z_interpolated using the blend operation
                              point.z =  simde_mm256_fmadd_ps(alpha, z0, simde_mm256_fmadd_ps(beta, z1, simde_mm256_mul_ps(gamma, z2)));

                             /*Comparing z-buffer value to determine update colour or not!*/
                              simde__m256 mask = simde_mm256_and_ps(simde_mm256_cmp_ps(point.z, Original_Z, SIMDE_CMP_LT_OQ), inside_mask);

                              if (simde_mm256_testz_ps(mask, mask)) {
                                        continue;
                              }

                              //calculate a set of normal for point.x, point.y, point.z
                              NormalSIMD normal = Tools::interpolateNormal(
                                        alpha, 
                                        beta, 
                                        gamma, 
                                        payloads[0].normal,
                                        payloads[1].normal, 
                                        payloads[2].normal
                              );

                              TexCoordSIMD texCoord = Tools::interpolateTexCoord(alpha, beta, gamma, 
                                        payloads[0].texCoords, 
                                        payloads[1].texCoords,
                                        payloads[2].texCoords);

                              /*If it's valid then set to 1.0f, or set to 0*/
                             ColorSIMD color(Original_Red , Original_Green, Original_Blue);
                             shader->applyFragmentShader(m_eye, lights, point, normal, texCoord, color);

                             writeZBuffer(start_pos, simde_mm256_blendv_ps(Original_Z, point.z, mask));
                             writePixel(start_pos, 
                                       simde_mm256_blendv_ps(Original_Red, color.r, mask), 
                                       simde_mm256_blendv_ps(Original_Green, color.g, mask),
                                       simde_mm256_blendv_ps(Original_Blue, color.b, mask)
                             );

                    }
                    for (; x < endX; ++x) {
                              // Check if the point (currentX, currentY) is inside the triangle
                              if (!insideTriangle(x + 0.5f, y + 0.5f, triangle)) {
                                        continue;
                              }

                              // Check if the point (currentX, currentY) is inside the triangle
                              auto [alpha, beta, gamma] = barycentric(x, y, triangle);

                              // For Z-buffer interpolation
                              //float w_reciprocal = 1.0f / (alpha + beta + gamma);
                              float z_interpolated = alpha * A_Point.z() + beta * B_Point.z() + gamma * C_Point.z();
                              //z_interpolated *= w_reciprocal;

                              // Test and write z-buffer (check depth before writing)
                              if (writeZBuffer(x, y, z_interpolated)) {

                                        /*interpolate normal*/
                                        auto interpolation_normal =
                                                  Tools::interpolateNormal(alpha, beta, gamma, payloads[0].normal,
                                                            payloads[1].normal, payloads[2].normal);

                                        /*interpolate uv*/
                                        auto interpolation_texCoord = Tools::interpolateTexCoord(
                                                  alpha, beta, gamma, payloads[0].texCoords, payloads[1].texCoords,
                                                  payloads[2].texCoords);

                                        fragment_shader_payload shading_on_xy(
                                                  Eigen::Vector3f(x, y, z_interpolated), interpolation_normal,
                                                  interpolation_texCoord);

                                        auto color = Tools::normalizedToRGB(
                                                  shader->applyFragmentShader(m_eye, lights, shading_on_xy));

                                        // Write pixel to the frame buffer
                                        writePixel(x, y, color);
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
      writePixel(x, y, color);
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

    writePixel(x, y, color);

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

      writePixel(x, y, color);
    }
  }
}
