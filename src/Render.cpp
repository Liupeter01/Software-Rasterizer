#include <Tools.hpp>
#include <type_traits>
#include <spdlog/spdlog.h>
#include <render/Render.hpp>
#include <opencv2/opencv.hpp>
#include <service/ThreadPool.hpp>

SoftRasterizer::RenderingPipeline::RenderingPipeline()
    : RenderingPipeline(800, 600) {}

SoftRasterizer::RenderingPipeline::RenderingPipeline(
          const std::size_t width, const std::size_t height)
          : m_width(width), m_height(height), m_channels(numbers) /*set to three*/
      , m_frameBuffer(m_height, m_width, CV_32FC3){

          /*init Thread Pool*/
  ThreadPool::get_instance();

  /*set channel ammount to three!*/
  m_channels.resize(numbers);

  /*resize std::vector of z-Buffer*/
  m_zBuffer.resize(width * height);

  /*init framebuffer*/
  clear(SoftRasterizer::Buffers::Color | SoftRasterizer::Buffers::Depth);
}

SoftRasterizer::RenderingPipeline::~RenderingPipeline() {
          /*Shutdown Thread Pool*/
          ThreadPool::get_instance()->terminate();
}

void SoftRasterizer::RenderingPipeline::clearFrameBuffer() {
  // #pragma omp parallel for
  for (long long i = 0; i < numbers; ++i) {
    m_channels[i] = cv::Mat::zeros(m_height, m_width, CV_32FC1);
  }

  m_frameBuffer = cv::Mat::zeros(m_height, m_width, CV_32FC3);
}

void SoftRasterizer::RenderingPipeline::clearZDepth() {
  std::for_each(m_zBuffer.begin(), m_zBuffer.end(), [](float &depth) {
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

void SoftRasterizer::RenderingPipeline::display(Primitive type) {
  /*draw pictures according to the specific type*/
  draw(type);

  cv::merge(m_channels, m_frameBuffer);
  m_frameBuffer.convertTo(m_frameBuffer, CV_8UC3, 1.0f);
  cv::imshow("SoftRasterizer", m_frameBuffer);
}

bool
SoftRasterizer::RenderingPipeline::addScene(std::shared_ptr<Scene> scene, std::optional<std::string> name){
          try {
                    if (scene == nullptr) {
                              return false;
                    }
                    if (name.has_value()) {
                              scene->m_sceneName = name.value();
                    }

                    /*Set Render's width and height info to scene*/
                    scene->setNDCMatrix(m_width, m_height);

                    if (m_scenes.find(scene->m_sceneName) != m_scenes.end()) {
                              spdlog::error("Add Scene Failed! Scene Already Exist");
                              return false;
                    }

                    m_scenes[scene->m_sceneName] = scene;
          }
          catch (const std::exception& e) {
                    spdlog::error("Add Scene Failed! Reason: {}", e.what());
                    return false;
          }
          return true;
}

inline void SoftRasterizer::RenderingPipeline::writePixel(
    const long long x, const long long y, const Eigen::Vector3f &color) {
  if (x >= 0 && x < m_width && y >= 0 && y < m_height) {
    auto pos = x + y * m_width;

    *(m_channels[0].ptr<float>(0) + pos) = color.x(); // R
    *(m_channels[1].ptr<float>(0) + pos) = color.y(); // G
    *(m_channels[2].ptr<float>(0) + pos) = color.z(); // B
  }
}

inline void SoftRasterizer::RenderingPipeline::writePixel(
    const long long x, const long long y, const Eigen::Vector3i &color) {
  writePixel(x, y, Eigen::Vector3f(color.x(), color.y(), color.z()));
}

inline void
SoftRasterizer::RenderingPipeline::writePixel(const long long start_pos,
                                              const ColorSIMD &color) {
  writePixel(start_pos, color.r, color.g, color.b);
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

inline void SoftRasterizer::RenderingPipeline::writeZBuffer(const long long start_pos,
          const float depth)
{
          m_zBuffer[start_pos] = depth;
}

inline const float 
SoftRasterizer::RenderingPipeline::readZBuffer(const long long x, const long long y){
          return m_zBuffer[x + y * m_width];
}

#if defined(__x86_64__) || defined(_WIN64)

template <typename _simd>
inline void SoftRasterizer::RenderingPipeline::writePixel(
    const long long start_pos, const _simd &r, const _simd &g, const _simd &b) {
  if constexpr (std::is_same_v<_simd, __m256>) {
    _mm256_storeu_ps(m_channels[0].ptr<float>(0) + start_pos, r); // R
    _mm256_storeu_ps(m_channels[1].ptr<float>(0) + start_pos, g); // G
    _mm256_storeu_ps(m_channels[2].ptr<float>(0) + start_pos, b); // B
  } else if constexpr (std::is_same_v<_simd, __m128>) {
    _mm_storeu_ps(m_channels[0].ptr<float>(0) + start_pos, r); // R
    _mm_storeu_ps(m_channels[1].ptr<float>(0) + start_pos, g); // G
    _mm_storeu_ps(m_channels[2].ptr<float>(0) + start_pos, b); // B
  }
}

template <typename _simd>
inline void
SoftRasterizer::RenderingPipeline::writeZBuffer(const long long start_pos,
                                                const _simd &depth) {
  if constexpr (std::is_same_v<_simd, __m256>) {
    _mm256_storeu_ps(reinterpret_cast<float *>(&m_zBuffer[start_pos]), depth);
  } else if constexpr (std::is_same_v<_simd, __m128>) {
    _mm_storeu_ps(reinterpret_cast<float *>(&m_zBuffer[start_pos]), depth);
  }
}

template <typename _simd>
inline std::tuple<_simd, _simd, _simd> 
SoftRasterizer::RenderingPipeline::readPixel(const long long start_pos){
          if constexpr (std::is_same_v<_simd, __m256>) {

                    return{
                              _mm256_loadu_ps(m_channels[0].ptr<float>(0) + start_pos), // R
                              _mm256_loadu_ps(m_channels[1].ptr<float>(0) + start_pos), // G
                              _mm256_loadu_ps(m_channels[2].ptr<float>(0) + start_pos) // B
                    };
          }
          else if constexpr (std::is_same_v<_simd, __m128>) {
                    return{
                              _mm_loadu_ps(m_channels[0].ptr<float>(0) + start_pos), // R
                              _mm_loadu_ps(m_channels[1].ptr<float>(0) + start_pos), // G
                              _mm_loadu_ps(m_channels[2].ptr<float>(0) + start_pos) // B
                    };

          }
          return {};
}

template <typename _simd>
inline _simd 
SoftRasterizer::RenderingPipeline::readZBuffer(const long long start_pos){

          if constexpr (std::is_same_v<_simd, __m256>) {
                    return  _mm256_loadu_ps(reinterpret_cast<float*>(&m_zBuffer[start_pos]));
          }
          else if constexpr (std::is_same_v<_simd, __m128>) {
                    return  _mm_loadu_ps(reinterpret_cast<float*>(&m_zBuffer[start_pos]));

          }
          return {};

}

#elif defined(__arm__) || defined(__aarch64__)
inline void SoftRasterizer::RenderingPipeline::writePixel(
    const long long start_pos, const simde__m256 &r, const simde__m256 &g,
    const simde__m256 &b) {
  simde_mm256_storeu_ps(m_channels[0].ptr<float>(0) + start_pos, r); // R
  simde_mm256_storeu_ps(m_channels[1].ptr<float>(0) + start_pos, g); // G
  simde_mm256_storeu_ps(m_channels[2].ptr<float>(0) + start_pos, b); // B
}

inline void
SoftRasterizer::RenderingPipeline::writeZBuffer(const long long start_pos,
                                                const simde__m256 &depth) {
  simde_mm256_storeu_ps(reinterpret_cast<float *>(&m_zBuffer[start_pos]),
                        depth);
}

simde__m256 SoftRasterizer::RenderingPipeline::insideTriangle(
    const simde__m256 &x, const simde__m256 &y,
    const SoftRasterizer::Triangle &triangle) {

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
  simde__m256 crossABP =
      simde_mm256_sub_ps(simde_mm256_mul_ps(simde_mm256_sub_ps(bx, ax),
                                            simde_mm256_sub_ps(py, ay)),
                         simde_mm256_mul_ps(simde_mm256_sub_ps(by, ay),
                                            simde_mm256_sub_ps(px, ax)));

  simde__m256 crossBCP =
      simde_mm256_sub_ps(simde_mm256_mul_ps(simde_mm256_sub_ps(cx, bx),
                                            simde_mm256_sub_ps(py, by)),
                         simde_mm256_mul_ps(simde_mm256_sub_ps(cy, by),
                                            simde_mm256_sub_ps(px, bx)));

  simde__m256 crossCAP =
      simde_mm256_sub_ps(simde_mm256_mul_ps(simde_mm256_sub_ps(ax, cx),
                                            simde_mm256_sub_ps(py, cy)),
                         simde_mm256_mul_ps(simde_mm256_sub_ps(ay, cy),
                                            simde_mm256_sub_ps(px, cx)));

  // Check if all cross products have the same sign (positive or negative)
  simde__m256 zero = simde_mm256_set1_ps(0.0f);
  simde__m256 signABP =
      simde_mm256_cmp_ps(crossABP, zero, SIMDE_CMP_GT_OQ); // > 0
  simde__m256 signBCP =
      simde_mm256_cmp_ps(crossBCP, zero, SIMDE_CMP_GT_OQ); // > 0
  simde__m256 signCAP =
      simde_mm256_cmp_ps(crossCAP, zero, SIMDE_CMP_GT_OQ); // > 0

  // Combine the signs: all positive or all negative
  simde__m256 allPositive =
      simde_mm256_and_ps(simde_mm256_and_ps(signABP, signBCP), signCAP);
  simde__m256 allNegative = simde_mm256_and_ps(
      simde_mm256_and_ps(simde_mm256_cmp_ps(crossABP, zero, SIMDE_CMP_LT_OQ),
                         simde_mm256_cmp_ps(crossBCP, zero, SIMDE_CMP_LT_OQ)),
      simde_mm256_cmp_ps(crossCAP, zero, SIMDE_CMP_LT_OQ));

  return simde_mm256_or_ps(allPositive, allNegative);
}

#else
#endif

void SoftRasterizer::RenderingPipeline::rasterizeWireframe(
    const SoftRasterizer::Triangle &triangle) {
  drawLine(triangle.b(), triangle.a(), triangle.m_color[0]);
  drawLine(triangle.b(), triangle.c(), triangle.m_color[1]);
  drawLine(triangle.a(), triangle.c(), triangle.m_color[2]);
}

inline bool SoftRasterizer::RenderingPipeline::insideTriangle(
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
  const float PAx = A.x() - x_pos, PAy = A.y() - y_pos;
  const float BCx = C.x() - B.x(), BCy = C.y() - B.y();
  const float PBx = B.x() - x_pos, PBy = B.y() - y_pos;
  const float PCx = C.x() - x_pos, PCy = C.y() - y_pos;

  // Compute areas directly using the 2D cross product (determinant)
  const float areaABC = ABx * ACy - ABy * ACx; // Area of triangle ABC
  const float areaPBC = PBx * PCy - PBy * PCx; // Area of triangle PBC
  const float areaPCA = PCx * PAy - PCy * PAx; // Area of triangle PCA

  // Calculate barycentric coordinates
  const float alpha = areaPBC / areaABC;
  const float beta = areaPCA / areaABC;

  return {alpha, beta, 1.0f - alpha - beta};
}

/**
 * @brief Calculates the barycentric coordinates (alpha, beta, gamma) for a
 * given point (x_pos, y_pos) with respect to a triangle. Also checks if the
 * point is inside the triangle using the `insideTriangle` function and applies
 * the result as a mask to ensure the coordinates are only valid for points
 * inside the triangle.
 *
 * @param x_pos SIMD register containing x positions of points.
 * @param y_pos SIMD register containing y positions of points.
 * @param triangle The triangle whose barycentric coordinates are to be
 * calculated.
 * @return A tuple of three simde__m256 values representing the barycentric
 * coordinates (alpha, beta, gamma) for the point (x_pos, y_pos). The
 * coordinates are zeroed out for points outside the triangle using a mask.
 */

#if defined(__x86_64__) || defined(_WIN64)
inline std::tuple<__m256, __m256, __m256>
SoftRasterizer::RenderingPipeline::barycentric(
    const __m256 &x_pos, const __m256 &y_pos,
    const SoftRasterizer::Triangle &triangle) {

  const Eigen::Vector3f A = triangle.a();
  const Eigen::Vector3f B = triangle.b();
  const Eigen::Vector3f C = triangle.c();

  __m256 ax = _mm256_set1_ps(A.x()), ay = _mm256_set1_ps(A.y());
  __m256 bx = _mm256_set1_ps(B.x()), by = _mm256_set1_ps(B.y());
  __m256 cx = _mm256_set1_ps(C.x()), cy = _mm256_set1_ps(C.y());
  const __m256 one = _mm256_set1_ps(1.0f);

  // Edges
  __m256 ABx = _mm256_sub_ps(bx, ax), ABy = _mm256_sub_ps(by, ay);
  __m256 ACx = _mm256_sub_ps(cx, ax), ACy = _mm256_sub_ps(cy, ay);
  __m256 PBx = _mm256_sub_ps(bx, x_pos), PBy = _mm256_sub_ps(by, y_pos);
  __m256 PCx = _mm256_sub_ps(cx, x_pos), PCy = _mm256_sub_ps(cy, y_pos);
  __m256 PAx = _mm256_sub_ps(ax, x_pos), PAy = _mm256_sub_ps(ay, y_pos);

  // Compute area of triangle ABC (cross product of AB ¡Á AC)
  __m256 inverse = _mm256_rcp_ps(
      _mm256_fmsub_ps(ABx, ACy, _mm256_mul_ps(ACx, ABy))); // AB x AC

  // Compute area of triangle PBC (cross product of PB ¡Á PC)
  __m256 areaPBC = _mm256_fmsub_ps(PBx, PCy, _mm256_mul_ps(PCx, PBy)); // PBxPC

  // Compute area of triangle PCA (cross product of PC ¡Á PA)
  __m256 areaPCA =
      _mm256_fmsub_ps(PCx, PAy, _mm256_mul_ps(PAx, PCy)); // PC x PA

  // Barycentric coordinates
  __m256 alpha = _mm256_mul_ps(areaPBC, inverse);
  __m256 beta = _mm256_mul_ps(areaPCA, inverse);
  __m256 gamma = _mm256_sub_ps(one, _mm256_add_ps(alpha, beta));

  return {alpha, beta, gamma};
}

#elif defined(__arm__) || defined(__aarch64__)

inline std::tuple<simde__m256, simde__m256, simde__m256>
SoftRasterizer::RenderingPipeline::barycentric(
    const simde__m256 &x_pos, const simde__m256 &y_pos,
    const SoftRasterizer::Triangle &triangle) {

  const Eigen::Vector3f A = triangle.a();
  const Eigen::Vector3f B = triangle.b();
  const Eigen::Vector3f C = triangle.c();

  simde__m256 ax = simde_mm256_set1_ps(A.x()), ay = simde_mm256_set1_ps(A.y());
  simde__m256 bx = simde_mm256_set1_ps(B.x()), by = simde_mm256_set1_ps(B.y());
  simde__m256 cx = simde_mm256_set1_ps(C.x()), cy = simde_mm256_set1_ps(C.y());

  // Edges
  simde__m256 ABx = simde_mm256_sub_ps(bx, ax),
              ABy = simde_mm256_sub_ps(by, ay);
  simde__m256 ACx = simde_mm256_sub_ps(cx, ax),
              ACy = simde_mm256_sub_ps(cy, ay);
  simde__m256 PBx = simde_mm256_sub_ps(bx, x_pos),
              PBy = simde_mm256_sub_ps(by, y_pos);
  simde__m256 PCx = simde_mm256_sub_ps(cx, x_pos),
              PCy = simde_mm256_sub_ps(cy, y_pos);
  simde__m256 PAx = simde_mm256_sub_ps(ax, x_pos),
              PAy = simde_mm256_sub_ps(ay, y_pos);

  // Compute area of triangle ABC (cross product of AB × AC)
  simde__m256 areaABC = simde_mm256_sub_ps(
      simde_mm256_mul_ps(ABx, ACy), simde_mm256_mul_ps(ACx, ABy)); // AB x AC

  simde__m256 inverse = simde_mm256_rcp_ps(areaABC);

  // Compute area of triangle PBC (cross product of PB × PC)
  simde__m256 areaPBC = simde_mm256_sub_ps(
      simde_mm256_mul_ps(PBx, PCy), simde_mm256_mul_ps(PCx, PBy)); // PB × PC

  // Compute area of triangle PCA (cross product of PC × PA)
  simde__m256 areaPCA = simde_mm256_sub_ps(
      simde_mm256_mul_ps(PCx, PAy), simde_mm256_mul_ps(PAx, PCy)); // PC × PA

  // Barycentric coordinates
  simde__m256 alpha = simde_mm256_mul_ps(areaPBC, inverse);
  simde__m256 beta = simde_mm256_mul_ps(areaPCA, inverse);

  return std::tuple<simde__m256, simde__m256, simde__m256>(
      alpha, beta,
      simde_mm256_sub_ps(simde_mm256_set1_ps(1.0f),
                         simde_mm256_add_ps(alpha, beta)));
}

#else
#endif

void SoftRasterizer::RenderingPipeline::draw(SoftRasterizer::Primitive type) {
  if ((type != SoftRasterizer::Primitive::LINES) &&
      (type != SoftRasterizer::Primitive::TRIANGLES)) {
    spdlog::error("Primitive Type is not supported!");
    throw std::runtime_error("Primitive Type is not supported!");
  }

  for (auto& [SceneName, SceneObj] : m_scenes) {

            /*Load All Triangle in one scene*/
            std::vector<SoftRasterizer::Scene::ObjTuple> stream = SceneObj->loadTriangleStream();
            std::vector<SoftRasterizer::light_struct> lights = SceneObj->loadLights();
            Eigen::Vector3f eye = SceneObj->loadEyeVec();

            /*Traversal All The Triangle*/
            for (auto& [shader, CurrentObj] : stream) {

                      // Check how many full groups of 8 we can process with AVX2
                      std::size_t totalTriangles = CurrentObj.size();

                      for (auto& triangle : CurrentObj) {

                                auto box_startX = triangle.box.startX, box_endX = triangle.box.endX;

                                // Split into AVX2-compatible chunks (use the largest multiple of 8 for AVX2 if possible)
                                auto avx2_chunks = (box_endX - box_startX) >> 3;
                                auto avx2_end = box_startX + (avx2_chunks << 3);              // Largest multiple of 8 for AVX2

                                /*use for fps sync*/
                                std::vector<std::future<void>> futures;

                                for (auto y = triangle.box.startY; y < triangle.box.endY; ++y) {

                                          // Submit AVX2 task for the first chunk of triangles (up to the AVX2 boundary)

                                          if (avx2_chunks > 0) {

                                                    futures.emplace_back(ThreadPool::get_instance()->commit(
                                                              [&shader, this, type, avx2_chunks, avx2_end, y, &triangle, &lights, &eye]()->void {
                                                                        if (type == SoftRasterizer::Primitive::TRIANGLES) {
                                                                                  auto start_pos = triangle.box.startX + y * m_width;

#if defined(__x86_64__) || defined(_WIN64)
                                                                                  auto packed_size = avx2_chunks * sizeof(__m256);

#elif defined(__arm__) || defined(__aarch64__)
                                                                                  auto packed_size = avx2_chunks * sizeof(simde__m256);

#else
#endif

                                                                                  std::vector<float> tempz(packed_size, std::numeric_limits<float>::infinity());
                                                                                  std::vector<float> tempr(packed_size, 0.f);
                                                                                  std::vector<float> tempg(packed_size, 0.f);
                                                                                  std::vector<float> tempb(packed_size, 0.f);

                                                                                  std::memcpy(tempz.data(), &m_zBuffer[start_pos],packed_size);
                                                                                  std::memcpy(tempr.data(), m_channels[0].ptr<float>(0) + start_pos,  packed_size);
                                                                                  std::memcpy(tempg.data(), m_channels[1].ptr<float>(0) + start_pos,  packed_size);
                                                                                  std::memcpy(tempb.data(), m_channels[2].ptr<float>(0) + start_pos,  packed_size);

                                                                                  rasterizeBatchAVX2(triangle.box.startX, avx2_end, y,
                                                                                            tempz.data(), tempr.data(), tempg.data(), tempb.data(),
                                                                                            lights, shader, triangle, eye);

                                                                                  std::memcpy(&m_zBuffer[start_pos], tempz.data(), packed_size);
                                                                                  std::memcpy(m_channels[0].ptr<float>(0) + start_pos, tempr.data(), packed_size);
                                                                                  std::memcpy(m_channels[1].ptr<float>(0) + start_pos, tempg.data(),  packed_size);
                                                                                  std::memcpy(m_channels[2].ptr<float>(0) + start_pos, tempb.data(),  packed_size);

                                                                        }
                                                                        return;
                                                              }));
                                          }

                                          futures.emplace_back(ThreadPool::get_instance()->commit(
                                                    [&shader, this, type, box_startX, avx2_end, y, &triangle, &lights, &eye]()->void {
                                                              if (type == SoftRasterizer::Primitive::TRIANGLES) {
                                                                        auto start_pos = avx2_end + y * m_width;
                                                                        auto scalar_size = triangle.box.endX - avx2_end;

                                                                        std::vector<float> tempz(ROUND_UP_TO_MULTIPLE_OF_8(scalar_size), std::numeric_limits<float>::infinity());
                                                                        std::memcpy(tempz.data(), &m_zBuffer[start_pos], sizeof(float) * scalar_size);
                                                                        rasterizeBatchScalar(avx2_end, triangle.box.endX, y, tempz.data(), lights, shader, triangle, eye);
                                                                        std::memcpy(&m_zBuffer[start_pos], tempz.data(), sizeof(float) * scalar_size);
                                                              }
                                                              return;
                                                    })
                                          );

                                }

                                // Block until all tasks are complete
                                for (auto& future : futures) {
                                          future.get();
                                }
                      }
            }
  }
}

inline 
void
SoftRasterizer::RenderingPipeline::rasterizeBatchAVX2(
          const int startx, const int endx, const int y,
          float* z, float* r, float* g, float* b,
          const std::vector<SoftRasterizer::light_struct>& lists,
          std::shared_ptr<SoftRasterizer::Shader> shader,
          const SoftRasterizer::Triangle& packed,
          const Eigen::Vector3f& eye){

          PointSIMD point;
#if defined(__x86_64__) || defined(_WIN64)
          point.y = _mm256_set1_ps(static_cast<float>(y));
#elif defined(__arm__) || defined(__aarch64__)
          point.y = simde_mm256_set1_ps(static_cast<float>(y));
#else
#endif

          auto A_Point = packed.m_vertex[0];
          auto B_Point = packed.m_vertex[1];
          auto C_Point = packed.m_vertex[2];

#if defined(__x86_64__) || defined(_WIN64)
          auto z0 = _mm256_set1_ps(A_Point.z());
          auto z1 = _mm256_set1_ps(B_Point.z());
          auto z2 = _mm256_set1_ps(C_Point.z());
#elif defined(__arm__) || defined(__aarch64__)
          auto z0 = simde_mm256_set1_ps(A_Point.z());
          auto z1 = simde_mm256_set1_ps(B_Point.z());
          auto z2 = simde_mm256_set1_ps(C_Point.z());
#else
#endif

          for (auto x = startx; x < endx; x += AVX2) {
#if defined(__x86_64__) || defined(_WIN64)
                    __m256 Original_Z =
                              _mm256_loadu_ps(&z[x - startx]);
                    __m256 Original_Blue =
                              _mm256_loadu_ps(&b[x - startx]);
                    __m256 Original_Green =
                              _mm256_loadu_ps(&g[x - startx]);
                    __m256 Original_Red =
                              _mm256_loadu_ps(&r[x - startx]);

                    point.x = _mm256_set_ps(x + 7.f, x + 6.f, x + 5.f, x + 4.f, x + 3.f,x + 2.f, x + 1.f, x + 0.f);

#elif defined(__arm__) || defined(__aarch64__)
                    simde__m256 Original_Z =
                              simde_mm256_loadu_ps(&z[x - startx]);
                    simde__m256 Original_Blue =
                              simde_mm256_loadu_ps(&b[x - startx]);
                    simde__m256 Original_Green =
                              simde_mm256_loadu_ps(&g[x - startx]);
                    simde__m256 Original_Red =
                              simde_mm256_loadu_ps(&r[x - startx]);

                    point.x = simde_mm256_set_ps(x + 7.f, x + 6.f, x + 5.f, x + 4.f, x + 3.f, x + 2.f, x + 1.f, x + 0.f);

#else
#endif

                    /*
                 * Calculates the barycentric coordinates(alpha, beta, gamma) for each
                 * point Checks if the point(x_pos, y_pos) is inside the triangle using
                 * the `insideTriangle` function.A mask is generated based on this check.
                 * (x_pos, y_pos) with respect to the triangle.The coordinates are
                 * calculated based on the edge vectors and point vectors.
                 *
                 * The coordinates are then masked to zero out any invalid values(those
                 * outside the triangle).
                 */
                    auto [alpha, beta, gamma] = barycentric(point.x, point.y, packed);


#if defined(__x86_64__) || defined(_WIN64)
                    // if sum is zero, then it means the point is not inside the triangle!!!
                    __m256 alpha_valid = _mm256_and_ps(_mm256_cmp_ps(alpha, zero, _CMP_GT_OQ),
                              _mm256_cmp_ps(alpha, one, _CMP_LT_OQ));
                    __m256 beta_valid = _mm256_and_ps(_mm256_cmp_ps(beta, zero, _CMP_GT_OQ),
                              _mm256_cmp_ps(beta, one, _CMP_LT_OQ));
                    __m256 gamma_valid = _mm256_and_ps(_mm256_cmp_ps(gamma, zero, _CMP_GT_OQ),
                              _mm256_cmp_ps(gamma, one, _CMP_LT_OQ));
                    __m256 inside_mask =
                              _mm256_and_ps(_mm256_and_ps(alpha_valid, beta_valid), gamma_valid);

                    /*when all points are not inside triangle! continue to next loop*/
                    if (_mm256_testz_ps(inside_mask, inside_mask)) {
                              continue;
                    }

                    // Compute the z_interpolated using the blend operation
                    point.z = _mm256_fmadd_ps(
                              alpha, z0, _mm256_fmadd_ps(beta, z1, _mm256_mul_ps(gamma, z2)));

                    /*Comparing z-buffer value to determine update colour or not!*/
                    __m256 mask = _mm256_and_ps(
                              _mm256_cmp_ps(point.z, Original_Z, _CMP_LT_OQ), inside_mask);

                    if (_mm256_testz_ps(mask, mask)) {
                              continue;
                    }

#elif defined(__arm__) || defined(__aarch64__)
                    // if sum is zero, then it means the point is not inside the triangle!!!
                    simde__m256 alpha_valid =
                              simde_mm256_and_ps(simde_mm256_cmp_ps(alpha, zero, SIMDE_CMP_GT_OQ),
                                        simde_mm256_cmp_ps(alpha, one, SIMDE_CMP_LT_OQ));
                    simde__m256 beta_valid =
                              simde_mm256_and_ps(simde_mm256_cmp_ps(beta, zero, SIMDE_CMP_GT_OQ),
                                        simde_mm256_cmp_ps(beta, one, SIMDE_CMP_LT_OQ));
                    simde__m256 gamma_valid =
                              simde_mm256_and_ps(simde_mm256_cmp_ps(gamma, zero, SIMDE_CMP_GT_OQ),
                                        simde_mm256_cmp_ps(gamma, one, SIMDE_CMP_LT_OQ));
                    simde__m256 inside_mask = simde_mm256_and_ps(
                              simde_mm256_and_ps(alpha_valid, beta_valid), gamma_valid);

                    /*when all points are not inside triangle! continue to next loop*/
                    if (simde_mm256_testz_ps(inside_mask, inside_mask)) {
                              continue;
                    }

                    // Compute the z_interpolated using the blend operation
                    // point.z = simde_mm256_fmadd_ps(
                    //     alpha, z0,
                    //     simde_mm256_fmadd_ps(beta, z1, simde_mm256_mul_ps(gamma, z2)));
                    auto alpha_z = simde_mm256_mul_ps(alpha, z0);
                    auto beta_z = simde_mm256_mul_ps(beta, z1);
                    auto gamma_z = simde_mm256_mul_ps(gamma, z2);

                    alpha_z = simde_mm256_add_ps(alpha_z, beta_z);
                    point.z = simde_mm256_add_ps(alpha_z, gamma_z);

                    /*Comparing z-buffer value to determine update colour or not!*/
                    simde__m256 mask = simde_mm256_and_ps(
                              simde_mm256_cmp_ps(point.z, Original_Z, SIMDE_CMP_LT_OQ),
                              inside_mask);

                    if (simde_mm256_testz_ps(mask, mask)) {
                              continue;
                    }
#else
#endif

                    // calculate a set of normal for point.x, point.y, point.z
                    NormalSIMD normal =
                              Tools::interpolateNormal(alpha, beta, gamma, packed.m_normal[0],
                                        packed.m_normal[1], packed.m_normal[2]);

                    TexCoordSIMD texCoord = Tools::interpolateTexCoord(
                              alpha, beta, gamma, packed.m_texCoords[0],
                              packed.m_texCoords[1], packed.m_texCoords[2]);

                    ColorSIMD color;
                    shader->applyFragmentShader(eye, lists, point, normal, texCoord, color);




#if defined(__x86_64__) || defined(_WIN64)
                    Original_Z = _mm256_blendv_ps(Original_Z, point.z, mask);
                    Original_Red = _mm256_blendv_ps(Original_Red, color.r, mask);
                    Original_Green = _mm256_blendv_ps(Original_Green, color.g, mask);
                    Original_Blue = _mm256_blendv_ps(Original_Blue, color.b, mask);

                    _mm256_storeu_ps(&z[x - startx], Original_Z);
                    _mm256_storeu_ps(&r[x - startx], Original_Red);
                    _mm256_storeu_ps(&g[x - startx], Original_Green);
                    _mm256_storeu_ps(&b[x - startx], Original_Blue);

#elif defined(__arm__) || defined(__aarch64__)
                    Original_Z = simde_mm256_blendv_ps(Original_Z, point.z, mask);
                    Original_Red = simde_mm256_blendv_ps(Original_Red, color.r, mask);
                    Original_Green = simde_mm256_blendv_ps(Original_Green, color.g, mask);
                    Original_Blue = simde_mm256_blendv_ps(Original_Blue, color.b, mask);

                    simde_mm256_storeu_ps(&z[x - startx], Original_Z);
                    simde_mm256_storeu_ps(&r[x - startx], Original_Red);
                    simde_mm256_storeu_ps(&g[x - startx], Original_Green);
                    simde_mm256_storeu_ps(&b[x - startx], Original_Blue);

#else
#endif
          }
}

inline 
void
SoftRasterizer::RenderingPipeline::rasterizeBatchScalar(
          const int startx, const int endx, const int y, float* z, 
          const std::vector<SoftRasterizer::light_struct>& lists,
          std::shared_ptr<SoftRasterizer::Shader> shader,
           const SoftRasterizer::Triangle& scalar,
          const Eigen::Vector3f& eye) {


          /*update triangle position!*/
          auto A_Point = scalar.m_vertex[0];
          auto B_Point = scalar.m_vertex[1];
          auto C_Point = scalar.m_vertex[2];

          for (auto x = startx; x < endx; x += SCALAR) {
                    // Check if the point (currentX, currentY) is inside the triangle
                    if (!insideTriangle(x + 0.5f, y + 0.5f, scalar)) {
                              continue;
                    }

                    // Check if the point (currentX, currentY) is inside the triangle
                    auto [alpha, beta, gamma] = barycentric(x, y, scalar);

                    // For Z-buffer interpolation
                    // float w_reciprocal = 1.0f / (alpha + beta + gamma);
                    auto new_z =
                              alpha * A_Point.z() + beta * B_Point.z() + gamma * C_Point.z();

                    if (new_z > z[x - startx]) {
                              continue;
                    }

                    z[x - startx] = new_z;

                    /*interpolate normal*/
                    auto interpolation_normal =
                              Tools::interpolateNormal(alpha, beta, gamma, scalar.m_normal[0],
                                        scalar.m_normal[1], scalar.m_normal[2]);

                    /*interpolate uv*/
                    auto interpolation_texCoord = Tools::interpolateTexCoord(
                              alpha, beta, gamma, scalar.m_texCoords[0],
                              scalar.m_texCoords[1], scalar.m_texCoords[2]);

                    writePixel(x, y, Tools::normalizedToRGB(
                              shader->applyFragmentShader(eye, lists, fragment_shader_payload(
                                        Eigen::Vector3f(x, y, new_z), interpolation_normal,
                                        interpolation_texCoord))));
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
