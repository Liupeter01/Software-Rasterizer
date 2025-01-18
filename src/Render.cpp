#include <Tools.hpp>
#include <opencv2/opencv.hpp>
#include <render/Render.hpp>
#include <service/ThreadPool.hpp>
#include <spdlog/spdlog.h>
#include <type_traits>
#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>

SoftRasterizer::RenderingPipeline::RenderingPipeline()
    : RenderingPipeline(800, 600) {}

SoftRasterizer::RenderingPipeline::RenderingPipeline(const std::size_t width,
                                                     const std::size_t height)
    : m_width(width), m_height(height), m_channels(numbers) /*set to three*/
      ,
      m_frameBuffer(m_height, m_width, CV_32FC3) {

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

bool SoftRasterizer::RenderingPipeline::addScene(
    std::shared_ptr<Scene> scene, std::optional<std::string> name) {
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
  } catch (const std::exception &e) {
    spdlog::error("Add Scene Failed! Reason: {}", e.what());
    return false;
  }
  return true;
}

inline void SoftRasterizer::RenderingPipeline::writePixel(
    const long long x, const long long y, const glm::vec3 &color) {
  if (x >= 0 && x < m_width && y >= 0 && y < m_height) {
    auto pos = x + y * m_width;

    *(m_channels[0].ptr<float>(0) + pos) = color.x; // R
    *(m_channels[1].ptr<float>(0) + pos) = color.y; // G
    *(m_channels[2].ptr<float>(0) + pos) = color.z; // B
  }
}

inline void SoftRasterizer::RenderingPipeline::writePixel(
    const long long x, const long long y, const glm::uvec3 &color) {
  writePixel(x, y, glm::vec3(color.x, color.y, color.z));
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

inline void
SoftRasterizer::RenderingPipeline::writeZBuffer(const long long start_pos,
                                                const float depth) {
  m_zBuffer[start_pos] = depth;
}

inline const float
SoftRasterizer::RenderingPipeline::readZBuffer(const long long x,
                                               const long long y) {
  return m_zBuffer[x + y * m_width];
}

template <typename _simd>
inline void SoftRasterizer::RenderingPipeline::writePixel(
    const long long start_pos, const _simd &r, const _simd &g, const _simd &b) {
#if defined(__x86_64__) || defined(_WIN64)
  if constexpr (std::is_same_v<_simd, __m256>) {
    _mm256_storeu_ps(m_channels[0].ptr<float>(0) + start_pos, r); // R
    _mm256_storeu_ps(m_channels[1].ptr<float>(0) + start_pos, g); // G
    _mm256_storeu_ps(m_channels[2].ptr<float>(0) + start_pos, b); // B

#elif defined(__arm__) || defined(__aarch64__)
  if constexpr (std::is_same_v<_simd, simde__m256>) {
    simde_mm256_storeu_ps(m_channels[0].ptr<float>(0) + start_pos, r); // R
    simde_mm256_storeu_ps(m_channels[1].ptr<float>(0) + start_pos, g); // G
    simde_mm256_storeu_ps(m_channels[2].ptr<float>(0) + start_pos, b); // B

#else
#endif
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
#if defined(__x86_64__) || defined(_WIN64)
  if constexpr (std::is_same_v<_simd, __m256>) {
    _mm256_storeu_ps(reinterpret_cast<float *>(&m_zBuffer[start_pos]), depth);

#elif defined(__arm__) || defined(__aarch64__)
  if constexpr (std::is_same_v<_simd, simde__m256>) {
    simde_mm256_storeu_ps(reinterpret_cast<float *>(&m_zBuffer[start_pos]),
                          depth);
#else
#endif
  } else if constexpr (std::is_same_v<_simd, __m128>) {
    _mm_storeu_ps(reinterpret_cast<float *>(&m_zBuffer[start_pos]), depth);
  }
}

template <typename _simd>
inline std::tuple<_simd, _simd, _simd>
SoftRasterizer::RenderingPipeline::readPixel(const long long start_pos) {

#if defined(__x86_64__) || defined(_WIN64)
  if constexpr (std::is_same_v<_simd, __m256>) {
    return {
        _mm256_loadu_ps(m_channels[0].ptr<float>(0) + start_pos), // R
        _mm256_loadu_ps(m_channels[1].ptr<float>(0) + start_pos), // G
        _mm256_loadu_ps(m_channels[2].ptr<float>(0) + start_pos)  // B
    };

#elif defined(__arm__) || defined(__aarch64__)
  if constexpr (std::is_same_v<_simd, simde__m256>) {
    return {
        simde_mm256_loadu_ps(m_channels[0].ptr<float>(0) + start_pos), // R
        simde_mm256_loadu_ps(m_channels[1].ptr<float>(0) + start_pos), // G
        simde_mm256_loadu_ps(m_channels[2].ptr<float>(0) + start_pos)  // B
    };
#else
#endif
  } else if constexpr (std::is_same_v<_simd, __m128>) {
    return {
        _mm_loadu_ps(m_channels[0].ptr<float>(0) + start_pos), // R
        _mm_loadu_ps(m_channels[1].ptr<float>(0) + start_pos), // G
        _mm_loadu_ps(m_channels[2].ptr<float>(0) + start_pos)  // B
    };
  }
  return {};
}

template <typename _simd>
inline _simd
SoftRasterizer::RenderingPipeline::readZBuffer(const long long start_pos) {
#if defined(__x86_64__) || defined(_WIN64)
  if constexpr (std::is_same_v<_simd, __m256>) {
    return _mm256_loadu_ps(reinterpret_cast<float *>(&m_zBuffer[start_pos]));

#elif defined(__arm__) || defined(__aarch64__)
  if constexpr (std::is_same_v<_simd, simde__m256>) {
    return simde_mm256_loadu_ps(
        reinterpret_cast<float *>(&m_zBuffer[start_pos]));
#else
#endif
  } else if constexpr (std::is_same_v<_simd, __m128>) {
    return _mm_loadu_ps(reinterpret_cast<float *>(&m_zBuffer[start_pos]));
  }
  return {};
}

void SoftRasterizer::RenderingPipeline::rasterizeWireframe(
    const SoftRasterizer::Triangle &triangle) {
  drawLine(triangle.b(), triangle.a(), triangle.m_color[0]);
  drawLine(triangle.b(), triangle.c(), triangle.m_color[1]);
  drawLine(triangle.a(), triangle.c(), triangle.m_color[2]);
}

inline bool SoftRasterizer::RenderingPipeline::insideTriangle(
    const std::size_t x_pos, const std::size_t y_pos,
    const SoftRasterizer::Triangle &triangle) {
  const glm::vec3 P = {static_cast<float>(x_pos), static_cast<float>(y_pos),
                       1.0f};

  glm::vec3 A = triangle.a();
  glm::vec3 B = triangle.b();
  glm::vec3 C = triangle.c();

  A.z = B.z = C.z = 1.0f;

  // Vectors representing the edges of the triangle
  glm::vec3 AB = B - A;
  glm::vec3 BC = C - B;
  glm::vec3 CA = A - C;

  // Vectors from the point to each vertex
  glm::vec3 AP = P - A;
  glm::vec3 BP = P - B;
  glm::vec3 CP = P - C;

  // Cross product results (we only need the z-components)
  const float crossABP_z = AB.x * AP.y - AB.y * AP.x;
  const float crossBCP_z = BC.x * BP.y - BC.y * BP.x;
  const float crossCAP_z = CA.x * CP.y - CA.y * CP.x;

  // Check if all cross products have the same sign
  return (crossABP_z > 0 && crossBCP_z > 0 && crossCAP_z > 0) ||
         (crossABP_z < 0 && crossBCP_z < 0 && crossCAP_z < 0);
}

inline std::tuple<float, float, float>
SoftRasterizer::RenderingPipeline::barycentric(
    const std::size_t x_pos, const std::size_t y_pos,
    const SoftRasterizer::Triangle &triangle) {

  const glm::vec3 A = triangle.a();
  const glm::vec3 B = triangle.b();
  const glm::vec3 C = triangle.c();

  // Compute edges
  const float ABx = B.x - A.x, ABy = B.y - A.y;
  const float ACx = C.x - A.x, ACy = C.y - A.y;
  const float PAx = A.x - x_pos, PAy = A.y - y_pos;
  const float BCx = C.x - B.x, BCy = C.y - B.y;
  const float PBx = B.x - x_pos, PBy = B.y - y_pos;
  const float PCx = C.x - x_pos, PCy = C.y - y_pos;

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

  const glm::vec3 A = triangle.a();
  const glm::vec3 B = triangle.b();
  const glm::vec3 C = triangle.c();

  __m256 ax = _mm256_set1_ps(A.x), ay = _mm256_set1_ps(A.y);
  __m256 bx = _mm256_set1_ps(B.x), by = _mm256_set1_ps(B.y);
  __m256 cx = _mm256_set1_ps(C.x), cy = _mm256_set1_ps(C.y);
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

  const glm::vec3 A = triangle.a();
  const glm::vec3 B = triangle.b();
  const glm::vec3 C = triangle.c();

  simde__m256 ax = simde_mm256_set1_ps(A.x), ay = simde_mm256_set1_ps(A.y);
  simde__m256 bx = simde_mm256_set1_ps(B.x), by = simde_mm256_set1_ps(B.y);
  simde__m256 cx = simde_mm256_set1_ps(C.x), cy = simde_mm256_set1_ps(C.y);

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

  static oneapi::tbb::affinity_partitioner ap;
  oneapi::tbb::enumerable_thread_specific<std::size_t> ets;

  for (auto &[SceneName, SceneObj] : m_scenes) {

    /*Load All Triangle in one scene*/
    std::vector<SoftRasterizer::Scene::ObjTuple> stream =
        SceneObj->loadTriangleStream();
    std::vector<SoftRasterizer::light_struct> lights = SceneObj->loadLights();
    const glm::vec3 eye = SceneObj->loadEyeVec();

    /*Traversal All The Triangle*/
    for (auto &[shader, CurrentObj] : stream) {

              std::size_t triangles_number = CurrentObj.size();

              //Find Non Overlapping Triangles
              oneapi::tbb::parallel_for(static_cast<std::size_t>(0), triangles_number, [&](std::size_t index) {
                        // Set a thread specific value
                        //ets.local() = index;

                        oneapi::tbb::this_task_arena::isolate([&]() {

                                 // ets.local() = index;
                                  auto& triangle = CurrentObj[index];

                                  auto box_startX = triangle.box.startX, box_endX = triangle.box.endX;

                                  // Split into AVX2-compatible chunks (use the largest multiple of 8 for AVX2 if possible)
                                  auto avx2_chunks = (box_endX - box_startX + 1) >> 3;
                                  auto avx2_size = (avx2_chunks << 3);
                                  auto avx2_end = box_startX + avx2_size;//  Largest multiple of 8 for AVX2

                                  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(triangle.box.startY, triangle.box.endY + 1),
                                            [&](const oneapi::tbb::blocked_range<std::size_t>& range) {

                                                      for (auto x = range.begin(); x != range.end(); ++x) {
                                                                rasterizeBatchAVX2(triangle.box.startX, avx2_end, x, lights, shader,
                                                                          triangle, eye);

                                                                rasterizeBatchScalar(avx2_end, triangle.box.endX, x, lights, shader,
                                                                          triangle, eye);
                                                      }
                                            }, ap);
                         });
              });
              //ets.clear();
    }
  }
}

inline void SoftRasterizer::RenderingPipeline::rasterizeBatchAVX2(
    const int startx, const int endx, const int y,
    const std::vector<SoftRasterizer::light_struct> &lists,
    std::shared_ptr<SoftRasterizer::Shader> shader,
    const SoftRasterizer::Triangle &packed, const glm::vec3 &eye) {

  if (startx + AVX2 > endx) {
    return;
  }

#if defined(__x86_64__) || defined(_WIN64)
  auto z0 = _mm256_set1_ps(packed.m_vertex[0].z);
  auto z1 = _mm256_set1_ps(packed.m_vertex[1].z);
  auto z2 = _mm256_set1_ps(packed.m_vertex[2].z);
#elif defined(__arm__) || defined(__aarch64__)
  auto z0 = simde_mm256_set1_ps(packed.m_vertex[0].z);
  auto z1 = simde_mm256_set1_ps(packed.m_vertex[1].z);
  auto z2 = simde_mm256_set1_ps(packed.m_vertex[2].z);
#else
#endif

  for (auto x = startx; x < endx; x += AVX2) {
    processFragByAVX2(x, y, z0, z1, z2, lists, shader, packed, eye);
  }
}

template <typename _simd>
inline void SoftRasterizer::RenderingPipeline::processFragByAVX2(
    const int x, const int y, const _simd &z0, const _simd &z1, const _simd &z2,
    const std::vector<SoftRasterizer::light_struct> &lists,
    std::shared_ptr<SoftRasterizer::Shader> shader,
    const SoftRasterizer::Triangle &packed, const glm::vec3 &eye) {
  const auto op_pos = x + y * m_width;
  PointSIMD point;

#if defined(__x86_64__) || defined(_WIN64)
  point.y = _mm256_set1_ps(static_cast<float>(y));
  point.x = _mm256_set_ps(x + 7.f, x + 6.f, x + 5.f, x + 4.f, x + 3.f, x + 2.f,
                          x + 1.f, x + 0.f);

#elif defined(__arm__) || defined(__aarch64__)
  point.y = simde_mm256_set1_ps(static_cast<float>(y));
  point.x = simde_mm256_set_ps(x + 7.f, x + 6.f, x + 5.f, x + 4.f, x + 3.f,
                               x + 2.f, x + 1.f, x + 0.f);
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
    return;
  }

  // Compute the z_interpolated using the blend operation
  point.z = _mm256_fmadd_ps(
      alpha, z0, _mm256_fmadd_ps(beta, z1, _mm256_mul_ps(gamma, z2)));

  // Start Reading From Now
  __m256 Original_Z = readZBuffer<__m256>(op_pos);
  auto [Original_Red, Original_Green, Original_Blue] =
      readPixel<__m256>(op_pos);

  /*Comparing z-buffer value to determine update colour or not!*/
  __m256 mask = _mm256_and_ps(_mm256_cmp_ps(point.z, Original_Z, _CMP_LT_OQ),
                              inside_mask);

  if (_mm256_testz_ps(mask, mask)) {
    return;
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
    return;
  }

  // Compute the z_interpolated using the blend operation
  point.z = simde_mm256_fmadd_ps(
      alpha, z0, simde_mm256_fmadd_ps(beta, z1, simde_mm256_mul_ps(gamma, z2)));

  // Start Reading From Now
  simde__m256 Original_Z = readZBuffer<simde__m256>(op_pos);
  auto [Original_Red, Original_Green, Original_Blue] =
      readPixel<simde__m256>(op_pos);

  /*Comparing z-buffer value to determine update colour or not!*/
  simde__m256 mask = simde_mm256_and_ps(
      simde_mm256_cmp_ps(point.z, Original_Z, SIMDE_CMP_LT_OQ), inside_mask);

  if (simde_mm256_testz_ps(mask, mask)) {
    return;
  }
#else
#endif

  // calculate a set of normal for point.x, point.y, point.z
  NormalSIMD normal =
      Tools::interpolateNormal(alpha, beta, gamma, packed.m_normal[0],
                               packed.m_normal[1], packed.m_normal[2]);

  TexCoordSIMD texCoord =
      Tools::interpolateTexCoord(alpha, beta, gamma, packed.m_texCoords[0],
                                 packed.m_texCoords[1], packed.m_texCoords[2]);

  ColorSIMD color;
  shader->applyFragmentShader(eye, lists, point, normal, texCoord, color);

#if defined(__x86_64__) || defined(_WIN64)
  Original_Z = _mm256_blendv_ps(Original_Z, point.z, mask);
  Original_Red = _mm256_blendv_ps(Original_Red, color.r, mask);
  Original_Green = _mm256_blendv_ps(Original_Green, color.g, mask);
  Original_Blue = _mm256_blendv_ps(Original_Blue, color.b, mask);

#elif defined(__arm__) || defined(__aarch64__)
  Original_Z = simde_mm256_blendv_ps(Original_Z, point.z, mask);
  Original_Red = simde_mm256_blendv_ps(Original_Red, color.r, mask);
  Original_Green = simde_mm256_blendv_ps(Original_Green, color.g, mask);
  Original_Blue = simde_mm256_blendv_ps(Original_Blue, color.b, mask);

#else
#endif
  writeZBuffer(op_pos, Original_Z);
  writePixel(op_pos, Original_Red, Original_Green, Original_Blue);
}

inline void SoftRasterizer::RenderingPipeline::rasterizeBatchScalar(
    const int startx, const int endx, const int y,
    const std::vector<SoftRasterizer::light_struct> &lists,
    std::shared_ptr<SoftRasterizer::Shader> shader,
    const SoftRasterizer::Triangle &scalar, const glm::vec3 &eye) {

  const auto read_length = endx - startx + 1;
  const auto read_pos = y * m_width + startx;

  auto z0 = scalar.m_vertex[0].z;
  auto z1 = scalar.m_vertex[1].z;
  auto z2 = scalar.m_vertex[2].z;

  alignas(64) float z[8]{std::numeric_limits<float>::infinity()};
  alignas(64) float r[8]{0.f};
  alignas(64) float g[8]{0.f};
  alignas(64) float b[8]{0.f};

  std::vector<std::future<void>> m_io;

  std::memcpy(z, &m_zBuffer[read_pos], sizeof(float) * read_length);
  std::memcpy(r, m_channels[0].ptr<float>(0) + read_pos,
              sizeof(float) * read_length);
  std::memcpy(g, m_channels[1].ptr<float>(0) + read_pos,
              sizeof(float) * read_length);
  std::memcpy(b, m_channels[2].ptr<float>(0) + read_pos,
              sizeof(float) * read_length);

  for (auto x = startx; x <= endx; x += SCALAR) {
    processFragByScalar(startx, x, y, z[x - startx], z0, z1, z2, z, r, g, b,
                        lists, shader, scalar, eye);
  }

  std::memcpy(&m_zBuffer[read_pos], z, sizeof(float) * read_length);
  std::memcpy(m_channels[0].ptr<float>(0) + read_pos, r,
              sizeof(float) * read_length);
  std::memcpy(m_channels[1].ptr<float>(0) + read_pos, g,
              sizeof(float) * read_length);
  std::memcpy(m_channels[2].ptr<float>(0) + read_pos, b,
              sizeof(float) * read_length);
}

inline void SoftRasterizer::RenderingPipeline::processFragByScalar(
    const int startx, const int x, const int y, const float old_z,
    const float z0, const float z1, const float z2, float *__restrict z,
    float *__restrict r, float *__restrict g, float *__restrict b,
    const std::vector<SoftRasterizer::light_struct> &lists,
    std::shared_ptr<SoftRasterizer::Shader> shader,
    const SoftRasterizer::Triangle &scalar, const glm::vec3 &eye) {

  // Check if the point (currentX, currentY) is inside the triangle
  if (!insideTriangle(x + 0.5f, y + 0.5f, scalar)) {
    return;
  }

  // Check if the point (currentX, currentY) is inside the triangle
  auto [alpha, beta, gamma] = barycentric(x, y, scalar);

  // For Z-buffer interpolation
  auto new_z = alpha * z0 + beta * z1 + gamma * z2;

  if (new_z > old_z) {
    return;
  }

  /*interpolate normal*/
  auto interpolation_normal =
      Tools::interpolateNormal(alpha, beta, gamma, scalar.m_normal[0],
                               scalar.m_normal[1], scalar.m_normal[2]);

  /*interpolate uv*/
  auto interpolation_texCoord =
      Tools::interpolateTexCoord(alpha, beta, gamma, scalar.m_texCoords[0],
                                 scalar.m_texCoords[1], scalar.m_texCoords[2]);

  auto color = Tools::normalizedToRGB(shader->applyFragmentShader(
      eye, lists,
      fragment_shader_payload(glm::vec3(x, y, new_z), interpolation_normal,
                              interpolation_texCoord)));

  // writePixel(x, y, color);
  z[x - startx] = new_z;
  r[x - startx] = static_cast<float>(color.x);
  g[x - startx] = static_cast<float>(color.y);
  b[x - startx] = static_cast<float>(color.z);
}

/* Bresenham algorithm*/
void SoftRasterizer::RenderingPipeline::drawLine(const glm::vec3 &p0,
                                                 const glm::vec3 &p1,
                                                 const glm::uvec3 &color) {

  auto x1 = p0.x;
  auto y1 = p0.y;
  auto x2 = p1.x;
  auto y2 = p1.y;

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
