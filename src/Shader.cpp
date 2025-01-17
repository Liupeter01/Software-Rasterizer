#include <Tools.hpp>
#include <glm/vec4.hpp>
#include <shader/Shader.hpp>
#include <spdlog/spdlog.h>

/*static variables*/
glm::vec3 SoftRasterizer::Shader::ka = glm::vec3(0.005f, 0.005f, 0.005f);
glm::vec3 SoftRasterizer::Shader::ks = glm::vec3(0.7937, 0.7937, 0.7937);

float SoftRasterizer::Shader::p = 150;
float SoftRasterizer::Shader::kh = 0.2;
float SoftRasterizer::Shader::kn = 0.1;

SoftRasterizer::fragment_shader_payload::fragment_shader_payload(
          const glm::vec3& _p, 
          const glm::vec3& _n,
          const glm::vec2& _texcoord,
          const glm::vec3& _color)
    : position(_p), normal(_n), texCoords(_texcoord), color(_color) {}

SoftRasterizer::Shader::Shader(const std::string &path)
    : Shader(std::make_shared<TextureLoader>(path)) {}

SoftRasterizer::Shader::Shader(std::shared_ptr<TextureLoader> _loader)
    : texture(_loader) {

  /*calculate __m256 width and height*/
#if defined(__x86_64__) || defined(_WIN64)
  width_256 = _mm256_set1_ps(texture->m_width);
  height_256 = _mm256_set1_ps(texture->m_height);

#elif defined(__arm__) || defined(__aarch64__)
  width_256 = simde_mm256_set1_ps(texture->m_width);
  height_256 = simde_mm256_set1_ps(texture->m_height);

#else
#endif

  /*calculate __m128 width and height*/
  width_128 = _mm_set1_ps(texture->m_width);
  height_128 = _mm_set1_ps(texture->m_height);

  registerShaders();
}

void SoftRasterizer::Shader::registerShaders() {

  /*Standard*/
  standard_shaders[0] = std::bind(
      &SoftRasterizer::Shader::standard_normal_fragment_shader_impl, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  standard_shaders[1] = std::bind(
      &SoftRasterizer::Shader::standard_texture_fragment_shader_impl, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  standard_shaders[2] = std::bind(
      &SoftRasterizer::Shader::standard_phong_fragment_shader_impl, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  standard_shaders[3] = std::bind(
      &SoftRasterizer::Shader::standard_displacement_fragment_shader_impl, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  standard_shaders[4] = std::bind(
      &SoftRasterizer::Shader::standard_bump_fragment_shader_impl, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

  /*SIMD*/
  simd_shaders[0] = std::bind(
      &SoftRasterizer::Shader::simd_normal_fragment_shader_impl, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
      std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);

  simd_shaders[1] = std::bind(
      &SoftRasterizer::Shader::simd_texture_fragment_shader_impl, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
      std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);

  simd_shaders[2] = std::bind(
      &SoftRasterizer::Shader::simd_phong_fragment_shader_impl, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
      std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);

  simd_shaders[3] = std::bind(
      &SoftRasterizer::Shader::simd_displacement_fragment_shader_impl, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
      std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);

  simd_shaders[4] = std::bind(
      &SoftRasterizer::Shader::simd_bump_fragment_shader_impl, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
      std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
}

bool SoftRasterizer::Shader::setFragmentShader(SHADERS_TYPE type) {
  if (static_cast<std::uint8_t>(type) >= standard_shaders.size()) {
    spdlog::error("Set FramentShader Error Due To Invalid Shader Type Input!");
    return false;
  }
  try {
    m_standard = standard_shaders[static_cast<std::uint8_t>(type)];
    m_simd = simd_shaders[static_cast<std::uint8_t>(type)];

  } catch (const std::exception &e) {
    spdlog::error("Set FramentShader Error! Reason {}", e.what());
    return false;
  }
  return true;
}

/*User Vertex Shader*/
SoftRasterizer::vertex_displacement SoftRasterizer::Shader::applyVertexShader(
    const glm::mat4&Model, const glm::mat4 &View,
    const glm::mat4&Projection, const fragment_shader_payload &payload) {
          return SoftRasterizer::vertex_displacement{
              Tools::to_vec3(Projection * View * Model * glm::vec4(payload.position, 1.0f)),
              Tools::to_vec3(glm::transpose(glm::inverse(Model)) * glm::vec4(payload.normal, 1.0f)) 
          };
}

/*Use Fragment Shader*/
glm::vec3 SoftRasterizer::Shader::applyFragmentShader(
    const glm::vec3&camera, const std::vector<light_struct> &lights,
    const fragment_shader_payload &payload) {
  return m_standard(camera, lights, payload);
}

void SoftRasterizer::Shader::applyFragmentShader(
    const glm::vec3&camera, const std::vector<light_struct> &lights,
    const PointSIMD &point, NormalSIMD &normal, TexCoordSIMD &texcoord,
    ColorSIMD &colour) {

#if defined(__x86_64__) || defined(_WIN64)
  texcoord.u = _mm256_mul_ps(texcoord.u, width_256);
  texcoord.v = _mm256_mul_ps(texcoord.v, height_256);

  texcoord.u = _mm256_max_ps(
      zero, _mm256_min_ps(texcoord.u, _mm256_sub_ps(width_256, one)));
  texcoord.v = _mm256_max_ps(
      zero, _mm256_min_ps(texcoord.v, _mm256_sub_ps(height_256, one)));

#elif defined(__arm__) || defined(__aarch64__)
  texcoord.u = simde_mm256_mul_ps(texcoord.u, width_256);
  texcoord.v = simde_mm256_mul_ps(texcoord.v, height_256);

  texcoord.u = simde_mm256_max_ps(
      zero, simde_mm256_min_ps(texcoord.u, simde_mm256_sub_ps(width_256, one)));
  texcoord.v = simde_mm256_max_ps(
      zero,
      simde_mm256_min_ps(texcoord.v, simde_mm256_sub_ps(height_256, one)));
#else
#endif

  m_simd(camera, lights, point, normal, texcoord, colour);
}

void SoftRasterizer::Shader::simd_normal_fragment_shader_impl(
    const glm::vec3&camera, const std::vector<light_struct> &lights,
    const PointSIMD &point, NormalSIMD &normal, TexCoordSIMD &texcoord,
    ColorSIMD &colour) {

#if defined(__x86_64__) || defined(_WIN64)
  __m256 color = _mm256_set1_ps(255.f);
  colour.r = _mm256_mul_ps(_mm256_add_ps(normal.x, one), point_five);
  colour.g = _mm256_mul_ps(_mm256_add_ps(normal.y, one), point_five);
  colour.b = _mm256_mul_ps(_mm256_add_ps(normal.z, one), point_five);

  colour.r =
      _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(colour.r, zero), one), color);
  colour.g =
      _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(colour.g, zero), one), color);
  colour.b =
      _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(colour.b, zero), one), color);

#elif defined(__arm__) || defined(__aarch64__)
  simde__m256 color = simde_mm256_set1_ps(255.f);
  colour.r = simde_mm256_mul_ps(simde_mm256_add_ps(normal.x, one), point_five);
  colour.g = simde_mm256_mul_ps(simde_mm256_add_ps(normal.y, one), point_five);
  colour.b = simde_mm256_mul_ps(simde_mm256_add_ps(normal.z, one), point_five);

  colour.r = simde_mm256_mul_ps(
      simde_mm256_min_ps(simde_mm256_max_ps(colour.r, zero), one), color);
  colour.g = simde_mm256_mul_ps(
      simde_mm256_min_ps(simde_mm256_max_ps(colour.g, zero), one), color);
  colour.b = simde_mm256_mul_ps(
      simde_mm256_min_ps(simde_mm256_max_ps(colour.b, zero), one), color);

#else
#endif
}

void SoftRasterizer::Shader::simd_texture_fragment_shader_impl(
    const glm::vec3&camera, const std::vector<light_struct> &lights,
    const PointSIMD &point, NormalSIMD &normal, TexCoordSIMD &texcoord,
    ColorSIMD &colour) {

  auto [R, G, B] = texture->getTextureColor(texcoord.u, texcoord.v);

#if defined(__x86_64__) || defined(_WIN64)
  auto ka_r = _mm256_set1_ps(ka.x);
  auto ka_g = _mm256_set1_ps(ka.y);
  auto ka_b = _mm256_set1_ps(ka.z);

  auto ks_r = _mm256_set1_ps(ks.x);
  auto ks_g = _mm256_set1_ps(ks.y);
  auto ks_b = _mm256_set1_ps(ks.z);

  auto camera_x = _mm256_set1_ps(camera.x);
  auto camera_y = _mm256_set1_ps(camera.y);
  auto camera_z = _mm256_set1_ps(camera.z);

  auto p_simd = _mm256_set1_ps(p);

#elif defined(__arm__) || defined(__aarch64__)
  auto ka_r = simde_mm256_set1_ps(ka.x);
  auto ka_g = simde_mm256_set1_ps(ka.y);
  auto ka_b = simde_mm256_set1_ps(ka.z);

  auto ks_r = simde_mm256_set1_ps(ks.x);
  auto ks_g = simde_mm256_set1_ps(ks.y);
  auto ks_b = simde_mm256_set1_ps(ks.z);

  auto camera_x = simde_mm256_set1_ps(camera.x);
  auto camera_y = simde_mm256_set1_ps(camera.y);
  auto camera_z = simde_mm256_set1_ps(camera.z);

  auto p_simd = simde_mm256_set1_ps(p);
#else
#endif

  colour.r = colour.g = colour.b = zero;

  for (const auto &light : lights) {
    auto [Lr, Lg, Lb] = BlinnPhong(
        normal, ka_r, ka_g, ka_b, R, G, B, ks_r, ks_g, ks_b, camera_x, camera_y,
        camera_z,
#if defined(__x86_64__) || defined(_WIN64)
        _mm256_set1_ps(light.position.x), _mm256_set1_ps(light.position.y),
        _mm256_set1_ps(light.position.z), _mm256_set1_ps(light.intensity.x),
        _mm256_set1_ps(light.intensity.y),
        _mm256_set1_ps(light.intensity.z),
#elif defined(__arm__) || defined(__aarch64__)
        simde_mm256_set1_ps(light.position.x),
        simde_mm256_set1_ps(light.position.y),
        simde_mm256_set1_ps(light.position.z),
        simde_mm256_set1_ps(light.intensity.x),
        simde_mm256_set1_ps(light.intensity.y),
        simde_mm256_set1_ps(light.intensity.z),
#else
#endif
        point.x, point.y, point.z, p_simd);

#if defined(__x86_64__) || defined(_WIN64)
    colour.r = _mm256_add_ps(colour.r, Lr);
    colour.g = _mm256_add_ps(colour.g, Lg);
    colour.b = _mm256_add_ps(colour.b, Lb);

#elif defined(__arm__) || defined(__aarch64__)
    colour.r = simde_mm256_add_ps(colour.r, Lr);
    colour.g = simde_mm256_add_ps(colour.g, Lg);
    colour.b = simde_mm256_add_ps(colour.b, Lb);
#else
#endif
  }

#if defined(__x86_64__) || defined(_WIN64)
  __m256 color = _mm256_set1_ps(255.f);
  colour.r =
      _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(colour.r, zero), one), color);
  colour.g =
      _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(colour.g, zero), one), color);
  colour.b =
      _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(colour.b, zero), one), color);

#elif defined(__arm__) || defined(__aarch64__)
  simde__m256 color = simde_mm256_set1_ps(255.f);
  colour.r = simde_mm256_mul_ps(
      simde_mm256_min_ps(simde_mm256_max_ps(colour.r, zero), one), color);
  colour.g = simde_mm256_mul_ps(
      simde_mm256_min_ps(simde_mm256_max_ps(colour.g, zero), one), color);
  colour.b = simde_mm256_mul_ps(
      simde_mm256_min_ps(simde_mm256_max_ps(colour.b, zero), one), color);

#else
#endif
}

void SoftRasterizer::Shader::simd_phong_fragment_shader_impl(
    const glm::vec3&camera, const std::vector<light_struct> &lights,
    const PointSIMD &point, NormalSIMD &normal, TexCoordSIMD &texcoord,
    ColorSIMD &colour) {

#if defined(__x86_64__) || defined(_WIN64)
  auto ka_r = _mm256_set1_ps(ka.x);
  auto ka_g = _mm256_set1_ps(ka.y);
  auto ka_b = _mm256_set1_ps(ka.z);

  auto ks_r = _mm256_set1_ps(ks.x);
  auto ks_g = _mm256_set1_ps(ks.y);
  auto ks_b = _mm256_set1_ps(ks.z);

  auto camera_x = _mm256_set1_ps(camera.x);
  auto camera_y = _mm256_set1_ps(camera.y);
  auto camera_z = _mm256_set1_ps(camera.z);

  auto p_simd = _mm256_set1_ps(p);

#elif defined(__arm__) || defined(__aarch64__)
  auto ka_r = simde_mm256_set1_ps(ka.x());
  auto ka_g = simde_mm256_set1_ps(ka.y());
  auto ka_b = simde_mm256_set1_ps(ka.z());

  auto ks_r = simde_mm256_set1_ps(ks.x());
  auto ks_g = simde_mm256_set1_ps(ks.y());
  auto ks_b = simde_mm256_set1_ps(ks.z());

  auto camera_x = simde_mm256_set1_ps(camera.x());
  auto camera_y = simde_mm256_set1_ps(camera.y());
  auto camera_z = simde_mm256_set1_ps(camera.z());

  auto p_simd = simde_mm256_set1_ps(p);
#else
#endif

  auto kd_r = colour.r;
  auto kd_g = colour.g;
  auto kd_b = colour.b;

  colour.r = colour.g = colour.b = zero;

  for (const auto &light : lights) {

    auto [Lr, Lg, Lb] = BlinnPhong(
        normal, ka_r, ka_g, ka_b, kd_r, kd_g, kd_b, ks_r, ks_g, ks_b, camera_x,
        camera_y, camera_z,
#if defined(__x86_64__) || defined(_WIN64)
        _mm256_set1_ps(light.position.x), _mm256_set1_ps(light.position.y),
        _mm256_set1_ps(light.position.z), _mm256_set1_ps(light.intensity.x),
        _mm256_set1_ps(light.intensity.y),
        _mm256_set1_ps(light.intensity.z),
#elif defined(__arm__) || defined(__aarch64__)
        simde_mm256_set1_ps(light.position.x),
        simde_mm256_set1_ps(light.position.y),
        simde_mm256_set1_ps(light.position.z),
        simde_mm256_set1_ps(light.intensity.x),
        simde_mm256_set1_ps(light.intensity.y),
        simde_mm256_set1_ps(light.intensity.z),
#else
#endif
        point.x, point.y, point.z, p_simd);

#if defined(__x86_64__) || defined(_WIN64)
    colour.r = _mm256_add_ps(colour.r, Lr);
    colour.g = _mm256_add_ps(colour.g, Lg);
    colour.b = _mm256_add_ps(colour.b, Lb);

#elif defined(__arm__) || defined(__aarch64__)
    colour.r = simde_mm256_add_ps(colour.r, Lr);
    colour.g = simde_mm256_add_ps(colour.g, Lg);
    colour.b = simde_mm256_add_ps(colour.b, Lb);
#else
#endif
  }

#if defined(__x86_64__) || defined(_WIN64)
  __m256 color = _mm256_set1_ps(255.f);
  colour.r =
      _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(colour.r, zero), one), color);
  colour.g =
      _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(colour.g, zero), one), color);
  colour.b =
      _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(colour.b, zero), one), color);

#elif defined(__arm__) || defined(__aarch64__)
  simde__m256 color = simde_mm256_set1_ps(255.f);
  colour.r = simde_mm256_mul_ps(
      simde_mm256_min_ps(simde_mm256_max_ps(colour.r, zero), one), color);
  colour.g = simde_mm256_mul_ps(
      simde_mm256_min_ps(simde_mm256_max_ps(colour.g, zero), one), color);
  colour.b = simde_mm256_mul_ps(
      simde_mm256_min_ps(simde_mm256_max_ps(colour.b, zero), one), color);

#else
#endif
}

void SoftRasterizer::Shader::simd_displacement_fragment_shader_impl(
    const glm::vec3&camera, const std::vector<light_struct> &lights,
    const PointSIMD &point, NormalSIMD &normal, TexCoordSIMD &texcoord,
    ColorSIMD &colour) {
  for (const auto &light : lights) {
  }

#if defined(__x86_64__) || defined(_WIN64)
  __m256 color = _mm256_set1_ps(255.f);
  colour.r =
      _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(colour.r, zero), one), color);
  colour.g =
      _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(colour.g, zero), one), color);
  colour.b =
      _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(colour.b, zero), one), color);

#elif defined(__arm__) || defined(__aarch64__)
  simde__m256 color = simde_mm256_set1_ps(255.f);
  colour.r = simde_mm256_mul_ps(
      simde_mm256_min_ps(simde_mm256_max_ps(colour.r, zero), one), color);
  colour.g = simde_mm256_mul_ps(
      simde_mm256_min_ps(simde_mm256_max_ps(colour.g, zero), one), color);
  colour.b = simde_mm256_mul_ps(
      simde_mm256_min_ps(simde_mm256_max_ps(colour.b, zero), one), color);

#else
#endif
}

void SoftRasterizer::Shader::simd_bump_fragment_shader_impl(
    const glm::vec3 &camera, const std::vector<light_struct> &lights,
    const PointSIMD &point, NormalSIMD &normal, TexCoordSIMD &texcoord,
    ColorSIMD &colour) {
  for (const auto &light : lights) {
  }

#if defined(__x86_64__) || defined(_WIN64)
  __m256 color = _mm256_set1_ps(255.f);
  colour.r =
      _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(colour.r, zero), one), color);
  colour.g =
      _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(colour.g, zero), one), color);
  colour.b =
      _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(colour.b, zero), one), color);

#elif defined(__arm__) || defined(__aarch64__)
  simde__m256 color = simde_mm256_set1_ps(255.f);
  colour.r = simde_mm256_mul_ps(
      simde_mm256_min_ps(simde_mm256_max_ps(colour.r, zero), one), color);
  colour.g = simde_mm256_mul_ps(
      simde_mm256_min_ps(simde_mm256_max_ps(colour.g, zero), one), color);
  colour.b = simde_mm256_mul_ps(
      simde_mm256_min_ps(simde_mm256_max_ps(colour.b, zero), one), color);

#else
#endif
}

/*Compute Displacement Mapping*/
SoftRasterizer::vertex_displacement
SoftRasterizer::Shader::calcDisplacementMapping(
    const fragment_shader_payload &payload, const float kh, const float kn) {

  auto n = payload.normal;
  glm::vec3 t((n.x * n.y) / std::sqrt(n.x * n.x + n.z * n.z),
            std::sqrt(n.x * n.x + n.z * n.z),
            (n.z * n.y) / std::sqrt(n.x * n.x + n.z * n.z));

  glm::vec3 b = glm::cross(n, t);
  glm::mat3 TBN(t.x, b.x, n.x, t.y, b.y, n.y, t.z, b.z, n.z);

  /*Calculating derivatived on Both UV directions seperately */
  auto origin_texture = texture->getTextureColor(payload.texCoords);
  auto origin_norm = glm::length(origin_texture);
  auto U_direction = texture->getTextureColor(glm::vec2(
      (payload.texCoords.x + 1) / texture->m_width, payload.texCoords.y));

  auto V_direction = texture->getTextureColor(glm::vec2(
      payload.texCoords.x, (payload.texCoords.y + 1) / texture->m_height));

  auto dU = kh * kn * (glm::length(U_direction) - origin_norm);
  auto dV = kh * kn * (glm::length(V_direction) - origin_norm);

  glm::vec3 ln(-dU, -dV, 1.0f);

  return SoftRasterizer::vertex_displacement{
      payload.position + kn * n * origin_norm, // update vertex position
      glm::normalize(TBN * ln)                  // update normal
  };
}

/*Compute Bump Mapping*/
glm::vec3
SoftRasterizer::Shader::calcBumpMapping(const fragment_shader_payload &payload,
                                        const float kh, const float kn) {

  auto n = payload.normal;

  glm::vec3 t((n.x * n.y) / std::sqrt(n.x * n.x + n.z * n.z),
            std::sqrt(n.x * n.x + n.z * n.z),
            (n.z * n.y) / std::sqrt(n.x * n.x + n.z * n.z));

  glm::vec3 b = glm::cross(n, t);
  glm::mat3 TBN(t.x, b.x, n.x, t.y, b.y, n.y, t.z, b.z, n.z);

  /*Calculating derivatived on Both UV directions seperately */
  auto origin_texture = texture->getTextureColor(payload.texCoords);
  auto origin_norm = glm::length(origin_texture);
  auto U_direction = texture->getTextureColor(glm::vec2(
      (payload.texCoords.x + 1) / texture->m_width, payload.texCoords.y));

  auto V_direction = texture->getTextureColor(glm::vec2(
      payload.texCoords.x, (payload.texCoords.y + 1) / texture->m_height));

  auto dU = kh * kn * (glm::length(U_direction) - origin_norm);
  auto dV = kh * kn * (glm::length(V_direction) - origin_norm);

  glm::vec3 ln(-dU, -dV, 1.0f);
  return glm::normalize(TBN * ln);
}

// Static function to compute the Blinn-Phong reflection model
glm::vec3 SoftRasterizer::Shader::BlinnPhong(
    const glm::vec3&camera, const fragment_shader_payload &shading_point,
    const light_struct &light, const glm::vec3&ka,
    const glm::vec3&kd, const glm::vec3&ks, const float p) {

          glm::vec3 normal = glm::normalize(shading_point.normal);
          glm::vec3 lightDir = light.position - shading_point.position;

  // Light distribution based on inverse square law (distance attenuation)
  float distanceSquared =
      std::sqrt(std::pow((light.position.x - shading_point.position.x), 2) +
                std::pow((light.position.y - shading_point.position.y), 2));

  glm::vec3 distribution = light.intensity / distanceSquared;

  // Ambient lighting
  glm::vec3 La = ka * light.intensity;

  // Diffuse reflection (Lambertian reflectance)
  float cosTheta = std::max(0.f, glm::dot(normal, glm::normalize(lightDir)));
  glm::vec3 Ld = cosTheta * kd * distribution;

  // Specular reflection (Blinn-Phong)
  glm::vec3 v = camera - shading_point.position;
  glm::vec3 h = glm::normalize(lightDir + v);
  float cosAlpha = std::max(0.f, glm::dot(normal, h));
  glm::vec3 Ls = std::pow(cosAlpha, p) * ks * distribution;

  // Combine all lighting components
  glm::vec3 result_color = La + Ld + Ls;

  // Calculate the final color based on the shading point color
  return result_color * shading_point.color;
}

/*Visualizing normal directions or checking surface normal directions in some
 * debugging scenarios*/
glm::vec3 SoftRasterizer::Shader::standard_normal_fragment_shader_impl(
    const glm::vec3&camera, const std::vector<light_struct> &lights,
    const fragment_shader_payload &payload) {

  return (glm::normalize(payload.normal) + glm::vec3(1.0f)) /
         2.0f;
}

glm::vec3 SoftRasterizer::Shader::standard_texture_fragment_shader_impl(
    const glm::vec3&camera, const std::vector<light_struct> &lights,
    const fragment_shader_payload &payload) {

  glm::vec3 result_color = {0, 0, 0};

  auto kd = texture->getTextureColor(payload.texCoords);

  fragment_shader_payload shader_arguments{
      payload.position, payload.normal, payload.texCoords,
      texture->getTextureColor(payload.texCoords)};

  /* *ambient*, *diffuse*, and *specular* */
  for (const auto &light : lights) {

    /*Blinn-Phong reflection model*/
    result_color += BlinnPhong(camera, shader_arguments, light, ka, kd, ks, p);
  }
  return result_color;
}

glm::vec3 SoftRasterizer::Shader::standard_phong_fragment_shader_impl(
    const glm::vec3&camera, const std::vector<light_struct> &lights,
    const fragment_shader_payload &payload) {

          glm::vec3 result_color = { 0, 0, 0 };

  auto kd = payload.color;

  fragment_shader_payload shader_arguments{payload.position, payload.normal,
                                           payload.texCoords, kd};

  /* *ambient*, *diffuse*, and *specular* */
  for (const auto &light : lights) {

    /*Blinn-Phong reflection model*/
    result_color += BlinnPhong(camera, shader_arguments, light, ka, kd, ks, p);
  }

  return result_color;
}

glm::vec3
SoftRasterizer::Shader::standard_displacement_fragment_shader_impl(
    const glm::vec3&camera, const std::vector<light_struct> &lights,
    const fragment_shader_payload &payload) {
          glm::vec3 result_color = {0, 0, 0};
          glm::vec3 kd = texture->getTextureColor(payload.texCoords);

  fragment_shader_payload shader_arguments{
      payload.position, payload.normal, payload.texCoords,
      texture->getTextureColor(payload.texCoords)};

  vertex_displacement new_vertex =
      calcDisplacementMapping(shader_arguments, kh, kn);

  shader_arguments.position = new_vertex.new_position;
  shader_arguments.normal = new_vertex.new_normal;

  /* *ambient*, *diffuse*, and *specular* */
  for (const auto &light : lights) {
    /*Blinn-Phong reflection model*/
    result_color += BlinnPhong(camera, shader_arguments, light, ka, kd, ks, p);
  }

  return result_color;
}

glm::vec3 SoftRasterizer::Shader::standard_bump_fragment_shader_impl(
    const glm::vec3&camera, const std::vector<light_struct> &lights,
    const fragment_shader_payload &payload) {

          glm::vec3 result_color = {0, 0, 0};
          glm::vec3 kd = texture->getTextureColor(payload.texCoords);

  fragment_shader_payload shader_arguments{
      payload.position, payload.normal, payload.texCoords,
      texture->getTextureColor(payload.texCoords)};

  shader_arguments.normal = calcBumpMapping(payload, kh, kn);

  /* *ambient*, *diffuse*, and *specular* */
  for (const auto &light : lights) {
    /*Blinn-Phong reflection model*/
    result_color += BlinnPhong(camera, shader_arguments, light, ka, kd, ks, p);
  }
  return result_color;
}
