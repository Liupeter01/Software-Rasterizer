#include <Shader.hpp>
#include <Tools.hpp>
#include <spdlog/spdlog.h>

/*static variables*/
Eigen::Vector3f SoftRasterizer::Shader::ka =
    Eigen::Vector3f(0.005f, 0.005f, 0.005f);
Eigen::Vector3f SoftRasterizer::Shader::ks =
    Eigen::Vector3f(0.7937, 0.7937, 0.7937);

float SoftRasterizer::Shader::p = 150;
float SoftRasterizer::Shader::kh = 0.2;
float SoftRasterizer::Shader::kn = 0.1;

SoftRasterizer::fragment_shader_payload::fragment_shader_payload(
    const Eigen::Vector3f &_p, const Eigen::Vector3f &_n,
    const Eigen::Vector2f &_texcoord, const Eigen::Vector3f &_color)
    : position(_p), normal(_n), texCoords(_texcoord), color(_color) {}

SoftRasterizer::Shader::Shader(const std::string &path)
    : Shader(std::make_shared<TextureLoader>(path)) {}

SoftRasterizer::Shader::Shader(std::shared_ptr<TextureLoader> _loader)
    : texture(_loader) {
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
    const Eigen::Matrix4f &Model, const Eigen::Matrix4f &View,
    const Eigen::Matrix4f &Projection, const fragment_shader_payload &payload) {
  return SoftRasterizer::vertex_displacement{
      Tools::to_vec3(Projection * View * Model *
                     Tools::to_vec4(payload.position, 1.0f)),
      Tools::to_vec3(Model.inverse().transpose() *
                     Tools::to_vec4(payload.normal))};
}

/*Use Fragment Shader*/
Eigen::Vector3f SoftRasterizer::Shader::applyFragmentShader(
    const Eigen::Vector3f &camera,
    const std::initializer_list<light_struct> &lights,
    const fragment_shader_payload &payload) {

  return m_standard(camera, lights, payload);
}

void SoftRasterizer::Shader::applyFragmentShader(
    const Eigen::Vector3f &camera,
    const std::initializer_list<light_struct> &lights, const PointSIMD &point,
    NormalSIMD &normal, const TexCoordSIMD &texcoord, ColorSIMD &colour) {

  m_simd(camera, lights, point, normal, texcoord, colour);
}

void SoftRasterizer::Shader::simd_normal_fragment_shader_impl(
    const Eigen::Vector3f &camera,
    const std::initializer_list<light_struct> &lights, const PointSIMD &point,
    NormalSIMD &normal, const TexCoordSIMD &texcoord, ColorSIMD &colour) {

#if defined(__x86_64__) || defined(_WIN64)
          __m256 color = _mm256_set1_ps(255.f);
          colour.r =_mm256_mul_ps(_mm256_add_ps(normal.x, one), point_five);
          colour.g =_mm256_mul_ps(_mm256_add_ps(normal.y, one), point_five);
          colour.b = _mm256_mul_ps(_mm256_add_ps(normal.z, one), point_five);

          colour.r = _mm256_mul_ps(
                    _mm256_min_ps(_mm256_max_ps(colour.r, zero), one), color);
          colour.g = _mm256_mul_ps(
                   _mm256_min_ps(_mm256_max_ps(colour.g, zero), one), color);
          colour.b =_mm256_mul_ps(
                    _mm256_min_ps(_mm256_max_ps(colour.b, zero), one), color);

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
    const Eigen::Vector3f &camera,
    const std::initializer_list<light_struct> &lights, const PointSIMD &point,
    NormalSIMD &normal, const TexCoordSIMD &texcoord, ColorSIMD &colour) {
  for (const auto &light : lights) {
  }
}

void SoftRasterizer::Shader::simd_phong_fragment_shader_impl(
    const Eigen::Vector3f &camera,
    const std::initializer_list<light_struct> &lights, const PointSIMD &point,
    NormalSIMD &normal, const TexCoordSIMD &texcoord, ColorSIMD &colour) {
  for (const auto &light : lights) {
  }
}

void SoftRasterizer::Shader::simd_displacement_fragment_shader_impl(
    const Eigen::Vector3f &camera,
    const std::initializer_list<light_struct> &lights, const PointSIMD &point,
    NormalSIMD &normal, const TexCoordSIMD &texcoord, ColorSIMD &colour) {
  for (const auto &light : lights) {
  }
}

void SoftRasterizer::Shader::simd_bump_fragment_shader_impl(
    const Eigen::Vector3f &camera,
    const std::initializer_list<light_struct> &lights, const PointSIMD &point,
    NormalSIMD &normal, const TexCoordSIMD &texcoord, ColorSIMD &colour) {
  for (const auto &light : lights) {
  }
}

/*Compute Displacement Mapping*/
SoftRasterizer::vertex_displacement
SoftRasterizer::Shader::calcDisplacementMapping(
    const fragment_shader_payload &payload, const float kh, const float kn) {

  Eigen::Vector3f n = payload.normal;
  Eigen::Vector3f t;
  t << (n.x() * n.y()) / std::sqrt(n.x() * n.x() + n.z() * n.z()),
      std::sqrt(n.x() * n.x() + n.z() * n.z()),
      (n.z() * n.y()) / std::sqrt(n.x() * n.x() + n.z() * n.z());

  Eigen::Vector3f b = n.cross(t);
  Eigen::Matrix3f TBN;
  TBN << t.x(), b.x(), n.x(), t.y(), b.y(), n.y(), t.z(), b.z(), n.z();

  /*Calculating derivatived on Both UV directions seperately */
  auto origin_texture = texture->getTextureColor(payload.texCoords);
  auto origin_norm = origin_texture.norm();
  auto U_direction = texture->getTextureColor(Eigen::Vector2f(
      (payload.texCoords.x() + 1) / texture->m_width, payload.texCoords.y()));

  auto V_direction = texture->getTextureColor(Eigen::Vector2f(
      payload.texCoords.x(), (payload.texCoords.y() + 1) / texture->m_height));

  auto dU = kh * kn * (U_direction.norm() - origin_norm);
  auto dV = kh * kn * (V_direction.norm() - origin_norm);

  Eigen::Vector3f ln(-dU, -dV, 1.0f);

  return SoftRasterizer::vertex_displacement{
      payload.position + kn * n * origin_norm, // update vertex position
      (TBN * ln).normalized()                  // update normal
  };
}

/*Compute Bump Mapping*/
Eigen::Vector3f
SoftRasterizer::Shader::calcBumpMapping(const fragment_shader_payload &payload,
                                        const float kh, const float kn) {

  Eigen::Vector3f n = payload.normal;

  Eigen::Vector3f t;
  t << (n.x() * n.y()) / std::sqrt(n.x() * n.x() + n.z() * n.z()),
      std::sqrt(n.x() * n.x() + n.z() * n.z()),
      (n.z() * n.y()) / std::sqrt(n.x() * n.x() + n.z() * n.z());

  Eigen::Vector3f b = n.cross(t);
  Eigen::Matrix3f TBN;
  TBN << t.x(), b.x(), n.x(), t.y(), b.y(), n.y(), t.z(), b.z(), n.z();

  /*Calculating derivatived on Both UV directions seperately */
  auto origin_texture = texture->getTextureColor(payload.texCoords);
  auto origin_norm = origin_texture.norm();
  auto U_direction = texture->getTextureColor(Eigen::Vector2f(
      (payload.texCoords.x() + 1) / texture->m_width, payload.texCoords.y()));

  auto V_direction = texture->getTextureColor(Eigen::Vector2f(
      payload.texCoords.x(), (payload.texCoords.y() + 1) / texture->m_height));

  auto dU = kh * kn * (U_direction.norm() - origin_norm);
  auto dV = kh * kn * (V_direction.norm() - origin_norm);

  Eigen::Vector3f ln(-dU, -dV, 1.0f);
  return (TBN * ln).normalized();
}

// Static function to compute the Blinn-Phong reflection model
Eigen::Vector3f SoftRasterizer::Shader::BlinnPhong(
    const Eigen::Vector3f &camera, const fragment_shader_payload &shading_point,
    const light_struct &light, const Eigen::Vector3f &ka,
    const Eigen::Vector3f &kd, const Eigen::Vector3f &ks, const float p) {

  Eigen::Vector3f normal = shading_point.normal.normalized();
  Eigen::Vector3f lightDir = light.position - shading_point.position;

  // Light distribution based on inverse square law (distance attenuation)
  float distanceSquared =
      std::sqrt(std::pow((light.position.x() - shading_point.position.x()), 2) +
                std::pow((light.position.y() - shading_point.position.y()), 2));

  Eigen::Vector3f distribution = light.intensity / distanceSquared;

  // Ambient lighting
  Eigen::Vector3f La = ka.cwiseProduct(light.intensity);

  // Diffuse reflection (Lambertian reflectance)
  float cosTheta = std::max(0.f, normal.dot(lightDir.normalized()));
  Eigen::Vector3f Ld = cosTheta * kd.cwiseProduct(distribution);

  // Specular reflection (Blinn-Phong)
  Eigen::Vector3f v = camera - shading_point.position;
  Eigen::Vector3f h = (lightDir + v).normalized();
  float cosAlpha = std::max(0.f, normal.dot(h));
  Eigen::Vector3f Ls = std::pow(cosAlpha, p) * ks.cwiseProduct(distribution);

  // Combine all lighting components
  Eigen::Vector3f result_color = La + Ld + Ls;

  // Calculate the final color based on the shading point color
  return result_color.cwiseProduct(shading_point.color);
}

/*Visualizing normal directions or checking surface normal directions in some
 * debugging scenarios*/
Eigen::Vector3f SoftRasterizer::Shader::standard_normal_fragment_shader_impl(
    const Eigen::Vector3f &camera,
    const std::initializer_list<light_struct> &lights,
    const fragment_shader_payload &payload) {

  return (payload.normal.normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) /
         2.0f;
}

Eigen::Vector3f SoftRasterizer::Shader::standard_texture_fragment_shader_impl(
    const Eigen::Vector3f &camera,
    const std::initializer_list<light_struct> &lights,
    const fragment_shader_payload &payload) {

  Eigen::Vector3f result_color = {0, 0, 0};

  Eigen::Vector3f kd = texture->getTextureColor(payload.texCoords);

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

Eigen::Vector3f SoftRasterizer::Shader::standard_phong_fragment_shader_impl(
    const Eigen::Vector3f &camera,
    const std::initializer_list<light_struct> &lights,
    const fragment_shader_payload &payload) {

  Eigen::Vector3f result_color = {0, 0, 0};

  Eigen::Vector3f kd = payload.color;

  fragment_shader_payload shader_arguments{payload.position, payload.normal,
                                           payload.texCoords, kd};

  /* *ambient*, *diffuse*, and *specular* */
  for (const auto &light : lights) {

    /*Blinn-Phong reflection model*/
    result_color += BlinnPhong(camera, shader_arguments, light, ka, kd, ks, p);
  }

  return result_color;
}

Eigen::Vector3f
SoftRasterizer::Shader::standard_displacement_fragment_shader_impl(
    const Eigen::Vector3f &camera,
    const std::initializer_list<light_struct> &lights,
    const fragment_shader_payload &payload) {
  Eigen::Vector3f result_color = {0, 0, 0};
  Eigen::Vector3f kd = texture->getTextureColor(payload.texCoords);

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

Eigen::Vector3f SoftRasterizer::Shader::standard_bump_fragment_shader_impl(
    const Eigen::Vector3f &camera,
    const std::initializer_list<light_struct> &lights,
    const fragment_shader_payload &payload) {

  Eigen::Vector3f result_color = {0, 0, 0};
  Eigen::Vector3f kd = texture->getTextureColor(payload.texCoords);

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
