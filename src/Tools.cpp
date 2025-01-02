#include <Tools.hpp>
#include <spdlog/spdlog.h>

// degree to radian
float SoftRasterizer::Tools::degreeToRadian(float degree) {
  return degree * (PI / 180.0f);
}

Eigen::Vector4f SoftRasterizer::Tools::to_vec4(const Eigen::Vector3f &v3,
                                               float w) {
  return Eigen::Vector4f(v3.x(), v3.y(), v3.z(), w);
}

Eigen::Vector3f SoftRasterizer::Tools::to_vec3(const Eigen::Vector4f &v4) {
  return Eigen::Vector3f(v4.x() / v4.w(), v4.y() / v4.w(), v4.z() / v4.w());
}

/**
 * @brief Calculates the rotation matrix using Rodrigues' rotation formula.
 *
 * This function computes a 4x4 rotation matrix that represents a rotation
 * of the given angle (in radian) around the specified axis in 3D space.
 * It utilizes Rodrigues' rotation formula to generate the matrix.
 *
 * @param axis A 3D vector representing the axis of rotation.
 * @param angle The angle (in radian) by which to rotate around the axis.
 *        A positive value rotates counterclockwise, and a negative value
 *        rotates clockwise around the axis.
 *
 * @return A 4x4 rotation matrix (Eigen::Matrix4f) representing the rotation.
 */
Eigen::Matrix4f
SoftRasterizer::Tools::calculateRotationMatrix(const Eigen::Vector3f &axis,
                                               float angle) {
  spdlog::info(
      "Calculating rotation matrix for angle: {} and axis: ({}, {}, {})", angle,
      axis.x(), axis.y(), axis.z());
  Eigen::Matrix4f rotationMatrix = Eigen::Matrix4f::Identity();

  auto rad = degreeToRadian(angle);

  auto cos = std::cos(rad);
  auto sin = std::sin(rad);

  const Eigen::Vector3f origin =
      Eigen::Vector3f(0.f, 0.f, 0.f); // origin point(0, 0, 0)
  const Eigen::Vector3f normalize_axis = axis.normalized(); // normalize axis

  const Eigen::Vector3f move_back = axis - origin; // move from origin to axis
  const Eigen::Vector3f transform_to = -move_back; // move from axis to origin

  /*Transform Matrix*/
  Eigen::Matrix4f move_back_matrix;    // Transform from origin to axis
  Eigen::Matrix4f transform_to_matrix; // Transform from axis to origin

  /*Rotate Matrix*/
  Eigen::Matrix4f rotationMatrix4 = Eigen::Matrix4f::Identity();

  transform_to_matrix << 1, 0, 0, transform_to.x(), 0, 1, 0, transform_to.y(),
      0, 0, 1, transform_to.z(), 0, 0, 0, 1;

  move_back_matrix << 1, 0, 0, move_back.x(), 0, 1, 0, move_back.y(), 0, 0, 1,
      move_back.z(), 0, 0, 0, 1;

  /*Rodrigues Rotation*/
  Eigen::Matrix3f Rodrigues;
  Rodrigues << 0, -normalize_axis.z(), normalize_axis.y(), normalize_axis.z(),
      0, -normalize_axis.x(), -normalize_axis.y(), normalize_axis.x(), 0;

  // cos E + (1-cos)(n * n^T)+sin * Rodrigues
  Rodrigues = cos * Eigen::Matrix3f::Identity() +
              (1.0f - cos) * normalize_axis * normalize_axis.transpose() +
              sin * Rodrigues;

  rotationMatrix4.block<3, 3>(0, 0) = Rodrigues;

  return move_back_matrix * rotationMatrix4 * transform_to_matrix *
         rotationMatrix;
}

/**
 * @brief Calculates the model matrix for a 3D object.
 *
 * This function computes a 4x4 model matrix that represents the transformation
 * of a 3D object in world space. The model matrix is obtained by combining the
 * translation, rotation, and scaling transformations.
 *
 * @param translation A 3D vector representing the translation (position) of the
 * object.
 * @param rotation A 4x4 matrix representing the rotation of the object.
 * @param scale A 3D vector representing the scaling factors along the x, y, and
 * z axes.
 *
 * @return A 4x4 model matrix (Eigen::Matrix4f) representing the combined
 * transformations.
 */
Eigen::Matrix4f
SoftRasterizer::Tools::calculateModelMatrix(const Eigen::Vector3f &translation,
                                            const Eigen::Matrix4f &rotation,
                                            const Eigen::Vector3f &scale) {
  Eigen::Matrix4f modelMatrix = Eigen::Matrix4f::Identity();

  Eigen::Matrix4f T, S;
  T << 1, 0, 0, translation.x(), 0, 1, 0, translation.y(), 0, 0, 1,
      translation.z(), 0, 0, 0, 1;

  S << scale.x(), 0, 0, 0, 0, scale.y(), 0, 0, 0, 0, scale.z(), 0, 0, 0, 0, 1;

  return T * rotation * S * modelMatrix;
}

/**
 * @brief Calculates the view matrix for a camera.
 *
 * This function computes a 4x4 view matrix that represents the transformation
 * from world coordinates to camera coordinates based on the camera's position
 * (eye), target point (center), and up direction (up).
 *
 * @param eye The position of the camera in world space.
 * @param center The point the camera is looking at in world space.
 * @param up The up direction vector for the camera, which defines the
 * orientation.
 *
 * @return A 4x4 view matrix (Eigen::Matrix4f) representing the camera's view
 * transformation.
 */
Eigen::Matrix4f
SoftRasterizer::Tools::calculateViewMatrix(const Eigen::Vector3f &eye,
                                           const Eigen::Vector3f &center,
                                           const Eigen::Vector3f &up) {
  const Eigen::Vector3f up_normalize = up.normalized(); // up normalize
  const Eigen::Vector3f look_at =
      (eye - center).normalized(); // look at normalize
  auto right = look_at.cross(up_normalize).normalized();

  auto _up = up_normalize;
  auto _z = (-look_at);

  // Rotation Matrix On each Axis
  //  (And we could ignore matrix inverse)
  Eigen::Matrix4f R;
  R << right.x(), right.y(), right.z(), 0, _up.x(), _up.y(), _up.z(), 0, _z.x(),
      _z.y(), _z.z(), 0, 0, 0, 0, 1;

  // Transform Matrix
  Eigen::Matrix4f T;
  T << 1, 0, 0, -eye.x(), 0, 1, 0, -eye.y(), 0, 0, 1, -eye.z(), 0, 0, 0, 1;

  return R * T;
}

/**
 * @brief Calculates an orthographic projection matrix.
 *
 * This function computes a 4x4 orthographic projection matrix that maps 3D
 * world space coordinates to 2D normalized device coordinates. The orthographic
 * projection is defined by the left, right, bottom, top, zNear, and zFar
 * parameters.
 *
 * @param left The coordinate for the left clipping plane.
 * @param right The coordinate for the right clipping plane.
 * @param bottom The coordinate for the bottom clipping plane.
 * @param top The coordinate for the top clipping plane.
 * @param zNear The distance to the near clipping plane.
 * @param zFar The distance to the far clipping plane.
 *
 * @return A 4x4 orthographic projection matrix (Eigen::Matrix4f).
 */
Eigen::Matrix4f SoftRasterizer::Tools::calculateOrthoMatrix(
    float left, float right, float bottom, float top, float zNear, float zFar) {
  const Eigen::Vector3f center((left + right) / 2.0f, (top + bottom) / 2.0f,
                               (zNear + zFar) / 2.0f);

  Eigen::Matrix4f T;
  T << 1, 0, 0, -center.x(), 0, 1, 0, -center.y(), 0, 0, 1, -center.z(), 0, 0,
      0, 1;

  Eigen::Matrix4f S;
  S << 2.0f / (right - left), 0, 0, 0, 0, 2.0f / (top - bottom), 0, 0, 0, 0,
      2.0f / (zNear - zFar), 0, 0, 0, 0, 1;
  return S * T;
}

/**
 * @brief Calculates a perspective projection matrix.
 *
 * This function computes a 4x4 perspective projection matrix using the field
 * of view (fovy), aspect ratio, near clipping plane (zNear), and far clipping
 * plane (zFar). The perspective matrix is used to simulate a camera's field
 * of view, where objects closer to the camera appear larger.
 *
 * @param fovy The vertical field of view in degrees.
 * @param aspect The aspect ratio of the view (width / height).
 * @param zNear The distance to the near clipping plane.
 * @param zFar The distance to the far clipping plane.
 *
 * @return A 4x4 perspective projection matrix (Eigen::Matrix4f).
 */
Eigen::Matrix4f SoftRasterizer::Tools::calculateProjectionMatrix(float fovy,
                                                                 float aspect,
                                                                 float zNear,
                                                                 float zFar) {
  Eigen::Matrix4f projectionMatrix = Eigen::Matrix4f::Identity();

  float top = std::tan(fovy / 2.0f) * std::abs(zNear);
  float bottom = -top;

  float right = top * aspect;
  float left = -right;

  Eigen::Matrix4f Ortho =
      calculateOrthoMatrix(left, right, bottom, top, zNear, zFar);

  /*convert projection matrix to ortho matrix*/
  Eigen::Matrix4f Persp2Ortho;
  Persp2Ortho << zNear, 0, 0, 0, 0, zNear, 0, 0, 0, 0, zNear + zFar,
      -zNear * zFar, 0, 0, 1, 0;

  return Ortho * Persp2Ortho;
}
