#pragma once
#ifndef _TOOLS_HPP_
#define _TOOLS_HPP_
#include <Eigen/Eigen>

namespace SoftRasterizer {
struct Tools {
  static constexpr float PI = 3.14159265358979323846f;

  // degree to radian
  static float degreeToRadian(float degree);

  static Eigen::Vector4f to_vec4(const Eigen::Vector3f &v3, float w = 1.0f);

  static Eigen::Vector3f to_vec3(const Eigen::Vector4f &v4);

  /**
   * @brief Calculates the rotation matrix using Rodrigues' rotation formula.
   *
   * This function computes a 4x4 rotation matrix that represents a rotation
   * of the given angle (in degrees) around the specified axis in 3D space.
   * It utilizes Rodrigues' rotation formula to generate the matrix.
   *
   * @param axis A 3D vector representing the axis of rotation.
   * @param angle The angle (in degrees) by which to rotate around the axis.
   *        A positive value rotates counterclockwise, and a negative value
   *        rotates clockwise around the axis.
   *
   * @return A 4x4 rotation matrix (Eigen::Matrix4f) representing the rotation.
   */
  static Eigen::Matrix4f calculateRotationMatrix(const Eigen::Vector3f &axis,
                                                 float angle);

  /**
   * @brief Calculates the model matrix for a 3D object.
   *
   * This function computes a 4x4 model matrix that represents the
   * transformation of a 3D object in world space. The model matrix is obtained
   * by combining the translation, rotation, and scaling transformations.
   *
   * @param translation A 3D vector representing the translation (position) of
   * the object.
   * @param rotation A 4x4 matrix representing the rotation of the object.
   * @param scale A 3D vector representing the scaling factors along the x, y,
   * and z axes.
   *
   * @return A 4x4 model matrix (Eigen::Matrix4f) representing the combined
   * transformations.
   */
  static Eigen::Matrix4f
  calculateModelMatrix(const Eigen::Vector3f &translation,
                       const Eigen::Matrix4f &rotation,
                       const Eigen::Vector3f &scale);

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
  static Eigen::Matrix4f calculateViewMatrix(const Eigen::Vector3f &eye,
                                             const Eigen::Vector3f &center,
                                             const Eigen::Vector3f &up);

  /**
   * @brief Calculates an orthographic projection matrix.
   *
   * This function computes a 4x4 orthographic projection matrix that maps 3D
   * world space coordinates to 2D normalized device coordinates. The
   * orthographic projection is defined by the left, right, bottom, top, zNear,
   * and zFar parameters.
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
  static Eigen::Matrix4f calculateOrthoMatrix(float left, float right,
                                              float bottom, float top,
                                              float zNear, float zFar);

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
  static Eigen::Matrix4f calculateProjectionMatrix(float fovy, float aspect,
                                                   float zNear, float zFar);
};
} // namespace SoftRasterizer

#endif //_TOOLS_HPP_
