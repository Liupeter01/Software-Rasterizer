#define TINYOBJLOADER_IMPLEMENTATION
#include <ObjLoader.hpp>
#include <Tools.hpp>
#include <iostream>
#include <spdlog/spdlog.h>
#include <tiny_obj_loader.h>
#include <unordered_map>

SoftRasterizer::ObjLoader::ObjLoader(const std::string &path,
                                     const std::string &meshName,
                                     Eigen::Matrix4f &&model)
    : m_path(path), m_meshName(meshName), m_model(std::move(model)) {}

SoftRasterizer::ObjLoader::ObjLoader(const std::string &path,
                                     const std::string &meshName,
                                     const Eigen::Matrix4f &rotation,
                                     const Eigen::Vector3f &translation,
                                     const Eigen::Vector3f &scale)
    : ObjLoader(path, meshName,
                Tools::calculateModelMatrix(translation, rotation, scale)) {}

SoftRasterizer::ObjLoader::ObjLoader(const std::string &path,
                                     const std::string &meshName,
                                     const Eigen::Vector3f &axis,
                                     const float angle,
                                     const Eigen::Vector3f &translation,
                                     const Eigen::Vector3f &scale)
    : ObjLoader(path, meshName,
                Tools::calculateModelMatrix(
                    translation, Tools::calculateRotationMatrix(axis, angle),
                    scale)) {}

void SoftRasterizer::ObjLoader::setObjFilePath(const std::string &path) {
  m_path = path;
}

static SoftRasterizer::Material
processMatrial(const std::vector<tinyobj::material_t> &_material) {

  SoftRasterizer::Material m{};

  for (std::size_t i = 0; i < _material.size(); ++i) {
    tinyobj::material_t material = _material[i];

    m.name = material.name;

    // Ambient Texture Map
    m.Ka = Eigen::Vector3f(material.ambient[0], material.ambient[1],
                           material.ambient[2]);

    //  Diffuse Texture Map
    m.Kd = Eigen::Vector3f(material.diffuse[0], material.diffuse[1],
                           material.diffuse[2]);

    // Specular Color
    m.Ks = Eigen::Vector3f(material.specular[0], material.specular[1],
                           material.specular[2]);

    m.illum = material.illum;
    m.d = material.dissolve;

    m.map_bump = material.bump_texname;
    m.map_d = material.alpha_texname;
    m.map_Ka = material.ambient_texname;
    m.map_Kd = material.diffuse_texname;
    m.map_Ks = material.specular_texname;
    m.map_Ns = material.specular_highlight_texname;
  }
  return m;
}

/*start processing with obj file and handle missing normal*/
static std::unique_ptr<SoftRasterizer::Mesh>
processingVertexData(const std::string &objName,
                     const tinyobj::attrib_t &attrib,
                     const std::vector<tinyobj::shape_t> &shapes,
                     const std::vector<tinyobj::material_t> &materials) {

  bool noNormal = true;
  std::string meshname;
  std::vector<uint32_t> indices;

  /*which is going to export to other function*/
  std::vector<SoftRasterizer::Vertex> vertices;
  std::vector<Eigen::Vector3i> faces;

  // handle Vertex deduplication
  std::unordered_map<SoftRasterizer::Vertex, uint32_t,
                     std::hash<SoftRasterizer::Vertex>>
      uniqueVertices = {};

  // Loop over shapes
  for (std::size_t s = 0; s < shapes.size(); s++) {
    meshname = shapes[s].name;

    spdlog::info("\n - Read Original Data From Wavefront Format Obj - \n"
                 "\t| Shape[{0}].name = {1}\n"
                 "\t| Size of shape[{0}] vertices: {2}\n"
                 "\t| Size of shape[{0}].mesh.indices: {3}\n"
                 "\t| Size of shape[{0}].mesh.num_faces: {4}",
                 s, meshname.empty() ? "unknown" : meshname,
                 attrib.vertices.size() / 3, shapes[s].mesh.indices.size(),
                 shapes[s].mesh.num_face_vertices.size());

    faces.resize(shapes[s].mesh.num_face_vertices.size());

    for (const auto &idx : shapes[s].mesh.indices) {
      SoftRasterizer::Vertex vertex;

      vertex.position =
          Eigen::Vector3f(attrib.vertices[3 * size_t(idx.vertex_index) + 0],
                          attrib.vertices[3 * size_t(idx.vertex_index) + 1],
                          attrib.vertices[3 * size_t(idx.vertex_index) + 2]);

      vertex.color =
          Eigen::Vector3f(attrib.colors[3 * size_t(idx.vertex_index) + 0],
                          attrib.colors[3 * size_t(idx.vertex_index) + 1],
                          attrib.colors[3 * size_t(idx.vertex_index) + 2]);

      // Check if `normal_index` is zero or positive. negative = no normal data
      if (idx.normal_index >= 0) {
        /*normal exist*/
        noNormal = false;

        vertex.normal =
            Eigen::Vector3f(attrib.normals[3 * size_t(idx.normal_index) + 0],
                            attrib.normals[3 * size_t(idx.normal_index) + 1],
                            attrib.normals[3 * size_t(idx.normal_index) + 2]);
      }

      // Check if `texcoord_index` is zero or positive. negative = no texcoord
      // data
      if (idx.texcoord_index >= 0) {
        vertex.texCoord = Eigen::Vector2f(
            attrib.texcoords[2 * size_t(idx.texcoord_index) + 0],
            // attrib.texcoords[2 * size_t(idx.texcoord_index) + 1]
            1.0f - attrib.texcoords[2 * size_t(idx.texcoord_index) + 1]);
      }

      if (uniqueVertices.count(vertex) == 0) {
        uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
        vertices.push_back(vertex);
      }
      indices.push_back(uniqueVertices[vertex]);
    }
  }
  spdlog::info("Default Vertex Normal {}!", noNormal ? "Not Exist" : "Exist");
  spdlog::info("Start to Transform Indices to Vertices and Calculating Normal "
               "When it's Not Exist");

  for (std::size_t i = 0; i < indices.size() / 3; i++) {
    uint32_t a_pos = indices[3 * i + 0];
    uint32_t b_pos = indices[3 * i + 1];
    uint32_t c_pos = indices[3 * i + 2];

    auto &A = vertices[a_pos];
    auto &B = vertices[b_pos];
    auto &C = vertices[c_pos];

    faces[i] = Eigen::Vector3i(a_pos, b_pos, c_pos);

    /*no normal found*/
    if (noNormal) {
      A.normal = SoftRasterizer::Tools::calculateNormalWithWeight(
          A.position, B.position, C.position);
      B.normal = SoftRasterizer::Tools::calculateNormalWithWeight(
          B.position, C.position, A.position);
      C.normal = SoftRasterizer::Tools::calculateNormalWithWeight(
          C.position, A.position, B.position);
    }
  }

  auto material = processMatrial(materials);

  return std::make_unique<SoftRasterizer::Mesh>(
      objName.empty() ? meshname : objName, material, std::move(vertices),
      std::move(faces));
}

std::optional<std::unique_ptr<SoftRasterizer::Mesh>>
SoftRasterizer::ObjLoader::startLoadingFromFile(const std::string &objName) {

  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn;
  std::string err;

  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                              m_path.c_str());

  if (!warn.empty()) {
    spdlog::warn("[TinyObjReader]: Warning {}", warn);
  }

  if (!err.empty()) {
    spdlog::error("[TinyObjReader]: Error Occured! {}", err);
    throw std::runtime_error("LoadObj Error");
  }

  if (!ret) {
    return std::nullopt;
  }

  std::unique_ptr<SoftRasterizer::Mesh> mesh =
      processingVertexData(objName, attrib, shapes, materials);

  spdlog::info("\n - After Transform to Customerlize Format - \n"
               "\t| Mesh Name {}\n"
               "\t| Size of vertices: {}\n"
               "\t| Size of Faces: {}",
               mesh->meshname, mesh->vertices.size(), mesh->faces.size());

  /*convert tiny obj loader format to customalized format*/
  return mesh;
}

SoftRasterizer::ObjLoader::~ObjLoader() {}

void SoftRasterizer::ObjLoader::updateModelMatrix(
    const Eigen::Matrix4f &model) {
  m_model = model;
}

const Eigen::Matrix4f &SoftRasterizer::ObjLoader::getModelMatrix() {
  return m_model;
}
