#include "glm/fwd.hpp"
#include "object/Material.hpp"
#include <light/SphereLight.hpp>
#include <memory>
#include <object/Sphere.hpp>
#include <opencv2/opencv.hpp>
#include <render/PathTracing.hpp>
#include <render/Rasterizer.hpp>
#include <render/RayTracing.hpp>
#include <scene/Scene.hpp>

int main() {
  int key = 0;
  float degree = 0.0f;

  // Create Ray Tracing Main Class
  auto render = std::make_shared<SoftRasterizer::RayTracing>(1024, 1024, 1);

  // Create A Scene
  auto scene = std::make_shared<SoftRasterizer::Scene>(
      "TestScene",
      /*eye=*/glm::vec3(0.0f, 0.0f, -0.9f),
      /*center=*/glm::vec3(0.0f, 0.0f, 0.0f),
      /*up=*/glm::vec3(0.0f, 1.0f, 0.0f),
      /*background color*/ glm::vec3(0.235294, 0.67451, 0.843137));

  /*Modify spot's Material Properties*/
  std::shared_ptr<SoftRasterizer::Material> crate =
      std::make_shared<SoftRasterizer::Material>();
  std::shared_ptr<SoftRasterizer::Material> spot =
      std::make_shared<SoftRasterizer::Material>();
  std::shared_ptr<SoftRasterizer::Material> diffuse =
      std::make_shared<SoftRasterizer::Material>();
  std::shared_ptr<SoftRasterizer::Material> light =
      std::make_shared<SoftRasterizer::Material>();
  std::shared_ptr<SoftRasterizer::Material> reflectrefract =
      std::make_shared<SoftRasterizer::Material>();

  diffuse->type = crate->type = spot->type =
      SoftRasterizer::MaterialType::DIFFUSE_AND_GLOSSY;
  diffuse->Ka = crate->Ka = spot->Ka = glm::vec3(0.005f);
  diffuse->Kd = crate->Kd = spot->Kd = glm::vec3(1.f);
  diffuse->Ks = crate->Ks = spot->Ks = glm::vec3(0.7937f);
  crate->specularExponent = 150.f; // no specular
  diffuse->specularExponent = spot->specularExponent = 150.f;

  /*Only self-illumination object gets emission*/
  light->type = SoftRasterizer::MaterialType::DIFFUSE_AND_GLOSSY;
  light->Kd = glm::vec3(1.0f);
  light->emission = glm::vec3(2.f); // and also intensity of the light

  /*Set REFLECTION_AND_REFRACTION Material*/
  reflectrefract->type =
      SoftRasterizer::MaterialType::REFLECTION_AND_REFRACTION;
  reflectrefract->ior = 2.0f; /*Air to Glass*/

  /*Set Diffuse Color*/
  auto diffuse_sphere = std::make_unique<SoftRasterizer::Sphere>(
      /*center=*/glm::vec3(-0.07f, 0.0f, 0.f),
      /*radius=*/0.1f);

  /*Set Refrflect Sphere Object*/
  auto refrflect_sphere = std::make_unique<SoftRasterizer::Sphere>(
      /*center=*/glm::vec3(-0.05f, 0.01f, 0.f),
      /*radius=*/0.1f);

  /*Add Light To Scene*/
  auto spherelight = std::make_unique<SoftRasterizer::SphereLight>(
      /*pos=*/glm::vec3(-0, 0.1, -0.5));

  scene->addGraphicObj(std::move(refrflect_sphere), "refrflect");
  scene->addGraphicObj(std::move(diffuse_sphere), "diffuse");
  scene->addGraphicObj(std::move(spherelight), "spherelight");

  /*Add a spot object*/
  scene->addGraphicObj(
      CONFIG_HOME "examples/models/spot/spot_triangulated_good.obj", "spot",
      glm::vec3(0, 1, 0), 0.f, glm::vec3(0.f), glm::vec3(0.3f));
  scene->addGraphicObj(CONFIG_HOME "examples/models/Crate/Crate1.obj", "Crate",
                       glm::vec3(0.f, 1.f, 0.f), 0.f, glm::vec3(0.0f),
                       glm::vec3(0.2f));

  scene->startLoadingMesh("spot");
  scene->startLoadingMesh("Crate");

  if (auto spotOpt = scene->getMeshObj("spot"); spotOpt)
    (*spotOpt)->setMaterial(spot);
  if (auto CrateOpt = scene->getMeshObj("Crate"); CrateOpt)
    (*CrateOpt)->setMaterial(crate);
  if (auto refrflectOpt = scene->getMeshObj("refrflect"); refrflectOpt)
    (*refrflectOpt)->setMaterial(reflectrefract);
  if (auto diffuseOpt = scene->getMeshObj("diffuse"); diffuseOpt)
    (*diffuseOpt)->setMaterial(diffuse);
  if (auto sperelightOpt = scene->getMeshObj("spherelight"); sperelightOpt)
    (*sperelightOpt)->setMaterial(light);

  /*Add a texture shader for spot object!*/
  scene->addShader("spot_shader",
                   CONFIG_HOME "examples/models/spot/spot_texture.png",
                   SoftRasterizer::SHADERS_TYPE::TEXTURE);
  scene->addShader("crate_shader",
                   CONFIG_HOME "examples/models/Crate/Crate1.png",
                   SoftRasterizer::SHADERS_TYPE::TEXTURE);

  scene->bindShader2Mesh("spot", "spot_shader");
  scene->bindShader2Mesh("Crate", "crate_shader");

  /*Register Scene To Render Main Frame*/
  render->addScene(scene);

  while (key != 27) {

    /*clear both shading and depth!*/
    render->clear(SoftRasterizer::Buffers::Color |
                  SoftRasterizer::Buffers::Depth);

    /*Model Matrix*/
    scene->setModelMatrix(
        "spot",
        /*axis=*/glm::vec3(0.f, 1.f, 0.f),
        /*degree=+ for Counterclockwise;- for Clockwise*/ degree,
        /*transform=*/glm::vec3(0.25f, 0.1f, 0.20f),
        /*scale=*/glm::vec3(0.2f));

    scene->setModelMatrix(
        "Crate",
        /*axis=*/glm::vec3(0.f, 1.f, 0.f),
        /*degree=+ for Counterclockwise;- for Clockwise*/ degree,
        /*transform=*/glm::vec3(0.25f, -0.13f, 0.15f),
        /*scale=*/glm::vec3(0.1f));

    scene->setModelMatrix("refrflect",
                          /*axis=*/glm::vec3(0.f, 1.f, 0.f),
                          /*degree=+ for Counterclockwise;- for Clockwise*/ 0,
                          /*transform=*/glm::vec3(0.1f, 0.0f, 0.0f),
                          /*scale=*/glm::vec3(1.0f));

    scene->setModelMatrix("diffuse",
                          /*axis=*/glm::vec3(0.f, 1.f, 0.f),
                          /*degree=+ for Counterclockwise;- for Clockwise*/ 0,
                          /*transform=*/glm::vec3(-0.25f, 0.1f, 0.5f),
                          /*scale=*/glm::vec3(1.0f));

    scene->setModelMatrix("spherelight", glm::vec3(0, 1, 0), 0,
                          glm::vec3(-0.0, 0.3, -0.3f), glm::vec3(1.0f));

    /*View Matrix*/
    scene->setViewMatrix(
        /*eye=*/glm::vec3(0.0f, 0.0f, -0.9f),
        /*center=*/glm::vec3(0.0f, 0.0f, 0.0f),
        /*up=*/glm::vec3(0.0f, 1.0f, 0.0f));

    /*Projection Matrix*/
    scene->setProjectionMatrix(
        /*fov=*/45.0f,
        /*near=*/0.1f,
        /*far=*/100.0f);

    render->display(SoftRasterizer::Primitive::TRIANGLES);

    key = cv::waitKey(0);
    if (key == 'a' || key == 'A') {
      degree += 10.0f;
    } else if (key == 'd' || key == 'D') {
      degree -= 10.0f;
    }

    /*reset the degree*/
    auto delta = degree - 360.f;
    if (delta >= -0.00000001f && delta <= 0.00000001f) {
      degree = 0.0f;
    }
  }
  return 0;
}
