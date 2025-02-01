#include <object/Sphere.hpp>
#include <opencv2/opencv.hpp>
#include <render/Rasterizer.hpp>
#include <render/RayTracing.hpp>
#include <scene/Scene.hpp>

int main() {
  int key = 0;
  float degree = 0.0f;

  // Create Ray Tracing Main Class
  auto render = std::make_shared<SoftRasterizer::RayTracing>(1024, 1024);

  // Create A Scene
  auto scene = std::make_shared<SoftRasterizer::Scene>(
      "TestScene",
      /*eye=*/glm::vec3(0.0f, 0.3f, -0.9f),
      /*center=*/glm::vec3(0.0f, 0.0f, 0.0f),
      /*up=*/glm::vec3(0.0f, 1.0f, 0.0f),
      /*background color*/ glm::vec3(0.235294, 0.67451, 0.843137));

  /*Set Diffuse Color*/
 auto diffuse_sphere = std::make_unique<SoftRasterizer::Sphere>(
          /*center=*/glm::vec3(-0.07f, 0.0f, 0.f),
          /*radius=*/0.1f);

 diffuse_sphere->getMaterial()->type =
           SoftRasterizer::MaterialType::DIFFUSE_AND_GLOSSY;
 diffuse_sphere->getMaterial()->color = glm::vec3(0.6f, 0.7f, 0.8f);
 diffuse_sphere->getMaterial()->Kd = glm::vec3(0.6f, 0.7f, 0.8f);
 diffuse_sphere->getMaterial()->Ka = glm::vec3(0.105f);
 diffuse_sphere->getMaterial()->Ks = glm::vec3(0.7937f);
 diffuse_sphere->getMaterial()->specularExponent = 150.f;

   /*Set Reflect Sphere Object*/
   auto reflect_sphere = std::make_unique<SoftRasterizer::Sphere>(
             /*center=*/glm::vec3(-0.05f, 0.01f, 0.f),
             /*radius=*/0.1f);

   /*Set REFLECTION_AND_REFRACTION Material*/
   reflect_sphere->getMaterial()->type = 
             SoftRasterizer::MaterialType::REFLECTION_AND_REFRACTION;
   reflect_sphere->getMaterial()->ior = 1.49f; /*Air to Glass*/

   scene->addGraphicObj(std::move(reflect_sphere), "reflect");
  scene->addGraphicObj(std::move(diffuse_sphere), "diffuse");

  scene->addGraphicObj(CONFIG_HOME "examples/models/bunny/bunny.obj", "bunny",
                       glm::vec3(0, 1, 0), 0.f, glm::vec3(0.f), glm::vec3(1.f));

  scene->startLoadingMesh("bunny");

  /*Modify Bunny's Material Properties*/
  auto bunny_obj = scene->getMeshObj("bunny");
  bunny_obj.value()->getMaterial()->type = 
            SoftRasterizer::MaterialType::DIFFUSE_AND_GLOSSY;
  bunny_obj.value()->getMaterial()->color = glm::vec3(1.0f);
  bunny_obj.value()->getMaterial()->Kd = glm::vec3(1.0f);
  bunny_obj.value()->getMaterial()->Ka = glm::vec3(0.015f);
  bunny_obj.value()->getMaterial()->Ks = glm::vec3(0.7937f);
  bunny_obj.value()->getMaterial()->specularExponent = 150.f;

  /*Add Light To Scene*/
  auto light1 = std::make_shared<SoftRasterizer::light_struct>();
  light1->position = glm::vec3{0.5f, -0.4f, -0.9f};
  light1->intensity = glm::vec3{1, 1, 1};

  auto light2 = std::make_shared<SoftRasterizer::light_struct>();
  light2->position = glm::vec3{-0.5f, -0.4f, -0.9f};
  light2->intensity = glm::vec3{1, 1, 1};

  scene->addLight("Light1", light1);
  scene->addLight("Light2", light2);

  /*Register Scene To Render Main Frame*/
  render->addScene(scene);

  while (key != 27) {

    /*clear both shading and depth!*/
    render->clear(SoftRasterizer::Buffers::Color |
                  SoftRasterizer::Buffers::Depth);

    /*Model Matrix*/
    scene->setModelMatrix(
        "bunny",
        /*axis=*/glm::vec3(0.f, 1.f, 0.f),
        /*degree=+ for Counterclockwise;- for Clockwise*/ degree,
        /*transform=*/glm::vec3(0.1f, -0.1f, 0.f),
        /*scale=*/glm::vec3(1.f));

    scene->setModelMatrix(
              "reflect",
              /*axis=*/glm::vec3(0.f, 1.f, 0.f),
              /*degree=+ for Counterclockwise;- for Clockwise*/ 0,
              /*transform=*/glm::vec3(0.05f, -0.02f, -0.2f),
              /*scale=*/glm::vec3(1.f));

    /*View Matrix*/
    scene->setViewMatrix(
        /*eye=*/glm::vec3(0.0f, 0.0f, -0.5f),
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
    } 
    else if (key == 'd' || key == 'D') {
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
