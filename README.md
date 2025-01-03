# Software-Rasterizer
## 0x00 Description

This project is a simple, custom software rasterizer built using modern C++ features and libraries like Eigen and OpenCV. The primary goal is to implement a basic rendering pipeline that can draw simple geometric shapes (like triangles) with basic shading and color. It leverages a custom rasterization class, `SoftRasterizer::RenderingPipeline`, to load vertices, colors, and face indices and render them onto a 2D plane using OpenCV for display.

The system is designed to be easily extendable and can serve as the foundation for more complex 3D rendering techniques, such as texture mapping, lighting models, and advanced shading algorithms. The rendering process is handled by clearing buffers and displaying geometric primitives in a continuous loop, allowing for real-time interaction.

This project is ideal for developers looking to learn more about computer graphics, rasterization, and the underlying workings of 3D rendering engines.



## 0x01 Example

### Set Software Resolution

```C++
SoftRasterizer::RenderingPipeline render(800, 600);
```



### Setup Vertices/Colour/Indices(faces)

~~i'm going to implement .obj file reader later on! ^_^~~

```c++
  SoftRasterizer::RenderingPipeline render(1000, 1000);

  /*set up all vertices*/
  std::vector<Eigen::Vector3f> pos{
      {0.f, 0.5f, 0.f}, {-0.5f, -0.5f, 0.f}, {0.5f, -0.5f, 0.f}};

  /*set up all shading param(colours)*/
  std::vector<Eigen::Vector3f> color{
      {1.0f, 0.f, 0.f}, {0.f, 1.0f, 0.f}, {0.f, 0.f, 1.0f}};

  /*set up all indices(faces)*/
  std::vector<Eigen::Vector3i> ind{{0, 1, 2}};
  render.loadVertices(pos);
  render.loadColours(color);
  render.loadIndices(ind);
```



### Software-Rasterizer Usage

#### How to Control Angles

1.  `A/a `On Keyboard: counterclockwise
2. `D/d` On Keyboard: clockwise

```c++
if (key == 'a' || key =='A') {
      degree -= 10.0f;
}
else if (key == 'd' || key == 'D') {
      degree += 10.0f;
}
```



#### Draw LINES

```c++
render.display(SoftRasterizer::Primitive::LINES);
```

![2025-01-02 20-19-33](./assets/lines.gif)



#### Draw Triangles

```c++
render.display(SoftRasterizer::Primitive::TRIANGLES);
```

![triangles](./assets/triangles.gif)



#### Z-buffer Support(Still Testing)

Construct two triangles with overlapping relationships and set their Z-buffer values.

```c++
std::vector<Eigen::Vector3f> pos{
    {0.f, 0.5f, -0.1f}, {-0.5f, -0.5f, -0.1f}, {0.5f, -0.5f, -0.1f},
    {0.f, 0.5f, 0.1f}, {0.7f, 0.5f, 0.1f}, {0.7f, -0.3f, 0.1f}
};

/*set up all shading param(colours)*/
std::vector<Eigen::Vector3f> color{
    {1.0f, 0.f, 0.f}, {0.f, 1.0f, 0.f}, {0.f, 0.f, 1.0f},
    {1.0f, 1.f, 1.f}, {1.f, 1.0f, 1.f}, {1.f, 1.f, 1.0f}
};

/*set up all indices(faces)*/
std::vector<Eigen::Vector3i> ind{ {0, 1, 2},{3, 4, 1} };
```

![z-buffer](./assets/z-buffer.gif)



## 0x02 Developer Quick Start

### Platform Support
Windows, Linux, MacOS(Intel and M Serious Chip)
### Prerequisites
- You have to set OpenCV_DIR by system path variable or cmake variable before building Software-Rasterizer
``` cmake
set(OpenCV_DIR "path/to/opencv")
```



### Building Software-Rasterizer

``` bash
git clone https://github.com/Liupeter01/Software-Rasterizer
cd Software-Rasterizer
git submodule update --init
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel x
```



## 0x03 Reference

### Bresenham algorithm

[drawing-lines-with-bresenhams-line-algorithm](https://stackoverflow.com/questions/10060046/drawing-lines-with-bresenhams-line-algorithm)
