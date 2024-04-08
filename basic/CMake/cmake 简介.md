# cmake 简介

存在多种make工具（不同平台、不同应用），cmake被设计出来实现”write once, run everywhere”

基础流程：

1. 写 CMake  的配置文件 `CMakeLists.txt`

2. 执行命令 

    ```
    cmake PATH
    ```

      或 

    ```
    ccmake PATH
    ```

      生成 

    ```
    Makefile
    ```

    1. PATH 是 CMakeLists.txt 所在的目录
    2. ccmake 比 cmake 多交互式页面

3. 使用 `make` 命令进行编译