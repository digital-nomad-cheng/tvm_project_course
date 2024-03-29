# tvm_project_course
Code for the project course on tvm. The goal of the course is to get familiar with tvm first. Then try to implement a new optimization for nvidia gpu using it and do some benchmarks from time, power consumption, and memory footprint perspectives. 
For detailed information about what I have done, refer to the [report](https://github.com/digital-nomad-cheng/tvm_project_course/blob/main/Project_Course_on_Apache_TVM_final_report.pdf).
## Organizations
+ byoc: Bring Your Own Codegen study, where I successfully use it to offload computation to [ncnn](https://github.com/Tencent/ncnn), a lightweight neural network inference engine designed for mobile devices.
  You can check the implementation [**here**](https://github.com/digital-nomad-cheng/tvm)
+ cmake: my cmake file backup for build tvm with various support
+ codegen: codegen stuff
+ cu_prog: cuda programs
+ docker: dockerfile and scripts for setting up development environment
+ ncnn: demo showing the usage of ncnn for later BYOC intergrating
+ schedule: benchmark different strategies in tvm for optimizing mobilenet_v2 and yolov8
+ relay: relay related learning materials
+ howto: scripts from official docs
+ tvmcon2023: scripts from tvm conference 2023
