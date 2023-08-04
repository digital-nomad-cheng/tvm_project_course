# launch docker container with gpu support
# docker run -it --gpus all apache/tvm:v0.12_cuda

# ...
sudo docker run -it --gpus all --cap-add=SYS_ADMIN --privileged \
	-v /home/vincent/Work/tvm_project_course/docker/mnt:/home/work/tvm_project_course/relay/mnt \
	apache/tvm:v0.13.0_debug
