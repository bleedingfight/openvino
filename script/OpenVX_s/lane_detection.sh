# OpenVINO默认安装目录，如果安装的时候为个人用户安装则在当前用户的intel目录下，否则在/opt/intel下
openvino_dir=~/intel
openvx_dir=${openvino_dir}/computer_vision_sdk/openvx
lane_source=${openvx_dir}/samples/samples/lane_detection
build_dir=${lane_source}/build
if [ ! -d ${build_dir} ];then
	mkdir ${build_dir}
fi
cd ${build_dir}
echo -e "Build Source"
cmake ..
make -j4
echo -e "Build Finished,Start Lane Detection"
## 执行检测，同时讲检测输出到log文件中
./lane_detection --input ../road_lane.mp4|tee log

