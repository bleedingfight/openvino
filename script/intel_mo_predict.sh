# OpenVINO 编译生成的ssd文件路径
ssd_bin=/home/amax/inference_engine_samples_build/intel64/Release/object_detection_sample_ssd
# mo优化生成的xml文件所在路径
mo_intel_dir=/home/amax/ssd_inception_v2_coco_2018_01_28/mo_intel
network=${mo_intel_dir}/frozen_inference_graph.xml
${ssd_bin} -i example.jpeg -m ${network} -d CPU

