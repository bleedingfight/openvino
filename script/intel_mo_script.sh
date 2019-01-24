# OpenVINO转换脚本所在位置
mo_script=/home/amax/intel/computer_vision_sdk_2018.4.420/deployment_tools/model_optimizer/mo_tf.py
# 配置文件所在路径
json_dir=/home/amax/intel/computer_vision_sdk_2018.4.420/deployment_tools/model_optimizer/extensions/front/tf
# 需要转化的训练模型目录
model_dir=/home/amax/ssd_inception_v2_coco_2018_01_28
# 训练好的模型的冻结文件
pb_file=${model_dir}/frozen_inference_graph.pb
# 配置文件
json_file=${json_dir}/ssd_v2_support.json
# pipeline文件
config=${model_dir}/pipeline.config
output_dir=${model_dir}/mo_intel
if [ ! -d "${mo_intel}" ];then
	mkdir ${output_dir}
else
	echo "文件夹已经存在"
fi
python3 ${mo_script} --input_model ${pb_file} --output=detection_boxes,detection_scores,num_detections --tensorflow_use_custom_operations_config ${json_file} --tensorflow_object_detection_api_pipeline_config ${config} --output_dir ${output_dir}
#python3 /home/amax/intel/computer_vision_sdk_2018.4.420/deployment_tools/model_optimizer/mo_tf.py --input_model /home/amax/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --output=detection_boxes,detection_scores,num_detections --tensorflow_use_custom_operations_config /home/amax/intel/computer_vision_sdk_2018.4.420/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /home/amax/ssd_inception_v2_coco_2018_01_28/pipeline.config --output_dir mo_intel
