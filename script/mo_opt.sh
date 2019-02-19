#!/bin/bash
# Intel MO config
openvino_dir=~/intel/computer_vision_sdk
deploy_dir=${openvino_dir}/deployment_tools
intel_mo_script=${deploy_dir}/model_optimizer/mo_tf.py
front_dir=${deploy_dir}/model_optimizer/extensions/front/tf
# trained config
trained_model=`pwd`
trained_dir=${trained_model}/export_models
pb_file=${trained_dir}/frozen_inference_graph.pb
json_list=`ls ${front_dir}/*.json`
a=0
for i in ${json_list};
do
	array_json[${a}]=${i}
	a=$[$a+1] 
	file=${i##/*/}
	echo "[${a}] ----- ${file}"
done
echo ${array_json[2]}
read -p "请选择模型:  " choose_num
echo ${array_json[0]}
# 选择模型文件
case ${choose_num} in
	"1")
		json=${array_json[0]}
		echo "###### 您选择的json文件为:${json##/*/}"
		;;
	"2")
		json=${array_json[1]}
		echo "###### 您选择的json文件为:${json##/*/}"
		;;
	"3")
		json=${array_json[2]$}
		echo "###### 您选择的json文件为:${json##/*/}"
		;;
	"4")
		json=${array_json[3]}
		echo "###### 您选择的json文件为:${json##/*/}"
		;;
	"5")
		json=${array_json[4]}
		echo "###### 您选择的json文件为:${json##/*/}"
		;;
	"6")
		json=${array_json[5]}
		echo "###### 您选择的json文件为:${json##/*/}"
		;;
	"7")
		json=${array_json[6]}
		echo "###### 您选择的json文件为:${json##/*/}"
		;;
	"8")
		json=${array_json[7]}
		echo "###### 您选择的json文件为:${json##/*/}"
		;;
	"9")
		json=${array_json[8]}
		echo "###### 您选择的json文件为:${json##/*/}"
		;;
	"10")
		json=${array_json[9]}
		echo "###### 您选择的json文件为:${json##/*/}"
		;;
	"11")
		json=${array_json[10]}
		echo "###### 您选择的json文件为:${json##/*/}"
		;;
	"12")
		json=${array_json[11]}
		echo "###### 您选择的json文件为:${json##/*/}"
		;;
	*)
	exit 1
	;;
	esac
config=${trained_dir}/pipeline.config
mo_model_dir=${trained_dir}/intel_mo
if [ ! -d ${mo_model_dir} ];then
    mkdir ${mo_model_dir}
fi

python3 ${intel_mo_script} --input_model ${pb_file} --output=detection_boxes,detection_scores,num_detections --tensorflow_use_custom_operations_config ${json} --tensorflow_object_detection_api_pipeline_config ${config} --output_dir ${mo_model_dir}
