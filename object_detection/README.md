# openvino
## 计算机环境
- Linux amax-Super-Server 4.13.0-32-generic #35~16.04.1-Ubuntu SMP Thu Jan 25 10:13:43 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux
## 使用tensorflow预训练目标检测网络优化(假设OpenVINO环境安装正常)
- 下载tensorflow预训练目标检测模型[ssd_mobilenet_v2](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
- 执行优化脚本`bash script/intel_mo_script.sh`,脚本内容如下：
```bash
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

```
输出如下：
```
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/home/amax/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb
	- Path for generated IR: 	/home/amax/ssd_inception_v2_coco_2018_01_28/mo_intel
	- IR output name: 	frozen_inference_graph
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	detection_boxes,detection_scores,num_detections
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP32
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	False
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Offload unsupported operations: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	/home/amax/ssd_inception_v2_coco_2018_01_28/pipeline.config
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/home/amax/intel/computer_vision_sdk_2018.4.420/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
Model Optimizer version: 	1.4.292.6ef7232d
/home/amax/anaconda3/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /home/amax/ssd_inception_v2_coco_2018_01_28/mo_intel/frozen_inference_graph.xml
[ SUCCESS ] BIN file: /home/amax/ssd_inception_v2_coco_2018_01_28/mo_intel/frozen_inference_graph.bin
[ SUCCESS ] Total execution time: 15.49 seconds. 

```
- 执行预测脚本`bash scrip/intel_mo_predict.sh`预测结果,脚本内容如下：
```bash
# OpenVINO 编译生成的ssd文件路径
ssd_bin=/home/amax/inference_engine_samples_build/intel64/Release/object_detection_sample_ssd
# mo优化生成的xml文件所在路径
mo_intel_dir=/home/amax/ssd_inception_v2_coco_2018_01_28/mo_intel
network=${mo_intel_dir}/frozen_inference_graph.xml
${ssd_bin} -i example.jpeg -m ${network} -d CPU

```

输出结果如下：

```
[ INFO ] InferenceEngine: 
	API version ............ 1.4
	Build .................. 17328
Parsing input parameters
[ INFO ] Files were added: 1
[ INFO ]     example.jpeg
[ INFO ] Loading plugin

	API version ............ 1.4
	Build .................. lnx_20181004
	Description ....... MKLDNNPlugin
[ INFO ] Loading network files:
	/home/amax/ssd_inception_v2_coco_2018_01_28/mo_intel/frozen_inference_graph.xml
	/home/amax/ssd_inception_v2_coco_2018_01_28/mo_intel/frozen_inference_graph.bin
[ INFO ] Preparing input blobs
[ INFO ] Batch size is 1
[ INFO ] Preparing output blobs
[ INFO ] Loading model to the plugin
[ WARNING ] Image is resized from (640, 747) to (300, 300)
[ INFO ] Batch size is 1
[ INFO ] Start inference (1 iterations)
[ INFO ] Processing output blobs
[0,1] element, prob = 0.913663    (27.3395,3.8907)-(640,743.81) batch id : 0 WILL BE PRINTED!
[ INFO ] Image out_0.bmp created!

total inference time: 30.407
Average running time of one iteration: 30.407 ms

Throughput: 32.8871 FPS

[ INFO ] Execution successful

```
输出结果如下：

![原始图像](../demo/example.jpeg)

![输出图像](../demo/out_0.bmp)

## 优化文件(百度网盘：[下载](提取码：))
文件结构如下：
```
├── checkpoint
├── frozen_inference_graph.pb
├── model.ckpt.data-00000-of-00001
├── model.ckpt.index
├── model.ckpt.meta
├── mo_intel
│   ├── frozen_inference_graph.bin
│   ├── frozen_inference_graph.mapping
│   └── frozen_inference_graph.xml
├── pipeline.config
├── saved_model
│   ├── saved_model.pb
│   └── variables
└── script
    ├── example.jpeg
    ├── intel_mo_predict.sh
    ├── intel_mo_script.sh
    └── out_0.bmp
```
