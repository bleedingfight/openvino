# 目标检测
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
原始图像如下：

<p align="center">
<img src="../demo/example.jpeg" width=300 height=300>
</p>
输出结果如下：
<p align="center">
<img src="../demo/out_0.bmp" width=300 height=300>
</p>


## 优化文件(百度网盘：[下载](https://pan.baidu.com/s/1ykvqy9A7af9BVGDhJc3w5g)(提取码：6rex))
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
## ssd v2网络推理
使用tensorflow训练的v2网络需要在训练后的模型中将config.pipeline中的`train = True` 改为`train = False`之后才能生成优化的bin和xml文件。
在新的脚本下执行`bash mo_opt.sh `

```
[1] ----- faster_rcnn_support_api_v1.7.json
[2] ----- faster_rcnn_support.json
[3] ----- mask_rcnn_support_api_v1.11.json
[4] ----- mask_rcnn_support_api_v1.7.json
[5] ----- mask_rcnn_support.json
[6] ----- rfcn_support.json
[7] ----- ssd_support.json
[8] ----- ssd_toolbox_detection_output.json
[9] ----- ssd_toolbox_multihead_detection_output.json
[10] ----- ssd_v2_support.json
[11] ----- yolo_v1_v2.json
[12] ----- yolo_v3.json
/home/amax/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/tf/mask_rcnn_support_api_v1.11.json
请选择模型:  10
/home/amax/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support_api_v1.7.json
###### 您选择的json文件为:ssd_v2_support.json
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/mnt/train_chess/train_output/ssd_mobilenet_v2_coco_2018_03_29_trainoutput/export_models/frozen_inference_graph.pb
	- Path for generated IR: 	/mnt/train_chess/train_output/ssd_mobilenet_v2_coco_2018_03_29_trainoutput/export_models/intel_mo
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
	- Use configuration file used to generate the model with Object Detection API: 	/mnt/train_chess/train_output/ssd_mobilenet_v2_coco_2018_03_29_trainoutput/export_models/pipeline.config
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/home/amax/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
Model Optimizer version: 	1.4.292.6ef7232d
/home/amax/anaconda3/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /mnt/train_chess/train_output/ssd_mobilenet_v2_coco_2018_03_29_trainoutput/export_models/intel_mo/frozen_inference_graph.xml
[ SUCCESS ] BIN file: /mnt/train_chess/train_output/ssd_mobilenet_v2_coco_2018_03_29_trainoutput/export_models/intel_mo/frozen_inference_graph.bin
[ SUCCESS ] Total execution time: 19.11 seconds. 

```
输出优化结果如下：
```bash
[ INFO ] InferenceEngine: 
	API version ............ 1.4
	Build .................. 17328
Parsing input parameters
[ INFO ] Files were added: 1
[ INFO ]     /mnt/train_chess/test_image/example.png
[ INFO ] Loading plugin

	API version ............ 1.4
	Build .................. lnx_20181004
	Description ....... MKLDNNPlugin
[ INFO ] Loading network files:
	/mnt/train_chess/train_output/ssd_mobilenet_v2_coco_2018_03_29_trainoutput/export_models/intel_mo/frozen_inference_graph.xml
	/mnt/train_chess/train_output/ssd_mobilenet_v2_coco_2018_03_29_trainoutput/export_models/intel_mo/frozen_inference_graph.bin
[ INFO ] Preparing input blobs
[ INFO ] Batch size is 1
[ INFO ] Preparing output blobs
[ INFO ] Loading model to the plugin
[ WARNING ] Image is resized from (640, 480) to (300, 300)
[ INFO ] Batch size is 1
[ INFO ] Start inference (1 iterations)
[ INFO ] Processing output blobs
[0,1] element, prob = 0.999961    (210.554,74.3187)-(253.47,118.206) batch id : 0 WILL BE PRINTED!
[1,1] element, prob = 0.999956    (255.702,317.677)-(301.527,363.577) batch id : 0 WILL BE PRINTED!
[2,1] element, prob = 0.999924    (463.739,223.401)-(506.011,267.229) batch id : 0 WILL BE PRINTED!
[3,1] element, prob = 0.999821    (257.218,226.447)-(301.906,269.522) batch id : 0 WILL BE PRINTED!
[4,1] element, prob = 0.999802    (261.671,119.751)-(305.027,164.275) batch id : 0 WILL BE PRINTED!
[5,1] element, prob = 0.999779    (467.137,329.944)-(509.943,372.557) batch id : 0 WILL BE PRINTED!
[6,1] element, prob = 0.999667    (261.124,21.5977)-(305.101,66.5238) batch id : 0 WILL BE PRINTED!
[7,1] element, prob = 0.999615    (122.574,7.85445)-(165.675,55.6883) batch id : 0 WILL BE PRINTED!
[8,1] element, prob = 0.999389    (106.529,423.158)-(152.407,469.356) batch id : 0 WILL BE PRINTED!
[9,1] element, prob = 0.999389    (409.77,224.246)-(450.765,268.325) batch id : 0 WILL BE PRINTED!
[10,1] element, prob = 0.999299    (570.467,226.986)-(614.907,270.094) batch id : 0 WILL BE PRINTED!
[11,1] element, prob = 0.998824    (570.509,162.438)-(615.206,205.074) batch id : 0 WILL BE PRINTED!
[12,1] element, prob = 0.998801    (413.24,114.065)-(456.125,158.549) batch id : 0 WILL BE PRINTED!
[13,1] element, prob = 0.996859    (108.133,167.935)-(154.817,209.461) batch id : 0 WILL BE PRINTED!
[14,1] element, prob = 0.995425    (205.089,378.243)-(252.144,427.253) batch id : 0 WILL BE PRINTED!
[15,1] element, prob = 0.995348    (563.026,69.8918)-(609.705,114.824) batch id : 0 WILL BE PRINTED!
[16,1] element, prob = 0.99208    (107.174,371.902)-(155.295,414.948) batch id : 0 WILL BE PRINTED!
[17,1] element, prob = 0.984267    (252.578,428.485)-(296.941,472.183) batch id : 0 WILL BE PRINTED!
[18,1] element, prob = 0.977793    (103.667,277.236)-(152.858,318.794) batch id : 0 WILL BE PRINTED!
[19,1] element, prob = 0.795092    (517.955,372.259)-(563.981,419.081) batch id : 0 WILL BE PRINTED!
[20,1] element, prob = 0.774198    (114.599,116.513)-(158.794,161.626) batch id : 0 WILL BE PRINTED!
[21,1] element, prob = 0.695678    (413.454,434.859)-(460.336,477.795) batch id : 0 WILL BE PRINTED!
[22,1] element, prob = 0.495147    (107.636,321.207)-(156.872,364.036) batch id : 0
[23,1] element, prob = 0.451679    (109.334,224.972)-(158.214,270.294) batch id : 0
[ INFO ] Image out_0.bmp created!

total inference time: 10.2547
Average running time of one iteration: 10.2547 ms

Throughput: 97.5167 FPS

[ INFO ] Execution successful

```
