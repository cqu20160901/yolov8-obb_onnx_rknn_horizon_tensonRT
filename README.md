# yolov8obb_onnx_rknn_horizon_tensonRT

yolov8obb 旋转目标检测部署，瑞芯微RKNN芯片部署、地平线Horizon芯片部署、TensorRT部署

导出 onnx 参考链接 [【yolov8obb 旋转目标检测 瑞芯微RKNN芯片部署、地平线Horizon芯片部署、TensorRT部署】](https://blog.csdn.net/zhangqian_1/article/details/139437315)

**特别说明：本示例提供的代码只适用按照对应参考导出的onnx，其它方式导出onnx自行写后处理。**

# 文件夹结构说明

yolov8obb_onnx：onnx模型、测试图像、测试结果、测试demo脚本

yolov8obb_TensorRT：TensorRT版本模型、测试图像、测试结果、测试demo脚本、onnx模型、onnx2tensorRT脚本(tensorRT-7.2.3.4)

yolov8obb_rknn：rknn模型、测试（量化）图像、测试结果、onnx2rknn转换测试脚本

yolov8obb_horizon：地平线模型、测试（量化）图像、测试结果、转换测试脚本、测试量化后onnx模型脚本

# 测试结果

python 测试结果

![test](https://github.com/cqu20160901/yolov8obb_onnx_rknn_horizon_tensonRT/assets/22290931/fc44788c-5736-4ab5-baf8-6fd04d906b82)


onnx 测试结果

![image](https://github.com/cqu20160901/yolov8obb_onnx_rknn_horizon_tensonRT/blob/main/yolov8obb_onnx/test_onnx_result.jpg)
