import onnxruntime as ort
import numpy as np
import cv2
import os
import sys 

# 拷贝来自 mimicmotion/dwpose/onnxdet.py
# img 原图
# input_size 推理输入尺寸
def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

# 拷贝来自 mimicmotion/dwpose/onnxdet.py
# outputs yolox输出(去掉Batch维度) 
# img_size 推理输入大小
def demo_postprocess(outputs, img_size, p6=False):

    # YOLOX 模型  anchor-free 目标检测框架
    # 将网络的输出从相对于网格单元的位置和大小（偏移量）转换为相对于输入(推理)图像的实际位置和大小
    # 在 YOLOX 中，特征图上的每一个点（或网格单元）都会对应一个预测框。特征图的每个点表示输入图像上的一个区域。这个区域的大小由特征图的步幅决定。
    
    
    # 在 YOLOX 中，如果没有启用 p6 模式，通常会使用三个尺度的特征图来生成预测框
    # 使用三个不同的步幅（8、16、32），特征图的尺寸如下：
    # 步幅 8：生成的特征图尺寸为 640/8 = 80x80。
    # 步幅 16：生成的特征图尺寸为 640/16 = 40x40。
    # 步幅 32：生成的特征图尺寸为 640/32 = 20x20

    # 每个特征图的 "每个位置" 都会生成"一个预测框", 每个特征图上的预测框数:
    # 步幅 8 的特征图：80x80 = 6400 个位置
    # 步幅 16 的特征图：40x40 = 1600 个位置
    # 步幅 32 的特征图：20x20 = 400 个位置

    # 总的预测框数是所有特征图上的预测框数量之和 6400 + 1600 + 400 = 8400

    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    # stride 表示网络的下采样倍率，例如 8, 16, 32 表示特征图的大小是原始输入图像的 1/8, 1/16, 1/32
    # 步幅是指输入图像经过卷积神经网络后，生成"特征图"时每个特征点在输入图像上对应的像素距离
    # stride = 8 意味着特征图上的每一个点对应输入图像上的 8x8 像素区域
    # 
    # 步幅表示特征图上每个点对应到输入图像上的实际大小，或换句话说，它表示"下采样的倍率"
    # 步幅越大，特征图的分辨率就越低，每个特征点所覆盖的输入图像区域就越大。通常，步幅的值为 8、16、32 等，这些值代表下采样的倍率

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides): # 遍历每个特征图尺度 
        # np.meshgrid 函数用于生成网格坐标（x, y），这些坐标表示每个预测框,相对于其对应的网格单元的位置
        print(f"create mesh goid = {hsize},{wsize}:{stride}")
        # create mesh goid = 80,80:8
        # create mesh goid = 40,40:16
        # create mesh goid = 20,20:32
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize)) 
        # np.arange(wsize) 一维的numpy.ndarray  
        # np.meshgrid 用法:  https://wangyeming.github.io/2018/11/12/numpy-meshgrid/
        #                   两个一维数组 两两成对 xv存放一对中的x yv存放一对的y (xv[i],yv[i])为一对
        # np.stack 用法: https://blog.csdn.net/Riverhope/article/details/78922006
        #        假设要转变的张量数组arrays的长度为N,其中的"每个张量"数组的形状为(A, B, C)。
        #           如果轴axis=0，则转变后的张量的形状为(N, A, B, C)。
        #           如果轴axis=1，则转变后的张量的形状为(A, N, B, C)。
        #           如果轴axis=2，则转变后的张量的形状为(A, B, N, C)
        #           从原来axis=0(1,2) "-1" (也就是-1,0,2) 维度 下各取一个元素 作为新的维度 

        # stack会增加一个维度，而concatenate不会增加维度

        #print(f"xv.shape = {xv.shape} yv.shape = {yv.shape}")
        #xv.shape = (80, 80) yv.shape = (80, 80)  stack (80, 80, 2)
        #xv.shape = (40, 40) yv.shape = (40, 40)  stack (40, 40, 2)
        #xv.shape = (20, 20) yv.shape = (20, 20)  stack (20, 20, 2)

        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)  # （1, 80*80, 2） 展开 
        grids.append(grid) # 每个特征图尺寸下的mesh grid(坐标)

        shape = grid.shape[:2] # e.g (1, 80*80)

        # expanded_strides 保存了每个网格点对应的步幅（stride），
        # 步幅是用来将网络的输出从特征图坐标尺度, 还原到输入图像坐标尺度的。
        expanded_strides.append(np.full((*shape, 1), stride))  # e,g (1, 80*80, 1)

    # 不增加维度  grids是列表 每个元素是(1, 80*80, 2), (1, 40*40, 2), (1, 20*20, 2)
    # axis=1 从原来每个array中 的axix=1 "-1"取一个 cat合并起来  也是作为 新array的axis=1维度(变化的维度)
    grids = np.concatenate(grids, 1)
    # grids = (1, 8400, 2)

    expanded_strides = np.concatenate(expanded_strides, 1)

    # 网络输出的前两个通道通常是预测框的"中心点坐标的偏移量"，这个偏移量"相对于网格点"计算。
    # 通过加上网格坐标，并乘以步幅，
    # 这些相对坐标被转换成了 "相对于输入图像的绝对坐标"

    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides

    # 网络输出的接下来的两个通道通常是预测框的宽度和高度，这些值是以对数空间输出的。
    # 通过指数化处理（np.exp）并乘以步幅，
    # 这些宽度和高度被转换成了输入图像的尺度。

    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def postprocess(output, ratio, conf_threshold=0.45, nms_threshold=0.45):
    # 解析输出 
    predictions = output[0] # 只取batch中第0个
    print(f"postprocess predictions = {predictions.shape}")
    
    # 还原到原始图像尺寸
    boxes = predictions[:, :4] # 4 个值：代表边界框的位置和大小（通常是中心点坐标 x、y 和宽度、高度）
    scores = predictions[:, 4] * predictions[:, 5] 
    
    # YOLOX 推理输出的边界框中心坐标最初并不是直接相对于推理输入图像的绝对坐标
    # 相对于特征图上的网格单元位置的偏移量
    # 
    # 输出的边界框坐标（中心点的 x、y 以及宽度和高度）都是相对于输入到模型中的图像尺寸（例如 640x640 
    # 宽度和高度是归一化的，它们的值通常在 [0, 1] 之间  是输入图像宽度和高度的比例 
    # 如果输出宽度输出为 0.5，输入图像的宽度为 640 像素 那么这个边界框的实际宽度就是 0.5 * 640 = 320 像素

    # 代表目标框中包含目标的置信度 objectness score
    # 80 个值：代表属于某个类别的概率（分类得分）
    
    # 筛选置信度
    mask = scores > conf_threshold
    print(f"postprocess mask = {np.sum(mask)} {mask.shape}")
    boxes = boxes[mask]
    scores = scores[mask]

    print(f"postprocess boxes = {boxes.shape}")
    print(f"postprocess scores = {scores.shape}")
    
    # 应用 NMS 不区分分类
    # cv2.dnn.NMSBoxes 函数用于执行非极大值抑制（NMS  box:[x, y, width, height]
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, nms_threshold)
    
    # 还原坐标到原始图像
    #ratio = min(input_size[0] / image_shape[0], input_size[1] / image_shape[1])
    boxes /= ratio
    
   
    return boxes[indices], scores[indices]


os.environ["ORT_CUDA_DEVICE_ID"] = "3" 
#sys.path.append('~/anaconda3/envs/mimicmotion/lib/python3.11/site-packages/torch/lib/')
#old_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
#os.environ['LD_LIBRARY_PATH'] = '~/anaconda3/envs/mimicmotion/lib/python3.11/site-packages/torch/lib/:' + old_ld_library_path
# 上面的配置 并不能让onnx runtime 去找 torch/lib/ 目录下的 libcudnn.so.8:

# 这样执行才能走GPU !!
# LD_LIBRARY_PATH=~/anaconda3/envs/mimicmotion/lib/python3.11/site-packages/torch/lib/ python test_onnx_runtime_gpu.py

# 加载 ONNX 模型
# providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
providers = ['CUDAExecutionProvider'] 
#session = ort.InferenceSession("models/DWPose/yolox_l.onnx")
#session = ort.InferenceSession("models/DWPose/yolox_l.onnx", providers=providers)
# 性能优化 ?? 有效果 ??
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession("models/DWPose/yolox_l.onnx", providers=providers, session_options=session_options)


# 推理Path
# 如果输出中包含 'CUDAExecutionProvider'，则表示 ONNX Runtime 可以使用 GPU
# 
used_providers = session.get_providers()
print(used_providers)
if 'CUDAExecutionProvider' in used_providers:
    print("模型正在使用 GPU 进行推理")
else:
    print("模型正在使用 CPU 进行推理")


# 模型元数据
# 需要调整 input_size、conf_threshold 和 nms_threshold
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_name = session.get_outputs()[0].name
print(f"model meta-data input_name ={input_name} input_shape = {input_shape} output_name = {output_name}")
input_size = input_shape[2:]   # 认为是方形的 
#  YOLOX-L: input_name =images input_shape = [1, 3, 640, 640] output_name = output

# 准备输入数据
#input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
image = cv2.imread("assets/example_data/images/demo1.jpg")
original_shape = image.shape[:2]

# 前处理 
input_data, ratio  = preprocess(image, input_size)
print(f"input_data = {input_data.shape}")
input_data = input_data[None, :, :, :]
print(f"input_data 2 = {input_data.shape}")

# 运行推理 
outputs = session.run(None, {input_name: input_data}) # key:value 对 

# 模型输出的格式 [batch, num_boxes, 5+num_classes]
outputs = outputs[0]
outputs = demo_postprocess(outputs, input_size)

# 后处理  使用参数默认值 作为nms和阈值
boxes, scores = postprocess(outputs, ratio)



for box, score in zip(boxes, scores):
    print(f"process {box} {score}")
    xC, yC, width, height = map(int, box)
    x1 = int(xC - width  / 2)
    y1 = int(yC - height / 2)
    x2 = int(xC + width  / 2)
    y2 = int(yC + height / 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imwrite("test_onnx_runtime_gpu.jpg", image)



# 批量推理
#input_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
#outputs = session.run(None, {input_name: input_data})
