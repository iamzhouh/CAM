import numpy

import ResNetCAM
import torch
import numpy as np
import cv2

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import torch.optim

batch_size = 1
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])   #归一化

test_dataset = datasets.MNIST(root = './dataset/mnist/', train=False, download = True, transform = transform)
test_loader = DataLoader(test_dataset, shuffle = True, batch_size = batch_size)


PATH = 'state_dict_model.pth'
model = ResNetCAM.Net()
model.load_state_dict(torch.load(PATH))

model.eval()

finalconv_name = 'rblock2'


features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
model._modules.get(finalconv_name).register_forward_hook(hook_feature)  # 得到GAP前的特征图

params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())    # 得到softmax的权重参数


def returnCAM(feature_conv, weight_softmax, class_idx):   # 计算类激活图
    # generate the class activation maps upsample to 256x256
    size_upsample = (28, 28)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


for data in test_loader:
    images, labels = data
    outputs = model(images)

    outputs = torch.nn.functional.softmax(outputs, dim=1).data.squeeze()   # softmax
    probs, idx = outputs.sort(0, True)  # 对class的概率排序
    probs = probs.numpy()    # 概率序列
    idx = idx.numpy()  # 标签序列

    # output the prediction
    for i in range(0, 10):
        print('{:.9f} -> {}'.format(probs[i], idx[i]))    #  打印每个class的概率

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])      # 得到CAM

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s'%idx[0])   # 输出类别

    img1 = images[0][0]*255
    b = numpy.zeros((28,28))
    g = numpy.zeros((28,28))

    img = numpy.zeros((28,28,3))
    img[:,:,0] = img1
    img[:,:,1] = img1
    img[:,:,2] = img1
    # 输入的图
    # cv2.imwrite('img.jpg', img)

    height, width, _ = img.shape

    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)   # 热力图
    result = heatmap * 0.7 + img * 0.3   # 图像融合
    # cv2.imwrite('CAM.jpg', result)
    res = np.hstack((img,heatmap,result))  # 行组合
    cv2.imwrite('result.jpg', res)


    break

