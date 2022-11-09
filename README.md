# yolov7-pose
Implementation of "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"

Pose estimation implimentation is based on [YOLO-Pose](https://arxiv.org/abs/2204.06806). 

在yoloV7-pose基础上添加了任意关键点数量 + 多类别分类代码。

这里是以4个关键点进行举例，其中添加了左右翻转数据增强，

点的交换是：point1和point2 point3和point4 point5和point6 依次类推。设置了20个点的交换，可以取前n个，n为偶数。



## Dataset preparison

``` shell

# data.txt   含义分别是： cls   	 x	           y          w       h        point1x		point1y		point2x	   	point2y		 point3x	  point3y	  point4x	   point4y		    ...
# 						类别  目标中心点x    目标中心点y   目标宽w   目标高h    目标点1x坐标  目标点1y坐标   目标点2x坐标  目标点2y坐标   目标点3x坐标  目标点3y坐标   目标点4x坐标  目标点4y坐标    依次类推 
0  0.5739299610894941  0.1724137931034483  0.3715953307392996  0.29064039408866993  0.38910505836575876  0.08374384236453201  0.7587548638132295  0.029556650246305417  0.7607003891050583  0.2660098522167488  0.39299610894941633  0.32019704433497537  
2  0.5739299610894941  0.1724137931034483  0.3715953307392996  0.29064039408866993  0.38910505836575876  0.08374384236453201  0.7587548638132295  0.029556650246305417  0.7607003891050583  0.2660098522167488  0.39299610894941633  0.32019704433497537  
0  0.5739299610894941  0.1724137931034483  0.3715953307392996  0.29064039408866993  0.38910505836575876  0.08374384236453201  0.7587548638132295  0.029556650246305417  0.7607003891050583  0.2660098522167488  0.39299610894941633  0.32019704433497537  


```
写一个train.txt 和 val.txt文件


``` shell

# train.txt
./train/images/-nfs-阿拉伯车牌字符-沙特阿拉伯卡口车牌-2-沙特阿拉伯卡口车牌-2-image1837.jpeg
./train/images/-nfs-车牌字符-埃及车牌-埃及车牌截图-2021-04-30 11-11-52屏幕截图.png
./train/images/-nfs-车牌字符-埃及车牌-埃及车牌截图-2021-04-30 13-57-27屏幕截图.png
./train/images/-nfs-车牌字符-埃及车牌-埃及车牌截图-2021-04-30 10-19-54屏幕截图.png
./train/images/-nfs-阿拉伯车牌字符-外国车牌现场_20210519_1-外国车牌现场_20210519_1-e0d92b0990a1249388bc77bdfa8e43ed.jpg
./train/images/-nfs-车牌字符-埃及车牌-埃及车牌截图-2021-04-30 13-51-28屏幕截图.png
./train/images/-nfs-车牌字符-约旦车牌-videoplayback-videoplayback_13_1460.jpg
./train/images/-nfs-车牌字符-埃及车牌-埃及车牌截图-2021-04-30 13-56-51屏幕截图.png
./train/images/-nfs-车牌字符-埃及车牌-埃及车牌截图-2021-04-30 10-27-50屏幕截图.png


```



## Training

[yolov7-w6-person.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-person.pt)

百度网盘：[yolov7-w6-person.pt](https://pan.baidu.com/s/12HOci-SMAQatxj3v2_sTnA)  提取码: 9nlk

``` shell
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train_Ncla_nPoint.py --data data/coco_kpts.yaml --cfg cfg/yolov7-w6-pose.yaml --weights weights/yolov7-w6-person.pt --batch-size 128 --img 640 --kpt-label --sync-bn --device 0,1,2,3,4,5,6,7 --name yolov7-w6-pose --hyp data/hyp.pose.yaml
```

## Deploy
TensorRT:[https://github.com/nanmi/yolov7-pose](https://github.com/nanmi/yolov7-pose)

## Testing

[yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

``` shell
python test_Ncla.py --data data/coco_kpts.yaml --img 640 --conf 0.5 --iou 0.25 --weights yolov7-w6-pose.pt --kpt-label
```


