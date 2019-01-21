# tf-pose-estimation

[Original Git here](https://github.com/ildoonet/tf-pose-estimation)  
'Openpose' for human pose estimation have been implemented using Tensorflow. It also provides several variants that have made some changes to the network structure for **real-time processing on the CPU or low-power embedded devices.**


**You can even run this on your macbook with descent FPS!**

Original Repo(Caffe) : https://github.com/CMU-Perceptual-Computing-Lab/openpose

| CMU's Original Model</br> on Macbook Pro 15" | Mobilenet Variant </br>on Macbook Pro 15" | Mobilenet Variant</br>on Jetson TX2 |
|:---------|:--------------------|:----------------|
| ![cmu-model](/etcs/openpose_macbook_cmu.gif)     | ![mb-model-macbook](/etcs/openpose_macbook_mobilenet3.gif) | ![mb-model-tx2](/etcs/openpose_tx2_mobilenet3.gif) |
| **~0.6 FPS** | **~4.2 FPS** @ 368x368 | **~10 FPS** @ 368x368 |
| 2.8GHz Quad-core i7 | 2.8GHz Quad-core i7 | Jetson TX2 Embedded Board | 

Implemented features are listed here : [features](./etcs/feature.md)

## Important Updates

2018.5.21 Post-processing part is implemented in c++. It is required compiling the part. See: https://github.com/ildoonet/tf-pose-estimation/tree/master/src/pafprocess
2018.2.7 Arguments in run.py script changed. Support dynamic input size.

## Install

### Dependencies

You need dependencies below.

- python
- tensorflow 1.4.1
- opencv, protobuf, python-tk

### Opensources

- slim
- slidingwindow
  - https://github.com/adamrehn/slidingwindow
  - I copied from the above git repo to modify few things.

### Install

Clone the repo and install 3rd-party libraries.

```bash
$ pip install tensorflow==1.4.1
$ pip install matplotlib==2.0.0
# apt install python-tk swig
$ pip install psutil scipy

$ git clone https://www.github.com/k5iogura/tf-pose-estimation_fs
$ cd tf-pose-estimation_fs
$ pip install -r requirements.txt
```

Build c++ library for post processing. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess
```
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python setup.py build_ext --inplace
```

#### Test installed package
![package_install_result](./etcs/imgcat0.gif)
```bash
$ cd tf-pose-estimation_fs
$ python -c 'import tf_pose; tf_pose.infer(image="./images/p1.jpg")'
```


## Models
### Download Tensorflow Graph File(pb file)

Notice: No more original cmu and mobilenet .pb file on dropbox area.  
Use Download script on models/pretrained.  

### Inference Time

| Dataset | Model              | Inference Time<br/>Macbook Pro i5 3.1G | Inference Time<br/>Jetson TX2  |
|---------|--------------------|----------------:|----------------:|
| Coco    | cmu                | 10.0s @ 368x368 | OOM   @ 368x368<br/> 5.5s  @ 320x240|
| Coco    | dsconv             | 1.10s @ 368x368 |
| Coco    | mobilenet_accurate | 0.40s @ 368x368 | 0.18s @ 368x368 |
| Coco    | mobilenet          | 0.24s @ 368x368 | 0.10s @ 368x368 |
| Coco    | mobilenet_fast     | 0.16s @ 368x368 | 0.07s @ 368x368 |

## Demo

### Test Inference

You can test the inference feature with a single image.

```
$ python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg
```

The image flag MUST be relative to the src folder with no "~", i.e:
```
--image ../../Desktop
```

Then you will see the screen as below with pafmap, heatmap, result and etc.

![inferent_result](./etcs/inference_result2.png)

### Realtime Webcam

```
$ python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0
```

Then you will see the realtime webcam screen with estimated poses as below. This [Realtime Result](./etcs/openpose_macbook13_mobilenet2.gif) was recored on macbook pro 13" with 3.1Ghz Dual-Core CPU.

## Python Usage

This pose estimator provides simple python classes that you can use in your applications.

See [run.py](run.py) or [run_webcam.py](run_webcam.py) as references.

```python
e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
humans = e.inference(image)
image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
```

## ROS Support

See : [etcs/ros.md](./etcs/ros.md)

## Training

See : [etcs/training.md](./etcs/training.md)

## References

### OpenPose

[1] https://github.com/CMU-Perceptual-Computing-Lab/openpose

[2] Training Codes : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

[3] Custom Caffe by Openpose : https://github.com/CMU-Perceptual-Computing-Lab/caffe_train

[4] Keras Openpose : https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation

[5] Keras Openpose2 : https://github.com/kevinlin311tw/keras-openpose-reproduce

### Lifting from the deep

[1] Arxiv Paper : https://arxiv.org/abs/1701.00295

[2] https://github.com/DenisTome/Lifting-from-the-Deep-release

### Mobilenet

[1] Original Paper : https://arxiv.org/abs/1704.04861

[2] Pretrained model : https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md

### Libraries

[1] Tensorpack : https://github.com/ppwwyyxx/tensorpack

### Tensorflow Tips

[1] Freeze graph : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py

[2] Optimize graph : https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2
