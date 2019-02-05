# A Place to Play with Tensorflow and Keras

If I keep myself organized, everything in src will have a jupyter notebook
demonstrating the architecture used/described therein.


### Things to try at some point

Text Detection CTPN
https://github.com/eragonruan/text-detection-ctpn
https://arxiv.org/abs/1609.03605

__YOLO - you only look once
Object identification and location__
Resources:
http://machinethink.net/blog/object-detection-with-yolo/
https://github.com/experiencor/basic-yolo-keras
https://arxiv.org/abs/1612.08242

https://github.com/joycex99/tiny-yolo-keras/blob/master/Tiny%20Yolo%20Keras.ipynb

Keras Yolo (and weights link):
https://sriraghu.com/category/machine-learning/keras/
yolo-tiny.weights?
https://drive.google.com/file/d/0B1tW_VtY7onibmdQWE1zVERxcjQ/view

YOLO website: https://pjreddie.com/darknet/yolo/

___
## TODO:

fix tensorflow / keras
  reinstall?

__RNN todo:__
- [ ] data generator for rnn
  - [ ] QUESTION: can batch_limit in `_bucket_to_fit()` and max_batch_size in `gen_data` be changed to batchsize? or is batchsize controled elsewhere (meaning these are just backup limits)
- [ ] adapt to json/yaml config format

__YOLO todo:__
- [ ] YOLO: get pretrained weights incorporate into network
- [ ] YOLO: run (on small image set) to ensure operation

__tflite conversion todo:__
- [ ] compare logloss/accuracy of tflite and original model  
- [ ] 
