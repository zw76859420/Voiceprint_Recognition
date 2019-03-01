### Voiceprint_Recognition
Just for DCASE 2019 and for studying AI
#### data
`Task name` : Acoustic Scene Classfication <br>
`Download` : You can download the datasets from here [datasets](https://pan.baidu.com/s/1NmJsWFmOv4M0A7nLYReVmw, "悬停显示") , and the code for datasets are "wzzg"
#### data preprocess
We use skip_frames in our preprocess stage, the skip is 2. In our future work, we will use sparse_frames and skip_frames together.
#### features
`mfcc174`
#### model
`Densenet`
#### cost function
`CrossEntropy`
### Operation process
 `read_file.py` --> `file_shuffle.py` --> `same_step.py` --> `feature_extract.py` --> `DenseNet.py`
#### reference link
  https://github.com/qiuqiangkong/audioset_classification
