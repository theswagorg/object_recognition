# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 21:24:48 2022

@author: PuraLumbreMusic
"""


from imageai.Classification import ImageClassification

prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath("resnet50_imagenet_tf.2.0.h5")
prediction.loadModel()

predictions, probabilities = prediction.classifyImage("people/train/cops/IMG_3877.jpg", result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
