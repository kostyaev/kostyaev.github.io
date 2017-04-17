---
layout: post
title: Why top-5 error is more fair metric than top-1 for ILSVRC models
category: article
tags: [ILSVRC, top-1, top-5, ImageNet, dataset]
comments: true
share: true
---

For the most publicly available models trained on ImageNet dataset [top-5](http://stats.stackexchange.com/questions/156471/imagenet-what-is-top-1-and-top-5-error-rate) and [top-1](http://stats.stackexchange.com/questions/156471/imagenet-what-is-top-1-and-top-5-error-rate) errors are reported. The best perfoming [model](https://github.com/KaimingHe/deep-residual-networks) from [ILSVRC 2015](http://image-net.org/challenges/LSVRC/2015/) has 6.7% top-5 error and 23% top-1 error evaluated using singe center crop. It's pretty interesting to find out that human error turned out to be not much better, top-5 error is around 5.1%  according to the result of some experiments descibed in the [article](http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/). An ensembe of models show even better results, six models of different depth leads to 3.57% top-5 error (1st place in ILSVRC 2015). Top-1 error still seems to be very big and one may say that top-5 rule is very forgiving and rules should be tightened and only top-1 should be considered. But top-1 error doesn't really give you an understanding of how good is you neural network in general, it can only be useful to compare the perfomance of different models. The reason is that ImageNet is a single label dataset containing images that actually can fall into several categories and the orded of those categories is ambiguous. To illustare this let's look at some of the example images from the training set, starting from example easy to classify:

Sport cars 
![race_cars](../../images/sport_cars.jpg){:height="250px" width="750px"}

Race cars 
![race_cars](../../images/race_cars.jpg){:height="250px" width="750px"}

Car wheels 
![car_wheels](../../images/car_wheels.jpg){:height="250px" width="750px"}


Now if you look at the next examples, you will find that they are pretty ambigous if you are asked to assign only one label, let's say, out of five. And for a number of categories there a lot of images with such ambiguity.  

![cars](../../images/cars_collage.jpg)

The major problem with the dataset is that for many of the images we can't say precisely if a category A or a category B describes it the best and the model in such situation is penalized during the training for the the mismatch between the predicted A category and B category given as a ground truth. So even if the image from the evaluation set is better in the sense of an ambiguity but still can be described with multiple labels, it will be harder for the network to determine a pravailing label because the training dataset doesn't train it to do so very well. And this explain such a big margin between top-1 and top-5 errors.
