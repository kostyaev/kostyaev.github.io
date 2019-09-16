---
layout: post
title: Preparing multilabel dataset for training ConvNet with Caffe
modified: 2016-02-17
categories: 
  - Caffe
tags: [caffe, lmdb, multilabel]
comments: true
share: true
---
Preparing multilabel training set for caffe framework is a bit nontrivial. So, if you have multiple, possibly varying number of ground truth labels for each training example then here is how you can do it using LMDB store. 
For LMDB data source you need to separate your data input and your labels by creating two LMDB (one for the data and the second one for the labels). You also have to define two data layers in your network definition, set the same batch size for both of them and disable shuffling for the alignment.

To share it I've created a small script available on [github](https://github.com/kostyaev/ml-utils/blob/master/create_multilabel_lmdb.py). You can run it as in this example:

~~~~~~~~
python create_multilabel_lmdb.py 
	--images /path/to/image_file/images.txt 
	--labels /path/to/labels_file/labels.npy 
	--imagesOut /path/to/image-lmdb 
	--labelsOut /path/to/label-lmdb 
	-n size_of_test_set 
	--maxPx 256 
	--minPx 227 
	--shuffle=true
	
~~~~~~~~

What this script do is reads the images text file having the format like this:

~~~~~~~~
 /path/to/dir/img1.jpg
 /path/to/dir/img2.jpg
 ...
~~~~~~~~

Reads the labels file, which is just a 2d numpy array serialized using numpy, here is the example of an array:

~~~~~~~~
[
	[1,0,1,0,1], 
	[0,1,0,0,1],
	[1,1,0,0,0]
]
~~~~~~~~

The first row indicates that the first image (in the images text file) has labels 1,3 and 5. And the second row says that the second image has labels 2 and 5, the third row - labels 1 and 2.
This script also shuffles the data, resizes images preserving an aspect ratio and prints mean image values at the end of the work.
If error occures (e.g if some image file is corrupted or missing), the procedure skips the corrupted image and its label and continues the progress.