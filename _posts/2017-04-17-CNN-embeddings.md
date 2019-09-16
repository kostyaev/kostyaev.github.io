---
layout: post
title: Playing with image embeddings
categories: 
  - Computer Vision
tags: [computer vision, features vectors, linear operations]
comments: true
share: true
sidebar_link: true
excerpt_separator: "<!--more-->"
---

Quite a while ago I worked on image retrieval and made a little experiment with algebraic operations on image embeddings extracted from convolutional network. I downloaded a set of publicly available photos, extracted feature vectors using pretrained ResNet 50 and applied cosine distance KNN search using linear combinations of some query vectors. 
<!--more-->
All documents and queries can be encoded with the following few lines of code using Caffe:

~~~ python
def encode(url):
    img = open_image(url)
    img = preprocess(img)
    data = np.asarray([img])
    if net.blobs['data'].data.shape[0] != 1:
        net.blobs['data'].reshape(1,3,224,224)
    result = net.forward(data=data)
    return net.blobs['pool5'].data[0].flatten().copy()
~~~

Encode your documents and queries and perform nearest neighbor (NN) search. For fast NN search I put vectors into [Annoy](https://github.com/spotify/annoy) index.

~~~ python
    
def search(query_vector):
    # get 9 ids of nearest vectors from index
    ids = index.get_nns_by_vector(query_vector, 9)
    # load images by ids and show them
    images=[]
    for i in ids:
        img = Image.open(id2file[i])
        if img is None:
            continue
        img = centeredCrop(np.array(resize(img, 128, 128)), 128, 128)
        images.append(img)
    plt.figure(2, figsize=(10,10))
    show(np.array(images))
~~~



~~~ python
search(sea)
~~~


![png](/images/vectors/output_29_0.png)



~~~ python
search(sea+woman)
~~~


![png](/images/vectors/output_30_0.png)



~~~ python
search(building + crowd*0.5 + sea*1.5)
~~~


![png](/images/vectors/output_31_0.png)



~~~ python
search(woman + car*3 + dress)
~~~


![png](/images/vectors/output_32_0.png)



~~~ python
search(coffee*1.3 + burger)
~~~


![png](/images/vectors/output_33_0.png)



~~~ python
search(man + dress)
~~~


![png](/images/vectors/output_34_0.png)



~~~ python
search(woman_in_dress)
~~~


![png](/images/vectors/output_35_0.png)



~~~ python
search(woman_in_dress - dress)
~~~


![png](/images/vectors/output_36_0.png)

