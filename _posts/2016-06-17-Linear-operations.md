---
layout: post
title: Playing with image embeddings
category: article
tags: [computer vision, features vectors, linear operations]
comments: true
share: true
---

Quite a while ago I worked on image retrieval and performed a little experiment with algebraic operations on image embeddings extracted from convolutional networks. I downloaded a set of publicly available photos, extracted feature vectors using pretrained ResNet 50 and applied cosine distance KNN search using linear combinations of query vectors:

 - [man](http://static1.squarespace.com/static/55366165e4b0e488db74b92b/572cd5bbc2ea5104e50cbf8c/572cd5be7c65e48ce9293423/1462556096928/20160229_manoscar9.jpg)
 - [woman](https://s-media-cache-ak0.pinimg.com/736x/e2/73/e7/e273e787cef28c8fe2eb359a97ae0f11.jpg)
  - [woman_in_dress](http://image.dhgate.com/albu_332362244_00-1.0x0/2013-hot-sale-fashion-flower-lady-dress-woman.jpg)
 - [car_with_woman](http://static3.therichestimages.com/cdn/780/410/90/c/wp-content/uploads/2015/06/Girl-Mercedes1.jpg)
 - [car](http://f.tqn.com/y/moneyfor20s/1/S/h/1/-/-/nice-car.jpg)
 - [sea](http://www.redorbit.com/media/uploads/2012/11/tide.jpg)
 - [coffee](http://s2.favim.com/orig/32/coffe-cool-cute-eat-food-Favim.com-253389.jpg)
 - [dress](http://gloimg.rosegal.com/rosegal/2015/201509/goods-img/1441933712518-P-3102218.jpg?20131202008)
 - [crowd](https://static-secure.guim.co.uk/sys-images/Guardian/Pix/pictures/2013/1/30/1359549064145/Crowd-of-people-008.jpg)
 - [burger](http://www.seriouseats.com/images/2013/06/20130614-256060-ultimate-cheesy-burger.jpg)
 - [building](http://www.e-architect.co.uk/images/jpgs/leeds/jessops_building_sheffield_aw170410_3.jpg)

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

Encode your documents and queries and perform nearest neighbor search. For NN search I put vectors into Anorm index.

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


![png](../../images/vectors/output_29_0.png)



~~~ python
search(sea+woman)
~~~


![png](../../images/vectors/output_30_0.png)



~~~ python
search(building + crowd*0.5 + sea*1.5)
~~~


![png](../../images/vectors/output_31_0.png)



~~~ python
search(woman + car*3 + dress)
~~~


![png](../../images/vectors/output_32_0.png)



~~~ python
search(coffee*1.3 + burger)
~~~


![png](../../images/vectors/output_33_0.png)



~~~ python
search(man + dress)
~~~


![png](../../images/vectors/output_34_0.png)



~~~ python
search(woman_in_dress)
~~~


![png](../../images/vectors/output_35_0.png)



~~~ python
search(woman_in_dress - dress)
~~~


![png](../../images/vectors/output_36_0.png)

