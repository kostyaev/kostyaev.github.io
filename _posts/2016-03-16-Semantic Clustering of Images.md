---
layout: post
title: Semantic clustering of images
description: "Semantic clustering of images with a convolutional neural network"
category: article
tags: [image, semantic, clustering, kmeans]
comments: true
share: true
---
Recent studies in the field of computer vision have shown the abilities of deep convolutional neural networks to learn high level image representation. These representations have reach semantic and can be very handy in various visual tasks. One of such task that can make use of high level features is semantic clusterization. Let's find some pictures in Google Web Search, feed them to CNN and apply k-means to the extracted feautures and take the biggest clusters, here is what I got for different queries :

Query "extreme":
![extreme](../../images/semantic/extreme_grid.png)

On the left is the query result and on the right - some groups of images. It's clear that pictures in the same group are closer in semantic than images across. Here we got two clusters, first for the American rock band and the second for some extreme activity. 

Query "destruction":

![distruction](../../images/semantic/distruction_grid.png)

Now we have three clusters and arguably three slightly different meanings: an explosion, destroyed buildings and a forest fire. Let's try a query that for sure should show images with different semantic.

Query "apple":

![apple](../../images/semantic/apple_grid.png)

As you might guess Google will return results containing some Apple products andd we can automatically move images with different meaning to their semantic group by applying clustering algorithm on CNN codes. Having multiple images in top-k biggest clusters we can pull out centroids and use them as different representation of the same concept. Let's do this for our examples:

![centroids](../../images/semantic/centroids.png)

Here centroids of the "distruction" clusters can be thought as different representations of a destruction. There are many other applications, for example, semantic clusters can help to find the most popular meaning for some text term. We can also apply clustering to filter duplicates as they tend to fall into the same group, in this case we need to select images from different groups. 

So coming to conclusion, deep convolutional neural networks can produce very good abstractions that can be further used in many visual recognition tasks that hardly be possible without them.  

