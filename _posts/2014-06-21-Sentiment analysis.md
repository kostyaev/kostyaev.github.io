---
layout: post
title: Sentiment analysis with CoreNLP
description: "How to do sentiment analysis with CoreNLP library"
modified: 2014-06-21
category: article
tags: [sentiment analysis, CoreNLP, scala]
comments: true
share: true
---

In the rise of social media customer’s opinions has become extremely valuable for businesses selling their products, financial markets and social researches. To extract opinions from customer’s reviews, comments or other kind of text data you might want to know what sentiment analysis is. 

> Sentiment analysis and opinion mining is the field of study that analyzes people's opinions, sentiments, evaluations, attitudes, and emotions from written language.[^1]

[^1]: [Sentiment Analysis and Opinion Mining Synthesis Lectures on Human Language Technologies](http://www.morganclaypool.com/doi/abs/10.2200/S00416ED1V01Y201204HLT016)

So the the basic task of sentiment analysis is classifying text into some emotive categories. The most common set of categories are: positive, neutral and negative.

### Methods for sentiment analysis
There is a number of methods for sentiment analysis that can be divided in two groups:

 * Lexicon-based methods
 * Machine learning methods
	* NB (Naive Bayes classifier)
	* biNB (Naive Bayes with bag of bigram features)
	* SVM (Support Vector Machine)
	* RNTN (Recursive Neural Tensor Network)
	* RNN (Recursive Neural Network)
	* MV-RNN (Matrix-Vector RNN)

The last method in the list is claimed to have the better performance than the others. [^2]

[^2]: [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)


### CoreNLP library
As you may already know there is great open source library for natural language processing named [CoreNLP](http://nlp.stanford.edu/software/corenlp.shtml) which includes RNTN for sentiment analysis. The library is written in Java, but there are also some wrappers for other languages.

### How to use it

Let’s dive into details on how to use CoreNLP in Scala. Since Scala is compatible with Java, you can simply import CoreNLP library in your Scala project. To include library and models for sentiment component in your project add the following dependency to the sbt build file:

{% highlight scala %}
libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.3.1" artifacts (Artifact("stanford-corenlp", "models"), Artifact("stanford-corenlp"))
{% endhighlight %}

When doing analysis with CoreNLP you get fine grained predictions as a result (i.e. 5 classes: very positive, positive, neutral, negative, very negative) which are less accurate than Positive/Neutral/Negative ones. In the code below, we create three categories and map result into them in the end of `getSentiment` method.

{% highlight scala %}

import java.util.Properties
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations

object SentimentCategory extends Enumeration {
  type SentimentCategory = Int
  val Negative = 0
  val Neutral = 1
  val Positive = 2
}

trait SentimentTools {
    val props = new Properties()
    props.setProperty("annotators", "tokenize, ssplit, parse, sentiment");
    val pipeline = new StanfordCoreNLP(props)

    def getSentiment(text : String): SentimentCategory = {
      var mainSentiment = 0
      if (text != null && text.length() > 0) {
        var longest = 0
        val annotation = pipeline.process(text)
        val list = annotation.get(classOf[CoreAnnotations.SentencesAnnotation])
        val it = list.iterator()
        while (it.hasNext)
        {
          val sentence = it.next()
          val tree = sentence.get(classOf[SentimentCoreAnnotations.AnnotatedTree])
          val sentiment = RNNCoreAnnotations.getPredictedClass(tree)
          val partText = sentence.toString()
          if (partText.length() > longest) {
            mainSentiment = sentiment
            longest = partText.length()
          }
        }
      }
      import SentimentCategory._
      if (mainSentiment < 2)
        Negative
      else if (mainSentiment == 2)
        Neutral
      else
        Positive
    }
}

{% endhighlight %}

