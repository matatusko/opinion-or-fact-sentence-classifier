## Opinion Classifier

Classifies sentences whether they represent a fact or personal opinion. Tested with different algorithms from sklearn, including random forest
classifier, support vector machines, logistic regression and neural network and each achieves over 90% accuracy.

## Dataset

Factual sentences are mostly sentences extracted from wikipedia articles - the whole process happens in the gather_and_prepare_data.py module.
Maybe not a perfectly annotated data, but one can safely assume most of the sentences found on wiki are pure fact (or at least structured as facts).

As for the opinions, I've found a great dataset called [Opinosis](http://kavita-ganesan.com/opinosis/#.Wmljc6iWYow), which consists of 
opinion sentences as extracted from various reviews on many different topics.

In total there were around 11,000 sentences in the dataset. Not the perfect amount, but still does the job.

## Features

As for the features, I'm using spaCy to extract finely grained part of speech tags (including info whether the verb is in past or present form, 
singular or plural noun etc. More info on [spaCy](https://spacy.io/) website. Also, I'm extracted entities and their labels as more features. 
I wasn't sure whether that would work, as usually people tend to go with BOW models, but the classifiers all achieve over 90% accuracy on a test data, 
and based on my small sample test with sentences I've created, it is rather accurate.

Just for the sake of it, I've built a BOW model on exactly the same dataset and run it through multiple ML algorithms, including Naive Bayes, Random Forest, SVM, Logistic Regression and NN. It was interesting to see that in the accuracy on a test set was way higher, reaching up to 97%. 

However, when tested on random sample sentences outside of database, the BOW model (all algos testes) did a horrible job classifying all the sentences on donuts mostly incorrectly. I assume the BOW model tends to overfit and works only with sentences which are roughly on similar topic or contain the words in the wordlist. On the other hand, the model based on sentence structure (number of labeled POS tags) generalizes more and provides better results on new examples, at least when trained on relatively small dataset.

## Samples for sentence structure model

Extracted from quick_tests.py module. Feel free to give it a try, all the models I've trained are included in the repository. 
Maybe some more fine-tuning with parameters would yield better results, but for a side-project I'm quite satisfied with what it does.


Using rf_classifier (random forest)
<br> --: <b>Sentence:</b> "As far as I am concerned, donuts are amazing." <br>is an OPINION!

Using svm_classifier (support vector machine)
<br> --: <b>Sentence:</b> "Donuts are a kind of ring-shaped, deep fried dessert." <br>is a FACT!

Using lr_classifier (logistic regression)
<br> --: <b>Sentence:</b> "Doughnut can also be spelled as "Donut", which is an American variant of the word." <br>is a FACT!

Using nn_classifier (neural network)
<br> --: <b>Sentence:</b> "This new graphics card I bought recently is pretty amazing, it has no trouble rendering my 3D donuts art in high quality." <br>is a FACT!

Using nn_classifier (neural network)
<br> --: <b>Sentence:</b> "I think this new graphics card is amazing, it has no trouble rendering my 3D donuts art in high quality." <br>is an OPINION!

## Samples for BOW model (using NN classifier which got 97% on test set)

<b>Sentence:</b> As far as I am concerned, donuts are amazing.
<br>
The above sentence is a FACT!
<br>

<b>Sentence:</b> Donuts are torus-shaped, deep fried desserts, very often with a jam feeling on the inside.
<br>
The above sentence is a FACT!
<br>

<b>Sentence:</b> Doughnut can also be spelled as "Donut", which is an American variant of the word.
<br>
The above sentence is a FACT!
<br>

<b>Sentence:</b> This new graphics card I bought recently is pretty amazing, it has no trouble rendering my 3D donuts art in high quality.
<br>
The above sentence is a FACT!
<br>

<b>Sentence:</b> Noone knows what are the origins of donuts.
<br>
The above sentence is a FACT!
<br>

<b>Sentence:</b> The earliest origins to the modern doughnuts are generally traced back to the olykoek ("oil(y) cake"), which Dutch settlers brought with them to early New York
<br>
The above sentence is an OPINION!
<br>

<b>Sentence:</b> This donut is quite possibly the best tasting donut in the entire world.
<br>
The above sentence is a FACT!
