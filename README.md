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

## Samples
Using rf_classifier (random forest)
Your sentence: "As far as I am concerned, donuts are amazing." is an OPINION!

Using svm_classifier (support vector machine)
Your sentence: "Donuts are a kind of ring-shaped, deep fried dessert." is a FACT!

Using lr_classifier (logistic regression)
Your sentence: "Doughnut can also be spelled as "Donut", which is an American variant of the word." is a FACT!

Using nn_classifier (neural network)
Your sentence: "This new graphics card I bought recently is pretty amazing, it has no trouble rendering my 3D donuts art in high quality." is a FACT!

Using nn_classifier (neural network)
Your sentence: "I think this new graphics card is amazing, it has no trouble rendering my 3D donuts art in high quality." is an OPINION!
