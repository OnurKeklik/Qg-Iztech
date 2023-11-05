Rule-based Automatic Question Generation Using Semantic Role Labeling
=============

Code for IEICE 2019 paper [Rule-based Automatic Question Generation Using Semantic Role Labeling](https://search.ieice.org/bin/pdf_link.php?category=D&lang=E&year=2019&fname=e102-d_7_1362)


The designed system is written in Python 3.6.
Parsers:
* -SENNA's part-of-speech tagging and chunking,
* -SpaCy's dependency parsing and NER,
* -AllenNLP's semantic role labeling


To install:
* python3 setup.py install
* pip3 install spacy==2.0.18
* python3 -m spacy download en
* pip3 install allennlp==0.6.0
* pip3 install scikit-learn==0.22.2

To run:
* cd pntl
* python3 tools.py

Edit \__main__ in tools.py and add your own sentences to get corresponding questions