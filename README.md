Rule-based Automatic Question Generation Using Semantic Role Labeling
=============


The designed system is written in Python 3.6.
Parsers:
-SENNA's part-of-speech tagging and chunking,
-SpaCy's dependency parsing and NER,
-AllenNLP's semantic role labeling


To install run:
python3 setup.py install
pip3 install spacy==2.0.18
python3 -m spacy download en
pip3 install allennlp==0.6.0
pip3 install scikit-learn==0.22.2