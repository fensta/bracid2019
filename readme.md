Summary
=======
This project implements BRACID as described in the [paper](https://r2s.hh.se/ReadingClub/2013-05-17/1.s10844-011-0193-0.pdf):

"Napierala, Krystyna, and Jerzy Stefanowski. "BRACID: a comprehensive approach to learning rules from imbalanced data." Journal of Intelligent Information Systems 39.2 (2012): 335-373."

Our implementation might differ in some details from the original implementation, but the general idea should be correct.
Moreover, our implementation is able to deal with multiclass classification problems by converting a multiclass problem into multiple binary problems according to the one-vs-all scheme. In that case, the predicted label of an example is set to be the one with the maximum confidence across all available class labels. 

For example, given that classes *A*, *B*, and *C* exist in a dataset and the model trained on the rules discovered by BRACID computes as confidences for the unlabeled example 0.3, 0.4, 0.1 for the classes *A*, *B*, and *C* respectively,
*B* is chosen as the final label. Confidences are obtained from the supports as follows:
 
 * for binary classification problems:
        `Confidence = max (support for minority, support for majority class) / (support for minority class + support for majority class)`
 
 * for multiclass classification problems (remember, it was converted into a binary task where all other classes are merged into one to which we refer as `majority class` below):
        `Confidence = support for minority class / (support for minority class + support for majority class)` 

Usage
=====
Most importantly, all relevant functions for BRACID currently reside in `scripts/utils.py`.

The functions for predicting the labels of unknown examples are:
 * for binary classification: `extract_rules_and_train_and_predict_binary`
 * for multiclass classification: `extract_rules_and_train_and_predict_multiclass()`

The functions for estimating the performance are:
 * for binary classification: `cv_binary()`
 * for multiclass classification: `cv_multiclass()`

All of the abovementioned functions serve as wrapper functions that internally call functions that:
1. given an input dataset, discover rules using BRACID
2. compute the support of the discovered rules based on the training set
3. predict the labels of the test set

All unit test are stored in `unit_tests/` and they also serve as sample code.

Citation
========
If you use this implementation in your paper, please cite our paper: XXX

License
=======
MIT license

Contact
=======
In case of questions, feel free to contact me: ***fensta (comma) git (where) gmail (comma) com***

Replace (where) by "@" and (comma) by ".", and remove the whitespaces in the email address above.

TODOs
=====
- [x] Implement BRACID
- [ ] Remove `print()` statements
- [ ] Refractor code and create a BRACID class
- [ ] Add reference to our paper 