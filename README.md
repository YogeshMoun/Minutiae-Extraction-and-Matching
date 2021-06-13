# Minutiae-Extraction-and-Matching

# Minutiae Extraction
The important fingerprint minutiae features are the ridge endpoints (a.k.a. Terminations) and Ridge Bifurcations.

![image](https://user-images.githubusercontent.com/13918778/35665327-9ddbd220-06da-11e8-8fa9-1f5444ee2036.png)
![image](https://github.com/YogeshMoun/Minutiae-Extraction-and-Matching/blob/main/result/minutiae_1.png)

The feature set for the image consists of the location of Terminations and Bifurcations and their orientations

# Fingerprint recognition¶
Fingerprint recognition is probably the most mature biometric technique, which finds lots of real - life applications since long ago. This notebook is divided in two main parts: Identification and Authentication / Verification with fingerprints.
we read the files and we prepare the dataset. Each image is converted to grayscale version and then enhancement is applied by using the following library for fingerprints enhancement in Python: Fingerprint-Enhancement-Python. It uses oriented Gabor filter (a linear filter in image processing used for texture analysis) to enhance the fingerprint image.

The dataset is split into training and test set, with ratio 80:20.
## Basics of Brute-Force Matcher
Brute-Force matcher is simple. It takes the descriptor of one feature in first set and is matched with all other features in second set using some distance calculation. And the closest one is returned.

For BF matcher, first we have to create the BFMatcher object using cv.BFMatcher(). It takes two optional params. First one is normType. It specifies the distance measurement to be used. By default, it is cv.NORM_L2. It is good for SIFT, SURF etc (cv.NORM_L1 is also there). For binary string based descriptors like ORB, BRIEF, BRISK etc, cv.NORM_HAMMING should be used, which used Hamming distance as measurement. If ORB is using WTA_K == 3 or 4, cv.NORM_HAMMING2 should be used.

Second param is boolean variable, crossCheck which is false by default. If it is true, Matcher returns only those matches with value (i,j) such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa. That is, the two features in both sets should match each other. It provides consistent result, and is a good alternative to ratio test proposed by D.Lowe in SIFT paper.

Once it is created, two important methods are BFMatcher.match() and BFMatcher.knnMatch(). First one returns the best match. Second method returns k best matches where k is specified by the user. It may be useful when we need to do additional work on that.

Like we used cv.drawKeypoints() to draw keypoints, cv.drawMatches() helps us to draw the matches. It stacks two images horizontally and draw lines from first image to second image showing best matches. There is also cv.drawMatchesKnn which draws all the k best matches. If k=2, it will draw two match-lines for each keypoint. So we have to pass a mask if we want to selectively draw it.

### Brute-Force Matching with ORB Descriptors
ORB (Oriented FAST and Rotated BRIEF) descriptor is used to find matching keypoints. As a matching function we use number of matching features whose distance is below a given threshold.
## Identification scenario
First we analyse the identificaion scenario, which corresponds to 1:M clasification problem. Biometric identification answers the question “Who are you?”. It is usually applied in a situation where an organization needs to identify a person. The organization captures a biometric from that individual and then searches in a database in order to correctly identify the person

After the identification scenario method is defined, we continue with testing it. By iterating through different distance thresholds we try to find the most appropriate threshold size. Taking into account the size of the training set for each person, we take 3 as rank, which means that the first 3 closest fingerprints are considered when classifing the query image. This is actially similar to what kNN does, where k = 3.

## Authentication scenario
Second, we analyse the authentication scenario, which corresponds to 1:1 problem or binary classification. The question that is asked is: “Can you prove who you are?”. A system will challenge someone to prove their identity and the person has to respond in order to allow them access to a system or service. For example, a person touches their finger on a sensor embedded in a smartphone, used by the authentication solution as part of a challenge/response system. "Is it my finger? Yes, then my smartphone is unlocked, or No it isn’t my finger and the smartphone remains locked."

For the authentication scenairo, the data structure for training is slightly different, whereas the test set remains the same. For the training set, the already computed features are divided in separate dictionaries (that act as databases) where the key denotes the class (person), and then every image features for the correspoding class are set in the dictionary as a value
![image](https://github.com/YogeshMoun/Minutiae-Extraction-and-Matching/blob/main/result/matched_1.png)

# Data :
The data consists of images of fingerprints, provided as part of [FVC2002](http://bias.csr.unibo.it/fvc2002/download.asp): the Second International Competition for Fingerprint Verification Algorithms. There are 4 available datasets and we chose to work with the second dataset: DB1, which consists of 80 images of fingerprints that belong to 10 different individuals (or classes), which gives 8 images per person.
# PyEER :
PyEER is a python package intended for biometric systems performance evaluation but it can be used to evaluate binary classification systems also. It has been developed with the idea of providing researchers and the scientific community in general with a tool to correctly evaluate and report the performance of their systems.
This package is used for computing the EER, as well as the FAR and FRR later.

EER represents a point where both FAR and FRR are equal.
## Installing
To install the PyEER package you only need to type into a terminal the following line:

**pip install pyeer**
## Input file formats
Genuine match scores and impostor match scores must be provided in separated files one score per line. Each line can have any number of columns but the scores must be in the last column. Additionally, impostor match scores can be provided in a different format which explained next
## Evaluation
The evaluation of 1:1 biometric authentication systems is usually done by estimating False Accept Rate (FAR) and False Reject Rate (FRR). Using these estimates, a ROC curve is generated. Compared to the standard ROC curve, where we plot the True Positive Rate (TPR) against the False Positive Rate (FAR), in the domain of biometric sytems, instead of TPR, we plot FRR (or 1 - TPR).

![image](https://github.com/YogeshMoun/Minutiae-Extraction-and-Matching/blob/main/result/roc_1.png)

# Execution of code :
Firstly , you have to change path of dataset inside the utils.py acc. to your dataset location.

**python main.py 101_2.tif**

Image name should be in [101-110]_{2|6}.tif , Because these images are in test set , so you can test these images and search into the database otherwise give error !
