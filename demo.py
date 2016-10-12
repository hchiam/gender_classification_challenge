# challenge instructions: https://github.com/llSourcell/gender_classification_challenge
# get classifiers: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# get scores: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

from sklearn import tree

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()

## CHALLENGE - create 3 more classifiers...
# get classifiers: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

#1
clf1 = GaussianNB()
#2
clf2 = KNeighborsClassifier(n_neighbors=3)
#3
clf3 = SVC()

#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']


#CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
prediction = clf.predict([[190, 70, 43]])

#1
clf1 = clf1.fit(X, Y)
prediction1 = clf1.predict([[190, 70, 43]])
#2
clf2 = clf2.fit(X, Y)
prediction2 = clf2.predict([[190, 70, 43]])
#3
clf3 = clf3.fit(X, Y)
prediction3 = clf3.predict([[190, 70, 43]])

#CHALLENGE compare their results and print the best one!
# get scores: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

print prediction
print prediction1
print prediction2
print prediction3