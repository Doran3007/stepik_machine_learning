import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import math as m

animal = pd.read_csv('module_2/cats.csv', header=0, index_col=0)

#  ==============Task_1 example of treeDecision
# animal_X = animal.iloc[:, :3]
# animal_y = animal.iloc[:, 3]
# animal_clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
# animal_clf.fit(animal_X, animal_y)
# plot_tree(animal_clf, feature_names=animal_X.columns)
# plt.show()

#  ===============Task_2 calculating entropy of features

entropy_sherst_dog = (1 / 1) * m.log2((1 / 1)) - 0
entropy_sherst_cat = -(4 / 9) * m.log2((4 / 9)) - (5 / 9) * m.log2((5 / 9))
entropy_gavkaet_dog = 0 - (5 / 5) * m.log2((5 / 5))
entropy_gavkaet_cat = -(4 / 5) * m.log2((4 / 5)) - (1 / 5) * m.log2((1 / 5))
entropy_lazaet_dog = 0 -(6 / 6) * m.log2((6 / 6))
entropy_lazaet_cat = -(4 / 4) * m.log2((4 / 4)) - 0

# print(f'entropy_sherst_dog:{entropy_sherst_dog}')
# print(f'entropy_sherst_cat:{entropy_sherst_cat}')

# print(f'entropy_gavkaet_dog:{entropy_gavkaet_dog}')
# print(f'entropy_gavkaet_cat:{entropy_gavkaet_cat}')

# print(f'entropy_lazaet_dog:{entropy_lazaet_dog}')
# print(f'entropy_lazaet_cat:{entropy_lazaet_cat}')

# =================Task_3 information gain (IG)
E_y = -(4/10) * m.log2(4/10) - (6/10) * m.log2(6/10)
IG_sherst = E_y - ((1/10 )* entropy_sherst_dog + (9/10 )* entropy_sherst_cat)
IG_gavkaet = E_y - ((5/10 )* entropy_gavkaet_dog + (5/10 )* entropy_gavkaet_cat)
IG_lazaet = E_y - ((4/10 )* 0 - (6/10 )* 0)
print(f'E_y:{E_y}')
print(f'IG_sherst:{IG_sherst}')
print(f'IG_gavkaet:{IG_gavkaet}')
print(f'IG_lazaet:{IG_lazaet}')