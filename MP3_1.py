import numpy as np
import math
from scipy import spatial
import statistics
import os
import matplotlib as mpl
from matplotlib import pyplot

smoothing = 0.2

# get training data and set up probability matrix
num_train = 5000
text = open("trainingimages","r")
train_data = text.read().splitlines()
text.close()
text = open("traininglabels","r")
train_label = text.read().splitlines()
text.close()
counter = np.zeros((10,1))
prob_matrix_0 = np.zeros((10,28,28))
prob_matrix_1 = np.zeros((10,28,28))
for i in range(5000):
    label_num = int(train_label[i])
    #if i < 10:
    #    print(label_num)
    counter[label_num][0] = counter[label_num][0] + 1
    for j in range(28):
        line = train_data[28 * i + j]
    #    if i <10:
    #        print(line)
        for k in range(28):
            if line[k] is " ":
                prob_matrix_0[label_num][j][k] = prob_matrix_0[label_num][j][k] + 1
            else:
                prob_matrix_1[label_num][j][k] = prob_matrix_1[label_num][j][k] + 1

#print(prob_matrix_1[0])
#compute likelihood
for i in range(10):
    #print(counter[i][0])
    prob_matrix_1[i] = (prob_matrix_1[i] + smoothing) / (counter[i][0] + 2 * smoothing)
    prob_matrix_0[i] = (prob_matrix_0[i] + smoothing) / (counter[i][0] + 2 * smoothing)

#compute prior
counter = counter / 5000.0

#classify test data
mis_classified = 0
num_test = 1000
text = open("testimages","r")
test_data = text.read().splitlines()
text.close()
text = open("testlabels","r")
test_label = text.read().splitlines()
text.close()
confusion = np.zeros((10,10))
counter_test = np.zeros((10,1))
max_min_prob = np.zeros((10,2))
max_min_index = np.zeros((10,2))
for i in range(10):
    max_min_prob[i][0] = -math.inf
    max_min_prob[i][1] = math.inf
for i in range(1000):
    prob_vector_test = np.zeros((10,1))
    label_num = int(test_label[i])
    counter_test[label_num][0] = counter_test[label_num][0] + 1
    prob_vector_test = prob_vector_test + np.log(counter)
    for j in range(28):
        line = test_data[28 * i + j]
        for k in range(28):
            if line[k] is " ":
                for l in range(10):
                   prob_vector_test[l][0] += np.log(prob_matrix_0[l][j][k])
            else:
                for l in range(10):
                   prob_vector_test[l][0] += np.log(prob_matrix_1[l][j][k])
    maximum_prob = np.argmax(prob_vector_test)
    maximum_prob_value = np.max(prob_vector_test)
    if maximum_prob_value > max_min_prob[maximum_prob][0]:
        max_min_prob[maximum_prob][0] = maximum_prob_value
        max_min_index[maximum_prob][0] = i
    if maximum_prob_value < max_min_prob[maximum_prob][1]:
        max_min_prob[maximum_prob][1] = maximum_prob_value
        max_min_index[maximum_prob][1] = i
    if maximum_prob != label_num:
        mis_classified += 1
    confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1

for i in range(10):
    print("Highest posteriori Prob for Num" + str(i))
    for row in range(28):
        print(test_data[28 * int(np.squeeze(max_min_index[i][0])) + row])

    print("Lowest posteriori Prob for Num" + str(i))
    for row in range(28):
        print(test_data[28 * int(np.squeeze(max_min_index[i][1])) + row])

##print out confusion matrix
#for i in range(10):
#    print("Confusion rate for number " + str(i) + ":")
#    for j in range(10):
#        if i != j:
#            rate = np.squeeze(confusion[i][j]) / np.squeeze(counter_test[i][0]) * 100
#            print("Confused as " + str(j) + ": "+"%.3f" % rate + "%")

for i in range(10):
    print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
print("Confusion matrix(number in percentage)")
np.set_printoptions(precision=2)
print(confusion/counter_test * 100)


#COnfuse 4 as 9; 5 as 3; 7 as 9; 8 as 3
#4
fig4 = pyplot.figure(1)

cmap4 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['darkblue','green','yellow','orange','red'],
                                           16384)

img4 = pyplot.imshow(prob_matrix_1[4],interpolation='nearest',
                    cmap = cmap4,
                    origin='upper')


pyplot.colorbar(img4,cmap=cmap4)


fig4.savefig("image4.png")

#9
fig9 = pyplot.figure(2)

cmap9 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['darkblue','green','yellow','orange','red'],
                                           16384)

img9 = pyplot.imshow(prob_matrix_1[9],interpolation='nearest',
                    cmap = cmap9,
                    origin='upper')


pyplot.colorbar(img9,cmap=cmap9)


fig9.savefig("image9.png")

#4 vs 9
fig49 = pyplot.figure(3)

cmap49 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['darkblue','green','yellow','orange','red'],
                                           16384)

img49 = pyplot.imshow(np.divide(prob_matrix_1[4],prob_matrix_1[9]),interpolation='nearest',
                    cmap = cmap49,
                    origin='upper')


pyplot.colorbar(img49,cmap=cmap49)


fig49.savefig("image49.png")

#5
fig5 = pyplot.figure(4)

cmap5 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['darkblue','green','yellow','orange','red'],
                                           16384)

img5 = pyplot.imshow(prob_matrix_1[5],interpolation='nearest',
                    cmap = cmap5,
                    origin='upper')


pyplot.colorbar(img5,cmap=cmap5)


fig5.savefig("image5.png")

#3
fig3 = pyplot.figure(5)

cmap3 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['darkblue','green','yellow','orange','red'],
                                           16384)

img3 = pyplot.imshow(prob_matrix_1[3],interpolation='nearest',
                    cmap = cmap3,
                    origin='upper')


pyplot.colorbar(img3,cmap=cmap3)


fig3.savefig("image3.png")

#5 vs 3
fig53 = pyplot.figure(6)

cmap53 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['darkblue','green','yellow','orange','red'],
                                           16384)

img53 = pyplot.imshow(np.divide(prob_matrix_1[5],prob_matrix_1[3]),interpolation='nearest',
                    cmap = cmap53,
                    origin='upper')


pyplot.colorbar(img53,cmap=cmap53)


fig53.savefig("image53.png")

#7
fig7 = pyplot.figure(7)

cmap7 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['darkblue','green','yellow','orange','red'],
                                           16384)

img7 = pyplot.imshow(prob_matrix_1[7],interpolation='nearest',
                    cmap = cmap7,
                    origin='upper')


pyplot.colorbar(img7,cmap=cmap7)


fig7.savefig("image7.png")

#7 vs 9
fig79 = pyplot.figure(8)

cmap79 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['darkblue','green','yellow','orange','red'],
                                           16384)

img79 = pyplot.imshow(np.divide(prob_matrix_1[7],prob_matrix_1[9]),interpolation='nearest',
                    cmap = cmap79,
                    origin='upper')


pyplot.colorbar(img79,cmap=cmap79)


fig79.savefig("image79.png")

#8
fig8 = pyplot.figure(9)

cmap8 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['darkblue','green','yellow','orange','red'],
                                           16384)

img8 = pyplot.imshow(prob_matrix_1[8],interpolation='nearest',
                    cmap = cmap8,
                    origin='upper')


pyplot.colorbar(img8,cmap=cmap8)


fig8.savefig("image8.png")

#8 vs 3
fig83 = pyplot.figure(10)

cmap83 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['darkblue','green','yellow','orange','red'],
                                           16384)

img83 = pyplot.imshow(np.divide(prob_matrix_1[8],prob_matrix_1[3]),interpolation='nearest',
                    cmap = cmap83,
                    origin='upper')


pyplot.colorbar(img83,cmap=cmap83)


fig83.savefig("image83.png")
