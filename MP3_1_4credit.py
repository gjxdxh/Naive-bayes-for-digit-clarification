import numpy as np
import math
from scipy import spatial
import statistics
import os
import matplotlib as mpl
from matplotlib import pyplot
import time

########## These are for disjoint features

# 2*2 feature
def two_two_disjoint():
    #load train data
    smoothing = 0.0001
    tic = time.process_time()
    text = open("trainingimages","r")
    train_data = text.read().splitlines()
    text.close()
    text = open("traininglabels","r")
    train_label = text.read().splitlines()
    text.close()
    counter = np.zeros((10,1))
    prob_matrix = np.zeros((10,16,14,14))

    #calculate probability matrix using train data

    for i in range(5000):
        label_num = int(train_label[i])
        counter[label_num][0] = counter[label_num][0] + 1
        for j in range(14):
            for k in range(14):
                init_row = 28 * i + 2 * j
                init_col = k * 2
                bit_vector = 0
                for m in range(2):
                    for n in range(2):
                        if train_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                prob_matrix[label_num][bit_vector][j][k] += 1

    for i in range(10):
        for j in range(16):
            prob_matrix[i][j] = (prob_matrix[i][j] + smoothing) / (counter[i][0] + 16 * smoothing)

    #calculate prior
    counter = counter / 5000
    toc = time.process_time()
    print ("----- Computation time for training = " + str((toc - tic)) + "s")

    #classify test data and print confusion matrix
    tic = time.process_time()
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

    for i in range(1000):
        prob_vector_test = np.zeros((10,1))
        label_num = int(test_label[i])
        counter_test[label_num][0] = counter_test[label_num][0] + 1
        prob_vector_test = prob_vector_test + np.log(counter)
        for j in range(14):
            for k in range(14):
                init_row = 28 * i + 2 * j
                init_col = k * 2
                bit_vector = 0
                for m in range(2):
                    for n in range(2):
                        if test_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                for l in range(10):
                    prob_vector_test[l][0] += np.log(prob_matrix[l][bit_vector][j][k])
        maximum_prob = np.argmax(prob_vector_test)
        if maximum_prob != label_num:
            mis_classified += 1
        confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1

    toc = time.process_time()
    print ("----- Computation time for testing = " + str((toc - tic)) + "s")

    for i in range(10):
        print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
    print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
    print("Confusion matrix(number in percentage)")
    np.set_printoptions(precision=2)
    print(confusion/counter_test * 100)

# 2*4 feature
def two_four_disjoint():
    smoothing = 0.001

    #load test data
    tic = time.process_time()

    text = open("trainingimages","r")
    train_data = text.read().splitlines()
    text.close()
    text = open("traininglabels","r")
    train_label = text.read().splitlines()
    text.close()
    counter = np.zeros((10,1))
    prob_matrix = np.zeros((10,256,7,14))

    #calculate probability matrix using train data
    for i in range(5000):
        label_num = int(train_label[i])
        counter[label_num][0] = counter[label_num][0] + 1
        for j in range(7):
            for k in range(14):
                init_row = 28 * i + 4 * j
                init_col = k * 2
                bit_vector = 0
                for m in range(4):
                    for n in range(2):
                        if train_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                prob_matrix[label_num][bit_vector][j][k] += 1

    for i in range(10):
        for j in range(256):
            prob_matrix[i][j] = (prob_matrix[i][j] + smoothing) / (counter[i][0] + 256 * smoothing)

    counter = counter / 5000
    toc = time.process_time()
    print ("----- Computation time for training = " + str((toc - tic)) + "s")

    #classify test data and see performace
    tic = time.process_time()
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
    for i in range(1000):
        prob_vector_test = np.zeros((10,1))
        label_num = int(test_label[i])
        counter_test[label_num][0] = counter_test[label_num][0] + 1
        prob_vector_test = prob_vector_test + np.log(counter)
        for j in range(7):
            for k in range(14):
                init_row = 28 * i + 4 * j
                init_col = k * 2
                bit_vector = 0
                for m in range(4):
                    for n in range(2):
                        if test_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                for l in range(10):
                    prob_vector_test[l][0] += np.log(prob_matrix[l][bit_vector][j][k])
        maximum_prob = np.argmax(prob_vector_test)
        if maximum_prob != label_num:
            mis_classified += 1
        confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1

    toc = time.process_time()
    print ("----- Computation time for testing = " + str((toc - tic)) + "s")

    for i in range(10):
        print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
    print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
    print("Confusion matrix(number in percentage)")
    np.set_printoptions(precision=2)
    print(confusion/counter_test * 100)

# 4*2 feature
def four_two_disjoint():
    smoothing = 0.005

    tic = time.process_time()
    #load test data
    text = open("trainingimages","r")
    train_data = text.read().splitlines()
    text.close()
    text = open("traininglabels","r")
    train_label = text.read().splitlines()
    text.close()
    counter = np.zeros((10,1))
    prob_matrix = np.zeros((10,256,14,7))

    #calculate probability matrix using train data
    for i in range(5000):
        label_num = int(train_label[i])
        counter[label_num][0] = counter[label_num][0] + 1
        for j in range(14):
            for k in range(7):
                init_row = 28 * i + 2 * j
                init_col = k * 4
                bit_vector = 0
                for m in range(2):
                    for n in range(4):
                        if train_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                prob_matrix[label_num][bit_vector][j][k] += 1

    for i in range(10):
        for j in range(256):
            prob_matrix[i][j] = (prob_matrix[i][j] + smoothing) / (counter[i][0] + 256 * smoothing)

    counter = counter / 5000
    toc = time.process_time()
    print ("----- Computation time for training = " + str((toc - tic)) + "s")

    #classify test data and see performace
    tic = time.process_time()
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
    for i in range(1000):
        prob_vector_test = np.zeros((10,1))
        label_num = int(test_label[i])
        counter_test[label_num][0] = counter_test[label_num][0] + 1
        prob_vector_test = prob_vector_test + np.log(counter)
        for j in range(14):
            for k in range(7):
                init_row = 28 * i + 2 * j
                init_col = k * 4
                bit_vector = 0
                for m in range(2):
                    for n in range(4):
                        if test_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                for l in range(10):
                    prob_vector_test[l][0] += np.log(prob_matrix[l][bit_vector][j][k])
        maximum_prob = np.argmax(prob_vector_test)
        if maximum_prob != label_num:
            mis_classified += 1
        confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1

    toc = time.process_time()
    print ("----- Computation time for testing = " + str((toc - tic)) + "s")

    for i in range(10):
        print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
    print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
    print("Confusion matrix(number in percentage)")
    np.set_printoptions(precision=2)
    print(confusion/counter_test * 100)

# 4*4 feature
def four_four_disjoint():
    smoothing = 0.001

    #load test data
    tic = time.process_time()
    text = open("trainingimages","r")
    train_data = text.read().splitlines()
    text.close()
    text = open("traininglabels","r")
    train_label = text.read().splitlines()
    text.close()
    counter = np.zeros((10,1))
    prob_matrix = np.zeros((10,2**16,7,7))

    #calculate probability matrix using train data
    for i in range(5000):
        label_num = int(train_label[i])
        counter[label_num][0] = counter[label_num][0] + 1
        for j in range(7):
            for k in range(7):
                init_row = 28 * i + 4 * j
                init_col = k * 4
                bit_vector = 0
                for m in range(4):
                    for n in range(4):
                        if train_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                prob_matrix[label_num][bit_vector][j][k] += 1

    for i in range(10):
        for j in range(2**16):
            prob_matrix[i][j] = (prob_matrix[i][j] + smoothing) / (counter[i][0] + 2**16 * smoothing)

    counter = counter / 5000

    toc = time.process_time()
    print ("----- Computation time for training = " + str((toc - tic)) + "s")

    #classify test data and see performace
    tic = time.process_time()

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
    for i in range(1000):
        prob_vector_test = np.zeros((10,1))
        label_num = int(test_label[i])
        counter_test[label_num][0] = counter_test[label_num][0] + 1
        prob_vector_test = prob_vector_test + np.log(counter)
        for j in range(7):
            for k in range(7):
                init_row = 28 * i + 4 * j
                init_col = k * 4
                bit_vector = 0
                for m in range(4):
                    for n in range(4):
                        if test_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                for l in range(10):
                    prob_vector_test[l][0] += np.log(prob_matrix[l][bit_vector][j][k])
        maximum_prob = np.argmax(prob_vector_test)
        if maximum_prob != label_num:
            mis_classified += 1
        confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1


    toc = time.process_time()
    print ("----- Computation time for testing = " + str((toc - tic)) + "s")

    for i in range(10):
        print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
    print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
    print("Confusion matrix(number in percentage)")
    np.set_printoptions(precision=2)
    print(confusion/counter_test * 100)

########## These are for overlapping features

# 2*2 feature
def two_two_overlap():
    #load train data
    tic = time.process_time();

    smoothing = 0.00001
    text = open("trainingimages","r")
    train_data = text.read().splitlines()
    text.close()
    text = open("traininglabels","r")
    train_label = text.read().splitlines()
    text.close()
    counter = np.zeros((10,1))
    prob_matrix = np.zeros((10,16,27,27))

    #calculate probability matrix using train data
    for i in range(5000):
        label_num = int(train_label[i])
        counter[label_num][0] = counter[label_num][0] + 1
        for j in range(27):
            for k in range(27):
                init_row = 28 * i + 1 * j
                init_col = k
                bit_vector = 0
                for m in range(2):
                    for n in range(2):
                        if train_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                prob_matrix[label_num][bit_vector][j][k] += 1

    for i in range(10):
        for j in range(16):
            prob_matrix[i][j] = (prob_matrix[i][j] + smoothing) / (counter[i][0] + 16 * smoothing)

    #calculate prior
    counter = counter / 5000
    toc = time.process_time();
    print ("----- Computation time for training = " + str((toc - tic)) + "s")

    #classify test data and print confusion matrix
    tic = time.process_time();

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
    for i in range(1000):
        prob_vector_test = np.zeros((10,1))
        label_num = int(test_label[i])
        counter_test[label_num][0] = counter_test[label_num][0] + 1
        prob_vector_test = prob_vector_test + np.log(counter)
        for j in range(27):
            for k in range(27):
                init_row = 28 * i + 1 * j
                init_col = k
                bit_vector = 0
                for m in range(2):
                    for n in range(2):
                        if test_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                for l in range(10):
                    prob_vector_test[l][0] += np.log(prob_matrix[l][bit_vector][j][k])
        maximum_prob = np.argmax(prob_vector_test)
        if maximum_prob != label_num:
            mis_classified += 1
        confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1
    toc = time.process_time();
    print ("----- Computation time for testing = " + str((toc - tic)) + "s")

    for i in range(10):
        print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
    print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
    print("Confusion matrix(number in percentage)")
    np.set_printoptions(precision=2)
    print(confusion/counter_test * 100)

# 2*4 feature
def two_four_overlap():
    smoothing = 0.0001
    tic = time.process_time();

    #load test data
    text = open("trainingimages","r")
    train_data = text.read().splitlines()
    text.close()
    text = open("traininglabels","r")
    train_label = text.read().splitlines()
    text.close()
    counter = np.zeros((10,1))
    prob_matrix = np.zeros((10,256,25,27))

    #calculate probability matrix using train data
    for i in range(5000):
        label_num = int(train_label[i])
        counter[label_num][0] = counter[label_num][0] + 1
        for j in range(25):
            for k in range(27):
                init_row = 28 * i + 1 * j
                init_col = k
                bit_vector = 0
                for m in range(4):
                    for n in range(2):
                        if train_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                prob_matrix[label_num][bit_vector][j][k] += 1

    for i in range(10):
        for j in range(256):
            prob_matrix[i][j] = (prob_matrix[i][j] + smoothing) / (counter[i][0] + 256 * smoothing)

    counter = counter / 5000

    toc = time.process_time();
    print ("----- Computation time for training = " + str((toc - tic)) + "s")
    #classify test data and see performace

    tic = time.process_time();
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
    for i in range(1000):
        prob_vector_test = np.zeros((10,1))
        label_num = int(test_label[i])
        counter_test[label_num][0] = counter_test[label_num][0] + 1
        prob_vector_test = prob_vector_test + np.log(counter)
        for j in range(25):
            for k in range(27):
                init_row = 28 * i +  j
                init_col = k
                bit_vector = 0
                for m in range(4):
                    for n in range(2):
                        if test_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                for l in range(10):
                    prob_vector_test[l][0] += np.log(prob_matrix[l][bit_vector][j][k])
        maximum_prob = np.argmax(prob_vector_test)
        if maximum_prob != label_num:
            mis_classified += 1
        confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1

    toc = time.process_time();
    print ("----- Computation time for testing = " + str((toc - tic)) + "s")

    for i in range(10):
        print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
    print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
    print("Confusion matrix(number in percentage)")
    np.set_printoptions(precision=2)
    print(confusion/counter_test * 100)

# 4*2 feature
def four_two_overlap():
    smoothing = 0.001

    tic = time.process_time()
    #load test data
    text = open("trainingimages","r")
    train_data = text.read().splitlines()
    text.close()
    text = open("traininglabels","r")
    train_label = text.read().splitlines()
    text.close()
    counter = np.zeros((10,1))
    prob_matrix = np.zeros((10,256,27,25))

    #calculate probability matrix using train data
    for i in range(5000):
        label_num = int(train_label[i])
        counter[label_num][0] = counter[label_num][0] + 1
        for j in range(27):
            for k in range(25):
                init_row = 28 * i + j
                init_col = k
                bit_vector = 0
                for m in range(2):
                    for n in range(4):
                        if train_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                prob_matrix[label_num][bit_vector][j][k] += 1

    for i in range(10):
        for j in range(256):
            prob_matrix[i][j] = (prob_matrix[i][j] + smoothing) / (counter[i][0] + 256 * smoothing)

    counter = counter / 5000

    toc = time.process_time();
    print ("----- Computation time for training = " + str((toc - tic)) + "s")
    #classify test data and see performace

    tic = time.process_time();

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
    for i in range(1000):
        prob_vector_test = np.zeros((10,1))
        label_num = int(test_label[i])
        counter_test[label_num][0] = counter_test[label_num][0] + 1
        prob_vector_test = prob_vector_test + np.log(counter)
        for j in range(27):
            for k in range(25):
                init_row = 28 * i + j
                init_col = k
                bit_vector = 0
                for m in range(2):
                    for n in range(4):
                        if test_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                for l in range(10):
                    prob_vector_test[l][0] += np.log(prob_matrix[l][bit_vector][j][k])
        maximum_prob = np.argmax(prob_vector_test)
        if maximum_prob != label_num:
            mis_classified += 1
        confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1

    toc = time.process_time();
    print ("----- Computation time for testing = " + str((toc - tic)) + "s")

    for i in range(10):
        print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
    print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
    print("Confusion matrix(number in percentage)")
    np.set_printoptions(precision=2)
    print(confusion/counter_test * 100)

# 4*4 feature
def four_four_overlap():
    smoothing = 0.0001
    tic = time.process_time();

    #load test data
    text = open("trainingimages","r")
    train_data = text.read().splitlines()
    text.close()
    text = open("traininglabels","r")
    train_label = text.read().splitlines()
    text.close()
    counter = np.zeros((10,1))
    prob_matrix = np.zeros((10,2**16,25,25))

    #calculate probability matrix using train data
    for i in range(5000):
        label_num = int(train_label[i])
        counter[label_num][0] = counter[label_num][0] + 1
        for j in range(25):
            for k in range(25):
                init_row = 28 * i + j
                init_col = k
                bit_vector = 0
                for m in range(4):
                    for n in range(4):
                        if train_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                prob_matrix[label_num][bit_vector][j][k] += 1

    for i in range(10):
        for j in range(2**16):
            prob_matrix[i][j] = (prob_matrix[i][j] + smoothing) / (counter[i][0] + 2**16 * smoothing)

    counter = counter / 5000
    toc = time.process_time();
    print ("----- Computation time for training = " + str((toc - tic)) + "s")

    #classify test data and see performace
    tic = time.process_time();

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
    for i in range(1000):
        prob_vector_test = np.zeros((10,1))
        label_num = int(test_label[i])
        counter_test[label_num][0] = counter_test[label_num][0] + 1
        prob_vector_test = prob_vector_test + np.log(counter)
        for j in range(25):
            for k in range(25):
                init_row = 28 * i + j
                init_col = k
                bit_vector = 0
                for m in range(4):
                    for n in range(4):
                        if test_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                for l in range(10):
                    prob_vector_test[l][0] += np.log(prob_matrix[l][bit_vector][j][k])
        maximum_prob = np.argmax(prob_vector_test)
        if maximum_prob != label_num:
            mis_classified += 1
        confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1
    toc = time.process_time();
    print ("----- Computation time for testing = " + str((toc - tic)) + "s")

    for i in range(10):
        print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
    print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
    print("Confusion matrix(number in percentage)")
    np.set_printoptions(precision=2)
    print(confusion/counter_test * 100)

# 2*3 feature
def two_three_overlap():
    smoothing = 0.001
    tic = time.process_time();

    #load test data
    text = open("trainingimages","r")
    train_data = text.read().splitlines()
    text.close()
    text = open("traininglabels","r")
    train_label = text.read().splitlines()
    text.close()
    counter = np.zeros((10,1))
    prob_matrix = np.zeros((10,2**6,26,27))

    #calculate probability matrix using train data
    for i in range(5000):
        label_num = int(train_label[i])
        counter[label_num][0] = counter[label_num][0] + 1
        for j in range(26):
            for k in range(27):
                init_row = 28 * i + 1 * j
                init_col = k
                bit_vector = 0
                for m in range(3):
                    for n in range(2):
                        if train_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                prob_matrix[label_num][bit_vector][j][k] += 1

    for i in range(10):
        for j in range(2**6):
            prob_matrix[i][j] = (prob_matrix[i][j] + smoothing) / (counter[i][0] + 2**6 * smoothing)

    counter = counter / 5000
    toc = time.process_time();
    print ("----- Computation time for training = " + str((toc - tic)) + "s")
    #classify test data and see performace
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
    for i in range(1000):
        prob_vector_test = np.zeros((10,1))
        label_num = int(test_label[i])
        counter_test[label_num][0] = counter_test[label_num][0] + 1
        prob_vector_test = prob_vector_test + np.log(counter)
        for j in range(26):
            for k in range(27):
                init_row = 28 * i +  j
                init_col = k
                bit_vector = 0
                for m in range(3):
                    for n in range(2):
                        if test_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                for l in range(10):
                    prob_vector_test[l][0] += np.log(prob_matrix[l][bit_vector][j][k])
        maximum_prob = np.argmax(prob_vector_test)
        if maximum_prob != label_num:
            mis_classified += 1
        confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1
    toc = time.process_time();
    print ("----- Computation time for testing = " + str((toc - tic)) + "s")

    for i in range(10):
        print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
    print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
    print("Confusion matrix(number in percentage)")
    np.set_printoptions(precision=2)
    print(confusion/counter_test * 100)

# 3*2 feature
def three_two_overlap():
    smoothing = 0.00001
    tic = time.process_time();

    #load test data
    text = open("trainingimages","r")
    train_data = text.read().splitlines()
    text.close()
    text = open("traininglabels","r")
    train_label = text.read().splitlines()
    text.close()
    counter = np.zeros((10,1))
    prob_matrix = np.zeros((10,2**6,27,26))

    #calculate probability matrix using train data
    for i in range(5000):
        label_num = int(train_label[i])
        counter[label_num][0] = counter[label_num][0] + 1
        for j in range(27):
            for k in range(26):
                init_row = 28 * i + j
                init_col = k
                bit_vector = 0
                for m in range(2):
                    for n in range(3):
                        if train_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                prob_matrix[label_num][bit_vector][j][k] += 1

    for i in range(10):
        for j in range(2**6):
            prob_matrix[i][j] = (prob_matrix[i][j] + smoothing) / (counter[i][0] + 2**6 * smoothing)

    counter = counter / 5000
    toc = time.process_time();
    print ("----- Computation time for training = " + str((toc - tic)) + "s")
    #classify test data and see performace

    tic = time.process_time();

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
    for i in range(1000):
        prob_vector_test = np.zeros((10,1))
        label_num = int(test_label[i])
        counter_test[label_num][0] = counter_test[label_num][0] + 1
        prob_vector_test = prob_vector_test + np.log(counter)
        for j in range(27):
            for k in range(26):
                init_row = 28 * i + j
                init_col = k
                bit_vector = 0
                for m in range(2):
                    for n in range(3):
                        if test_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                for l in range(10):
                    prob_vector_test[l][0] += np.log(prob_matrix[l][bit_vector][j][k])
        maximum_prob = np.argmax(prob_vector_test)
        if maximum_prob != label_num:
            mis_classified += 1
        confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1
    toc = time.process_time();
    print ("----- Computation time for testing = " + str((toc - tic)) + "s")
    for i in range(10):
        print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
    print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
    print("Confusion matrix(number in percentage)")
    np.set_printoptions(precision=2)
    print(confusion/counter_test * 100)

# 3*3 feature
def three_three_overlap():
    smoothing = 0.001

    #load test data
    tic = time.process_time();
    text = open("trainingimages","r")
    train_data = text.read().splitlines()
    text.close()
    text = open("traininglabels","r")
    train_label = text.read().splitlines()
    text.close()
    counter = np.zeros((10,1))
    prob_matrix = np.zeros((10,2**9,26,26))

    #calculate probability matrix using train data
    for i in range(5000):
        label_num = int(train_label[i])
        counter[label_num][0] = counter[label_num][0] + 1
        for j in range(26):
            for k in range(26):
                init_row = 28 * i + j
                init_col = k
                bit_vector = 0
                for m in range(3):
                    for n in range(3):
                        if train_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                prob_matrix[label_num][bit_vector][j][k] += 1

    for i in range(10):
        for j in range(2**9):
            prob_matrix[i][j] = (prob_matrix[i][j] + smoothing) / (counter[i][0] + 2**9 * smoothing)

    counter = counter / 5000
    toc = time.process_time();
    print ("----- Computation time for training= " + str((toc - tic)) + "s")
    #classify test data and see performace

    tic = time.process_time()
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
    for i in range(1000):
        prob_vector_test = np.zeros((10,1))
        label_num = int(test_label[i])
        counter_test[label_num][0] = counter_test[label_num][0] + 1
        prob_vector_test = prob_vector_test + np.log(counter)
        for j in range(26):
            for k in range(26):
                init_row = 28 * i + j
                init_col = k
                bit_vector = 0
                for m in range(3):
                    for n in range(3):
                        if test_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 2 + 0
                        else:
                            bit_vector = bit_vector * 2 + 1
                for l in range(10):
                    prob_vector_test[l][0] += np.log(prob_matrix[l][bit_vector][j][k])
        maximum_prob = np.argmax(prob_vector_test)
        if maximum_prob != label_num:
            mis_classified += 1
        confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1

    toc = time.process_time();
    print ("----- Computation time for testing = " + str((toc - tic)) + "s")
    for i in range(10):
        print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
    print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
    print("Confusion matrix(number in percentage)")
    np.set_printoptions(precision=2)
    print(confusion/counter_test * 100)
#3*2
def generalized_overlap_ternary(x_size,y_size):
    smoothing = 0.00000001

    #load test data
    text = open("trainingimages","r")
    train_data = text.read().splitlines()
    text.close()
    text = open("traininglabels","r")
    train_label = text.read().splitlines()
    text.close()
    counter = np.zeros((10,1))
    prob_matrix = np.zeros((10,3**(x_size * y_size),(28-y_size+1),(28-x_size+1)))

    #calculate probability matrix using train data
    for i in range(5000):
        label_num = int(train_label[i])
        counter[label_num][0] = counter[label_num][0] + 1
        for j in range(28-y_size+1):
            for k in range(28-x_size+1):
                init_row = 28 * i + j
                init_col = k
                bit_vector = 0
                for m in range(y_size):
                    for n in range(x_size):
                        if train_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 3 + 0
                        elif train_data[init_row + m][init_col + n] is "+":
                            bit_vector = bit_vector * 3 + 1
                        else:
                            bit_vector = bit_vector * 3 + 2
                prob_matrix[label_num][bit_vector][j][k] += 1

    for i in range(10):
        for j in range(3**(x_size * y_size)):
            prob_matrix[i][j] = (prob_matrix[i][j] + smoothing) / (counter[i][0] + 3**(x_size * y_size) * smoothing)

    counter = counter / 5000

    #classify test data and see performace
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
    for i in range(1000):
        prob_vector_test = np.zeros((10,1))
        label_num = int(test_label[i])
        counter_test[label_num][0] = counter_test[label_num][0] + 1
        prob_vector_test = prob_vector_test + np.log(counter)
        for j in range(28-y_size + 1):
            for k in range(28 - x_size +1):
                init_row = 28 * i + j
                init_col = k
                bit_vector = 0
                for m in range(y_size):
                    for n in range(x_size):
                        if test_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 3 + 0
                        elif test_data[init_row + m][init_col + n] is "+":
                            bit_vector = bit_vector * 3 + 1
                        else:
                            bit_vector = bit_vector * 3 + 2
                for l in range(10):
                    prob_vector_test[l][0] += np.log(prob_matrix[l][bit_vector][j][k])
        maximum_prob = np.argmax(prob_vector_test)
        if maximum_prob != label_num:
            mis_classified += 1
        confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1

    for i in range(10):
        print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
    print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
    print("Confusion matrix(number in percentage)")
    np.set_printoptions(precision=2)
    print(confusion/counter_test * 100)

def generalized_disjoint_ternary(x_size,y_size):
    smoothing = 0.001

    #load test data
    text = open("trainingimages","r")
    train_data = text.read().splitlines()
    text.close()
    text = open("traininglabels","r")
    train_label = text.read().splitlines()
    text.close()
    counter = np.zeros((10,1))
    prob_matrix = np.zeros((10,3**(x_size * y_size),(math.floor(28/y_size)),(math.floor(28/x_size))))

    #calculate probability matrix using train data
    for i in range(5000):
        label_num = int(train_label[i])
        counter[label_num][0] = counter[label_num][0] + 1
        for j in range(math.floor(28/y_size)):
            for k in range(math.floor(28/x_size)):
                init_row = 28 * i + j * y_size
                init_col = k * x_size
                bit_vector = 0
                for m in range(y_size):
                    for n in range(x_size):
                        if train_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 3 + 0
                        elif train_data[init_row + m][init_col + n] is "+":
                            bit_vector = bit_vector * 3 + 1
                        else:
                            bit_vector = bit_vector * 3 + 2
                prob_matrix[label_num][bit_vector][j][k] += 1

    for i in range(10):
        for j in range(3**(x_size * y_size)):
            prob_matrix[i][j] = (prob_matrix[i][j] + smoothing) / (counter[i][0] + 3**(x_size * y_size) * smoothing)

    counter = counter / 5000

    #classify test data and see performace
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
    for i in range(1000):
        prob_vector_test = np.zeros((10,1))
        label_num = int(test_label[i])
        counter_test[label_num][0] = counter_test[label_num][0] + 1
        prob_vector_test = prob_vector_test + np.log(counter)
        for j in range(math.floor(28/y_size)):
            for k in range(math.floor(28/x_size)):
                init_row = 28 * i + j * y_size
                init_col = k * x_size
                bit_vector = 0
                for m in range(y_size):
                    for n in range(x_size):
                        if test_data[init_row + m][init_col + n] is " ":
                            bit_vector = bit_vector * 3 + 0
                        elif test_data[init_row + m][init_col + n] is "+":
                            bit_vector = bit_vector * 3 + 1
                        else:
                            bit_vector = bit_vector * 3 + 2
                for l in range(10):
                    prob_vector_test[l][0] += np.log(prob_matrix[l][bit_vector][j][k])
        maximum_prob = np.argmax(prob_vector_test)
        if maximum_prob != label_num:
            mis_classified += 1
        confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1

    for i in range(10):
        print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
    print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
    print("Confusion matrix(number in percentage)")
    np.set_printoptions(precision=2)
    print(confusion/counter_test * 100)

def three_hori_conse():
    smoothing = 0.1

    #load test data
    text = open("trainingimages","r")
    train_data = text.read().splitlines()
    text.close()
    text = open("traininglabels","r")
    train_label = text.read().splitlines()
    text.close()
    counter = np.zeros((10,1))
    prob_matrix = np.zeros((10,2,28,28))

    #calculate probability matrix using train data
    for i in range(5000):
        label_num = int(train_label[i])
        counter[label_num][0] = counter[label_num][0] + 1
        for j in range(28):
            for k in range(28):
                if k+1 <28 and k+2 <28 and train_data[28*i+j][k] != " " and train_data[28*i+j][k+1] != " " and train_data[28*i+j][k+2] != " ":
                    prob_matrix[label_num][1][j][k] += 1
                else:
                    prob_matrix[label_num][0][j][k] +=1

    for i in range(10):
        for j in range(2):
            prob_matrix[i][j] = (prob_matrix[i][j] + smoothing) / (counter[i][0] + 2 * smoothing)

    counter = counter / 5000

    #classify test data and see performace
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
    for i in range(1000):
        prob_vector_test = np.zeros((10,1))
        label_num = int(test_label[i])
        counter_test[label_num][0] = counter_test[label_num][0] + 1
        prob_vector_test = prob_vector_test + np.log(counter)
        for j in range(28):
            for k in range(28):
                if k+1 <28 and k+2 <28 and test_data[28*i+j][k] != " " and test_data[28*i+j][k+1] != " " and test_data[28*i+j][k+2] != " ":
                    bit_vector = 1
                else:
                    bit_vector = 0
                for l in range(10):
                    prob_vector_test[l][0] += np.log(prob_matrix[l][bit_vector][j][k])
        maximum_prob = np.argmax(prob_vector_test)
        if maximum_prob != label_num:
            mis_classified += 1
        confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1

    for i in range(10):
        print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
    print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
    print("Confusion matrix(number in percentage)")
    np.set_printoptions(precision=2)
    print(confusion/counter_test * 100)

def three_verti_conse():
    smoothing = 0.1

    #load test data
    text = open("trainingimages","r")
    train_data = text.read().splitlines()
    text.close()
    text = open("traininglabels","r")
    train_label = text.read().splitlines()
    text.close()
    counter = np.zeros((10,1))
    prob_matrix = np.zeros((10,2,28,28))

    #calculate probability matrix using train data
    for i in range(5000):
        label_num = int(train_label[i])
        counter[label_num][0] = counter[label_num][0] + 1
        for j in range(28):
            for k in range(28):
                if j+1 <28 and j+2 <28 and train_data[28*i+j][k] != " " and train_data[28*i+j+1][k] != " " and train_data[28*i+j+2][k] != " ":
                    prob_matrix[label_num][1][j][k] += 1
                else:
                    prob_matrix[label_num][0][j][k] +=1

    for i in range(10):
        for j in range(2):
            prob_matrix[i][j] = (prob_matrix[i][j] + smoothing) / (counter[i][0] + 2 * smoothing)

    counter = counter / 5000

    #classify test data and see performace
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
    for i in range(1000):
        prob_vector_test = np.zeros((10,1))
        label_num = int(test_label[i])
        counter_test[label_num][0] = counter_test[label_num][0] + 1
        prob_vector_test = prob_vector_test + np.log(counter)
        for j in range(28):
            for k in range(28):
                if j+1 <28 and j+2 <28 and test_data[28*i+j][k] != " " and test_data[28*i+j+1][k] != " " and test_data[28*i+j+2][k] != " ":
                    bit_vector = 1
                else:
                    bit_vector = 0
                for l in range(10):
                    prob_vector_test[l][0] += np.log(prob_matrix[l][bit_vector][j][k])
        maximum_prob = np.argmax(prob_vector_test)
        if maximum_prob != label_num:
            mis_classified += 1
        confusion[label_num][maximum_prob] = confusion[label_num][maximum_prob] + 1

    for i in range(10):
        print("Classification rate for Number " + str(i) + ": " + str(np.squeeze(confusion[i][i]) / np.squeeze(counter_test[i][0]) * 100) + "%")
    print("Overall confusion rate = " + str(mis_classified/1000.0 * 100) + "%")
    print("Confusion matrix(number in percentage)")
    np.set_printoptions(precision=2)
    print(confusion/counter_test * 100)

"""
two_two_disjoint()
two_four_disjoint()
four_two_disjoint()
four_four_disjoint()
"""

two_two_overlap()
two_four_overlap()
four_two_overlap()
four_four_overlap()
two_three_overlap()
three_two_overlap()
three_three_overlap()
