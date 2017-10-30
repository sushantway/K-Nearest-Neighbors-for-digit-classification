import os
import struct
import numpy as np


def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()



def c_question_graph_plot():
    #I put the values by running the code for different training values. List Y has the accuracy values
    #for the different training dataset values in x
    k = 1
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    y = [43.8,64.6,62.8,65.6,67.4,68.6,61.8,63.9,66.3,66.9]
    x = [30,1000,2000,3000,4000,5000,6000,7500,9000,10000]
    plt.xlabel('Number of Training images')
    plt.ylabel('Accuracy')
    plt.plot(x, y, linewidth=2.0)
    plt.show()
    print("plotted")

def d_question_graph_plot():
    #I put the values by running the code for different K values. List y1,y2,y3,y5,y10 has the accuracy values
    #for the different training dataset values in x
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    k = [1,2,3,5,10]
    x = [30,1000,2000,3000,4000,5000,6000,7500,9000,10000]
    y1 = [43.8,64.6,62.8,65.6,67.4,68.6,61.8,63.9,66.3,66.9]
    y2 = [33.9,66.0,62.3,65.0,67.3,68.8,67.9,68.9,69.7,70.3]
    y3 = [32.5,66.0,64.9,66.8,69.0,70.8,70.4,71.5,71.7,72.9]
    y5 = [33.2,67.4,65.4,68.3,70.5,71.8,71.1,72.3,73.2,74.0]
    y10 = [26.6,66.7,65.3,69.8,71.4,72.2,72.7,73.2,73.7,73.9]
    plt.xlabel('Number of Training images')
    plt.ylabel('Accuracy')
    plt.plot(x, y1, label='k=1',linewidth=2.0)
    plt.plot(x, y2, label='k=2',linewidth=2.0)
    plt.plot(x, y3, label='k=3',linewidth=2.0)
    plt.plot(x, y5, label='k=5',linewidth=2.0)
    plt.plot(x, y10, label='k=10',linewidth=2.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def e_question_graph_plot():
    #I changed the K values for below piece of code and got the accuracy values in Y. X has the K values.
    mnist_train = read("training")
    #image = mnist.next()
    #print(image)

    #show(image[1])
    train_labels1 = []
    train_images1 = []
    train_labels = []
    train_images = []
    test_labels = []
    test_images = []
    k = 10
    for i in range(2000):
        label_image = mnist_train.next()
        train_labels1.append(label_image[0])
        train_images1.append(label_image[1])

    for j in range(1000):
        #print(j)
        train_labels.append(train_labels1[j])
        train_images.append(train_images1[j])

    for m in range(1000,2000):
        #print(k)
        test_labels.append(train_labels1[m])
        test_images.append(train_images1[m])



    #kNN(train_images, train_labels, test_images, test_labels, k)
    from matplotlib import pyplot
    import matplotlib.pyplot as plt
    x = [1,2,3,5,7,10]
    y = [68.4,65.3,67.3,68.3,68.4,69.0]
    plt.xlabel('Values for K')
    plt.ylabel('Accuracy')
    plt.plot(x, y,linewidth=2.0)
    plt.show()
    





def calculate_euclidian(image1, image2):
    dist = np.linalg.norm(image1-image2)
    return dist

def kNN(train_images, train_labels, test_images, test_labels, k):
    #print(train_labels)
    #print(test_labels)
    acc = []
    euclidian_distance_matrix = []
    temp_list = []
    correct = 0.0
    count_0 = 0.0
    count_1 = 0.0
    count_2 = 0.0
    count_3 = 0.0
    count_4 = 0.0
    count_5 = 0.0
    count_6 = 0.0
    count_7 = 0.0
    count_8 = 0.0
    count_9 = 0.0
    for i in range(len(test_images)):
        euclidian_distance_list = []
        for j in range(len(train_images)):
            euclidian_distance = calculate_euclidian(test_images[i],train_images[j])
            euclidian_distance_list.append(euclidian_distance)
        vals = np.array(euclidian_distance_list)
        sort_index = np.argsort(vals)
        temp_list = sort_index[:k]
        #print("temp_list",temp_list)
        #print(temp_list)
        temp_list2 = []
        temp_list3 = []
        for m in range(len(temp_list)):
            temp_list2.append(train_labels[temp_list[m]])
        temp_list3.append(temp_list2)
        #print("matched labels")
        #print(temp_list3)
        a = np.array(temp_list3)
        #print("a",a)
        b = np.squeeze(a)
        #print("b",b)
        #np.reshape(a, len(a))
        #print(b)
        if(k!=1):
            counts = np.bincount(b)
            euclidian_distance_matrix.append(np.argmax(counts))
        else:
            euclidian_distance_matrix.append(b)
        #print np.argmax(counts)
    #print(euclidian_distance_matrix)
    #print(test_labels)
    for t in range(len(euclidian_distance_matrix)):
        print("Correct Label:",test_labels[t],"Predicted label:",euclidian_distance_matrix[t])
        if(euclidian_distance_matrix[t] == test_labels[t]):
            correct += 1
            if(test_labels[t] == 0):
                count_0 += 1
            elif(test_labels[t] == 1):
                count_1 +=1
            elif(test_labels[t] == 2):
                count_2 +=1
            elif(test_labels[t] == 3):
                count_3 +=1
            elif(test_labels[t] == 4):
                count_4 +=1
            elif(test_labels[t] == 5):
                count_5 +=1
            elif(test_labels[t] == 6):
                count_6 +=1
            elif(test_labels[t] == 7):
                count_7 +=1
            elif(test_labels[t] == 8):
                count_8 +=1
            elif(test_labels[t] == 9):
                count_9 +=1

    #print(correct)
    #print(len(euclidian_distance_matrix))
    y = np.array(test_labels)
    acc.append((count_0/(y == 0).sum())*100)
    acc.append((count_1/(y == 1).sum())*100)
    acc.append((count_2/(y == 2).sum())*100)
    acc.append((count_3/(y == 3).sum())*100)
    acc.append((count_4/(y == 4).sum())*100)
    acc.append((count_5/(y == 5).sum())*100)
    acc.append((count_6/(y == 6).sum())*100)
    acc.append((count_7/(y == 7).sum())*100)
    acc.append((count_8/(y == 8).sum())*100)
    acc.append((count_9/(y == 9).sum())*100)
    print("Accuracy for 10 classes is as below:")
    print(acc)
    acc_av = (correct/len(euclidian_distance_matrix))*100
    print("Average accuracy is:")
    print(acc_av)
    #return acc_av

    
   


def main():
    mnist_train = read("training")
    mnist_test = read("testing")
    #image = mnist.next()
    #print(image)

    #show(image[1])

    train_labels = []
    train_images = []
    test_labels = []
    test_images = []
    k = 5
    for i in range(6000):
        label_image = mnist_train.next()
        train_labels.append(label_image[0])
        train_images.append(label_image[1])
    

    for i in range(1000):
        label_image1 = mnist_test.next()
        test_labels.append(label_image1[0])
        test_images.append(label_image1[1])
    
    #print("Value of k is:")
    #print(k);
    #print("Number of training images is 1000")
    #print("Number of test images is 1000")
    kNN(train_images, train_labels, test_images, test_labels, k)
    #c_question_graph_plot()
    #d_question_graph_plot()
    #e_question_graph_plot()
     
  
  
if __name__== "__main__":
  main()


