#!/bin/python
import sys
from icecream import ic
# from preprocessing import Data

# def read_files(tarfname):
#     """Read the training and development data from the speech tar file.
#     The returned object contains various fields that store the data, such as:

#     train_data,dev_data: array of documents (array of words)
#     train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
#     train_labels,dev_labels: the true string label for each document (same length as data)

#     The data is also preprocessed for use with scikit-learn, as:

#     count_vec: CountVectorizer used to process the data (for reapplication on new data)
#     train_x,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
#     le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
#     target_labels: List of labels (same order as used in le)
#     train_y,devy: array of int labels, one for each document
#     """
#     data = Data(tarfname)
#     return data

def read_unlabeled(tarfname, speech, preprocess = False, w2v = False):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the speech.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    unlabeled.fnames = []
    for m in tar.getmembers():
            if "unlabeled" in m.name and ".txt" in m.name:
                    unlabeled.fnames.append(m.name)
                    unlabeled.data.append(read_instance(tar, m.name))
    if preprocess:
        unlabeled.X = speech.svd.transform(speech.featurization.transform(unlabeled.data))
        ic(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    ic(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    fnames = []
    for line in tf:
            line = line.decode("utf-8")
            (ifname,label) = line.strip().split("\t")
            #ic ifname, ":", label
            content = read_instance(tar, ifname)
            labels.append(label)
            fnames.append(ifname)
            data.append(content)
    return data, fnames, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, speech):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the speech object,
    this function write the predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The speech object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = speech.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("FileIndex,Category\n")
    for i in range(len(unlabeled.fnames)):
            fname = unlabeled.fnames[i]
            # iid = file_to_id(fname)
            f.write(str(i+1))
            f.write(",")
            #f.write(fname)
            #f.write(",")
            f.write(labels[i])
            f.write("\n")
    f.close()

def file_to_id(fname):
    return str(int(fname.replace("unlabeled/","").replace("labeled/","").replace(".txt","")))

def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("FileIndex,Category\n")
    i = 0
    with open(tsvfile, 'r') as tf:
            for line in tf:
                (ifname,label) = line.strip().split("\t")
                # iid = file_to_id(ifname)
                i += 1
                f.write(str(i))
                f.write(",")
                #f.write(ifname)
                #f.write(",")
                f.write(label)
                f.write("\n")
    f.close()

def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts OBAMA_PRIMARY2008 for all the instances.
    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("FileIndex,Category\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (ifname,label) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("OBAMA_PRIMARY2008")
            f.write("\n")
    f.close()

def read_instance(tar, ifname):
    inst = tar.getmember(ifname)
    ifile = tar.extractfile(inst)
    content = ifile.read().strip()
    return content


    

