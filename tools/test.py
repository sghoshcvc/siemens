import numpy as np
import logging
import torch
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate_cnn_batch(cnn, dataset_loader):
    logger = logging.getLogger('Siemens-Experiment::test')
    # set the CNN in eval mode
    # fh = logging.FileHandler('log1.txt')
    # logger.addHandler(fh)
    cnn.eval()
    logger.info('Computing net output:')
    gpu_id =1
    predictions = []
    labels =[]

    for lc, (img, class_id) in enumerate(tqdm.tqdm(dataset_loader)):
        #if word_img.shape[0]<8:
         #   continue
        # if sample_idx > 10000:
        #     break
        #print(embedding[0].shape)
        #print(embedding[1].shape)
        if gpu_id is not None:
            # in one gpu!!
            img = img.cuda()

        img = torch.autograd.Variable(img)


        # st = lc * word_img.shape[0]
        # en = st + word_img.shape[0]
        # print(st, en)
        ''' BCEloss ??? '''
        # output = torch.sigmoid(cnn(word_img))
        img = Variable(img.float().cuda())
        # labels = Variable(class_ids.cuda())
        retval = cnn(img)
        preds = F.softmax(retval, dim=1)
        _, pred_ids = torch.max(preds, dim=1)
        predictions.append(pred_ids.cpu().numpy())
        labels.append(class_id.cpu().numpy())
    accuracy, sensitivity, specificity = get_matrices(labels,predictions)
    cnn.train()
    return accuracy,sensitivity,specificity


def get_matrices(y_true, y_prediction):

    cnf_matrix = confusion_matrix(y_true, y_prediction)
    # print(cnf_matrix)
    #[[1 1 3]
    # [3 2 2]
    # [1 3 1]]

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    return ACC,TPR,TNR
