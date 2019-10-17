''' script to plot evaluation for each subject and for all '''
import os
import numpy as np
import sklearn.metrics
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import scipy.ndimage.filters
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import matplotlib.pyplot as plt
rpy2.robjects.numpy2ri.activate()
import subprocess
import paramiko


def get_PSPI(aus):
    ''' get PSPI from aus
        PSPI = AU4 + max(AU6,AU7) + max(AU9,AU10) + AU43
        return a 1-d array
    '''
    return aus[:,0] + np.max(aus[:,1:3], axis=1) + aus[:,3] + aus[:,8]

class MySFTPClient(paramiko.SFTPClient):
    def put_dir(self, source, target):
        ''' Uploads the contents of the source directory to the target path. The
            target directory needs to exists. All subdirectories in source are 
            created under target.
        '''
        for item in os.listdir(source):
            if os.path.isfile(os.path.join(source, item)):
                self.put(os.path.join(source, item), '%s/%s' % (target, item))
            else:
                self.mkdir('%s/%s' % (target, item), ignore_existing=True)
                self.put_dir(os.path.join(source, item), '%s/%s' % (target, item))
    def get_dir(self, source, target):
        ''' Downloads the contents of the source directory to the target path. The
            target directory needs to exists. All subdirectories in source are 
            created under target.
        '''
        for f in self.listdir_attr(source):
            item = f.filename
            if item[-4:]!='.npz':
                continue
            self.get(os.path.join(source, item), '%s/%s' % (target, item))


    def mkdir(self, path, mode=511, ignore_existing=False):
        ''' Augments mkdir by adding an option to not fail if the folder exists  '''
        try:
            super(MySFTPClient, self).mkdir(path, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise

def download_data(server_results_dir, results_dir, servername = 'desagpu2.ucsd.edu', server_dir = "shoulder_pain_detection_weightall/newnorm_VAS3_using_PSPIinAUpred/newnorm_PSPIAU/"):
    # download result folders from server
    subprocess.run(["mkdir", results_dir])
    transport = paramiko.Transport((servername,22))
    if servername == 'sadkangaroo.001www.com':
        transport.connect(username = 'carina', password = '1388Ice!')
    else:
        transport.connect(username = 'xix068', password = '1388Ice!')
    sftp = MySFTPClient.from_transport(transport)
    sftp.mkdir(results_dir, ignore_existing=True)
    if servername == 'sadkangaroo.001www.com':
        sftp.get_dir("xiaojing/" + server_dir + server_results_dir , results_dir)
    else:
        sftp.get_dir("/mnt/cube/projects/" + server_dir + server_results_dir , results_dir)
    sftp.close()

def process_results(pred, label, mask = None):
    # rescale predictions. labels [VAS, OPR, SEN?, AFF?, ..] => [VAS,VAS,VAS,VAS, ..]
    # input: pred, label - numpy array of shape (num_samples, num_scores) or (num_samaples,), mask - list of len(num_scores)   
    # output: pred (num_samples, num_features), label(num_samples, num_features), label_OPR(num_samples,1)
    if len(pred.shape)==1:
        pred,label = pred.reshape(-1,1), label.reshape(-1,1)
    if pred.shape[1]>=2:
        label_OPR = label[:,1:2]
    else: # if there is only one column in pred/label, label_OPR is label of VAS
        label_OPR = label
    if pred.shape[1]%4==0:
        repeat = int(pred.shape[1]/4)
        if (label[:,0]!=label[:,1]).sum()==0:
            pred, label = pred, np.tile(label[:,0], (4*repeat,1)).T # this line for test of 4 VAS output model
        else:
            pred, label = pred * np.asarray([1,2,10/16,10/16]*repeat), np.tile(label[:,0], (4*repeat,1)).T
    elif pred.shape[1]==3:
        pred, label = pred, np.tile(label[:,0], (3,1)).T
    elif pred.shape[1]%10==0:
        pred = pred * np.asarray([1] + [5/16]*9)
        predall = []
        labelall = []
        for ite in range(0,pred.shape[1], 10):
            predtmp, labeltmp = np.stack((pred[:,ite], get_PSPI(pred[:,ite+1:ite+10])), axis=1), np.tile(label[:,0], (2,1)).T   
            predall.append(predtmp)
            labelall.append(labeltmp)
        pred = np.concatenate(predall, axis=1) 
        label = np.concatenate(labelall, axis=1) 
    # only use some columns/scores
    if mask:
        pred = pred[:,mask]
        label = label[:,mask]
    return pred, label, label_OPR

def OLC(pred, label=None, weighted = 0, l2constraint = 0, unbiased = 0, ensemble = 1, independent = False, verbose = False, weights = None, bias = None):
    # optimal linear combination of pred for label
    # input mode 1 - train: weighted MSE as objective? use l2 constraint on objective? make pred unbiased? OLC or just first column? scores independent?
    # input mode 2 - test: pass in pred, weights, bias to calculate OLC of pred
    # output: OLC pred, weights, bias
   
    if weights is None: # training model. Need label, ... , independent
        if ensemble==3:
            numfeat = pred.shape[1]
            if numfeat%4==0: # note here numfeat shouldn't be, e.g. 12
                repeat = numfeat/4
            elif numfeat%10==0:
                repeat = numfeat/10
            elif numfeat%3==0:
                repeat = numfeat/3
            repeat = int(repeat)

            pred = pred.reshape((pred.shape[0],repeat,-1))
            pred = pred.mean(axis=1)
            label = label[:,:int(numfeat/repeat)]

            # pred = pred.reshape((pred.shape[0]*repeat,-1))
            # label = label.reshape((label.shape[0]*repeat),-1)
        # unbiased or not
        if unbiased == 0:
            bias = 0
        else:
            bias = (pred - label).mean(axis=0)
        # weighted
        if weighted==1:
            classes, classweights = np.unique(label[:,0], return_counts=True)
            classweights = np.reciprocal(classweights.astype(float))
            sampleweights = classweights[np.searchsorted(classes, label[:,0])]
            sampleweights = sampleweights * sampleweights.shape[0] / np.sum(sampleweights) # normalize 
            sampleweights = np.tile(sampleweights, (pred.shape[1],1)).T
        else:
            sampleweights = np.ones(pred.shape)
        # constrained
        weights = np.inner(((pred-label-bias) * sampleweights).T, (pred-label-bias).T)
        weights = weights + l2constraint * np.identity(weights.shape[0])
        if independent:
            weights = weights * np.identity(weights.shape[0])
        # unconstrained
        # weights = np.inner(pred.T, pred.T)
        # weights = np.linalg.inv(weights)
        # weights = np.inner(weights, (pred*label).sum(axis=0))
        if verbose:
            print("diff matrix: \n", weights)
        weights = np.linalg.inv(weights)
        if verbose:
            print("inv matrix: \n", weights)
        weights = weights.sum(axis=0)
        


        # ensemble or not
        if ensemble == 0 and type(weights)!=int:
            weights[1:]=0
        elif ensemble == 2 and type(weights)!=int:
            weights = np.ones(weights.shape)
        elif ensemble==3 and type(weights)!=int:
            weights = np.tile(weights, repeat)
            if type(bias)!=int:
                bias = np.tile(bias, repeat)
            pred = np.tile(pred, (1,repeat))
            label = np.tile(label, (1,repeat))
            # pred = pred.reshape((-1, weights.shape[0]))
            # label = label.reshape((-1, weights.shape[0]))
        if type(weights)!=int:
            weights = weights * weights.shape[-1] / weights.sum(axis=-1) # normalize weights


    # generate optimal predictions  
    pred = ((pred - bias)* weights).mean(axis=1)

    return pred, weights, bias

allauc = []
allmae = []
allmse = []
allicc = []
allpcc = []
allwmae = []
allwmse = []
allMSE = []
allMAE = []
allAUC = []
avg_pred = []
avg_label = []
pred_all = []
label_all = []
opr_all = []

# plt.figure(figsize=(20,5))
# parameters
servername = 'desagpu3.ucsd.edu'
server_dir = "xiaojing/shoulder_pain_detection_weightall/newnorm_PSPIAU/"
# server_dir = "xiaojing/shoulder_pain_detection_VAS/VAS3_using_AUmax_allsigmoid/"
# server_dir = "yundong/pain_detection/s1/predict_PSPI/Using_feat4096/LSTM_discard/"

# server_dir = "NTAP_pain_detection/newnorm_test_pretrained_VAS3/"
# server_dir0 = "shoulder_pain_detection_weightall/newnorm_VAS3_using_PSPIinAUpred/newnorm_PSPIAU/"

regression = 1
unbiased = 1
knowOPR = 0 # 0 - don't use true OPR, 1 - OLC using OPR as input, 2 - average OLC and OPR, 3 - OLC using OLC and OPR as input in stage 4
ensemble = 0 # 0 - just use VAS, no ensemble, 1 - optimal linear combination, 2 - average, 3 - average same score's pred, then OLC
weighted = 0
l2constraint = 0
mask = None
independent = [False, False] # wheter s3 and s4 variables are assumed independent
verbose = False
# for kernel_size in range(1,2,2):
kernel_size = 5
# for results_dir in ['results']:
for rseed in [0,2,4,6,8]:
    results_dir = 'results_sf' + str(rseed)
    download_data(results_dir, results_dir, servername=servername, server_dir=server_dir)

    # results_dir0 = 'results0_sf' + str(rseed)
    # download_data(results_dir, results_dir0, servername=servername, server_dir=server_dir0)


    results_allsubj = []
    labels_allsubj = []
    for root, directory, files in os.walk(results_dir):
        for subj in sorted(files):
            if subj[-3:]!='npz':
                continue
            if knowOPR == 3 and len(subj)<7 or knowOPR<3 and len(subj)==7:
                continue

            # tmp_result = np.load(results_dir0+'/'+subj[0] + '.npz')        
            tmp_result = np.load(results_dir+'/'+subj[0] + '.npz')
            pred_train, label_train = np.concatenate((tmp_result['pred_train'], tmp_result['pred_val']),axis=0), np.concatenate((tmp_result['label_train'], tmp_result['label_val']),axis=0)
            # pred_train, label_train = tmp_result['pred_val'], tmp_result['label_val']
            # pred_train, label_train = np.concatenate((tmp_result['pred_train'], tmp_result['pred_val'], tmp_result['pred']),axis=0), np.concatenate((tmp_result['label_train'], tmp_result['label_val'], tmp_result['label']),axis=0)
            # pred_train, label_train = tmp_result['pred'], tmp_result['label']

            if server_dir[:7] == 'yundong' and len(pred_train.shape)>=2:
                pred_train, label_train = np.concatenate((pred_train[:,-1:],pred_train[:,:-1]), axis = 1), np.concatenate((label_train[:,-1:],label_train[:,:-1]), axis = 1), 

            pred_train, label_train, label_train_OPR = process_results(pred_train, label_train, mask)
            
            
            if knowOPR == 1:
                pred_train, label_train = np.concatenate((pred_train, label_train_OPR*2), axis=1), np.concatenate((label_train,label_train[:,0:1]), axis=1)
            pred_train, weights, bias = OLC(pred_train, label_train, weighted=weighted, l2constraint=l2constraint, unbiased=unbiased, ensemble=ensemble, independent=independent[0], verbose=verbose)

            print(weights)

            if unbiased==1:
                bias_OPR = (label_train_OPR[:,0] * 2 - label_train[:,0]).mean(axis=0)
            else:
                bias_OPR = 0

            if knowOPR == 3:

                tmp_result = np.load(results_dir+'/'+subj)
                pred_train, label_train = tmp_result['pred_val'], tmp_result['label_val']
                pred_train, label_train, label_train_OPR = process_results(pred_train, label_train, mask)
                pred_train, weights, bias = OLC(pred_train, weights = weights, bias = bias)

                # linear combination on top of pred_train
                pred_train, label_train = np.concatenate((pred_train.reshape(-1,1), label_train_OPR*2), axis=1), label_train[:,0:2]
                pred_train, weights2, bias2 = OLC(pred_train, label_train, weighted=weighted, l2constraint=l2constraint, unbiased=unbiased, ensemble=ensemble, independent=independent[1], verbose=verbose)    
                print(weights2)

            tmp_result = np.load(results_dir+'/'+subj)
            pred_test, label_test = tmp_result['pred'], tmp_result['label']
            if server_dir[:7] == 'yundong' and len(pred_train.shape)>=2:
                pred_test, label_test = np.concatenate((pred_test[:,-1:],pred_test[:,:-1]), axis = 1), np.concatenate((label_test[:,-1:],label_test[:,:-1]), axis = 1), 

            # pred_test, label_test = np.concatenate((tmp_result['pred_train'], tmp_result['pred_val']),axis=0), np.concatenate((tmp_result['label_train'], tmp_result['label_val']),axis=0)         
            pred_test, label_test, label_test_OPR = process_results(pred_test, label_test, mask)
            if knowOPR == 1:
                pred_test = np.concatenate((pred_test, label_test_OPR*2), axis=1)
            pred_test, weights, bias = OLC(pred_test, weights = weights, bias = bias)
            if knowOPR == 2:
                pred_test = (pred_test + label_test_OPR[:,0] * 2 - bias_OPR)/2
            if knowOPR == 3:
                pred_test, label_test = np.concatenate((pred_test.reshape(-1,1), label_test_OPR*2), axis=1), label_test[:,0:2]
                pred_test, weights2, bias2 = OLC(pred_test, weights = weights2, bias = bias2)
            label_test = label_test[:,0]
            pred_test0 = label_test_OPR[:,0]*2 - bias_OPR
            # pred_test = pred_test0
            # pred_test = np.ones(label_test.shape) * label_test.mean()


            # tmp_result = np.load(results_dir0+'/'+subj)
            # pred_test, label_test0 = tmp_result['pred'], tmp_result['label']
            # if len(pred_test.shape)==2:
            #     pred_test, label_test0 = pred_test[:,0] * 2, label_test0[:,0]

            # if np.max(label_test)<=1:
            #     pred_test = pred_test * 15
            #     label_test = label_test * 15
            
            # apply filter on sequence
            # pred_test = scipy.signal.medfilt(pred_test, kernel_size=kernel_size)
            # pred_test = scipy.ndimage.filters.maximum_filter1d(pred_test, 15)
            # pred_test[pred_test<0] =0

            # plt.plot(pred_test,'r')
            # plt.plot(label_test,'b')
            # plt.show()

            # break
            
            results_allsubj.append(pred_test)
            labels_allsubj.append(label_test)
            opr_all.append(pred_test0)


    ACC = []
    WACC = []
    MSE  = []
    WMSE = []
    MAE  = []
    WMAE = []
    AUC = []
    ICC = []
    PCC = []
    CNT = []
    for subj_left_id in range(len(results_allsubj)):
        pred_test = results_allsubj[subj_left_id]
        label_test = labels_allsubj[subj_left_id]

        acc = np.sum(np.round(pred_test) == label_test) / pred_test.shape[0]
        
        classes, classweights = np.unique(label_test, return_counts=True)
        classweights = np.reciprocal(classweights.astype(float))
        sampleweights = classweights[np.searchsorted(classes, label_test)]
        sampleweights = sampleweights * sampleweights.shape[0] / np.sum(sampleweights) 
        weighted_acc = np.sum((np.round(pred_test) == label_test) * sampleweights) / pred_test.shape[0]

        r_icc = importr("irr")
        B = np.stack((pred_test, label_test),axis=1)
        nr,nc = B.shape
        Br = ro.r.matrix(B, nrow=nr, ncol=nc)
        ro.r.assign("B", Br)
        icclist = r_icc.icc(Br,"twoway")
        icc = icclist.rx2('value')[0]
        
        pcc = np.corrcoef(label_test, pred_test)[0,1]

        mse = ((pred_test - label_test)**2).mean(axis=0)
        weighted_mse = ((pred_test - label_test)**2 * sampleweights).mean(axis=0)
        
        mae = np.abs(pred_test - label_test).mean(axis=0)
        weighted_mae = (np.abs(pred_test - label_test) * sampleweights).mean(axis=0)

        cnt = len(pred_test)

        auc = sklearn.metrics.roc_auc_score((label_test>3)+0.0, pred_test) if len(np.unique((label_test>3)+0.0))>1 else 0

        print('{} Acc: {:.4f} Weighted Acc: {:.4f} MSE: {:.4f} Weighted MSE: {:.4f}  MAE: {:.4f} Weighted MAE: {:.4f} AUC: {:.4f}'.format('test', acc, weighted_acc, mse, weighted_mse, mae, weighted_mae, auc))
        
        ACC.append(acc)
        WACC.append(weighted_acc)
        MSE.append(mse)
        WMSE.append(weighted_mse)
        MAE.append(mae)
        WMAE.append(weighted_mae)
        AUC.append(auc)
        ICC.append(icc)
        PCC.append(pcc)
        CNT.append(cnt)

    # if regression==1:
    #     plt.subplot(211)
    #     plt.bar(range(1,len(results_allsubj)+1), MAE)
    #     plt.ylabel('MAE')
    #     plt.subplot(212)
    #     plt.bar(range(1,len(results_allsubj)+1), WMAE)
    #     plt.ylabel('Weighted MAE')
    # else:
    #     plt.subplot(211)
    #     plt.bar(range(1,len(results_allsubj)+1), AUC)
    #     plt.ylabel('AUC')
    #     plt.subplot(212)
    #     plt.bar(range(1,len(results_allsubj)+1), WACC)
    #     plt.ylabel('Weighted ACC')
    # plt.subplot(223)
    # plt.bar(range(1,len(results_allsubj)+1), MSE)
    # plt.ylabel('MSE')
    # plt.subplot(224)
    # plt.bar(range(1,len(results_allsubj)+1), WMSE)
    # plt.ylabel('Weighted MSE')
    # plt.show()

    pred_test = np.concatenate(results_allsubj, axis=0)
    label_test = np.concatenate(labels_allsubj, axis=0)

    avg_pred.append(pred_test)
    avg_label.append(label_test)

    pred_all.append(pred_test)
    label_all.append(label_test)

    # pred_test[pred_test>label_test.max()]  = label_test.max()
    # pred_test[pred_test<0] = 0

    ## scatterplot for each iteration
    # plt.subplot(1,5,int(rseed/2)+1)
    # plt.scatter(label_test, pred_test)


    acc = np.sum(np.round(pred_test) == label_test) / pred_test.shape[0]
    cacc = np.mean((pred_test>0.5) == (label_test>0.5))

    classes, classweights = np.unique(label_test, return_counts=True)
    classweights = np.reciprocal(classweights.astype(float))
    sampleweights = classweights[np.searchsorted(classes, label_test)]
    sampleweights = sampleweights * sampleweights.shape[0] / np.sum(sampleweights) 
    weighted_acc = np.sum((np.round(pred_test) == label_test) * sampleweights) / pred_test.shape[0]
    weighted_cacc = np.mean(((pred_test>0.5) == (label_test>0.5)) * sampleweights)

    r_icc = importr("irr")
    B = np.stack((pred_test, label_test),axis=1)
    nr,nc = B.shape
    Br = ro.r.matrix(B, nrow=nr, ncol=nc)
    ro.r.assign("B", Br)
    icclist = r_icc.icc(Br,"twoway")
    icc = icclist.rx2('value')[0]

    micc = sum(ICC) / len(ICC)
    wicc = sum([ICC[i] * CNT[i] for i in range(len(CNT))]) / sum(CNT)

    mse = ((pred_test - label_test)**2).mean(axis=0)
    weighted_mse = ((pred_test - label_test)**2 * sampleweights).mean(axis=0)
    mae = np.abs(pred_test - label_test).mean(axis=0)
    weighted_mae = (np.abs(pred_test - label_test) * sampleweights).mean(axis=0)
    auc = sklearn.metrics.roc_auc_score((label_test>3)+0.0, pred_test)
    # ICC, r_var, e_var, session_effect_F, dfc, dfe = ICC_rep_anova(np.stack((label_test, pred_test), axis=1).transpose())
    pcc = np.corrcoef(label_test, pred_test)[0,1]
    mpcc = sum(PCC) / len(PCC)
    wpcc = sum([PCC[i] * CNT[i] for i in range(len(CNT))]) / sum(CNT)
    print('{} Classification Acc: {:.4f} Weighted Classification Acc: {:.4f} \n \
        Acc: {:.4f} Weighted Acc: {:.4f} \n \
        MSE: {:.4f} Weighted MSE: {:.4f} \n \
        MAE: {:.4f} Weighted MAE: {:.4f} \n \
        AUC: {:.4f} ICC: {:.4f} mean ICC: {:.4f} weighted ICC: {:.4f} \n \
        PCC: {:.4f} mean PCC: {:.4f} weighted PCC: {:.4f} '.format('All test',\
        cacc, weighted_cacc, acc, weighted_acc,\
        mse, weighted_mse, mae, weighted_mae, auc, icc, micc, wicc, pcc, mpcc, wpcc))

    # print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(mae, mse, icc, pcc, cacc))
    print('{:.2f}\t{:.2f}'.format(mae, icc))
    print('& {:.2f} & {:.2f} & {:.2f} & {:.2f} '.format(mae, mse, icc, pcc))

    allmae.append(mae)
    allmse.append(mse)
    allauc.append(auc)
    allicc.append(icc)
    allpcc.append(pcc)

    allwmae.append(weighted_mae)
    allwmse.append(weighted_mse)

    allMSE.append(MSE)
    allMAE.append(MAE)
    allAUC.append(AUC)

std = np.abs(np.concatenate(avg_pred) - np.concatenate(avg_label)).std(axis=0)
print('std=',std)

avg_pred = sum(avg_pred)/len(avg_pred)
avg_label = sum(avg_label)/len(avg_label)
auc = sklearn.metrics.roc_auc_score((avg_label>3)+0.0, avg_pred)
# l1, = plt.plot(allmae, label='MAE')
# l2, = plt.plot(allmse, label='MSE')
# l3, = plt.plot(allauc, label='AUC')
# l4, = plt.plot(allicc, label='ICC')
# l5, = plt.plot(allpcc, label='PCC')
# plt.legend(handles=[l1,l2,l3,l4,l5])
# plt.show()

# X = np.arange(1,len(results_allsubj)+1)
# for i in range(2):
#     MAE = allMAE[i]
#     AUC = allAUC[i]
#     if regression==1:
#         plt.subplot(211)
#         plt.bar(X+i/float(4), MAE, width = 0.25)
#         plt.ylabel('MAE')
#         plt.subplot(212)
#         plt.bar(X+i/float(4), AUC, width = 0.25)
#         plt.ylabel('AUC')
#     else:
#         plt.subplot(211)
#         plt.bar(range(1,len(results_allsubj)+1), AUC)
#         plt.ylabel('AUC')
#         plt.subplot(212)
#         plt.bar(range(1,len(results_allsubj)+1), WACC)
#         plt.ylabel('Weighted ACC')
# plt.show()


# pred_test = np.concatenate(pred_all)
# label_test = np.concatenate(label_all)
# opr_test = np.concatenate(opr_all)
# ae1 = np.abs(label_test-pred_test)
# ae2 = np.abs(label_test-opr_test)
# scipy.io.savemat('abs.mat', {'a':ae1, 'b':ae2})
# t, p=scipy.stats.wilcoxon(ae1, ae2)
# print('t=',t,'p=',p)
# # print percentiles of absolute error
# print('OPR as VAS MAE: ', ae1.mean())
# print('OPR as VAS MSE: ', (ae1**2).mean())
# print('pred within 2: ', (np.abs(np.round(pred_test) - label_test)<=2).sum()/len(pred_test))
# print('OPR within 2: ', (np.abs(opr_test - label_test)<=2).sum()/len(pred_test))
# diff = ae1 - ae2
# # diff = np.abs(pred_test - label_test)
# print('percent diff < 0 = ', sum(diff<0)/len(diff))
# plt.hist(diff, 30)
# for pencentile in [50, 75, 85, 95]:
#     p = np.percentile(diff, pencentile)
#     print(str(pencentile)+'% percentile:',p)
# print(diff.mean())
# print(np.abs(pred_test - label_test).mean())

## scatter plot
# plt.subplot(121)
# plt.scatter(label_test, pred_test)
# plt.subplot(122)
# plt.scatter(label_test, opr_test)
# plt.show()


## scatter plot with histograms
# scatter_axes = plt.subplot2grid((3, 3), (1, 0), rowspan=2, colspan=2)
# x_hist_axes = plt.subplot2grid((3, 3), (0, 0), colspan=2,
#                                sharex=scatter_axes)
# y_hist_axes = plt.subplot2grid((3, 3), (1, 2), rowspan=2,
#                                sharey=scatter_axes)

# scatter_axes.plot(label_test, pred_test, '.')
# x_hist_axes.hist(label_test)
# y_hist_axes.hist(pred_test, orientation='horizontal')

# plt.show()

allmae = np.asarray(allmae)
allmse = np.asarray(allmse)
allicc = np.asarray(allicc)
allpcc = np.asarray(allpcc)

allwmae = np.asarray(allwmae)

# allmae = np.asarray(MAE)
# allmse = np.asarray(MSE)
# allicc = np.asarray(ICC)
# allpcc = np.asarray(PCC)
allauc = np.asarray(AUC)
# allauc = auc
print(allauc.mean(), allauc.std())

print('${:.2f}\pm{:.2f}$ & ${:.2f}\pm{:.2f}$ & ${:.2f}\pm{:.2f}$ & ${:.2f}\pm{:.2f}$ & ${:.2f}\pm{:.2f}$'.format(allmae.mean(), allmae.std(), allmse.mean(), allmse.std(), allicc.mean(), allicc.std(), allpcc.mean(), allpcc.std(), allwmae.mean(), allwmae.std()))