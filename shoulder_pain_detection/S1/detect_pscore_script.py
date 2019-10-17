''' script to predict PSPI from image '''
from __future__ import print_function
from __future__ import division
import sys
sys.path.insert(0, './../../')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import McMasterDataset
import imp
import sklearn.metrics
imp.reload(McMasterDataset)
from McMasterDataset import *
from vgg_face import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')



def set_rseed(rseed):
    torch.manual_seed(rseed)
    torch.cuda.manual_seed(rseed)
    np.random.seed(rseed)
    random.seed(rseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "NN1":
        """ 1-hidden layer NN
        """
        input_size = 9
        model_ft = NN1(input_size, 2 * input_size, num_classes)

    elif model_name == "face1_vgg":
        """ vgg-vd-16 trained on vggface
        """
        model_ft = VGG_16()
        model_ft.load_weights()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc8.in_features
        model_ft.fc8 = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG16_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, weights={}, is_inception=False):
    since = time.time()

    val_loss_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae = [0, np.Infinity]
    best_mse = [0, np.Infinity]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            running_pred_label = np.empty((0,20))

            # Iterate over data.

            for sample in dataloaders[phase]:
                inputs = sample['image']
                labels = torch.cat((sample['framePSPI'].view(-1,1), sample['au']),dim=1).float()

                classes1, classweights1 = np.unique(labels.data.numpy()[:,0:1], return_counts=True)
                classweights1 = np.reciprocal(classweights1.astype(float))
                sampleweights = torch.from_numpy(classweights1[np.searchsorted(classes1, labels.data.numpy()[:,0:1])]).float().to(device).view(-1,1)
              
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss1 = criterion(outputs, labels/torch.FloatTensor([16] + [5]*9).to(device)) * sampleweights
                    loss1 = loss1.sum(0)
                    loss = loss1.mean()
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()

                outputs = outputs * torch.FloatTensor([16] + [5]*9).to(device)
                running_pred_label = np.concatenate((running_pred_label, np.concatenate([outputs.data.cpu().numpy(), labels.data.cpu().numpy()],axis=1)))
            
            pred_test = running_pred_label[:,0:10]
            label_test = running_pred_label[:,10:]

            sampleweights = weights[phase]['classweights'][np.searchsorted(weights[phase]['classes'], label_test[:,0:1])]
            sampleweights = sampleweights * sampleweights.shape[0] / np.sum(sampleweights) 

            epoch_weighted_mses = ((pred_test - label_test)**2 * sampleweights).mean(axis=0)
            epoch_weighted_maes = (np.abs(pred_test - label_test) * sampleweights).mean(axis=0)
            # epoch_weighted_mse = epoch_weighted_mses.mean()
            # epoch_weighted_mae = epoch_weighted_maes.mean()
            epoch_weighted_mse = epoch_weighted_mses[0]
            epoch_weighted_mae = epoch_weighted_maes[0]

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} loss: {:.4f} weighted MSE: {:.4f} weighted MAE: {:.4f}'.format(phase, epoch_loss, epoch_weighted_mse, epoch_weighted_mae))

            # deep copy the model
            if phase == 'val' and epoch_weighted_mae < best_mae[1]:
                best_mae = [epoch, epoch_weighted_mae]
                # best_model_wts = copy.deepcopy(model.state_dict())
                # patience = 0
            if phase == 'val' and epoch_weighted_mse < best_mse[1]:
                best_mse = [epoch, epoch_weighted_mse]
                best_model_wts = copy.deepcopy(model.state_dict())
                patience = 0
            if phase == 'val':
                val_loss_history.append([epoch_weighted_mse, epoch_weighted_mae, epoch_loss])
            if phase == 'train':
                train_loss_history.append([epoch_weighted_mse, epoch_weighted_mae, epoch_loss])

        patience += 1
        if patience >= 20:
            break
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val MAE: {:4f} at epoch {:0f}'.format(best_mae[1], best_mae[0]))
    print('Best val MSE: {:4f} at epoch {:0f}'.format(best_mse[1], best_mse[0]))
    f=open('BestEpoch.txt', "a")
    f.write('\nBest val MAE: {:4f} at epoch {:0f} \n'.format(best_mae[1], best_mae[0]))
    f.write('Best val MSE: {:4f} at epoch {:0f} \n'.format(best_mse[1], best_mse[0]))
    f.close()

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, np.asarray(train_loss_history), np.asarray(val_loss_history)


def test_model(model, dataloader, weights, result_dir):
    since = time.time()

    model.eval()   # Set model to evaluate mode

    running_pred_label = np.empty((0,20))

    # Iterate over data.
    for sample in dataloader:
        inputs = sample['image'] # 4D tensor ---> 8 x 3 x 224 x 224
        labels = torch.cat((sample['framePSPI'].view(-1,1), sample['au']),dim=1).float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        subjids = sample['subj_id']
        videoids = sample['video_id'] # list of string
        imageids = sample['image_id'] # list of string

        # forward
        # track history if only in train
        last_layer = list(model_ft.children())[-1]
        try:
            last_layer = last_layer[-1]
        except:
            last_layer = last_layer
        my_embedding = torch.zeros((inputs.shape[0],last_layer.in_features))
        def fun(m, i, o): 
            my_embedding.copy_(i[0].data)
        h = last_layer.register_forward_hook(fun)
        with torch.set_grad_enabled(False):
            # Get model outputs
            outputs = model(inputs) * torch.FloatTensor([16] + [5]*9).to(device)
        h.remove()
        # save results
        for curr_id in range(len(imageids)):
            if not os.path.isdir(os.path.join(result_dir, subjids[curr_id])):
                os.mkdir(os.path.join(result_dir, subjids[curr_id]))
            if not os.path.isdir(os.path.join(result_dir, subjids[curr_id], videoids[curr_id])):
                os.mkdir(os.path.join(result_dir, subjids[curr_id], videoids[curr_id]))
            f=open(os.path.join(result_dir, subjids[curr_id], videoids[curr_id], imageids[curr_id][:-4]+'.txt'), "w")
            f.write(str(outputs.data.cpu().numpy()[curr_id][0]))
            f.close()
            np.savez(os.path.join(result_dir, subjids[curr_id], videoids[curr_id], imageids[curr_id][:-4] + '.npz'), output = outputs.data.cpu().numpy()[curr_id], feat_map = my_embedding.data.cpu().numpy()[curr_id])

        # statistics
        running_pred_label = np.concatenate((running_pred_label, np.concatenate([outputs.data.cpu().numpy(), labels.data.cpu().numpy()],axis=1)))
    
    pred_test = running_pred_label[:,0:10]
    label_test = running_pred_label[:,10:]

    sampleweights = weights['classweights'][np.searchsorted(weights['classes'], label_test[:,0:1])]
    sampleweights = sampleweights * sampleweights.shape[0] / np.sum(sampleweights) 
    
    epoch_acc = (np.round(pred_test) == label_test).mean(0)[0]
    epoch_weighted_acc = ((np.round(pred_test) == label_test) * sampleweights).mean(0)[0]
    epoch_mse = ((pred_test - label_test)**2).mean(0)[0]
    epoch_weighted_mse = ((pred_test - label_test)**2 * sampleweights).mean(0)[0]
    epoch_mae = np.abs(pred_test - label_test).mean(0)[0]
    epoch_weighted_mae = (np.abs(pred_test - label_test) * sampleweights).mean(0)[0]
    
    print('{} Acc: {:.4f} Weighted Acc: {:.4f} MSE: {:.4f} Weighted MSE: {:.4f} MAE: {:.4f} Weighted MAE: {:.4f}'.format('test', 
            epoch_acc, epoch_weighted_acc, epoch_mse, epoch_weighted_mse, epoch_mae, epoch_weighted_mae))
    
    f=open('BestEpoch.txt', "a")
    f.write('Test MAE: {:4f} weighted MAE {:4f} \n'.format(epoch_mae, epoch_weighted_mae))
    f.close()

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return pred_test, label_test






print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
import random

for rseed in [0,2,4,6,8]:

    # rseed = 0
    set_rseed(rseed)

    image_dir = "./../../data/UNBCMcMaster_cropped/Images0.3"
    label_dir = "./../../data/UNBCMcMaster"
    if not os.path.isdir('./models_sf' + str(rseed)):
        os.mkdir('./models_sf' + str(rseed))
    if not os.path.isdir('./results_sf' + str(rseed)):
        os.mkdir('./results_sf' + str(rseed))
    # Number of classes in the dataset
    num_classes = 10
    # Batch size for training (change depending on how much memory you have)
    batch_size = 32
    # Number of epochs to train for
    num_epochs = 100
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "face1_vgg" 

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            # BGR2RGB(),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
        'val': transforms.Compose([
            # BGR2RGB(),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
        'test': transforms.Compose([
            # BGR2RGB(),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets

    # load subject folder names from root directory
    subjects = []
    for d in next(os.walk(image_dir))[1]:
        subjects.append(d[:3])
    subjects = sorted(subjects)
    random.shuffle(subjects)
    print(subjects)

    fold_size = 5
    folds = []
    for i in range(5):
        folds += [subjects[i*fold_size: (i+1)*fold_size]]

    # start cross validation
    for subj_left_id, subj_left_out in enumerate(folds):
        set_rseed(rseed)
        test_subj = subj_left_out
        train_id= range(len(folds))
        train_id.pop(subj_left_id)
        val_id = random.choice(train_id)
        val_subj = folds[val_id]

        if os.path.isfile('./results_sf' + str(rseed) + '/' + str(subj_left_id) + '.npz'):
            continue

        # Initialize the model for this run
        model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

        print('-'*10 + "cross-validation: " + 
        "(" + str(subj_left_id+1) + "/5)" + '-'*10)

        datasets = {x: McMasterDataset(image_dir, label_dir, val_subj, test_subj, x, data_transforms[x]) for x in ['train', 'val', 'test']}
        
        # Create training and validation dataloaders

        # class_sample_count = [sum(1 for x in datasets['train'] if x['framelabel']==c) for c in range(2)]
        # weights = 1 / torch.Tensor(class_sample_count)
        # sample_weights = weights[[x['framelabel'] for x in datasets['train']]]
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, batch_size)
        weights = {}
        for phase in ['train', 'val', 'test']:
            labels = [x['framePSPI'] for x in datasets[phase]]
            labels = np.stack(labels)
            classes, classweights = np.unique(labels, return_counts=True)
            classweights = np.reciprocal(classweights.astype(float))
            sampleweights = classweights[np.searchsorted(classes, labels)]
            classweights = classweights * sampleweights.shape[0] / np.sum(sampleweights) 
            weights[phase] = {'classes':classes, 'classweights': classweights}
    
        shuffle = {'train': True, 'val': False, 'test': False}
        dataloaders_dict = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=4, worker_init_fn=lambda l: [np.random.seed((rseed + l)), random.seed(rseed + l), torch.manual_seed(rseed+ l)]) for x in ['train', 'val', 'test']}

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Send the model to GPU
        model_ft = model_ft.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        # Observe that all parameters are being optimized
        last_layer = list(model_ft.children())[-1]
        try:
            last_layer = last_layer[-1]
        except:
            last_layer = last_layer
        ignored_params = list(map(id, last_layer.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                        model_ft.parameters())



        optimizer_ft = optim.Adam([
                    {'params': base_params},
                    {'params': last_layer.parameters(), 'lr': 1e-4}
                ], lr=1e-5, weight_decay=5e-4)



        # Setup the loss fxn
        criterion = nn.MSELoss(reduction='none')
        # criterion = nn.CrossEntropyLoss(weight = weights.to(device))

        # Train and evaluate
        if os.path.isfile('./models_sf' + str(rseed) + '/' + str(subj_left_id) + '.pth'):
            model_ft.load_state_dict(torch.load('./models_sf' + str(rseed) + '/' + str(subj_left_id) + '.pth',map_location=device))
        else:
            model_ft, train_hist, val_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, weights = weights, is_inception=(model_name=="inception"))
            torch.save(model_ft.state_dict(), './models_sf' + str(rseed) + '/' + str(subj_left_id) + '.pth')
            for i in range(3):
                plt.subplot(1,3,i+1)
                plt.plot(train_hist[:,i])
                plt.plot(val_hist[:,i])
            plt.legend(['train_mse', 'val_mse','train_mae','val_mae','train_loss','val_loss'])

            plt.savefig('./models_sf' + str(rseed) + '/' + str(subj_left_id) + '.png')
            plt.clf()

        # test
        if not os.path.isdir('./results_sf' + str(rseed) + '/' + str(subj_left_id)):
            os.mkdir('./results_sf' + str(rseed) + '/' + str(subj_left_id))
        pred_train, label_train = test_model(model_ft, dataloaders_dict['train'], weights['train'], './results_sf' + str(rseed) + '/' + str(subj_left_id))
        pred_val, label_val = test_model(model_ft, dataloaders_dict['val'], weights['val'], './results_sf' + str(rseed) + '/' + str(subj_left_id))
        pred_test, label_test = test_model(model_ft, dataloaders_dict['test'], weights['test'], './results_sf' + str(rseed) + '/' + str(subj_left_id))
        np.savez('./results_sf' + str(rseed) +'/' + str(subj_left_id) + '.npz', pred = pred_test, label = label_test, pred_val = pred_val, label_val = label_val,pred_train = pred_train, label_train = label_train)
