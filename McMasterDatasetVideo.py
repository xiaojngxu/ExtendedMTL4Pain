from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import os
import cv2
import numpy as np
import scipy.io


def get_PSPI(aus):
    ''' get PSPI from aus
        return a 1-d array
    '''
    aus = np.reshape(aus, (-1,9))
    return aus[:,0] + np.max(aus[:,1:3], axis=1) + aus[:,3] + aus[:,8]


def get_stat_feature(framePSPI):
    ''' get mean, max, min, std, 95 percentile, 85, 75, 50, 25, half_rect_mean of list
    '''
    framePSPI = np.asarray(framePSPI)
    feature = [np.nanmean(framePSPI, axis=0), 
        np.nanmax(framePSPI, axis=0),
        np.nanmin(framePSPI, axis=0),
        np.nanstd(framePSPI, axis=0),
        np.nanpercentile(framePSPI, 95, axis=0),
        np.nanpercentile(framePSPI, 85, axis=0),
        np.nanpercentile(framePSPI, 75, axis=0),
        np.nanpercentile(framePSPI, 50, axis=0),
        np.nanpercentile(framePSPI, 25, axis=0)]

    feature = np.asarray(feature)
    return feature


class McMasterDatasetVideo(Dataset):
    """ McMaster Shoulder Pain Dataset. Load one video as one sample"""

    def __init__(self, video_dir, label_dir, val_subj_id, test_subj_id, subset, transform=None, labeltransform=None, frame_score_dir = None, iau_dir = None):
        """
        Args:
            video_dir (string): Path to the video data "UNBCMcMaster_cropped/Images0.3"
            label_dir (string): Path to the label (pain level, etc.) "UNBCMcMaster"
            val_subj_id ([string]): list of paths containing validation data
            test_subj_id ([string]): list of paths containing test data
            subset (string): train, val, test
            transform (callable, optional): Optional transfomr to be applied on a sample
            labeltransform (callable, optional): Optional transfomr to be applied on a sample label
            frame_score_dir (string): Path to the frame predictions
            iau_dir (string): Path to iMotions AUs "FACET_mat"
        """
        self.seqVASpath = os.path.join(label_dir, 'Sequence_Labels','VAS')
        self.frameVASpath = os.path.join(label_dir, 'Frame_Labels','PSPI')
        self.AUpath = os.path.join(label_dir, 'Frame_Labels', 'FACS')
        self.framescorepath = frame_score_dir
        self.video_path = video_dir
        self.video_files = [(dir_s, dir_v) for dir_s in next(os.walk(video_dir))[1] for dir_v in next(os.walk(os.path.join(video_dir, dir_s)))[1] if 
          ((dir_s[:3] in test_subj_id and subset=='test') or 
          (dir_s[:3] in val_subj_id and subset=='val') or 
          (not(dir_s[:3] in val_subj_id+test_subj_id) and subset=='train'))]
        self.transform = transform
        self.labeltransform = labeltransform
        if iau_dir:
            self.iau_dir = iau_dir
        else:
            self.iau_dir = os.path.join(label_dir, 'FACET_output')

    def __len__(self):
        return sum([len(self.video_files)])

    def __getitem__(self, idx):
        """
        example of sample[]:
        'images': list of images, 
        'image_ids': list of string ('jh043t1afaff014.png'), 
        'video_id': 'jh043t1afaff', 
        'feat2048': (4096,), 
        'feat65': (9,10),
        'videolabel': 1, 
        'subj_id': '043-jh043', 
        'videoVAS': 2.0, 
        'framePSPIs': (9,) array([ 0.17647633,  0.7358513 , -0.15468097,  0.14511908,  0.41399366,
        0.3002513 ,  0.24338341,  0.17509282,  0.06955796])
        'framePSPIs2': (9,) array([ 0.15224148,  0.56571096, -0.02417694,  0.08819982,  0.31657189,
        0.22623274,  0.19287531,  0.13503292,  0.09428431])
        'videoAFF': 4.0, 
        'videoOPR': 1.0, 
        'videoSEN': 3.0, 
        'aus': (9,9)
        'iaus': (9,10)
        """
        subj_id = self.video_files[idx][0]
        video_id = self.video_files[idx][1]
        images = []
        image_ids = []
        framePSPIs = []
        framePSPIs2 = []
        feat2048 = -np.inf
        feat65 = []
        aus = []
        auSeqs = []
        for img in next(os.walk(os.path.join(self.video_path, subj_id, video_id)))[2]:
            if img[-3:]!='png':
                continue
            # load image
            if self.transform != None:
                image = cv2.imread(os.path.join(self.video_path, subj_id, video_id, img)) # H x W x 3
                images.append(image)
            image_id = img
            image_ids.append(image_id)

            # frameAU
            name = os.path.join(self.AUpath,subj_id,video_id,img[:-4]+'_facs')
            f=open(name + '.txt', "r")
            scorestr = f.readlines()
            f.close()
            scorestr = [x.strip() for x in scorestr]
            au = np.zeros((64,))
            for line in scorestr:
                words = [x.strip() for x in line.split(' ') if x]
                aunumberstr = words[0]
                auintensitystr = words[1]
                aunumber = float(aunumberstr[0:aunumberstr.find('e')]) * (10** int(aunumberstr[aunumberstr.find('+')+1:]))   
                auintensity = float(auintensitystr[0:auintensitystr.find('e')]) * (10** int(auintensitystr[auintensitystr.find('+')+1:]))   
                au[int(aunumber)-1] = auintensity
            
            auSeq = au[[3,5,6,8,9,11,14,19,24,25,26,42]]
            au = au[[3,5,6,9,11,19,24,25,42]]
            # au = au[[3,5,6,8,9,11,19,24,25,26,42]]

            aus.append(au)
            auSeqs.append(auSeq)

            # framePSPI
            name = os.path.join(self.frameVASpath,subj_id,video_id,img[:-4]+'_facs')
            f=open(name + '.txt', "r")
            scorestr = f.read()
            f.close()
            framePSPI = float(scorestr[0:scorestr.find('e')]) * (10** int(scorestr[scorestr.find('+')+1:]))   

            if self.framescorepath: # prediction framePSPI instead
                name = os.path.join(self.framescorepath,subj_id,video_id,img[:-4])
                if os.path.isfile(name+'.txt'):
                    f = open(name + '.txt', "r")
                    scorestr = f.read()
                    f.close()
                    framePSPI = float(scorestr)
                if os.path.isfile(name + '.npz'):
                    # ft2048 = np.load(name + '.npz')["feat_map"]
                    # feat2048 = np.maximum(feat2048, ft2048)
                    ft65 = np.load(name + '.npz')['output']
                    feat65.append(ft65)
                    if len(ft65.ravel())==9:
                        framePSPIs2.append(float(get_PSPI(ft65[1:])))
                if os.path.isfile(name + '.npy'):
                    ft65 = np.load(name + '.npy')
                    feat65.append(ft65)
                    if len(ft65.ravel())==9:
                        framePSPIs2.append(float(get_PSPI(ft65[1:])))
                    framePSPI = float(ft65[-1])
                
                
            framePSPIs.append(framePSPI)
            
        framePSPIs = get_stat_feature(framePSPIs)
        if len(framePSPIs2)>0:
            framePSPIs2 = get_stat_feature(framePSPIs2)
        aus = get_stat_feature(aus)
        if len(feat65)!=0:
            feat65 = get_stat_feature(feat65)
        if self.framescorepath:
            name = os.path.join(self.framescorepath, subj_id, video_id + '.npz')
            if os.path.isfile(name):
                feat65 = np.load(name)['output'] 

        # iMotions AU
        name = os.path.join(self.iau_dir,video_id+'.mat')
        iaus = scipy.io.loadmat(name)
        iaus = iaus['data']
        # print(iaus.shape) # (161,34)
        iaus = get_stat_feature(iaus)
        iaus = iaus[:,[13,15,16,18,19,24,27,28,30]] # the index of AU: 4 6 7 10 12 20 25 26 43 in header: 14 16 17 19 20 25 28 29 31
        iaus = (iaus + 5)/2
        iaus = np.concatenate((get_PSPI(iaus).reshape(-1,1)/15, iaus/5), axis=1) # shape = (9,10)

        # sequence level labels
        name = os.path.join(self.seqVASpath,subj_id,video_id)
        f=open(name + '.txt', "r")
        scorestr = f.read()
        f.close()
        videoVAS = float(scorestr[0:scorestr.find('e')]) * (10** int(scorestr[scorestr.find('+')+1:]))
        videolabel = 0 + (videoVAS>0)

        name = os.path.join(os.path.split(self.seqVASpath)[0], 'SEN',subj_id,video_id)
        f=open(name + '.txt', "r")
        scorestr = f.read()
        f.close()
        videoSEN = float(scorestr[0:scorestr.find('e')]) * (10** int(scorestr[scorestr.find('+')+1:]))
        
        name = os.path.join(os.path.split(self.seqVASpath)[0], 'OPR',subj_id,video_id)
        f=open(name + '.txt', "r")
        scorestr = f.read()
        f.close()
        videoOPR = float(scorestr[0:scorestr.find('e')]) * (10** int(scorestr[scorestr.find('+')+1:]))
        
        name = os.path.join(os.path.split(self.seqVASpath)[0], 'AFF',subj_id,video_id)
        f=open(name + '.txt', "r")
        scorestr = f.read()
        f.close()
        videoAFF = float(scorestr[0:scorestr.find('e')]) * (10** int(scorestr[scorestr.find('+')+1:]))

        sample = {'images': images, 'image_ids': image_ids, 'video_id': video_id, 'feat2048': feat2048, 'feat65': feat65,
            'videolabel': videolabel, 'subj_id': subj_id, 'videoVAS': videoVAS, 'framePSPIs': framePSPIs, 'framePSPIs2': framePSPIs2,
            'videoAFF': videoAFF, 'videoOPR': videoOPR, 
            'videoSEN': videoSEN, 'aus': aus, 'iaus': iaus, 'auSeq': auSeqs}
        if self.transform:
            sample['images'] = [self.transform(image) for image in images]
        if self.labeltransform:
            sample['aus'] = self.labeltransform(aus)
            sample['feat65'] = feat65.ravel()
        return sample


class CenterCrop(object):
    """Center crop the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        image = image[int(h/2-new_h/2):int(h/2+new_h/2),int(w/2-new_w/2):int(w/2+new_w/2),:]

        return image

class GetPain(object):
    """Get pain levels from images

    Args:
        model_img (pytorch model)
    """

    def __init__(self, model_img):
        self.model_img = model_img

    def __call__(self, image):

        image = self.model_img(image.unsqueeze(0))

        return image

