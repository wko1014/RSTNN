import numpy as np
import glob, re, mne, os.path, scipy.io
from mne.filter import filter_data

# Load dataset function
def load_DATA(subject, session_fold):
    def load_dataset_KU(subject, session):
        """
        Saved after preprocessing
        (Laplacian filtering, baseline correction, band-pass filtering (4~40Hz)
        """
        path = "Define/Your/Own/Path/"
        train_data = np.load(path + "Sess%02d_sub%02d_train.npy" % (session, subject))
        test_data = np.load(path + "Sess%02d_sub%02d_test.npy" % (session, subject))
        # Labels are one-hot encoded.
        train_label = np.load(path + "Sess%02d_sub%02d_trlbl.npy" % (session, subject))
        test_label = np.load(path + "Sess%02d_sub%02d_tslbl.npy" % (session, subject))

        train_data, test_data = np.moveaxis(train_data, -1, 0), np.moveaxis(test_data, -1, 0)
        train_label, test_label = np.swapaxes(train_label, 0, 1), np.swapaxes(test_label, 0, 1)

        np.random.seed(951014)
        rand_idx = np.random.permutation(train_data.shape[0])
        train_data = train_data[rand_idx, :, :]
        train_label = train_label[rand_idx, :]

        tmp = int(train_data.shape[0] * 0.1)
        valid_data = train_data[:tmp, :, :]
        valid_label = train_label[:tmp, :]
        train_data = train_data[tmp:, :, :]
        train_label = train_label[tmp:, :]
        return train_data, train_label, valid_data, valid_label, test_data, test_label

    
    train_eeg, train_label, valid_eeg, valid_label, test_eeg, test_label = load_dataset_KU(subject=subject,
                                                                                            session=session_fold)
 
        # Gaussian Normalization
        meanVector, stdVector = np.zeros(shape=(train_eeg.shape[1], 1)), np.zeros(shape=(train_eeg.shape[1], 1))

        for i in range(train_eeg.shape[1]):
            meanVector[i, 0], stdVector[i, 0] = np.mean(np.mean(train_eeg[:, i, :], 0), -1), np.std(np.std(train_eeg[:, i, :], 0), -1)
        meanVector, stdVector = np.tile(meanVector, test_eeg.shape[-1]), np.tile(stdVector, test_eeg.shape[-1])

        for i in range(train_eeg.shape[0]):
            train_eeg[i, :, :] -= meanVector
            train_eeg[i, :, :] /= stdVector

        for i in range(valid_eeg.shape[0]):
            valid_eeg[i, :, :] -= meanVector
            valid_eeg[i, :, :] /= stdVector

        for i in range(test_eeg.shape[0]):
            test_eeg[i, :, :] -= meanVector
            test_eeg[i, :, :] /= stdVector

        train_eeg, valid_eeg, test_eeg = np.expand_dims(train_eeg, -1), np.expand_dims(valid_eeg, -1), np.expand_dims(test_eeg, -1)

    return train_eeg, train_label, valid_eeg, valid_label, test_eeg, test_label

class callDataset():
    def __init__(self, sbjIdx, sessIdx):
        assert 0 < sbjIdx < 55, "We only have subject 1 to 54."
        assert 0 < sessIdx < 3, "We only have session 1 and 2."

        self.sbjIdx, self.sessIdx = sbjIdx, sessIdx
        self.path = "Define/Your/Own/Path"

    def loadData(self):
        Xtr = np.load(self.path + "/TIME_Sess{:>02d}_sub{:>02d}_train.npy".format(self.sessIdx, self.sbjIdx))
        Ytr = np.load(self.path + "/TIME_Sess{:>02d}_sub{:>02d}_trlbl.npy".format(self.sessIdx, self.sbjIdx))

        Xts = np.load(self.path + "/TIME_Sess{:>02d}_sub{:>02d}_test.npy".format(self.sessIdx, self.sbjIdx))
        Yts = np.load(self.path + "/TIME_Sess{:>02d}_sub{:>02d}_tslbl.npy".format(self.sessIdx, self.sbjIdx))

        # Divide validation set
        numVals = int(Xtr.shape[-1]/10)
        Xvl, Yvl = Xtr[:, :, :numVals], Ytr[:, :numVals]
        Xtr, Ytr = Xtr[:, :, numVals:], Ytr[:, numVals:]

        return Xtr, Ytr, Xvl, Yvl, Xts, Yts

    def GaussNorm(self):
        Xtr, Ytr, Xvl, Yvl, Xts, Yts = self.loadData()
        # Calculate mean and standard deviation for Gaussian normalization
        meantr, stdtr = np.mean(Xtr, axis=(1, 2), keepdims=True), np.std(Xtr, axis=(1, 2), keepdims=True)

        def myNorm(X): return (X - meantr)/stdtr

        Xtr, Xvl, Xts = myNorm(Xtr), myNorm(Xvl), myNorm(Xts)

        return Xtr, Ytr, Xvl, Yvl, Xts, Yts

    def prepareData(self, is_GaussNorm=True):
        if is_GaussNorm: Xtr, Ytr, Xvl, Yvl, Xts, Yts = self.GaussNorm()
        else: Xtr, Ytr, Xvl, Yvl, Xts, Yts = self.loadData()

        Xtr, Xvl, Xts = np.moveaxis(Xtr, -1, 0), np.moveaxis(Xvl, -1, 0), np.moveaxis(Xts, -1, 0)
        Xtr, Xvl, Xts = np.expand_dims(Xtr, -1), np.expand_dims(Xvl, -1), np.expand_dims(Xts, -1)
        Ytr, Yvl, Yts = np.moveaxis(Ytr, 0, 1), np.moveaxis(Yvl, 0, 1), np.moveaxis(Yts, 0, 1)

        return Xtr, Ytr, Xvl, Yvl, Xts, Yts

