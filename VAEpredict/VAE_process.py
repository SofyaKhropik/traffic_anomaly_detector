import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from VAEpredict.VAE.VAEarchitecture import VAE


test_ratio = 0.001
quant = 0.995


class TSDataset(Dataset):
    def __init__(self, dset, cont_vars=None, cat_vars=None, lbl_as_feat=True):
        super().__init__()
        self.lbl_as_feat = lbl_as_feat
        self.df = dset
        self.cont_vars = cont_vars
        self.cat_vars = cat_vars
       

        # Finally, make two Numpy arrays for continuous and categorical
        # variables, respectively:
        if self.lbl_as_feat:
            self.cont = self.df[self.cont_vars].copy().to_numpy(dtype=np.float32)
        else:
            self.cont = self.df[self.cont_vars].copy().to_numpy(dtype=np.float32)
            self.lbl = self.df['Битрейт (КБ/сек)'].copy().to_numpy(dtype=np.float32)
        self.cat = self.df[self.cat_vars].copy().to_numpy(dtype=np.int64)

    def __getitem__(self, idx):
        if self.lbl_as_feat:  # for VAE training
            return torch.tensor(self.cont[idx]), torch.tensor(self.cat[idx])
        else:  # for supervised prediction
            return torch.tensor(self.cont[idx]), torch.tensor(self.cat[idx]), torch.tensor(self.lbl[idx])

    def __len__(self):
        return self.df.shape[0]




def predict_loss(test_data):
    test_data['Stable'] = "True"
    test_data['Processing_time'] = np.random.randint(1, 4, size=len(test_data))
    test_data['Source'] = test_data['Source'].str.replace('.', '')
    test_data['Destination'] = test_data['Destination'].str.replace('.', '')
    columns_to_drop = ['No.', 'Info']
    test_data = test_data.drop(columns=columns_to_drop, errors='ignore')
    #print(test_data.head())
    cont_vars = ['Length','Processing_time','Source','Destination']
    cat_vars = ['Stable','Protocol']

    label_encoders = [LabelEncoder() for _ in cat_vars]
    for col, enc in zip(cat_vars, label_encoders):
        test_data[col] = enc.fit_transform(test_data[col])
    scaler = preprocessing.StandardScaler().fit(test_data[cont_vars])
    tst_data_scaled = test_data.copy()
    tst_data_scaled[cont_vars] = scaler.transform(test_data[cont_vars])
    
    dataset = TSDataset(dset=tst_data_scaled, cont_vars=['Length', ], cat_vars=['Stable'], lbl_as_feat=True)
    
    trained_model = VAE.load_from_checkpoint('VAEpredict/VAE/vae_weights-v10.ckpt')
#trained_model.cuda() # перенос модели на графический вычислитель в случае использования GPU
    trained_model.freeze()
#pred = model(x.cuda()) # перенос модели на графический вычислитель в случае использования GPU

    losses = []
# run predictions for the training set examples
    for i in range(len(dataset)):
        x_cont, x_cat = dataset[i]
        x_cont.unsqueeze_(0)
        x_cat.unsqueeze_(0)
    #    recon, mu, logvar, x = trained_model.forward((x_cont.cuda(), x_cat.cuda())) # в случае использования GPU
        recon, mu, logvar, x = trained_model.forward((x_cont, x_cat))
        recon_loss, kld = trained_model.loss_function(x, recon, mu, logvar)
        losses.append(recon_loss + trained_model.hparams.kld_beta * kld)

    data_with_losses_test = dataset.df
    data_with_losses_test['loss'] = torch.asarray(losses)

    mean, sigma = data_with_losses_test['loss'].mean(), data_with_losses_test['loss'].std()

    thresh = data_with_losses_test['loss'].quantile(quant)  # threshold для аномалий зависит от квантиля выборки (quant)

    data_with_losses_test['anomaly'] = data_with_losses_test['loss'] > thresh

    data_with_losses_unscaled_test = data_with_losses_test.copy()

    for enc, var in zip(label_encoders, cat_vars):
        data_with_losses_unscaled_test[var] = enc.inverse_transform(data_with_losses_test[var])
    data_with_losses_unscaled_test = pd.DataFrame(data_with_losses_unscaled_test, columns=data_with_losses_test.columns)


    anomalies_value = data_with_losses_unscaled_test.loc[data_with_losses_unscaled_test['anomaly'], ['loss',
                                                                                                    'Length']]
    normals_value = data_with_losses_unscaled_test.loc[~data_with_losses_unscaled_test['anomaly'], ['loss',
                                                                                                    'Length']]

    return data_with_losses_unscaled_test, anomalies_value, normals_value
