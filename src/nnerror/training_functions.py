

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset, random_split

from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm

from .networks.im2spec_models import *

def norm_0to1(arr):
    arr = np.asarray(arr)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr


# def train_model(model, imgs_train, spectra_train, n_epochs = 100):

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     n_epochs = 100

#     train_loss = []

#     model.train()

#     train_images = torch.tensor(imgs_train, dtype=torch.float32)
#     train_spectra = torch.tensor(spectra_train, dtype=torch.float32)


#     for epoch in range(n_epochs):
        
#         optimizer.zero_grad()
#         outputs = model(train_images)
#         loss = criterion(outputs, train_spectra)

#         loss.backward()
#         optimizer.step()

#         train_loss.append(loss.item())


#     model.eval()

#     return model, train_loss


def l1_regularization(model, l1_lambda = 1e-4): # l1_lambda : regularization_strength
    l1_loss = sum(p.abs().sum() for p in model.parameters())  # Sum of absolute values of parameters
    return l1_lambda * l1_loss

   
    
class ELBOLoss(nn.Module):  
    
    def __init__(self, recon_loss_fn = nn.MSELoss(), beta_elbo = 0.1): # beta_elbo is regularization strength.
        super().__init__()  
        
        self.beta_elbo = beta_elbo
        self.recon_loss_fn = recon_loss_fn 

    def forward(self, output, train_spectra):
        """
        Computes the ELBO (Evidence Lower Bound) loss for Variational Autoencoder (VAE).
        
        Args:
            output: (pred_spectra, mu, logvar)
            train_spectra: Ground truth spectra

        Returns:
            Total ELBO loss (torch.Tensor)
        """
        pred_spectra, mu, logvar = output  

        # Reconstruction loss
        recon_loss = self.recon_loss_fn(pred_spectra, train_spectra)

            # KL Divergence loss (Summed over latent dimensions)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

            # Mean KL loss across batch
        kl_loss = torch.mean(kl_loss)

        return recon_loss + self.beta_elbo * kl_loss

    
def vae_loss_mse(output, train_spectra, beta_elbo = 1e-3):
    
    pred_spectra, mu, logvar = output  
    
    # Reconstruction Loss (Mean Squared Error)
    recon_loss = F.mse_loss(pred_spectra, train_spectra, reduction='mean')

    # KL Divergence Loss (Regularization)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta_elbo*kl_loss



class EarlyStopping:
    
    def __init__(self, skip_epochs = 100, patience = 5, min_delta = 0):
    
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = np.inf
        self.skip_epochs = skip_epochs
        self.counter = 0
        
    def __call__(self, val_loss, epoch):
        
        if epoch < self.skip_epochs:
            return False
        
        elif val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0  # Reset patience counter
        else:
            self.counter += 1  # Increase counter if no improvement
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.counter} epochs!")
                return True
            
        return False


class EarlyStopping_ensemble_swatrigger:
    
    def __init__(self, patience = 5, min_delta = 0, skip_epochs = 100, swa_epoch_th = 100, n_models = 1):
    
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = [np.inf for _ in range(n_models)]
        self.counter = [0 for _ in range(n_models)]
        self.val_loss = [np.inf for _ in range(n_models)]
        self.epoch_count = [0 for _ in range(n_models)]
        self.skip_epochs = skip_epochs
        
        self.swa_epoch_th = 100
        self.trigger_swa = [False for _ in range(n_models)]
        
    def __call__(self, model_idx):
        
        # Skip initial epochs
        if self.epoch_count[model_idx] < self.skip_epochs:
            return False
        
        elif self.val_loss[model_idx] < self.best_val_loss[model_idx] - self.min_delta:
            self.best_val_loss[model_idx] = self.val_loss[model_idx]
            self.counter[model_idx] = 0  # Reset patience counter
            self.trigger_swa[model_idx] = True
            
        else:
            self.counter[model_idx] += 1  # Increase counter if no improvement
            self.trigger_swa[model_idx] = False
            if self.counter[model_idx] >= self.patience:
                #print(f"Early stopping triggered model_id {model_idx} after {self.counter[model_idx]} epochs!")
                return True
            
        return False
    
    def enter_val_loss(self, val_epoch_loss, model_idx):
        
        self.val_loss[model_idx] = val_epoch_loss
        self.epoch_count[model_idx] += 1
        
    def trigger_swa_output(self, model_idx):
        
        if self.epoch_count[model_idx] <= self.swa_epoch_th:
            return False
        
        else:
            return self.trigger_swa[model_idx]
        
class EarlyStopping_ensemble:
    
    def __init__(self, patience = 5, min_delta = 0, skip_epochs = 100, n_models = 1):
    
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = [np.inf for _ in range(n_models)]
        self.counter = [0 for _ in range(n_models)]
        self.val_loss = [np.inf for _ in range(n_models)]
        self.epoch_count = [0 for _ in range(n_models)]
        self.skip_epochs = skip_epochs
        
        
        
    def __call__(self, model_idx):
        
        # Skip initial epochs
        if self.epoch_count[model_idx] < self.skip_epochs:
            return False
        
        elif self.val_loss[model_idx] < self.best_val_loss[model_idx] - self.min_delta:
            self.best_val_loss[model_idx] = self.val_loss[model_idx]
            self.counter[model_idx] = 0  # Reset patience counter
           
            
        else:
            self.counter[model_idx] += 1  # Increase counter if no improvement
         
            if self.counter[model_idx] >= self.patience:
                #print(f"Early stopping triggered model_id {model_idx} after {self.counter[model_idx]} epochs!")
                return True
            
        return False
    
    def enter_val_loss(self, val_epoch_loss, model_idx):
        
        self.val_loss[model_idx] = val_epoch_loss
        self.epoch_count[model_idx] += 1
        
       
    

def train_model_ensemble(model, dataset, lr = [0.1, 0.1, 0.1, 0.1, 0.1], n_epochs = 100, patience = 10,n_batches =3, 
                         l1_rglr = False, vae = False, beta_elbo = 0.1, weight_decay = 1e-6, swa = False, swa_epoch = 50, batchsize = None):

    """
    if batchsize is provided, n_batches is disregarded
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    
    criterion = nn.MSELoss()
        
    val_criterion  = nn.MSELoss()
    
    optimizers = [torch.optim.Adam(model.models[i].parameters(), lr=lr[i], weight_decay=weight_decay) for i in range(len(model.models))]
    
    
    # SWA model wrapper
    if swa:
        swa_model_list = [AveragedModel(submodel) for submodel in model.models]
        swa_model = Swa_Ensemble(swa_model_list)
        
    swa_triggered = False
    

    model.train()
    
    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    #Keep batchsize atleast 1
    train_batch_size = max(1, len(train_dataset)//n_batches)
    val_batch_size = max(1, len(val_dataset)//n_batches)
    
    tr_dataloader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = val_batch_size, shuffle = True)
    
    if batchsize is not None:
        tr_dataloader = DataLoader(train_dataset, batch_size = batchsize, shuffle = True)
        val_dataloader = DataLoader(val_dataset, batch_size = batchsize, shuffle = True)

    n_models = len(model.models)
    earlystopping = EarlyStopping_ensemble(patience = patience, min_delta = 0, n_models = n_models)
    
           
    
    progress_bar = tqdm(total=n_epochs, desc="im2spec Training", leave=False)
    train_loss = [[] for _ in range(n_models)]
    val_loss = [[] for _ in range(n_models)]
    
           
    for epoch in range(n_epochs):
            
        #train_loss_vector = []
        #val_loss_vector = []
        for idx, submodel in enumerate(model.models):
            
            if earlystopping(idx):
                continue

            tr_epoch_loss = 0
            val_epoch_loss = 0
            
            # Training
            submodel.train()

            for train_images, train_spectra in tr_dataloader:
                
                train_images, train_spectra = train_images.to(device), train_spectra.to(device)                           
                      
                optimizers[idx].zero_grad()
                
                output = submodel(train_images)
            
                    
                if vae:
                    loss = vae_loss_mse(output, train_spectra, beta_elbo = beta_elbo)
                else:
                    loss = criterion(output, train_spectra)
                
                # to implement l1_regularization
                if l1_rglr:
                    loss += l1_regularization(submodel, 1e-4)
                    
                tr_epoch_loss += loss.item()

                loss.backward()
                optimizers[idx].step()            
            
            tr_epoch_loss /= len(tr_dataloader)

            
            train_loss[idx].append(tr_epoch_loss)
        
            # Update swa_model
            
            if swa and epoch >= swa_epoch:
                swa_model.models[idx].update_parameters(submodel)
                swa_triggered = True
            
        
            # Validation
        
            submodel.eval()
            
            for val_images, val_spectra in val_dataloader:
                
                val_images, val_spectra = val_images.to(device), val_spectra.to(device)
                
                                    
                output = submodel.predict(val_images)
            
                loss = val_criterion(output, val_spectra)

                val_epoch_loss += loss.item()

            val_epoch_loss /= len(val_dataloader)

          
            val_loss[idx].append(val_epoch_loss)
            earlystopping.enter_val_loss(val_epoch_loss, idx)
            
        # update progress bar
        progress_bar.update(1)
            
    progress_bar.close()
    
    
    if swa_triggered:
        
        # Recompute BN and the update batch_stats.
        for submodel in swa_model.models:
            update_bn(tr_dataloader, submodel, device = device)
            submodel.eval()
            
        model = swa_model
        model.eval()
        
    else:
        model.eval()
        
        
    return model, train_loss, val_loss, swa_triggered


def train_error_ensemble(model, dataset, n_batches= 3, lr = 0.1, patience = 10, n_epochs = 100, batchsize = None):

    """
    if batchsize is provided, n_batches is disregarded
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    
    optimizers = [torch.optim.Adam(submodel.parameters(), lr=lr, weight_decay = 1e-6) for submodel in model.models]

    train_loss = []
    val_loss = []

    model.train()

    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    
    #Keep batchsize atleast 1
    train_batch_size = max(1, len(train_dataset)//n_batches)
    val_batch_size = max(1, len(val_dataset)//n_batches)
    
    tr_dataloader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = val_batch_size, shuffle = True)
    
    if batchsize is not None:
        tr_dataloader = DataLoader(train_dataset, batch_size = batchsize, shuffle = True)
        val_dataloader = DataLoader(val_dataset, batch_size = batchsize, shuffle = True)

    n_models = len(model.models)
    earlystopping = EarlyStopping_ensemble(patience = patience, min_delta = 0, n_models = n_models)
    
    train_loss = [[] for _ in range(n_models)]
    val_loss = [[] for _ in range(n_models)]
    

    for epoch in range(n_epochs):
      

        for idx, submodel in enumerate(model.models):
            
            if earlystopping(idx):
                continue
         
            tr_epoch_loss = 0
            val_epoch_loss = 0
            
            # Training
            submodel.train()

            for train_images, train_error_vector in tr_dataloader:
                
                train_images, train_error_vector = train_images.to(device), train_error_vector.to(device)                           
                      
                optimizers[idx].zero_grad()
                
                output = submodel(train_images)
            
                
                loss = criterion(output, train_error_vector[:, idx])
                tr_epoch_loss += loss.item()

                loss.backward()
                optimizers[idx].step()            
            
            tr_epoch_loss /= len(tr_dataloader)
            
            train_loss[idx].append(tr_epoch_loss)
            
            # Validation
        
            submodel.eval()
            
            for val_images, val_error_vector in val_dataloader:
                
                val_images, val_error_vector = val_images.to(device), val_error_vector.to(device)
                
                                    
                output = submodel.predict(val_images)
            
                loss = criterion(output, val_error_vector[:, idx])

                val_epoch_loss += loss.item()

            val_epoch_loss /= len(val_dataloader)

            
            val_loss[idx].append(val_epoch_loss)
            earlystopping.enter_val_loss(val_epoch_loss, idx)
            
            
       

    model.eval()

    return model, train_loss, val_loss



def train_model(model, dataset, n_batches= 3, lr = 0.1, patience = 10, n_epochs = 100, partial_train = True, 
                batchsize = None):

    """
        if batchsize is provided, n_batches is disregarded
    """
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = []
    val_loss = []

    

    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    
    #Keep batchsize atleast 1
    train_batch_size = max(1, len(train_dataset)//n_batches)
    val_batch_size = max(1, len(val_dataset)//n_batches)
    
    tr_dataloader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = val_batch_size, shuffle = True)
    
    #if batchsize provided disregard n_batches
    if batchsize is not None:
        tr_dataloader = DataLoader(train_dataset, batch_size = batchsize, shuffle = True)
        val_dataloader = DataLoader(val_dataset, batch_size = batchsize, shuffle = True)

    
    earlystopping = EarlyStopping(skip_epochs = 100, patience = patience, min_delta = 0)
    
    
    

    for epoch in tqdm(range(n_epochs)):     



        tr_epoch_loss = 0
        val_epoch_loss = 0

        
        # Training
        if partial_train:
            if not hasattr(model, "train_only_decoder"):
                raise AttributeError("Model does not have 'train_only_decoder' method. Set partial_train=False")
            
            model.train_only_decoder()
        else:
            model.train()

        for train_images, train_label in tr_dataloader:

            train_images, train_label = train_images.to(device), train_label.to(device)                           

            optimizer.zero_grad()

            output = model(train_images)

            loss = criterion(output, train_label)
            tr_epoch_loss += loss.item()

            loss.backward()
            optimizer.step()            

        tr_epoch_loss /= len(tr_dataloader)

        train_loss.append(tr_epoch_loss)

        # Validation

        model.eval()

        for val_images, val_label in val_dataloader:

            val_images, val_label = val_images.to(device), val_label.to(device)


            output = model.predict(val_images)

            loss = criterion(output, val_label)
            
            

            val_epoch_loss += loss.item()

        val_epoch_loss /= len(val_dataloader)


        val_loss.append(val_epoch_loss)
        
        if earlystopping(val_epoch_loss, epoch):
            break
        
            
            
       

    model.eval()

    return model, train_loss, val_loss



def acquisition_fn(error_mean, error_std, beta = 1, index_exclude = [], optimize = "minimize", sample_next_points = 1):
    
    aq_fn = error_mean + beta*error_std

    aq_fn = np.asarray(aq_fn)
    
    

    if optimize == "maximize":
        
        aq_fn[index_exclude] = - 10
        aq_ind = np.argsort(aq_fn)[::-1][:sample_next_points]  
        
    else:
        aq_fn[index_exclude] = 10
        aq_ind = np.argsort(aq_fn)[:sample_next_points]

    return aq_ind, aq_fn


def append_training_set(images, spectra, next_index, imgs_train, spectra_train, indices_train):

    imgs_train = np.append(imgs_train, images[next_index].reshape(1, images.shape[1], images.shape[2]), axis = 0)

    spectra_train = np.append(spectra_train, spectra[next_index].reshape(1, spectra.shape[1]), axis = 0)
    
    indices_train = np.append(indices_train, next_index)

    return imgs_train, spectra_train, indices_train


def predict_spectra(model, images, ensemble = True):
    
    images = torch.tensor(images, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    images = images.to(device)

    model.eval()
    
    if ensemble == True:
        pred_spectra = []
        
        for submodel in model.models:
            outputs = submodel.predict(images)
            pred_spectra_i = outputs.cpu().detach().squeeze().numpy()
            pred_spectra.append(pred_spectra_i)
    
    else:        
        outputs = model.predict(images)
        pred_spectra = outputs.cpu().detach().squeeze().numpy()
    
    return pred_spectra

def predict_embedding(model, images):
    
    images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) # add the channel_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    images = images.to(device)

    model.eval()
    
    with torch.no_grad():      
        outputs = model.encoder(images)
        
    pred_spectra = outputs.cpu().detach().squeeze().numpy()

    return pred_spectra

def predict_vae_embedding(model, images):
    
    images = torch.tensor(images, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    images = images.to(device)

    model.eval()
    
    with torch.no_grad():      
        outputs = model.embedding(images)
        
    pred_spectra = outputs.cpu().detach().squeeze().numpy()

    return pred_spectra
    
    
def predict_posterior(model, images, output_type = "prediction"):

    if output_type == "latent" :
        pred_spectra = predict_embedding(model, images)
        
    elif output_type == "prediction":
        pred_spectra = predict_spectra(model, images, ensemble = False)
       
    elif output_type == "vae_latent":
        pred_spectra = predict_vae_embedding(model, images)
        
    else:
        raise ValueError('Invalid output_type. Valid: prediction, latent, vae_latent')
        
    return pred_spectra


def distance_distribution(spectra, spectra_train, distance_type = "L2"):
    
    spectra = np.asarray(spectra)
    spectra_train = np.asarray(spectra_train)
    
    mean_spectra = spectra_train.mean(axis = 0)
    
    all_distance = []
    
    for i in range(spectra.shape[0]):
        
        distance = calc_distance(spectra[i], mean_spectra, distance_type = distance_type)
        
        all_distance.append(distance)     
        
    all_distance =  np.asarray(all_distance)
        
    return all_distance
         
    
def calc_distance(X, Y, distance_type = "L2"):

    if distance_type == 'L1':
    
        distance = np.sum(np.abs(X - Y), axis=-1)
        
    elif distance_type == 'L2':
    
        distance = np.sqrt(np.sum((X - Y) ** 2, axis=-1))
    
    
    elif distance_type == 'cos':
    
        cosine_similarity = np.dot(X, Y.T) / (np.linalg.norm(X) * np.linalg.norm(Y))
        distance = 1 - cosine_similarity
    
    else:
        raise ValueError('Invalid distance_type. Valid: L1, L2, cos')
    
    return distance
        

    
def distance_acq_fn(distances, beta = 0.5, lambda_ = 1, optimize = "custom_fn", sample_next_points = 10, exclude_indices = []):

    distances = np.ravel(np.asarray(distances))
    acq_vals = norm_0to1(distances) 
    
    
    
    if optimize == "minimize":
        
        aq_vals[exclude_indices] = 2
        aq_ind = np.argsort(acq_vals)
        

    elif optimize == "maximize":
        
        aq_vals[exclude_indices] = -1
        aq_ind = np.argsort(acq_vals)[::-1]

    elif optimize == "custom_fn":

                    # EXPLORATION + EXPLOITATION
        acq_vals = (1-np.exp(-lambda_ * np.abs(acq_vals-(1-beta))))
        acq_vals = norm_0to1(acq_vals) 
        
        #acq_vals = beta*(1- np.exp(-lambda_ * distances)) + (1-beta)*np.exp(-lambda_ * distances)

        acq_vals[exclude_indices] = -1
        aq_ind = np.argsort(acq_vals)[::-1]

    else:
        raise ValueError('Invalid optimization type')
    

    aq_ind = aq_ind[:sample_next_points]


    return aq_ind, acq_vals

    
    
    
def err_estimation(model, images, spectra):

    images = torch.tensor(images, dtype=torch.float32)
    spectra = torch.tensor(spectra, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    images, spectra = images.to(device), spectra.to(device)

    model.eval()
    
    outputs = model.predict(images)
    #print(outputs.shape)
    error_vector = torch.abs(outputs.squeeze(1) - spectra)
    #print(error_vector.shape)
    
    # copy to host cpu before detaching
    error_vector = error_vector.cpu()
    error_vector = error_vector.detach().squeeze().numpy()
    error_mean = np.mean(error_vector, axis = -1)
    error_std = np.std(error_vector, axis = -1)
   
    
    return error_mean, error_std, error_vector

def error_dataset(model, images, spectra, norm = True):

    error_vector = []
    preds_spectra = []
    for submodel in model.models:

        error_mean, _, pred_spectra = err_estimation(submodel, images, spectra)

        if norm:
            error_mean = norm_0to1(error_mean)
        
        error_vector.append(error_mean)
        preds_spectra.append(pred_spectra)

    error_vector = np.asarray(error_vector).T

    return error_vector
   

def predict_error(error_ensemble_model, images):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = torch.tensor(images, dtype=torch.float32)
    images = images.to(device)
    error_ensemble_model.to(device)
    
    errors = []
    for image in images:
        
        image = image.unsqueeze(0)
        error_vector = error_ensemble_model.predict(image)
        
        # Get to host cpu before detach
        errors.append([error.cpu().detach().numpy() for error in error_vector])

    errors = np.asarray(errors).squeeze()[:, np.newaxis]
    error_mean = np.mean(errors, axis = -1)
    error_std = np.std(errors, axis = -1)
    
    return error_mean, error_std, errors


def sort_model_idx(training_loss, last_epochs = 10):
    
    loss_all  = []
    
    for i in range(len(training_loss)):
            
        avg_ending_loss = np.asarray(training_loss[i][-last_epochs:-1]).mean()
        
        loss_all.append(avg_ending_loss)
        
        
    loss_all =  np.asarray(loss_all)
    model_indices = np.argsort(loss_all)
    
    return model_indices



