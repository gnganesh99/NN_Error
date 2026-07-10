

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

def norm_0to1(arr, eps=1e-8):
    """Normalize an array to the [0, 1] range."""
    arr = np.asarray(arr)
    arr_min = arr.min()
    arr_range = arr.max() - arr_min
    if arr_range < eps:
        print(f"Warning: norm_0to1 array range {arr_range} < eps {eps}; returning zeros.")
        return np.zeros_like(arr, dtype=float)
    return (arr - arr_min) / arr_range


# Model training function with swa updates
  
def train_model(model, dataset, n_batches= 3, lr = 0.1, patience = 10, n_epochs = 100, partial_train = False,
                batchsize = None, val_dataset = None, use_swa = False, last_swa_epochs = 0.1):
 
    """
    Train a model with optional encoder freezing, early stopping, and a
    live tqdm progress bar showing per-epoch train and validation loss.
 
    Args:
        model: nn.Module to train. If `partial_train=True`, the model must
            implement a `train_only_decoder()` method.
        dataset: Training dataset. If `val_dataset` is None, this is split
            80/20 into train/val internally.
        n_batches: Number of batches per epoch. Used to derive batch size
            as `len(train_dataset) // n_batches` (and likewise for val).
            Ignored if `batchsize` is provided. Default 3.
        lr: Learning rate for the Adam optimizer. Default 0.1.
        patience: Number of epochs without val-loss improvement to wait
            before early stopping. Default 10.
        n_epochs: Maximum number of training epochs. Default 100.
        partial_train: If True, only the decoder is trained (encoder
            frozen) via `model.train_only_decoder()`. If False, the full
            model is trained via `model.train()`. Default True.
        batchsize: Explicit batch size. If provided, `n_batches` is
            ignored and this value is used for both train and val
            DataLoaders. Default None.
        val_dataset: Optional separate validation dataset. If None, the
            training `dataset` is split 80/20 into train/val. Default None.
        use_swa: If True, use Stochastic Weight Averaging during the final portion
            of training. Default False.
        last_swa_epochs: If `use_swa=True`, the final portion of training during which SWA
            updates are applied. If >1, this is interpreted as a number of epochs;
            if <=1, this is interpreted as a fraction of `n_epochs`. Default 0.1 (10% of training).
    Returns:
        Tuple of (model, train_loss, val_loss) where:
            model: The trained model, set to eval mode. If `use_swa=True`
                and SWA was triggered, this is the original model object with
                the SWA-averaged weights copied into it.
            train_loss: List of per-epoch average training losses.
            val_loss: List of per-epoch average validation losses.
 
    Raises:
        AttributeError: If `partial_train=True` and the model does not
            implement a `train_only_decoder` method.
 
    Notes:
        - Uses MSELoss and the Adam optimizer.
        - Validation uses `model.predict(...)` rather than `model(...)`.
        - Early stopping starts after `skip_epochs=100` epochs.
    """
 
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
 
    criterion = nn.MSELoss()
 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    train_loss = []
    val_loss = []
 
    swa_triggered = False
    swa_model = None
    swa_sclr = None  # Swa scheduler to tune the optimizer learning rate during swa phase. This creates stability in model updates.
    if use_swa:
        swa_model = AveragedModel(model)
        swa_sclr = SWALR(optimizer, swa_lr = lr*0.1, anneal_epochs = 15, anneal_strategy = "cos")
 
        if last_swa_epochs > 1:
            swa_epoch = max(0, n_epochs - last_swa_epochs)
        else:
            swa_epoch = int(n_epochs * (1 - last_swa_epochs))
 
 
    if val_dataset is not None:
 
        train_size = len(dataset)
        val_size = len(val_dataset)
 
        train_dataset = dataset
        val_dataset = val_dataset
 
    else:
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
 
 
    pbar = tqdm(range(n_epochs))
 
    for epoch in pbar:
 
 
 
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
 
        if swa_triggered:
            swa_sclr.step()
            
 
        tr_epoch_loss /= len(tr_dataloader)
 
        train_loss.append(tr_epoch_loss)
 
 
 
        model.eval()
 
        with torch.no_grad():
            for val_images, val_label in val_dataloader:
           
                val_images, val_label = val_images.to(device), val_label.to(device)
 
                output = model.predict(val_images)
 
                loss = criterion(output, val_label)
 
                val_epoch_loss += loss.item()
 
        val_epoch_loss /= len(val_dataloader)
 
 
        val_loss.append(val_epoch_loss)
 
 
 
        if use_swa and epoch >= swa_epoch:
            if not swa_triggered:                
                print(f"SWA triggered at epoch {epoch}")
                swa_triggered = True
            swa_model.update_parameters(model)
 
        pbar.set_postfix(train_loss=f"{tr_epoch_loss:.4f}", val_loss=f"{val_epoch_loss:.4f}")
 
        if earlystopping(val_epoch_loss, epoch):
            break
 
 
    if use_swa and swa_triggered:
        torch.optim.swa_utils.update_bn(tr_dataloader, swa_model, device = device)
        model =  swa_model.module
        #model.load_state_dict(swa_model.module.state_dict())
 
    model.eval()
 
    return model, train_loss, val_loss
 

def l1_regularization(model, l1_lambda = 1e-4): # l1_lambda : regularization_strength
    """
    Compute the L1 penalty for all parameters in a PyTorch model.

    Args:
        model: Model whose parameters are regularized.
        l1_lambda: Multiplicative regularization strength.

    Returns:
        Scalar tensor containing `l1_lambda * sum(abs(parameters))`.
    """
    l1_loss = sum(p.abs().sum() for p in model.parameters())  # Sum of absolute values of parameters
    return l1_lambda * l1_loss



class ELBOLoss(nn.Module):

    """ELBO-style loss module for variational spectral reconstruction."""
    def __init__(self, recon_loss_fn = nn.MSELoss(), beta_elbo = 0.1): # beta_elbo is regularization strength.
        """Initialize ELBOLoss."""
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
    """
    Compute a VAE-style reconstruction loss with KL regularization.

    Args:
        output: Tuple of `(pred_spectra, mu, logvar)` from the model.
        train_spectra: Target spectra tensor.
        beta_elbo: Weight applied to the KL divergence term.

    Returns:
        Scalar tensor containing MSE reconstruction loss plus weighted KL loss.
    """

    pred_spectra, mu, logvar = output

    # Reconstruction Loss (Mean Squared Error)
    recon_loss = F.mse_loss(pred_spectra, train_spectra, reduction='mean')

    # KL Divergence Loss (Regularization)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta_elbo*kl_loss



class EarlyStopping:
    """
    Early-stopping helper for a single model.

    The first `skip_epochs` epochs are ignored. After that, validation loss
    must improve by at least `min_delta` before `patience` non-improving
    epochs are reached.
    """

    def __init__(self, skip_epochs = 100, patience = 5, min_delta = 0):

        """Initialize EarlyStopping."""
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = np.inf
        self.skip_epochs = skip_epochs
        self.counter = 0

    def __call__(self, val_loss, epoch):

        """Apply the callable helper."""
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
    """
    Per-model early stopping helper with SWA trigger tracking.

    Each ensemble member has independent validation-loss state. A model can
    request SWA updates after `swa_epoch_th` once its validation loss improves.
    """

    def __init__(self, patience = 5, min_delta = 0, skip_epochs = 100, swa_epoch_th = 100, n_models = 1):

        """Initialize EarlyStopping_ensemble_swatrigger."""
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
        """Apply the callable helper."""
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

        """Store the latest validation loss for one ensemble member."""
        self.val_loss[model_idx] = val_epoch_loss
        self.epoch_count[model_idx] += 1

    def trigger_swa_output(self, model_idx):

        """Return whether SWA should update for one ensemble member."""
        if self.epoch_count[model_idx] <= self.swa_epoch_th:
            return False

        else:
            return self.trigger_swa[model_idx]

class EarlyStopping_ensemble:
    """
    Per-model early stopping helper for an ensemble.

    Each ensemble member keeps its own best validation loss, patience counter,
    and epoch count so stopped members can be skipped independently.
    """

    def __init__(self, patience = 5, min_delta = 0, skip_epochs = 100, n_models = 1):

        """Initialize EarlyStopping_ensemble."""
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = [np.inf for _ in range(n_models)]
        self.counter = [0 for _ in range(n_models)]
        self.val_loss = [np.inf for _ in range(n_models)]
        self.epoch_count = [0 for _ in range(n_models)]
        self.skip_epochs = skip_epochs



    def __call__(self, model_idx):

        # Skip initial epochs
        """Apply the callable helper."""
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

        """Store the latest validation loss for one ensemble member."""
        self.val_loss[model_idx] = val_epoch_loss
        self.epoch_count[model_idx] += 1




def train_model_ensemble(model, dataset, lr = [0.1, 0.1, 0.1, 0.1, 0.1], n_epochs = 100, patience = 10,
                         n_batches =3, l1_rglr = False, vae = False, beta_elbo = 0.1,
                         weight_decay = 1e-6, swa = False, swa_epoch = 50, batchsize = None,
                         val_dataset = None):

    """
    Train each member of an im2spec ensemble on the same dataset.

    Args:
        model: Ensemble model with a `models` attribute containing submodels.
        dataset: Training dataset. If `val_dataset` is None, this is split
            80/20 into train/validation subsets with `random_split`.
        lr: Learning rate for each submodel optimizer. Must have at least one
            value per ensemble member.
        n_epochs: Maximum number of epochs to train.
        patience: Number of non-improving validation epochs before a submodel
            is skipped by early stopping.
        n_batches: Number of batches used to derive train and validation batch
            sizes. Ignored when `batchsize` is provided.
        l1_rglr: If True, add L1 regularization to the training loss.
        vae: If True, use `vae_loss_mse`; otherwise use MSE loss.
        beta_elbo: KL-divergence weight used when `vae=True`.
        weight_decay: Adam optimizer weight decay.
        swa: If True, maintain stochastic weight averaged submodels after
            `swa_epoch`.
        swa_epoch: First epoch at which SWA parameter updates are applied.
        batchsize: Explicit batch size for both train and validation loaders.
        val_dataset: Optional separate validation dataset. When provided, the
            full `dataset` is used for training.

    Returns:
        Tuple `(model, train_loss, val_loss, swa_triggered)`, where train and
        validation losses are lists of per-submodel epoch-loss histories.
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

    if val_dataset is not None:

        train_size = len(dataset)
        val_size = len(val_dataset)

        train_dataset = dataset
        val_dataset = val_dataset
    else:
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


def train_error_ensemble(model, dataset, n_batches= 3, lr = 0.1, patience = 10, n_epochs = 100,
                         batchsize = None, val_dataset = None):

    """
    Train an ensemble of error-prediction models.

    Args:
        model: Ensemble model with a `models` attribute containing submodels.
        dataset: Dataset yielding `(image, error_vector)` pairs. If
            `val_dataset` is None, this is split 80/20 into train/validation
            subsets with `random_split`.
        n_batches: Number of batches used to derive train and validation batch
            sizes. Ignored when `batchsize` is provided.
        lr: Learning rate used for every submodel optimizer.
        patience: Number of non-improving validation epochs before a submodel
            is skipped by early stopping.
        n_epochs: Maximum number of epochs to train.
        batchsize: Explicit batch size for both train and validation loaders.
        val_dataset: Optional separate validation dataset. When provided, the
            full `dataset` is used for training.

    Returns:
        Tuple `(model, train_loss, val_loss)`, where train and validation
        losses are lists of per-submodel epoch-loss histories.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()

    optimizers = [torch.optim.Adam(submodel.parameters(), lr=lr, weight_decay = 1e-6) for submodel in model.models]

    train_loss = []
    val_loss = []

    model.train()

    if val_dataset is not None:

        train_size = len(dataset)
        val_size = len(val_dataset)

        train_dataset = dataset
        val_dataset = val_dataset

    else:

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



# def train_model(model, dataset, n_batches= 3, lr = 0.1, patience = 10, n_epochs = 100, partial_train = True,
#                 batchsize = None, val_dataset =  None):

#     """
#     Train a model with optional encoder freezing, early stopping, and a
#     live tqdm progress bar showing per-epoch train and validation loss.

#     Args:
#         model: nn.Module to train. If `partial_train=True`, the model must
#             implement a `train_only_decoder()` method.
#         dataset: Training dataset. If `val_dataset` is None, this is split
#             80/20 into train/val internally.
#         n_batches: Number of batches per epoch. Used to derive batch size
#             as `len(train_dataset) // n_batches` (and likewise for val).
#             Ignored if `batchsize` is provided. Default 3.
#         lr: Learning rate for the Adam optimizer. Default 0.1.
#         patience: Number of epochs without val-loss improvement to wait
#             before early stopping. Default 10.
#         n_epochs: Maximum number of training epochs. Default 100.
#         partial_train: If True, only the decoder is trained (encoder
#             frozen) via `model.train_only_decoder()`. If False, the full
#             model is trained via `model.train()`. Default True.
#         batchsize: Explicit batch size. If provided, `n_batches` is
#             ignored and this value is used for both train and val
#             DataLoaders. Default None.
#         val_dataset: Optional separate validation dataset. If None, the
#             training `dataset` is split 80/20 into train/val. Default None.

#     Returns:
#         Tuple of (model, train_loss, val_loss) where:
#             model: The trained model, set to eval mode.
#             train_loss: List of per-epoch average training losses.
#             val_loss: List of per-epoch average validation losses.

#     Raises:
#         AttributeError: If `partial_train=True` and the model does not
#             implement a `train_only_decoder` method.

#     Notes:
#         - Uses MSELoss and the Adam optimizer.
#         - Validation uses `model.predict(...)` rather than `model(...)`.
#         - Early stopping starts after `skip_epochs=100` epochs.
#     """


#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     criterion = nn.MSELoss()

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     train_loss = []
#     val_loss = []



#     if val_dataset is not None:

#         train_size = len(dataset)
#         val_size = len(val_dataset)

#         train_dataset = dataset
#         val_dataset = val_dataset

#     else:
#         train_size = int(0.8*len(dataset))
#         val_size = len(dataset) - train_size

#         train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#     #Keep batchsize atleast 1
#     train_batch_size = max(1, len(train_dataset)//n_batches)
#     val_batch_size = max(1, len(val_dataset)//n_batches)

#     tr_dataloader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True)
#     val_dataloader = DataLoader(val_dataset, batch_size = val_batch_size, shuffle = True)

#     #if batchsize provided disregard n_batches
#     if batchsize is not None:
#         tr_dataloader = DataLoader(train_dataset, batch_size = batchsize, shuffle = True)
#         val_dataloader = DataLoader(val_dataset, batch_size = batchsize, shuffle = True)


#     earlystopping = EarlyStopping(skip_epochs = 100, patience = patience, min_delta = 0)


#     pbar = tqdm(range(n_epochs))

#     for epoch in pbar:



#         tr_epoch_loss = 0
#         val_epoch_loss = 0


#         # Training
#         if partial_train:
#             if not hasattr(model, "train_only_decoder"):
#                 raise AttributeError("Model does not have 'train_only_decoder' method. Set partial_train=False")

#             model.train_only_decoder()
#         else:
#             model.train()

#         for train_images, train_label in tr_dataloader:

#             train_images, train_label = train_images.to(device), train_label.to(device)

#             optimizer.zero_grad()

#             output = model(train_images)

#             loss = criterion(output, train_label)
#             tr_epoch_loss += loss.item()

#             loss.backward()
#             optimizer.step()

#         tr_epoch_loss /= len(tr_dataloader)

#         train_loss.append(tr_epoch_loss)

#         # Validation

#         model.eval()

#         for val_images, val_label in val_dataloader:

#             val_images, val_label = val_images.to(device), val_label.to(device)


#             output = model.predict(val_images)

#             loss = criterion(output, val_label)



#             val_epoch_loss += loss.item()

#         val_epoch_loss /= len(val_dataloader)


#         val_loss.append(val_epoch_loss)

#         pbar.set_postfix(train_loss=f"{tr_epoch_loss:.4f}", val_loss=f"{val_epoch_loss:.4f}")

#         if earlystopping(val_epoch_loss, epoch):
#             break




#     model.eval()

#     return model, train_loss, val_loss



def acquisition_fn(error_mean, error_std, beta = 1, index_exclude = [], optimize = "minimize", sample_next_points = 1):
    """
    Select candidate indices from an uncertainty-weighted acquisition value.

    Args:
        error_mean: Mean error estimate for each candidate.
        error_std: Error uncertainty estimate for each candidate.
        beta: Weight applied to `error_std`.
        index_exclude: Candidate indices to suppress before ranking.
        optimize: `"minimize"` selects the smallest acquisition values;
            `"maximize"` selects the largest.
        sample_next_points: Number of candidate indices to return.

    Returns:
        Tuple `(aq_ind, aq_fn)` containing selected indices and all acquisition
        values after exclusions are applied.
    """

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
    """
    Append one acquired image/spectrum pair to the current training set.

    Args:
        images: Full image-patch array, shape `(N, H, W)`.
        spectra: Full spectra array, shape `(N, S)`.
        next_index: Index to append from `images` and `spectra`.
        imgs_train: Current training image array.
        spectra_train: Current training spectra array.
        indices_train: Current array of selected training indices.

    Returns:
        Tuple `(imgs_train, spectra_train, indices_train)` with the selected
        sample appended.
    """

    imgs_train = np.append(imgs_train, images[next_index].reshape(1, images.shape[1], images.shape[2]), axis = 0)

    spectra_train = np.append(spectra_train, spectra[next_index].reshape(1, spectra.shape[1]), axis = 0)

    indices_train = np.append(indices_train, next_index)

    return imgs_train, spectra_train, indices_train


def predict_spectra(model, images, ensemble = True):
    """
    Predict spectra for one model or every submodel in an ensemble.

    Args:
        model: Model with `predict(...)`, or ensemble with a `models` list.
        images: Image-patch array converted to a float tensor internally.
        ensemble: If True, call each submodel and return a list of predictions.
            If False, call `model.predict(...)` once.

    Returns:
        NumPy prediction array for a single model, or a list of arrays for an
        ensemble.
    """

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
    """
    Compute encoder embeddings for image patches.

    Args:
        model: Model exposing an `encoder` module.
        images: Image-patch array without an explicit channel dimension.

    Returns:
        NumPy array of encoder outputs.
    """

    images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) # add the channel_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = images.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model.encoder(images)

    pred_spectra = outputs.cpu().detach().squeeze().numpy()

    return pred_spectra

def predict_vae_embedding(model, images):
    """
    Compute VAE latent embeddings for image patches.

    Args:
        model: Model exposing an `embedding(...)` method.
        images: Image-patch array.

    Returns:
        NumPy array of latent embeddings.
    """

    images = torch.tensor(images, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = images.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model.embedding(images)

    pred_spectra = outputs.cpu().detach().squeeze().numpy()

    return pred_spectra


def predict_posterior(model, images, output_type = "prediction"):
    """
    Predict spectra or latent representations for posterior analysis.

    Args:
        model: Trained model.
        images: Image-patch array.
        output_type: One of `"prediction"`, `"latent"`, or `"vae_latent"`.

    Returns:
        NumPy array returned by the selected prediction helper.

    Raises:
        ValueError: If `output_type` is not recognized.
    """

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
    """
    Compute distance from each spectrum to the mean training spectrum.

    Args:
        spectra: Candidate spectra array.
        spectra_train: Training spectra used to compute the reference mean.
        distance_type: Distance metric passed to `calc_distance`.

    Returns:
        One distance value per input spectrum.
    """

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
    """
    Compute a distance between two spectra.

    Args:
        X: Spectrum or vector.
        Y: Reference spectrum or vector.
        distance_type: One of `"L1"`, `"L2"`, or `"cos"`.

    Returns:
        Scalar distance value.

    Raises:
        ValueError: If `distance_type` is not recognized.
    """

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
    """
    Build a distance-based acquisition function and rank candidate indices.

    Args:
        distances: Distance or error values for all candidates.
        beta: Exploration/exploitation control used by `"custom_fn"`.
        lambda_: Smoothness parameter used by `"custom_fn"`.
        optimize: One of `"custom_fn"`, `"minimize"`, or `"maximize"`.
        sample_next_points: Number of candidate indices to return.
        exclude_indices: Candidate indices to suppress before ranking.

    Returns:
        Tuple `(aq_ind, acq_vals)` containing selected indices and normalized
        acquisition values.

    Raises:
        ValueError: If `optimize` is not recognized.
    """

    distances = np.ravel(np.asarray(distances))
    acq_vals = norm_0to1(distances)



    if optimize == "minimize":

        acq_vals[exclude_indices] = 2
        aq_ind = np.argsort(acq_vals)


    elif optimize == "maximize":

        acq_vals[exclude_indices] = -1
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




def err_estimation(model, images, spectra,
                   error_type = "L1"):
    """
    Estimate prediction error between model spectra and target spectra.

    Args:
        model: Model exposing `predict(...)`.
        images: Image-patch array.
        spectra: Target spectra array.
        error_type: One of `"L1"`, `"L2"`, or `"cos"`.

    Returns:
        Tuple `(error_mean, error_std, error_vector)` as NumPy arrays.

    Raises:
        ValueError: If `error_type` is not recognized.
    """

    images = torch.tensor(images, dtype=torch.float32)
    spectra = torch.tensor(spectra, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, spectra = images.to(device), spectra.to(device)

    model.eval()

    outputs = model.predict(images)
    #print(outputs.shape)
    if error_type == "L1":
        error_vector = torch.abs(outputs.squeeze(1) - spectra)
    elif error_type == "L2":
        error_vector = (outputs.squeeze(1) - spectra) ** 2
    elif error_type == "cos":
        cosine_similarity = F.cosine_similarity(outputs.squeeze(1), spectra, dim=-1)
        error_vector = (1 - cosine_similarity).unsqueeze(-1) #.expand_as(spectra)
    else:
        raise ValueError('Invalid error_type. Valid: L1, L2, cos')
    #print(error_vector.shape)

    # copy to host cpu before detaching
    error_vector = error_vector.cpu()
    error_vector = error_vector.detach().numpy()

    error_mean = np.mean(error_vector, axis = -1)
    error_std = np.std(error_vector, axis = -1)


    return error_mean, error_std, error_vector

def error_dataset(model, images, spectra, norm = True):
    """
    Build an error-vector target for training an error ensemble.

    Args:
        model: Ensemble model with a `models` list.
        images: Image-patch array.
        spectra: Target spectra array.
        norm: If True, normalize each submodel's mean error to [0, 1].

    Returns:
        Array of shape `(N, n_models)` containing per-submodel error targets.
    """

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
    """
    Predict error values with an error ensemble.

    Args:
        error_ensemble_model: Ensemble model with a `predict(...)` method.
        images: Image-patch array.

    Returns:
        Tuple `(error_mean, error_std, errors)` from the ensemble predictions.
    """


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
    """
    Rank ensemble members by their average loss near the end of training.

    Args:
        training_loss: List of per-model loss histories.
        last_epochs: Number of trailing epochs to average, excluding the final
            element according to the current slice behavior.

    Returns:
        Model indices sorted by increasing average trailing loss.
    """

    loss_all  = []

    for i in range(len(training_loss)):

        avg_ending_loss = np.asarray(training_loss[i][-last_epochs:-1]).mean()

        loss_all.append(avg_ending_loss)


    loss_all =  np.asarray(loss_all)
    model_indices = np.argsort(loss_all)

    return model_indices
