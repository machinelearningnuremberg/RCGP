
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:03:42 2021

@author: hsjomaa
"""
## Original packages
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import copy
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import json
import time
## Our packages
import gpytorch
import logging
from DeepKernelGPHelpers import totorch,prepare_data,Metric, EI
from torch.nn.functional import adaptive_max_pool1d
from scipy.optimize import curve_fit

# np.random.seed(1203)
RandomQueryGenerator= np.random.RandomState(413)
RandomSupportGenerator= np.random.RandomState(413)
RandomTaskGenerator = np.random.RandomState(413)

def sigmoid_fit(x, a, k, b, nu, q):
    y = a + k/np.power((1+q * np.exp(-10*x)), 1/max(0.0001, nu))
    return y

def sigmoid_fit_torch(x, z):
    y = z[:,0]/1 + z[:,1]/torch.pow((1+(z[:,4]/1)* torch.exp(-z[:,2]* x[:,0])), 1/(0.0001 + torch.nn.functional.relu(z[:,3])))
    return torch.clamp(y, min=-10000, max=10000)

class DeepKernelGP(nn.Module):

    def __init__(self,X,Y,Z,kernel,backbone_fn, config, support,log_dir,seed):
        super(DeepKernelGP, self).__init__()
        torch.manual_seed(seed)
        ## GP parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X,self.Y,self.Z = X,Y,Z
        self.feature_extractor = backbone_fn().to(self.device)
        self.config=config
        self.get_model_likelihood_mll(len(support),kernel,backbone_fn)

        logging.basicConfig(filename=log_dir, level=logging.DEBUG)

    def get_model_likelihood_mll(self, train_size,kernel,backbone_fn):

        train_x=torch.ones(train_size, self.feature_extractor.out_features).to(self.device)
        train_y=torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, config=self.config,
                             dims=self.feature_extractor.out_features)
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

    def train(self, support, load_model,optimizer, checkpoint=None,epochs=1000, verbose = False):

        if load_model:
            assert(checkpoint is not None)
            print("KEYS MATCHED")
            self.load_checkpoint(os.path.join(checkpoint,"weights"))

        inputs,labels = prepare_data(support,support,self.X,self.Y,self.Z)
        inputs,labels = totorch(inputs,device=self.device), totorch(labels.reshape(-1,),device=self.device)
        losses = [np.inf]
        best_loss = np.inf
        starttime = time.time()
        initial_weights = copy.deepcopy(self.state_dict())
        patience=0
        max_patience = self.config["patience"]
        for _ in range(epochs):
            optimizer.zero_grad()
            z = self.feature_extractor(inputs)
            self.model.set_train_data(inputs=z, targets=labels)
            predictions = self.model(z)
            try:
                loss = -self.mll(predictions, self.model.train_targets)
                loss.backward()
                optimizer.step()
            except Exception as ada:
                logging.info(f"Exception {ada}")
                break

            if verbose:
                print("Iter {iter}/{epochs} - Loss: {loss:.5f}   noise: {noise:.5f}".format(
                    iter=_+1,epochs=epochs,loss=loss.item(),noise=self.likelihood.noise.item()))
            losses.append(loss.detach().to("cpu").item())
            if best_loss>losses[-1]:
                best_loss = losses[-1]
                weights = copy.deepcopy(self.state_dict())
            if np.allclose(losses[-1],losses[-2],atol=self.config["loss_tol"]):
                patience+=1
            else:
                patience=0
            if patience>max_patience:
                break
        self.load_state_dict(weights)
        logging.info(f"Current Iteration: {len(support)} | Incumbent {max(self.Y[support])} | Duration {np.round(time.time()-starttime)} | Epochs {_} | Noise {self.likelihood.noise.item()}")
        return losses,weights,initial_weights

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint,map_location=torch.device(self.device))
        self.model.load_state_dict(ckpt['gp'],strict=False)
        self.likelihood.load_state_dict(ckpt['likelihood'],strict=False)
        self.feature_extractor.load_state_dict(ckpt['net'],strict=False)


    def predict(self,support, query_range=None, noise_fn=None):

        card = len(self.Y)
        if noise_fn:
            self.Y = noise_fn(self.Y)
        x_support,y_support = prepare_data(support,support,
                                           self.X,self.Y,self.Z)
        if query_range is None:
            x_query,_ = prepare_data(np.arange(card),support,
                                           self.X,self.Y,self.Z)
        else:
            x_query,_ = prepare_data(query_range,support,
                                           self.X,self.Y,self.Z)
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        z_support = self.feature_extractor(totorch(x_support,self.device)).detach()
        self.model.set_train_data(inputs=z_support, targets=totorch(y_support.reshape(-1,),self.device), strict=False)

        with torch.no_grad():
            z_query = self.feature_extractor(totorch(x_query,self.device)).detach()
            pred    = self.likelihood(self.model(z_query))


        mu    = pred.mean.detach().to("cpu").numpy().reshape(-1,)
        stddev = pred.stddev.detach().to("cpu").numpy().reshape(-1,)

        return mu,stddev

class DKT(nn.Module):
    def __init__(self, train_data,valid_data, kernel,backbone_fn, config):
        super(DKT, self).__init__()
        ## GP parameters
        self.train_data = train_data
        self.valid_data = valid_data
        self.fixed_context_size = config["fixed_context_size"]
        self.minibatch_size = config["minibatch_size"]
        self.n_inner_steps = config["n_inner_steps"]
        self.checkpoint_path = config["checkpoint_path"]
        os.makedirs(self.checkpoint_path,exist_ok=False)
        json.dump(config, open(os.path.join(self.checkpoint_path,"configuration.json"),"w"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.basicConfig(filename=os.path.join(self.checkpoint_path,"log.txt"), level=logging.DEBUG)
        self.feature_extractor = backbone_fn().to(self.device)
        self.config=config
        self.get_model_likelihood_mll(self.fixed_context_size,kernel,backbone_fn)
        self.mse = nn.MSELoss()
        self.curr_valid_loss = np.inf
        self.get_tasks()
        self.setup_writers()

        self.train_metrics = Metric()
        self.valid_metrics = Metric(prefix="valid: ")
        print(self)


    def setup_writers(self,):
        train_log_dir = os.path.join(self.checkpoint_path,"train")
        os.makedirs(train_log_dir,exist_ok=True)
        self.train_summary_writer = SummaryWriter(train_log_dir)

        valid_log_dir = os.path.join(self.checkpoint_path,"valid")
        os.makedirs(valid_log_dir,exist_ok=True)
        self.valid_summary_writer = SummaryWriter(valid_log_dir)

    def get_tasks(self,):
        pairs = []
        for space in self.train_data.keys():
            for task in self.train_data[space].keys():
                pairs.append([space,task])
        self.tasks = pairs
        ##########
        pairs = []
        for space in self.valid_data.keys():
            for task in self.valid_data[space].keys():
                pairs.append([space,task])
        self.valid_tasks = pairs


    def get_model_likelihood_mll(self, train_size,kernel,backbone_fn):

        train_x=torch.ones(train_size, self.feature_extractor.out_features).to(self.device)
        train_y=torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, config=self.config,dims = self.feature_extractor.out_features)
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

    def epoch_end(self):
        RandomTaskGenerator.shuffle(self.tasks)

    def train_loop(self, epoch, optimizer, scheduler_fn=None):
        if scheduler_fn:
            scheduler = scheduler_fn(optimizer,len(self.tasks))
        self.epoch_end()
        assert(self.training)
        for task in self.tasks:
            inputs, labels = self.get_batch(task)
            for _ in range(self.n_inner_steps):
                optimizer.zero_grad()
                z = self.feature_extractor(inputs)
                self.model.set_train_data(inputs=z, targets=labels, strict=False)
                predictions = self.model(z)
                loss = -self.mll(predictions, self.model.train_targets)
                loss.backward()
                optimizer.step()
                mse = self.mse(predictions.mean, labels)
                self.train_metrics.update(loss,self.model.likelihood.noise,mse)
            if scheduler_fn:
                scheduler.step()

        training_results = self.train_metrics.get()
        for k,v in training_results.items():
            self.train_summary_writer.add_scalar(k, v, epoch)
        for task in self.valid_tasks:
            mse,loss = self.test_loop(task,train=False)
            self.valid_metrics.update(loss,np.array(0),mse,)

        logging.info(self.train_metrics.report() + " " + self.valid_metrics.report())
        validation_results = self.valid_metrics.get()
        for k,v in validation_results.items():
            self.valid_summary_writer.add_scalar(k, v, epoch)
        self.feature_extractor.train()
        self.likelihood.train()
        self.model.train()

        if validation_results["loss"] < self.curr_valid_loss:
            self.save_checkpoint(os.path.join(self.checkpoint_path,"weights"))
            self.curr_valid_loss = validation_results["loss"]
        self.valid_metrics.reset()
        self.train_metrics.reset()

    def predict(self, x_support, y_support, x_query, y_query, GP_only=False):
        if not GP_only:
            z_support = self.feature_extractor(x_support).detach()
        else:
            z_support = x_support
        self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            if not GP_only:
                z_query = self.feature_extractor(x_query).detach()
            else:
                z_query = x_query
            pred = self.likelihood(self.model(z_query))
        mu = pred.mean.detach().to("cpu").numpy().reshape(-1, )
        stddev = pred.stddev.detach().to("cpu").numpy().reshape(-1, )
        return mu, stddev


    def test_loop(self, task, train, optimizer=None): # no optimizer needed for GP
        (x_support, y_support),(x_query,y_query) = self.get_support_and_queries(task,train)
        z_support = self.feature_extractor(x_support).detach()
        self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_query).detach()
            pred    = self.likelihood(self.model(z_query))
            loss = -self.mll(pred, y_query)
            lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_query)

        return mse,loss

    def get_batch(self,task):
        # we want to fit the gp given context info to new observations
        # task is an algorithm/dataset pair
        space,task = task
        x,response = np.array(self.train_data[space][task]["X"]),\
                          MinMaxScaler().fit_transform(np.array(self.train_data[space][task]["y"])).reshape(-1,)

        card, dim = x.shape

        support = RandomSupportGenerator.choice(np.arange(card),
                                              replace=False,size=self.fixed_context_size)
        remaining = np.setdiff1d(np.arange(card),support)
        indexes = RandomQueryGenerator.choice(
            remaining,replace=False,size=self.minibatch_size if len(remaining)>self.minibatch_size else len(remaining))

        inputs = x[indexes, :]
        labels = response[indexes]
        inputs,labels = totorch(inputs,device=self.device), totorch(labels.reshape(-1,),device=self.device)
        return inputs, labels

    def get_support_and_queries(self,task, train=False):

        # task is an algorithm/dataset pair
        space,task = task

        hpo_data = self.valid_data if not train else self.train_data
        x,response = np.array(hpo_data[space][task]["X"]), \
                          MinMaxScaler().fit_transform(np.array(hpo_data[space][task]["y"])).reshape(-1,)
        card, dim = x.shape

        support = RandomSupportGenerator.choice(np.arange(card),
                                              replace=False,size=self.fixed_context_size)
        indexes = RandomQueryGenerator.choice(
            np.setdiff1d(np.arange(card),support),replace=False,size=self.minibatch_size)

        support_x,support_y = x[support], response[support]
        query_x,query_y = x[indexes], response[indexes]

        return (totorch(support_x,self.device),totorch(support_y.reshape(-1,),self.device)),\
    (totorch(query_x,self.device),totorch(query_y.reshape(-1,),self.device))

    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict         = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict         = self.feature_extractor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net':nn_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])

class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,config,dims ):
        super(ExactGPLayer, self).__init__(train_inputs=train_x, train_targets=train_y, likelihood=likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()

        self.normalizer = torch.nn.BatchNorm1d(num_features=dims, affine=False)
        ## RBF kernel
        if(config["kernel"]=='rbf' or config["kernel"]=='RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dims if config["ard"] else None))
        elif(config["kernel"]=='52'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=config["nu"],ard_num_dims=dims if config["ard"] else None))
        ## Spectral kernel
        else:
            raise ValueError("[ERROR] the kernel '" + str(config["kernel"]) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        x = self.normalizer(x)
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class batch_mlp(nn.Module):
    def __init__(self, d_in, output_sizes, nonlinearity="relu",dropout=0.0):

        super(batch_mlp, self).__init__()
        assert(nonlinearity=="relu")
        self.nonlinearity = nn.Tanh()
        self.fc = nn.ModuleList([nn.Linear(in_features=d_in, out_features=output_sizes[0])])
        for d_out in output_sizes[1:]:
            self.fc.append(nn.Linear(in_features=self.fc[-1].out_features, out_features=d_out))
        self.out_features = output_sizes[-1]
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):

        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.dropout(x)
            x = self.nonlinearity(x)
        x = self.fc[-1](x)
        x = self.dropout(x)
        return x

class StandardDeepGP(nn.Module):
    def __init__(self, configuration):

        super(StandardDeepGP, self).__init__()
        self.A = batch_mlp(configuration["dim"], configuration["output_size_A"],dropout=configuration["dropout"])
        self.out_features = configuration["output_size_A"][-1]

    def forward(self, x):
        # e,r,x,z = x
        hidden = self.A(x) ### NxA
        return hidden


class dyhpo_mlp(nn.Module):
    def __init__(self, d_in, output_sizes, nonlinearity="relu", lc_features=3, budget_features=1):

        super(dyhpo_mlp, self).__init__()
        assert (nonlinearity == "relu")
        self.nonlinearity = nn.ReLU()
        self.lc_features = lc_features
        self.budget_features = budget_features
        self.fc1 = nn.Linear(in_features=d_in-1, out_features=output_sizes[0])
        self.fc2 = nn.Linear(in_features=self.fc1.out_features + self.lc_features + self.budget_features,
                             out_features=output_sizes[-1])
        self.cnn = nn.Sequential(nn.Conv1d(in_channels=1, kernel_size=(3,), out_channels=4), nn.AdaptiveMaxPool1d(1))
        self.out_features = output_sizes[-1]

    def forward(self, x, l_curve):

        x = self.fc1(x)
        x = self.nonlinearity(x)
        y = self.cnn(l_curve)
        x = self.fc2(torch.cat((x, torch.squeeze(y, dim=-1)), -1))
        x = self.nonlinearity(x)
        return x


class DyHPODGP(nn.Module):
    def __init__(self, configuration):
        super(DyHPODGP, self).__init__()
        self.A = dyhpo_mlp(configuration["dim"], configuration["output_size_A"])
        self.out_features = configuration["output_size_A"][-1]

    def forward(self, x, l_curve):
        hidden = self.A(x, l_curve)  ### NxA
        return hidden

class DyHPO(nn.Module):

    def __init__(self, kernel, backbone_fn, config, support, log_dir, seed):
        super(DyHPO, self).__init__()
        torch.manual_seed(seed)
        ## GP parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = backbone_fn().to(self.device)
        self.config = config
        self.get_model_likelihood_mll(len(support))
        self.X = None
        self.Y = None
        self.Z = None
        logging.basicConfig(filename=log_dir, level=logging.DEBUG)

    def get_model_likelihood_mll(self, train_size):

        train_x = torch.ones(train_size, self.feature_extractor.out_features).to(self.device)
        train_y = torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, config=self.config,
                             dims=self.feature_extractor.out_features)
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)

    def prepare_data(self, support_x, support_y):
        inputs = []
        labels = []
        l_curves = []
        length = np.amax([len(b) for b in support_y.values()])
        for config in support_y.keys():
            for idx, r in enumerate(support_y[config]):
                inputs.append(np.array(list(config) + [(idx+1)/100]))
                labels.append(r)
                l_curves.append(np.array(support_y[config][:idx+1] + [0] * (length - idx+1)))
        return np.array(inputs), np.array(labels), np.expand_dims(np.array(l_curves), axis=1)


    def train(self, support_x, support_y, optimizer, checkpoint=None, epochs=1000, verbose=False):

        inputs, labels, l_curves = self.prepare_data(support_x, support_y)
        self.X = inputs
        self.max_Y = max(np.abs(labels))
        if self.max_Y == 0: self.max_Y =1
        labels /= self.max_Y
        self.Y = labels
        self.Z = l_curves
        inputs, labels = torch.tensor(inputs, dtype=torch.float).to(self.device), torch.tensor(labels, dtype=torch.float).to(self.device)
        labels = labels/max(labels)
        losses = [np.inf]
        best_loss = np.inf
        starttime = time.time()
        initial_weights = copy.deepcopy(self.state_dict())
        for _ in range(epochs):
            optimizer.zero_grad()
            z = self.feature_extractor(inputs, torch.tensor(l_curves, dtype=torch.float).to(self.device))
            self.model.set_train_data(inputs=z, targets=labels, strict=False)
            predictions = self.model(z)
            try:
                loss = -self.mll(predictions, self.model.train_targets)
                loss.backward()
                optimizer.step()
            except Exception as ada:
                logging.info(f"Exception {ada}")
                break

            if verbose:
                print("Iter {iter}/{epochs} - Loss: {loss:.5f}   noise: {noise:.5f}".format(
                    iter=_ + 1, epochs=epochs, loss=loss.item(), noise=self.likelihood.noise.item()))
            losses.append(loss.detach().to("cpu").item())
            if best_loss > losses[-1]:
                best_loss = losses[-1]
                weights = copy.deepcopy(self.state_dict())
        self.load_state_dict(weights)
        logging.info(
            f"Current Iteration: {len(self.Y)} | Incumbent {max(self.Y)} | Duration {np.round(time.time() - starttime)} | Epochs {_} | Noise {self.likelihood.noise.item()}")
        return losses, weights, initial_weights

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint, map_location=torch.device(self.device))
        self.model.load_state_dict(ckpt['gp'], strict=False)
        self.likelihood.load_state_dict(ckpt['likelihood'], strict=False)
        self.feature_extractor.load_state_dict(ckpt['net'], strict=False)

    def predict(self, support_x, support_y, query_x, l_curves_x,  noise_fn=None):
        inputs, labels, l_curves = self.prepare_data(support_x, support_y)
        self.X = inputs
        self.Y = labels/self.max_Y
        self.Z = l_curves
        inputs, labels = torch.tensor(inputs, dtype=torch.float).to(self.device), torch.tensor(labels, dtype=torch.float).to(self.device)
        card = len(self.Y)
        if noise_fn:
            self.Y = noise_fn(self.Y)
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        z_support = self.feature_extractor(inputs, torch.tensor(l_curves, dtype=torch.float).to(self.device)).detach()
        self.model.set_train_data(inputs=z_support, targets=labels, strict=False)

        with torch.no_grad():
            z_query = self.feature_extractor(torch.tensor(query_x, dtype=torch.float).to(self.device), torch.tensor(l_curves_x, dtype=torch.float).
                                             to(self.device)).detach()
            pred = self.likelihood(self.model(z_query))

        mu = pred.mean.detach().to("cpu").numpy().reshape(-1, )
        stddev = pred.stddev.detach().to("cpu").numpy().reshape(-1, )

        return mu, stddev

class lcgp_mlp(nn.Module):
    def __init__(self, d_in, output_sizes, nonlinearity="relu"):

        super(lcgp_mlp, self).__init__()
        assert (nonlinearity == "relu")
        self.nonlinearity = nn.ReLU()
        self.fc1 = nn.Linear(in_features=d_in, out_features=output_sizes[0])
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=output_sizes[1])
        self.fc3 = nn.Linear(in_features=self.fc2.out_features, out_features=output_sizes[-1])
        self.out_features = output_sizes[-1]

    def forward(self, x):

        x = self.fc1(x)
        x = self.nonlinearity(x)
        x = self.fc2(x)
        x = self.nonlinearity(x)
        x = self.fc3(x)
        # x = self.nonlinearity(x)
        return x


class LCGPDGP(nn.Module):
    def __init__(self, configuration):
        super(LCGPDGP, self).__init__()
        self.A = lcgp_mlp(configuration["dim"], configuration["output_size_A"])
        self.out_features = configuration["output_size_A"][-1]
        self.in_features = configuration["dim"]

    def forward(self, x):
        # e,r,x,z = x
        hidden = self.A(x)  ### NxA
        return hidden

class LCGP(nn.Module):

    def __init__(self, kernel, backbone_fn, config, support, log_dir, seed):
        super(LCGP, self).__init__()
        torch.manual_seed(seed)
        ## GP parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = backbone_fn().to(self.device)
        self.config = config
        self.scaler = None
        self.get_model_likelihood_mll(len(support), kernel, backbone_fn)
        self.X = None
        self.Y = None
        self.Z = None
        logging.basicConfig(filename=log_dir, level=logging.DEBUG)

    def get_model_likelihood_mll(self, train_size, kernel, backbone_fn):

        train_x = torch.ones(train_size, self.feature_extractor.in_features).\
            to(self.device)
        train_y = torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, config=self.config,
                             dims=self.feature_extractor.in_features)
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)
        self.fe_loss = nn.MSELoss()

    def prepare_data(self, support_x, support_y, pretrain=False, budgets=None):
        inputs = []
        labels = []
        l_curves = []
        budgets_all = []
        for cfg in support_y.keys():
            lc = support_y[cfg]
            budgets_all.extend(np.divide(budgets[cfg], np.ones(len(budgets[cfg]))*100).tolist())
            # labels.append(lc[-1])

            # if pretrain:
            #     labels.extend([y_pred[b-1] for b in budgets[cfg]])
            # else:
            labels.extend([lc[b-1] for b in budgets[cfg]])
            for budget in budgets[cfg]:
                parameters, _ = curve_fit(sigmoid_fit, np.arange(len(lc)), lc, maxfev=10000000)
                y_pred = sigmoid_fit(np.arange(100), *parameters)
                inputs.append(np.array(list(cfg)))
                l_curves.append(np.array(parameters))
        return np.array(inputs), np.array(labels), np.array(l_curves), np.array(budgets_all)


    def train(self, support_x, support_y, optimizer, optimizer_nn, checkpoint=None, epochs=1000, verbose=False, pretrain=False,
              budgets=None):

        inputs, labels, l_curves, budgets_list = self.prepare_data(support_x, support_y, pretrain, budgets=budgets)
        budgets_list = torch.reshape(torch.tensor(budgets_list, dtype=torch.float).to(self.device),
                                     shape=[len(labels), 1])
        budget = torch.ones(len(labels), 1).to(self.device)*100
        self.X = inputs
        # self.max_Y = max(labels)
        # if self.max_Y == 0:
        #     self.max_Y = 1
        # labels /= self.max_Y
        self.scaler = MinMaxScaler()
        self.scaler.fit(X=labels.reshape(-1, 1))
        self.Y = self.scaler.transform(X=labels.reshape(-1, 1)).reshape(-1)
        self.Z = l_curves
        inputs, labels = torch.tensor(inputs, dtype=torch.float).to(self.device), torch.tensor(self.Y, dtype=torch.float).to(self.device)
        l_curves = torch.tensor(l_curves, dtype=torch.float).to(self.device)
        # labels = labels/max(labels)
        losses = [np.inf]
        best_loss = np.inf
        starttime = time.time()
        initial_weights = copy.deepcopy(self.state_dict())
        patience = 0
        max_patience = self.config["patience"]
        self.feature_extractor.train()
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            # for k in range(25):
            # z = self.feature_extractor(torch.hstack((inputs, budgets_list)))
            #     nn_loss = self.fe_loss(z, l_curves)
            #     nn_loss.backward()
            #     optimizer_nn.step()

            # z = torch.reshape(z.detach(), shape=[len(labels), 5]).to(self.device)
            # print(z)
            # predicted_val = torch.reshape(sigmoid_fit_torch(budget, z), shape=[len(labels), 1]).to(self.device)
            # print(predicted_val)
            # train_inputs = torch.hstack((predicted_val, budgets_list, inputs))
            train_inputs = torch.hstack((budgets_list, inputs))
            # print(train_inputs)
            self.model.set_train_data(inputs=train_inputs, targets=labels, strict=False)
            predictions = self.model(train_inputs)
            # print(predictions)
            # try:
            loss = -self.mll(predictions, self.model.train_targets)
            loss.backward()
            optimizer.step()
            # except Exception as ada:
            #     print("Exception %s" % ada)
            #     break

            if verbose:
                print("Iter {iter}/{epochs} - Loss: {loss:.5f}   noise: {noise:.5f}".format(
                    iter=_ + 1, epochs=epochs, loss=loss.item(),
                    noise=self.likelihood.noise.item()))
            losses.append(loss.detach().to("cpu").item())
            if best_loss > losses[-1]:
                best_loss = losses[-1]
                weights = copy.deepcopy(self.state_dict())

            # print("Weights: %s" % self.state_dict())
            # if np.allclose(losses[-1], losses[-2], atol=self.config["loss_tol"]):
            #     patience += 1
            # else:
            #     patience = 0
            # if patience > max_patience:
            #     break
        self.load_state_dict(weights)
        logging.info(
            f"Current Iteration: {len(self.Y)} | Incumbent {max(self.Y)} | Duration {np.round(time.time() - starttime)} | Epochs {_} | Noise {self.likelihood.noise.item()}")
        return losses, weights, initial_weights

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint, map_location=torch.device(self.device))
        self.model.load_state_dict(ckpt['gp'], strict=False)
        self.likelihood.load_state_dict(ckpt['likelihood'], strict=False)
        self.feature_extractor.load_state_dict(ckpt['net'], strict=False)

    def predict(self, support_x, support_y, query_x,  noise_fn=None, budgets=None, budget=None):
        inputs, labels, l_curves, budgets_list = self.prepare_data(support_x, support_y, budgets=budgets)
        # budgets_list = [[b] for b in budgets.values()]
        self.X = inputs

        self.max_Y = max(labels)
        if self.max_Y == 0: self.max_Y = 1
        # labels /= max(labels)
        self.Y = self.scaler.transform(labels.reshape(-1,1)).reshape(-1)
        self.Z = l_curves
        inputs, labels = torch.tensor(inputs, dtype=torch.float).to(self.device), torch.tensor(self.Y, dtype=torch.float).to(self.device)
        budget_tensor = torch.ones(len(labels), 1).to(self.device) * budget
        card = len(self.Y)
        if noise_fn:
            self.Y = noise_fn(self.Y)
        self.model.eval()
        # self.feature_extractor.eval()
        self.likelihood.eval()
        budgets_list = torch.reshape(torch.tensor(budgets_list, dtype=torch.float).to(self.device), shape=[len(labels), 1])
        # z_support = self.feature_extractor(torch.hstack((inputs, budgets_list))).detach()

        # predicted_val = torch.reshape(sigmoid_fit_torch(budget_tensor, z_support), shape=[len(labels), 1])
        # train_inputs = torch.hstack((predicted_val, budgets_list, inputs))
        train_inputs = torch.hstack((budgets_list, inputs))
        self.model.set_train_data(inputs=train_inputs, targets=labels, strict=False)

        with torch.no_grad():
            query_x = torch.tensor(query_x, dtype=torch.float).to(self.device)
            query_x = query_x[None, :]
            # z_query = self.feature_extractor(torch.hstack((query_x, torch.tensor([[budget]]).to(self.device)))).detach()
            # predicted_val = sigmoid_fit_torch(torch.tensor([[budget]]).to(self.device), z_query)
            # pred_q = torch.hstack((torch.tensor([[predicted_val]], dtype=torch.float).
            #               to(self.device), torch.tensor([[budget/100]],
            #                 dtype=torch.float).to(self.device), query_x))
            pred_q = torch.hstack((torch.tensor([[budget / 100]], dtype=torch.float).to(self.device), query_x))
            pred = self.likelihood(self.model(pred_q))

        mu = pred.mean.detach().to("cpu").numpy().reshape(-1, )
        stddev = pred.stddev.detach().to("cpu").numpy().reshape(-1, )

        return mu, stddev
