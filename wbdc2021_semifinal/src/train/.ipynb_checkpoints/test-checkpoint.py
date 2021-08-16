import numpy as np
import torch
import pandas as pd
from torch import nn
import dgl.function as fn
import torch.nn.functional as F
import gc
import random
from tqdm import tqdm
from sklearn.metrics import auc,roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.models.deepfm import FM,DNN
from deepctr_torch.layers  import CIN,InteractingLayer,CrossNet,CrossNetMix
from deepctr_torch.models.basemodel import *
from collections import defaultdict
from torch.optim import Optimizer
import torchtext
import os
import pickle
import warnings
from model.my_deep_v2 import *

print('finish')