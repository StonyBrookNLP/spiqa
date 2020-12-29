"""
squad related task quantization tool
"""

from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering
from transformers.data.metrics.squad_metrics import *
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import transformer_visualization as tv

import argparse as ag
import os
import sys
import random
import functools
import operator
from subprocess import call
from math import isnan, fsum, log
from textwrap import wrap
import urllib.request
import json
from itertools import compress, product

RES_FIG_PATH = "./res_fig/"
PARAM_PATH = "./params/"
DATA_PATH = "./data/"
FILT_PARAM_PATH = "./filtered_params/"
MAX_SEQ_LEN = 320
ATT_SIZE = [12, 12, MAX_SEQ_LEN, MAX_SEQ_LEN]
HS_SIZE = [ATT_SIZE[0]+1, 1, MAX_SEQ_LEN, 64*ATT_SIZE[1]]

