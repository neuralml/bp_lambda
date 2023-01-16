import sys
from sacred import Ingredient
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandparentdir = os.path.dirname(os.path.dirname(currentdir))

"""
src_path = os.path.join(grandparentdir, 'src')
if sys.path[0] != src_path:
    sys.path.insert(0, src_path)
"""
    
if sys.path[0] != '../src':
    sys.path.insert(0, '../src')
    
from dataset.mnist import load_mnist
from dataset.linedraw_continuous import load_linedraw_continuous
from dataset.copy_repeat import load_copy_repeat

seqmnist = Ingredient('dataset',)
load_seqmnist = seqmnist.capture(load_mnist)

linedraw = Ingredient('dataset',)
load_linedraw = linedraw.capture(load_linedraw_continuous)

copy_repeat = Ingredient('dataset',)
load_copy_repeat = copy_repeat.capture(load_copy_repeat)
