import sys
from sacred import Ingredient
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandparentdir = os.path.dirname(os.path.dirname(currentdir))

src_path = os.path.join(grandparentdir, 'src')
if sys.path[0] != src_path:
    sys.path.insert(0, src_path)

from model.init import init_rnn

model = Ingredient('model')
init_model = model.capture(init_rnn)

