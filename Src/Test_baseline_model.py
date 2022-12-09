
import numpy as np
import pandas as pd
import tensorflow as tf


new_model = tf.keras.models.load_model('Models/Finished_trained_models/BaselineModelIteration2')

new_model.summary()
