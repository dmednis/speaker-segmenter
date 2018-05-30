import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

loss = []
val_loss = []

acc = []
val_acc = []

for e in tf.train.summary_iterator("./logs/seg.logs"):
    for v in e.summary.value:
        if v.tag == 'loss':
            loss.append(v.simple_value)
        if v.tag == 'val_loss':
            val_loss.append(v.simple_value)
        if v.tag == 'acc':
            acc.append(v.simple_value)
        if v.tag == 'val_acc':
            val_acc.append(v.simple_value)


def smooth(scalars, weight=0.6):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(loss)
# ax1.plot(val_loss)
ax1.plot(smooth(val_loss))
ax1.set_title('apmācības/validācijas kļūda', fontsize='large')
ax1.set_ylabel('kļūda', fontsize='large')
ax1.set_xlabel('iterācija', fontsize='large')
ax1.legend(['apmācība', 'validācija'], loc='center right', fontsize='large')

ax2.plot(acc)
# ax2.plot(val_acc)
ax2.plot(smooth(val_acc))
ax2.set_title('apmācības/validācijas precizitāte', fontsize='large')
ax2.set_ylabel('precizitāte', fontsize='large')
ax2.set_xlabel('iterācija', fontsize='large')
ax2.legend(['apmācība', 'validācija'], loc='lower right', fontsize='large')

plt.show()