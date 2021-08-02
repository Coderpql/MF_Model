import numpy as np
import plotRes

L_loss = []
name = 'res/loss/loss.png'
for i in range(1, 7):
    loss = np.load('res/loss/loss_' + str(i) + '.npy')
    L_loss.append(loss)

plotRes.CreatePic(L_loss, name)
