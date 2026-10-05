from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


filename_emo = Path("pixel_data/260616_R4/t3/pixels_f_emo_full.npy")
filename_gcamp = Path("pixel_data/260616_R4/t3/pixels_f_gcamp_full.npy")


data_emo = np.load(filename_emo)
data_gcamp = np.load(filename_gcamp)    


# plot emo 
plt.plot(data_emo[:,0,0])
plt.title("Emo Data")
plt.savefig("p.png")
plt.close()

# plot gcamp
plt.plot(data_gcamp[:,0,0])
plt.title("GCamp Data")
plt.savefig("g.png")
plt.close()



a = 1


