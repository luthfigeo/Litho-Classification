import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from sklearn.preprocessing import label_binarize
from mpl_toolkits.basemap import Basemap
from pyproj import Proj
import seaborn as sns

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

dset = pd.DataFrame()
Facies = pd.DataFrame()
dataset =[]


df = pd.read_csv("Result_3class_311_100_.csv")
well = df['WELL'].unique()

for i in range(len(well)):
    dset=df.where(df['WELL']==well[i]).dropna()
    F = dset['LITHOLOGY2'].value_counts(normalize=True) * 100
    F = F.sort_index()
    F['WELL'] = well[i]
    F= pd.DataFrame(F).transpose().set_index('WELL')
    Facies = Facies.append(F)
    
Facies = Facies.fillna(0)
Facies['Lava'] = Facies[0]
Facies['Pflow'] = Facies[1]
Facies['Pfall'] = Facies[2]
# =============================================================================
# Facies['Lava'] = Facies[0]+Facies[1]
# Facies['Pflow'] = Facies[3]+Facies[4]+Facies[9]
# Facies['Pfall'] = Facies[7]+Facies[8]
# Facies['Sed'] = Facies[2]+Facies[5]+Facies[6]
# =============================================================================
#Facies.to_csv('FaciesRecap.csv', sep=',')
thick = pd.read_csv("TD.csv").set_index('WELL').sort_index()

xy = pd.read_csv("allcoordinate.csv").set_index('WELL').sort_index()
projection = Proj("+proj=utm +zone=48S, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
xy['X'], xy['Y'] = projection(xy['X'].values, xy['Y'].values, inverse=True)

prod = pd.read_csv("Production.csv").set_index('WELL').fillna(0)
dataset = Facies.iloc[:,0:3]
dataset = dataset.merge(thick, left_index=True, right_index=True, how='inner')
dataset = dataset.merge(xy, left_index=True, right_index=True, how='inner')
dataset = dataset.merge(prod, left_index=True, right_index=True, how='inner')
x = dataset.iloc[:,4]
y = dataset.iloc[:,5]
oil = dataset.iloc[:,7]
gas = dataset.iloc[:,8]
thick = dataset.iloc[:,3]
wells = dataset.index

#Lava
alphas = dataset.iloc[:,0]/dataset.iloc[:,0].max()
rgba_colors = np.zeros((len(dataset),4))
rgba_colors[:,0] = 1.0
rgba_colors[:, 3] = alphas
ftitle= "Lava Flow"

#PyroFlow
alphas = dataset.iloc[:,1]/dataset.iloc[:,1].max()
rgba_colors = np.zeros((len(dataset),4))
rgba_colors[:,0] = 1.0
rgba_colors[:,2] = 144/255
rgba_colors[:, 3] = alphas
ftitle = "Pyroclastic Flow"

#PyroFall
alphas = dataset.iloc[:,2]/dataset.iloc[:,2].max()
rgba_colors = np.zeros((len(dataset),4))
rgba_colors[:,0] = 211/255
rgba_colors[:,1] = 110/255
rgba_colors[:,2] = 112/255
rgba_colors[:, 3] = alphas
ftitle = "Pyroclastic Fall"

#Sedimen
alphas = dataset.iloc[:,3]/dataset.iloc[:,3].max()
rgba_colors = np.zeros((len(dataset),4))
rgba_colors[:,0] = 249/255
rgba_colors[:,1] = 166/255
rgba_colors[:,2] = 2/255
rgba_colors[:, 3] = alphas

xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()

fig=plt.figure()
fig.set_size_inches((20, 20))

ax1=fig.add_subplot(121)

bmap=Basemap(projection='merc',\
        llcrnrlat=-6.55, llcrnrlon=108.3,\
        urcrnrlat=-6.45, urcrnrlon=108.5,\
        #lon_0=(60+179.95)/2.,\
        ax=ax1,resolution='c')
bmap.drawcoastlines(color='black', linewidth=0.25) 
#bmap.etopo(ax = ax1)

fig=plt.figure()
fig.set_size_inches((10, 10))
for i in range (len(x)):
    x_ = x[i]
    y_ = y[i]
    oil_ = oil[i]
    gas_ = gas[i]
    thick_ = thick[i]
    rgba_colors_ = rgba_colors[i]
    plt.scatter(x_, y_, s=(gas_/thick_*40+50), color=rgba_colors_, edgecolors="grey", linewidth=1)
    plt.text(x_ , y_ , wells[i], fontsize=10)
    
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(xmin-500,xmax+1000)
plt.ylim(ymin-500,ymax+500)
plt.title(ftitle)
 
plt.show()

datacorr = pd.DataFrame()
datacorr['LAVA'] = dataset.iloc[:,0]/dataset.iloc[:,0].max()
datacorr['PYROCLASTIC FLOW'] = dataset.iloc[:,1]/dataset.iloc[:,1].max()
datacorr['PYROCLASTIC FALL'] = dataset.iloc[:,2]/dataset.iloc[:,2].max()
#datacorr['OIL'] = oil
datacorr['GAS'] = gas
datacorr = datacorr.dropna()
corr = datacorr.corr()

sns.heatmap(corr, cmap = 'YlGn', xticklabels=corr.columns,yticklabels=corr.columns, annot=True, linewidth = 5)
   