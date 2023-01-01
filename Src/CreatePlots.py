import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


# Plots for correlation of runtime againts number embedding dimension
embeddings = [21,97,55,43,75,98,115,11,24,35,110,9,99,25,63,116,113,96,99]
runningtime = [119,359,195,165,249,350,377,88,121,165,387,85,349,122,214,378,375,478,676]

print("The correlation between embedding dimension and runningtime is: ", pearsonr(embeddings,runningtime))

embeddings_clean = np.array([21,97,55,43,75,98,115,11,24,35,110,9,99,25,63,116,113])
runningtime_clean = np.array([119,359,195,165,249,350,377,88,121,165,387,85,349,122,214,378,375])

print("The correlation between embedding dimension and runningtime without outliers is: ", pearsonr(embeddings_clean,runningtime_clean))

#calculate equation for trendline
z = np.polyfit(embeddings, runningtime, 1)
p = np.poly1d(z)

plt.scatter(embeddings,runningtime)
plt.plot(embeddings, p(embeddings))
plt.title("Relationsship between embedding dimension and runningtime")
plt.xlabel("Embedding dimension")
plt.ylabel("Runningtime")
plt.show()

############### V2

embeddings = [55,117,105,88,127,20,92,13,70,9,37,21,46,18,91,83,71,38,15,91]
runningtime = [199,384,376,277,413,115,285,95,231,87,152,115,180,113,271,259,233,152,96,281]

print("The correlation between embedding dimension and runningtime is: ", pearsonr(embeddings,runningtime))


#calculate equation for trendline
z = np.polyfit(embeddings, runningtime, 1)
p = np.poly1d(z)

plt.scatter(embeddings,runningtime)
plt.plot(embeddings, p(embeddings))
plt.title("Relationsship between embedding dimension and runningtime")
plt.xlabel("Embedding dimension")
plt.ylabel("Runningtime")
plt.show()
