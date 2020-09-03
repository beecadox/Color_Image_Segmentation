# -------------------------
# needed imports
# ---------------------------
import time
import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth, MiniBatchKMeans, AgglomerativeClustering
from skimage.color import label2rgb
from sklearn.feature_extraction import grid_to_graph
from myfunctions import show_image, resize_image, noise_on_image
from matplotlib import pyplot as plt
import skimage.metrics as sm
from openpyxl.workbook import Workbook
from openpyxl.styles import Font


# -------------------------
# custom functions
# ---------------------------
# show original image but with the masked clipped on it
def show_masked_image(img_mask, img):
    plt.subplot(1, 2, 1)
    plt.imshow(img_mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.suptitle("Mask vs Masked Image")
    plt.show()


# show masked clustered image and choosen cluster
def cluster_vs_mask_show(clustered_mask, chosen_cluster):
    plt.subplot(1, 2, 1)
    plt.imshow(clustered_mask)
    plt.subplot(1, 2, 2)
    plt.imshow(chosen_cluster, cmap="gray")
    plt.suptitle("Mask vs Cluster")
    plt.show()


# function used to compare clusters with the mask of the image
def compare_cluster_with_mask(our_mask, clustered_image):
    # get all the unique colors of the clustered image - length should be equal to the number of clusters used
    cluster_colors = np.unique(clustered_image.reshape(-1, clustered_image.shape[2]), axis=0)
    # clip the mask on the clustered image
    clustered_masked_image = (clustered_image * (our_mask / 255)).clip(0, 255).astype(np.uint8)
    max_ssim = 0
    max_ssim_cluster = 0
    # for each color in the clustered image
    for cluster in range(len(cluster_colors)):
        # find the pixels where the color is the current color
        b = np.where((clustered_masked_image[:, :, 0] == cluster_colors[cluster][0]) &
                     (clustered_masked_image[:, :, 1] == cluster_colors[cluster][1]) &
                     (clustered_masked_image[:, :, 2] == cluster_colors[cluster][2]))
        # count the number of pixels
        num_of_pixels = len(b[0])
        # in case the number of pixels that the current color appears is greater than the max number of pixels
        # we currently have change the max value and the number of the color of the cluster
        if max_ssim < num_of_pixels:
            max_ssim = num_of_pixels
            max_ssim_cluster = cluster
    # after choosing the cluster that appears the most in the masked area change the cluster to an binary image
    # with white color where the cluster is
    temp_mask = cv2.inRange(clustered_image, cluster_colors[max_ssim_cluster], cluster_colors[max_ssim_cluster])
    # calculate the similarity between the two images ( mask and cluster) - use the whole cluster for that
    # not only the masked area
    structural_score = sm.structural_similarity(cv2.cvtColor(our_mask, cv2.COLOR_BGR2GRAY), temp_mask,
                                                multichannel=False, full=False)

    cluster_vs_mask_show(clustered_masked_image, temp_mask)
    return structural_score


images = ["boat.jpg", "violet_flower.jpg", "cactus.jpg", "yellow_flower.jpg"]  # images
masks = ["boat_binary.jpg", "violet_flower_binary.jpg", "cactus_binary.jpg", "yellow_flower_binary.jpg"]  # masks
num_clusters = [4, 3, 3, 3]  # number of clusters for each image
noise_percentage = [0, 0.05, 0.1, 0.15, 0.2]

# writing the scores in an xls file
print("Creating a workbook to append an xls file")
headers = ['Image Name', 'Noise Percentage', 'Clustering Algorithm', 'Structural Similarity Score']
workbook_name = 'OutputFiles/clusteringscores.xls'
wb = Workbook()
sheet = wb.active
sheet.title = 'Clustering Scores'  # title of sheet
sheet.append(headers)  # first line of the sheet
row = sheet.row_dimensions[1]
row.font = Font(bold=True)

# for each image
for i in range(len(images)):
    image = "images/segmentation/" + images[i]
    mask = "images/segmentation/" + masks[i]
    # Loading original image in BGR
    originImg = cv2.imread(image)
    originImg = resize_image(originImg, 10)
    maskImg = cv2.imread(mask)
    maskImg = resize_image(maskImg, 10)
    masked_image = (originImg * (maskImg / 255)).clip(0, 255).astype(np.uint8)  # clip image to the mask
    show_image("Original image", originImg)
    show_masked_image(maskImg, masked_image)
    n_clusters_ = num_clusters[i]

    for noise in noise_percentage:  # for each noise percentage
        noisy_image = noise_on_image(originImg, noise)

        originShape = noisy_image.shape
        flatImg = np.reshape(noisy_image, [-1, 3])  # flatten the image for k-means - meanshift

        t = time.time()
        # here run the meanshift approach
        # Estimate bandwidth for meanshift algorithm
        bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

        # Performing meanshift on flatImg
        print('Using MeanShift algorithm, it takes time!')
        ms.fit(flatImg)
        # (r,g,b) vectors corresponding to the different clusters after meanshift
        labels = ms.labels_

        # Displaying segmented image
        segmentedImg = np.reshape(labels, originShape[:2])
        segmentedImg = label2rgb(segmentedImg) * 255  # need this to work with cv2. imshow
        tmpRunTime = time.time() - t
        show_image(images[i].rsplit('.')[0] + "-shiftSegments-Noise-" + str(noise), segmentedImg)
        ssim_shift = compare_cluster_with_mask(maskImg, segmentedImg)
        print("MeanShift : \n\tImage Name : ", images[i], "\n\tNoise Percentage : ", str(noise), "\n\tRuntime : ",
              str(tmpRunTime), "\n\tClosest Cluster SSIM Score : ", str(round(ssim_shift, 4)))

        # using k-means algorithm
        print('Using kmeans algorithm, it is faster!....')
        t = time.time()
        km = MiniBatchKMeans(n_clusters=n_clusters_)
        km.fit(flatImg)
        labels = km.labels_
        # display clustered image
        segmentedImg = np.reshape(labels, originShape[:2])
        segmentedImg = label2rgb(segmentedImg) * 255
        segmentedImg = segmentedImg.astype(np.uint8)
        tmpRunTime = time.time() - t
        show_image(images[i].rsplit('.')[0] + "-kmeansSegments-Noise-" + str(noise), segmentedImg)
        ssim_kmeans = compare_cluster_with_mask(maskImg, segmentedImg)
        print("K-Means : \n\tImage Name : ", images[i], "\n\tNoise Percentage : ", str(noise), "\n\tRuntime : ",
              str(tmpRunTime), "\n\tClosest Cluster SSIM Score : ", str(round(ssim_kmeans, 4)))


        # not working 100% right
        # using agglomerative clustering algorithm
        print('Using Agglomerative Clustering algorithm!')
        t = time.time()
        connectivity = grid_to_graph(*noisy_image.shape)
        print("Compute structured hierarchical clustering...")
        AggM = AgglomerativeClustering(n_clusters=n_clusters_, linkage='average')
        X = np.reshape(noisy_image, (-1, 1))
        AggM.fit(X)
        labels = AggM.labels_
        # Displaying clustered image
        segmentedImg = labels.reshape(originShape[:2])
        segmentedImg = label2rgb(segmentedImg) * 255  # need this to work with cv2. imshow
        segmentedImg = segmentedImg[:, :, 0, :].reshape(originShape[0], originShape[1], originShape[2]).astype(np.uint8)
        tmpRunTime = time.time() - t
        show_image(images[i].rsplit('.')[0] + "-aggloSegments-Noise-" + str(noise), segmentedImg)
        ssim_agglo = compare_cluster_with_mask(maskImg, segmentedImg)
        print("Agglomerative : \n\tImage Name : ", images[i], "\n\tNoise Percentage : ", str(noise), "\n\tRuntime : ",
              str(tmpRunTime), "\n\tClosest Cluster SSIM Score : ", str(round(ssim_agglo, 4)))

        data_for_xls = [[images[i], str(noise), 'K-means', str(round(ssim_kmeans, 4))],
                        [images[i], str(noise), 'Agglomerative', str(round(ssim_agglo, 4))],
                        [images[i], str(noise), 'MeanShift', str(round(ssim_shift, 4))]]

        for data in data_for_xls:
            sheet.append(data)
    cv2.destroyAllWindows()

wb.save(filename=workbook_name)
cv2.destroyAllWindows()
