import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from math import log10, sqrt 

def mean_squared_error(x1, x2):
    return np.mean(np.square(x1 - x2))


def psnr(x1, x2): 
    mse = mean_squared_error(x1, x2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 


def calc_cdf(count):
    cdf = np.cumsum(count) / sum(count)
    return cdf


def histogram_equalization_channel(image:np.ndarray, nbins=255):
    hist, bin_centers = np.histogram(image.ravel(), bins=nbins)
    # bin_width = bin_centers[1] - bin_centers[0]
    cdf = calc_cdf(hist)

    flattened_img = image.ravel()
    out = []
    for pix in flattened_img:
        out.append(cdf[pix-1])

    out = (np.array(out).reshape(image.shape) * 255).astype('uint8')
    return out 


def histogram_equalization(img:np.ndarray):
    '''
    Image in RGB Space
    '''
    # ? Converting from BGR color space to YUV color space
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = histogram_equalization_channel(img_yuv[:,:,0])
    out = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return out 



def plot_histogram(channel_intensities:np.ndarray, title:str, save=True):
    hist, bins = np.histogram(channel_intensities.ravel(), bins=256, range=(0,256))
    cdf = hist.cumsum()
    cdf_norm = cdf * float(hist.max()) / cdf.max()


    plt.hist(channel_intensities.ravel(), bins=256, range=(0,255))
    plt.stairs(cdf_norm)
    plt.title(title)
    plt.xlim([0, 256])
    plt.xlabel('pixel intensities')
    plt.ylabel('frequency')
    plt.legend(('cdf','histogram'), loc = 'upper left')
    if save:
        plt.savefig(f'data/{title}.png')

    plt.show()


def equalize_histogram_cv(img:np.ndarray):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    out = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return out

def resize_with_aspect_ratio(image, width=None, height=None):
    # Get the original image dimensions
    h, w = image.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = w / h

    if width is None:
        # Calculate height based on the specified width
        new_height = int(height / aspect_ratio)
        resized_image = cv2.resize(image, (height, new_height))
    else:
        # Calculate width based on the specified height
        new_width = int(width * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, width))

    return resized_image


if __name__ == '__main__':
    img = cv2.imread('data/DSC_0232.JPG')

    # ? Plotting Luminance before equalization
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    plot_histogram(img_yuv[:, :, 0], "Y' Values Before Histogram Equalization")

    # ? Plotting Luminance after equalization
    plot_histogram(histogram_equalization_channel(img_yuv[:,:,0]), "Y' Values After Histogram Equalization")

    # ? Applying Histtogram_equalization and saving our image
    out = histogram_equalization(img)
    cv2.imwrite('data/our_histogram_output.png', out)

    # ? Comparing with OpenCV
    cv2_result = equalize_histogram_cv(img)
    res = np.hstack((out,cv2_result))

    # ? Resizing to decrease storage size
    cv2.imwrite('data/histogram_compare_with_cv2.png', resize_with_aspect_ratio(res, height=6000))




    print(f'Mean Square Error: {mean_squared_error(out, cv2_result):.2f}')
    print(f'PSNR: {psnr(out, cv2_result):.2f} db')
