import numpy as np
import skimage.io as sio
from skimage.measure import compare_ssim
import os, sys
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def computePSNR_from_directory(path_HR='./test_HR_images/1/', path_SR='./test_SR_images/'):

    files_HR = sorted(os.listdir(path_HR), key=str.lower)
    files_SR = sorted(os.listdir(path_SR), key=str.lower)

    files_HR = [path_HR+filename for filename in files_HR]
    files_SR = [path_SR+filename for filename in files_SR]
    
    files = [[files_HR[x], files_SR[x]] for x in range(len(files_HR))]
    
    psnr = [computePSNR_from_images(file) for file in files]
    mean_psnr = np.mean(psnr)
    
    return mean_psnr

def computePSNR_from_images(files):
    
    HRfile, SRfile = files
    
    m = 4

    HR = np.array(sio.imread(HRfile), dtype=np.float32)/ 256
    HRy = 65.738 * HR[...,0] + 129.057 * HR[...,1] + 25.064 * HR[...,2]
    HRy = HRy[m:-m,m:-m]
    
    try:
        SR = np.array(sio.imread(SRfile), dtype=np.float32)/ 256
    except:
        return 0, 0
    SRy = 65.738 * SR[...,0] + 129.057 * SR[...,1] + 25.064 * SR[...,2]
    SRy = SRy[m:-m,m:-m]
    
    mse = np.mean(np.square(HRy - SRy))
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)

    return psnr

def computeSSIM_from_directory(path_HR='./test_HR_images/1/', path_SR='./test_SR_images/'):
    files_HR = sorted(os.listdir(path_HR), key=str.lower)
    files_SR = sorted(os.listdir(path_SR), key=str.lower)

    files_HR = [path_HR + filename for filename in files_HR]
    files_SR = [path_SR + filename for filename in files_SR]

    files = [[files_HR[x], files_SR[x]] for x in range(len(files_HR))]

    ssim = [computeSSIM_from_images(file) for file in files]
    mean_ssim = np.mean(ssim)

    return mean_ssim

def computeSSIM_from_images(files):

    HRfile, SRfile = files

    m = 2

    HR = np.array(sio.imread(HRfile), dtype=np.float32) / 256
    # HRy = 65.738 * HR[..., 0] + 129.057 * HR[..., 1] + 25.064 * HR[..., 2]
    HRy = 0.299 * HR[...,0] + 0.587 * HR[...,1] + 0.114 * HR[...,2]
    HRy = HRy[m:-m, m:-m]

    try:
        SR = np.array(sio.imread(SRfile), dtype=np.float32) / 256
    except:
        return 0, 0

    # SRy = 65.738 * SR[..., 0] + 129.057 * SR[..., 1] + 25.064 * SR[..., 2]
    SRy = 0.299 * SR[..., 0] + 0.587 * SR[..., 1] + 0.114 * SR[..., 2]
    SRy = SRy[m:-m, m:-m]

    mmin = np.min([np.min(HRy),np.min(SRy)])
    mmax = np.max([np.max(HRy),np.max(SRy)])

    ssim = compare_ssim(HRy, SRy, data_range=mmax-mmin)

    return ssim

def computePSNR_per_iterations():

    HR = './test_HR/1/img_005_SRF_4_HR.png'

    def SR(idx):
        return './test_SR_SRGAN/img_005_SRF_4_HR-' + str(idx) + '.png'

    idx = np.arange(100, 100000+100, 100)

    log = open('srgan.txt', 'w')

    for i in idx:
        log.write(str(computePSNR_from_images((HR, SR(i))))+'\n')
        sys.stdout.write('\r%i' % i)
        sys.stdout.flush()

    log.close()

def plotPSNR_iteration_graph():

    file1 = open('./easrgan.txt').read().strip().split('\n')
    file1 = [float(x) for x in file1]
    file1.insert(0, 23.53)

    file2 = open('./srgan.txt').read().strip().split('\n')
    file2 = [float(x) for x in file2]
    file2.insert(0, 23.53)

    idx = np.arange(0, 100000+100, 100)
    plt.plot(idx[:160], file1[:160])
    plt.plot(idx[:160], file2[:160])
    plt.gca().legend(('EaSRGAN', 'SRGAN'))
    plt.grid()
    plt.xlabel('Number of Iterations')
    plt.ylabel('PSNR')
    plt.savefig('./easrgan-stability.png')
    # plt.show()

    print(np.std(file1))
    print(np.std(file2))



if __name__ == '__main__':
    plotPSNR_iteration_graph()
    pass



























