import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
from scipy import misc
from skimage.transform import resize
import datetime
import time

import model
from compute_psnr import computePSNR_from_directory, computePSNR_from_images
from utils import rgb2ycbcr, ycbcr2rgb


def HR_loader(batch_size):
    
    hr_transform = transforms.Compose([
        transforms.RandomCrop(size=96),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Lambda(lambda z: z[0]),
        # transforms.Lambda(lambda z: z.unsqueeze(0))
    ])

    dataset = torchvision.datasets.ImageFolder('/home/bishshoy/ssd0/RAISE/', transform=hr_transform)
    hr_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    
    return hr_loader


def save_resnet_weights(net, suffix):
    torch.save(net.state_dict(), './resnet-weights/saved-model-' + str(suffix))


def save_srgan_weights(G, D, suffix):
    torch.save(G.state_dict(), './srgan-weights/saved-G-' + str(suffix))
    torch.save(D.state_dict(), './srgan-weights/saved-D-' + str(suffix))
    

def read_lr(file):
    lr_file = open(file, 'r')
    new_lr = np.float64(lr_file.read())
    lr_file.close()
    
    return new_lr


def write_lr(lr, file):
    lr_file = open(file, 'w+')
    lr_file.write(str(lr))
    lr_file.close()


def test_resnet(net=None,
                saved_model='./resnet-weights/saved-model-1000000',
                hr_path='./test_HR_images/', sr_path='./test_SR_images/',
                save_file_prefix='', save_file_suffix='',
                quiet=False):
    if net is None:
        net = model.ResNet()
        net.load_state_dict(torch.load(saved_model))
        net.cuda()

    evalset = torchvision.datasets.ImageFolder(hr_path, transform=transforms.ToTensor())
    eval_loader = torch.utils.data.DataLoader(evalset, batch_size=1, shuffle=False, num_workers=1)

    if not quiet: print('Computing SR...')

    for i, data in enumerate(eval_loader):

        net.eval()

        HR, _ = data

        LR_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=[HR.shape[2] // 4, HR.shape[3] // 4], interpolation=3),
            transforms.ToTensor()
        ])

        LR = torch.FloatTensor(1, 3, HR.shape[2] // 4, HR.shape[3] // 4)
        LR[0] = LR_transform(HR[0])

        LR_r, LR_g, LR_b = LR[0].numpy()
        LR_y, LR_cb, LR_cr = rgb2ycbcr(LR_r, LR_g, LR_b)

        LR_Y = torch.from_numpy(LR_y).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            SR_Y = net(LR_Y.cuda())

        SR_y = SR_Y[0][0].cpu().numpy()
        SR_cb = resize(LR_cb, (HR.shape[2], HR.shape[3]), order=3, mode='edge', preserve_range=True)
        SR_cr = resize(LR_cr, (HR.shape[2], HR.shape[3]), order=3, mode='edge', preserve_range=True)

        SR_r, SR_g, SR_b = ycbcr2rgb(SR_y, SR_cb, SR_cr)
        SR_rgb = np.dstack((SR_r, SR_g, SR_b))

        name_of_image = evalset.imgs[i][0].split('/')[-1]

        save_filename = sr_path + str(save_file_prefix) + name_of_image[:-4] + str(save_file_suffix) + '.png'

        SR_RGB = np.array(np.round(np.clip(255 * SR_rgb, 0, 255)), dtype=np.uint8)
        misc.imsave(save_filename, SR_RGB)

        if not quiet: print(name_of_image)


def test_srgan(net,
               hr_path='./test_HR_images/', sr_path='./test_SR_images/',
               save_file_prefix='', save_file_suffix='',
               ):

    evalset = torchvision.datasets.ImageFolder(hr_path, transform=transforms.ToTensor())
    eval_loader = torch.utils.data.DataLoader(evalset, batch_size=1, shuffle=False, num_workers=1)

    for i, data in enumerate(eval_loader):

        net.eval()

        HR, _ = data

        LR_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=[HR.shape[2] // 4, HR.shape[3] // 4], interpolation=3),
            transforms.ToTensor()
        ])

        LR = torch.FloatTensor(1, 3, HR.shape[2] // 4, HR.shape[3] // 4)
        LR[0] = LR_transform(HR[0])

        with torch.no_grad():
            SR = net(LR.cuda())

        name_of_image = evalset.imgs[i][0].split('/')[-1]

        save_filename = sr_path + str(save_file_prefix) + name_of_image[:-4] + str(save_file_suffix) + '.png'

        torchvision.utils.save_image(SR.data, save_filename)


def train_resnet():

    batch_size = 16
    max_iter = int(1e6)

    hr_loader = HR_loader(batch_size)
    
    net = model.ResNet()
    net.cuda()

    # edge = model.Edge()
    
    HR_to_LR = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.Resize(size=24, interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])
    
    # Initialize lr file
    lr_file = './lr'
    lr = 1e-4
    write_lr(lr, lr_file)

    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, net.parameters()), lr=lr)

    # Set-up variables for training prints
    writer = SummaryWriter(log_dir='./runs/resnet/')
    global_step = 1
    print_freq = 100
    eval_freq = 1000
    save_freq = 10000
    start_time = time.time()

    print('ResNet Training starts...')

    end_flag = False
    while not end_flag:

        for i, data in enumerate(hr_loader):

            if global_step == max_iter:
                end_flag = True
                break
            else:
                global_step += 1

            net.train()
            net.zero_grad()

            HR, _ = data
            LR = torch.FloatTensor(HR.shape[0], 1, 24, 24)

            for k in range(HR.shape[0]):
                LR[k] = HR_to_LR(HR[k])

            HR = Variable(HR.cuda())
            LR = Variable(LR.cuda())

            SR = net(LR)
            # EDHR = edge(HR)
            # EDSR = edge(SR)

            mse_loss = torch.mean((HR - SR) ** 2)
            # edge_loss = torch.mean((EDSR - EDHR) ** 2)

            loss = mse_loss

            writer.add_scalar('MSE_Loss', mse_loss, global_step)
            # writer.add_scalar('Edge_Loss', edge_loss, global_step)

            loss.backward()
            optimizer.step()

            # Training prints
            if global_step % print_freq == 0:

                # Compute ETA
                speed = int((print_freq * batch_size)/ (time.time() - start_time))
                start_time = time.time()
                ETA = str(datetime.timedelta(seconds=int(batch_size * (max_iter - global_step)/ speed)))

                # Print
                print('Iter:%d, Loss:(%.4f,%.4f), Speed:%d, ETA:%s' % (global_step, mse_loss.data, 0, speed, ETA))

            if global_step % eval_freq == 0:

                # Eval images periodically
                test_resnet(net, quiet=True)

                # Compute PSNR
                psnr = computePSNR_from_directory()
                writer.add_scalar('Eval_PSNR', psnr, global_step)

                # Prints
                print('PSNR %0.4f' % psnr, 'global_step', global_step)

                # Manual update of lr
                lr = read_lr(lr_file)
                if lr != optimizer.param_groups[0]['lr']:
                    print('New LR detected', lr)
                    optimizer.param_groups[0]['lr'] = lr
                writer.add_scalar('Learning_Rate', lr, global_step)

                # Reset timing counter for accurate timing display
                start_time = time.time()

            if global_step % save_freq == 0:
                save_resnet_weights(net, global_step)
                
    print('Finished Training ResNet')


def train_srgan():

    batch_size = 16
    max_iter = int(1e5)

    trainloader = HR_loader(batch_size)

    G = model.ResNet()
    G.load_state_dict(torch.load('./saved-resnet-mse'))
    G.cuda()

    D = model.VGG13()
    D.cuda()

    edge = model.Edge(mode='L', s=1.4)

    HR_to_LR = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.Resize(size=24, interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

    # Initialize lr file
    lr_file = './lr'
    lr = 1e-5
    write_lr(lr, lr_file)

    G_optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, G.parameters()), lr=lr)
    D_optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, D.parameters()), lr=lr)

    # Set-up variables for training prints
    writer = SummaryWriter(log_dir='./runs/srgan')
    global_step = 1
    print_freq = 100
    eval_freq = 1000
    save_freq = 1000
    start_time = time.time()
    
    print('Training GAN...')

    end_flag = False
    while not end_flag:

        for i, data in enumerate(trainloader):

            if global_step == max_iter:
                end_flag = True
                break
            else:
                global_step += 1

            G.train()
            D.train()

            HR, _ = data
            LR = torch.FloatTensor(HR.shape[0], 1, 24, 24)

            for k in range(HR.shape[0]):
                LR[k] = HR_to_LR(HR[k])

            HR = Variable(HR.cuda())
            LR = Variable(LR.cuda())

            # Compute Discriminator losses
            D.zero_grad()
            SR = G(LR)
            EDHR = edge(HR)
            EDSR = edge(SR.detach())

            lossD_fake = torch.log(1 - D(EDSR.detach()) + 1e-12)
            lossD_real = torch.log(D(EDHR.detach()) + 1e-12)
            lossD = torch.mean(-(lossD_fake + lossD_real))/ 2

            # Compute Generator losses
            G.zero_grad()
            SR = G(LR)
            EDHR = edge(HR)
            EDSR = edge(SR)

            content_loss = torch.mean((EDHR - EDSR) ** 2)
            adversarial_loss = 1e-3 * torch.mean(-torch.log(D(EDSR) + 1e-12))
            lossG = content_loss + adversarial_loss

            writer.add_scalar('lossD', lossD, global_step)
            writer.add_scalar('content_loss', content_loss, global_step)
            writer.add_scalar('adversarial_loss', adversarial_loss, global_step)
            writer.add_scalar('lossG', lossG, global_step)

            # Train both
            lossD.backward()
            D_optimizer.step()

            lossG.backward()
            G_optimizer.step()

            # Train generator for smoothness
            G.zero_grad()
            G_optimizer.param_groups[0]['lr'] = 1e-3
            SR = G(LR)
            SMHR = (1 - edge(HR)) * HR
            SMSR = (1 - edge(SR)) * SR

            mse_loss = torch.mean((SMHR - SMSR) ** 2)
            mse_loss.backward()
            G_optimizer.step()
            G_optimizer.param_groups[0]['lr'] = lr

            # Training prints
            if global_step % print_freq == 0:

                # Compute Speed and ETA
                speed = int((print_freq * batch_size)/ (time.time() - start_time))
                start_time = time.time()
                ETA = str(datetime.timedelta(seconds=int(batch_size * (max_iter - global_step)/ speed)))

                # Print
                print('Iter:%d, Loss:(%0.4f,%0.4f,%0.4f,%0.4f), Speed:%d, ETA:%s' % (
                    global_step,
                    lossD,
                    content_loss,
                    adversarial_loss,
                    lossG,
                    speed, 
                    ETA
                ))

            if global_step % eval_freq == 0:

                # Eval images periodically
                test_resnet(G, quiet=True)
                test_resnet(G, hr_path='./test_HR/', sr_path='./test_SR/', save_file_suffix='-'+str(global_step), quiet=True)

                # Compute PSNR
                psnr = computePSNR_from_directory()
                writer.add_scalar('Eval_PSNR', psnr, global_step)

                # Prints
                print('PSNR %0.4f' % psnr, 'global_step', global_step)

                # Manual update of lr
                lr = read_lr(lr_file)
                if lr != G_optimizer.param_groups[0]['lr']:
                    print('New LR detected', lr)
                    G_optimizer.param_groups[0]['lr'] = lr
                    D_optimizer.param_groups[0]['lr'] = lr
                writer.add_scalar('Learning_Rate', lr, global_step)

            if global_step % save_freq == 0:
                save_srgan_weights(G, D, global_step)

    print('Finished Training SRGAN')


def train_purana_gan():

    batch_size = 16
    max_iter = int(1e5)

    trainloader = HR_loader(batch_size)

    G = model.ResNet()
    G.load_state_dict(torch.load('/home/bishshoy/saved-resnet-mse-rgb'))
    G.cuda()

    D = model.VGG13()
    D.cuda()

    VGG = model.VGG19()
    VGG.cuda()

    HR_to_LR = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize(size=24, interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

    # Initialize lr file
    lr = 1e-5

    G_optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, G.parameters()), lr=lr)
    D_optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, D.parameters()), lr=lr)

    # Set-up variables for training prints
    global_step = 1
    eval_freq = 100

    end_flag = False
    while not end_flag:

        for i, data in enumerate(trainloader):

            if global_step == max_iter:
                end_flag = True
                break
            else:
                global_step += 1

            G.train()
            D.train()
            VGG.eval()

            HR, _ = data
            LR = torch.FloatTensor(HR.shape[0], 3, 24, 24)

            for k in range(HR.shape[0]):
                LR[k] = HR_to_LR(HR[k])

            HR = Variable(HR.cuda())
            LR = Variable(LR.cuda())

            # Compute Discriminator losses
            D.zero_grad()
            SR = G(LR)

            lossD_fake = torch.log(1 - D(SR.detach()) + 1e-12)
            lossD_real = torch.log(D(HR) + 1e-12)
            lossD = torch.mean(-(lossD_fake + lossD_real)) / 2

            # Compute Generator losses
            G.zero_grad()
            SR = G(LR)
            content_loss = 6.1e-3 * torch.sum(torch.pow(VGG(SR) - VGG(HR), 2)) / 4096
            adversarial_loss = torch.mean(-torch.log(D(SR) + 1e-12))
            lossG = content_loss + 1e-3 * adversarial_loss

            # Train both
            lossD.backward()
            D_optimizer.step()

            lossG.backward()
            G_optimizer.step()

            if global_step % eval_freq == 0:

                # Eval images periodically
                test_srgan(G)
                test_srgan(G, hr_path='./test_HR/', sr_path='./test_SR/', save_file_suffix='-' + str(global_step))

                # Compute PSNR
                psnr = computePSNR_from_directory()
                psnr_image = computePSNR_from_images(['./test_HR/1/img_005_SRF_4_HR.png', './test_SR/img_005_SRF_4_HR.png'[:-4] + '-' + str(global_step) + '.png'])

                # Prints
                print(psnr, psnr_image, global_step)

    print('Finished Training SRGAN')


def main():
    
    torch.backends.cudnn.benchmark = True
    train_purana_gan()

    # for i in range(int(1e4), int(1e6+1e4), int(1e4)):
    #     saved_model = './resnet-weights-saved/saved-model-' + str(i)
    #     test_resnet(saved_model=saved_model, quiet=True)
    #     print(computePSNR_from_directory())














if __name__ == '__main__':
    main()

