import torch
import numpy as np


# 이미지 파일 확장자 검수 함수
def check_image_file(filename: str):
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP"])

# 전처리 과정 함수
def preprocess(img):
    # uInt8 -> float32로 변환
    x = np.array(img).astype(np.float32)
    x = x.transpose([2,0,1])
    # Normalize x 값
    x /= 255.
    # 넘파이 x를 텐서로 변환
    x = torch.from_numpy(x)
    # x의 차원의 수 증가
    x = x.unsqueeze(0)
    # x 값 반환
    return x

def calc_patch_size(func):
    def wrapper(args):
        if args.scale == 2:
            args.patch_size = 10
        elif args.scale == 3:
            args.patch_size = 7
        elif args.scale == 4:
            args.patch_size = 6
        else:
            raise Exception('Scale Error', args.scale)
        return func(args)
    return wrapper

def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.

def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
        cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])

def convert_ycbcr_to_rgb(img, dim_order='hwc'):
    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    else:
        r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921
        g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[2] / 256. + 135.576
        b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])

def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

# Copy from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# Copy from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
    
# Copy from https://github.com/luoming1994/SRResNet/blob/master/src/data.py

def numpyRGB2Image(img_np):
    """
    img_np is a RGB mode image, numpy type,dtype = np.uint8
    return a Image.Image type,RGB mode
    """
    img_rgb = Image.fromarray(img_np.transpose(1,2,0))
    return img_rgb
    
    
def numpyYCbCr2Image(img_np):
    """
    img_np is a YCbCr mode image, numpy type,dtype = np.uint8
    return a Image.Image type,RGB mode
    """
    Y  = Image.fromarray(img_np[0])     # 2dim
    Cb = Image.fromarray(img_np[1])
    Cr = Image.fromarray(img_np[2])
    img_YCbCr = Image.merge('YCbCr',(Y,Cb,Cr))
    img_rgb = img_YCbCr.convert('RGB')
    return img_rgb
    

def loadImgRGB2Numpy(filepath,down_scale = None,up_scale =None):
    """
    load a rgb image to numpy(channel * H * W)
    dtype = np.uint8 (make the data easy to Storage)
    down_scale: if is not None,down scale 
    up_scale:   if is not None,up sacle 
    """
    img = Image.open(filepath)  
    if down_scale is not None:
        W,H = img.size
        img = img.resize((int(W*down_scale),int(H*down_scale) ), Image.BICUBIC)
    if up_scale is not None:
        W,H = img.size
        img = img.resize((W*up_scale,H*up_scale),Image.BICUBIC)
    img = np.array(img).transpose(2, 0, 1)  # Image=>numpy.array
    
    return img

def loadImgYCbCr2Numpy(filepath,down_scale = None,up_scale =None):
    """
    load image Y channel to numpy(1 * H * W)
    dtype = np.uint8 (make the data easy to Storage)
    down_scale: if is not None,down scale 
    up_scale:   if is not None,up sacle 
    """
    img = Image.open(filepath)  
    if down_scale is not None:
        W,H = img.size
        img = img.resize((int(W*down_scale),int(H*down_scale)),Image.BICUBIC)
    if up_scale is not None:
        W,H = img.size
        img = img.resize((W*up_scale,H*up_scale),Image.BICUBIC)
    img_YCbCr = img.convert('YCbCr')        # change image mode
    img_YCbCr = np.array(img_YCbCr).transpose(2, 0, 1)  # Image=>numpy.array
        
    return img_YCbCr


def cut2normal(img_np,cut_size = 24):
    """
    cut a numpy(channel * H * W ) to normal size
    """
    shape = img_np.shape
    assert len(shape) == 3,"img_np is not 3 dim"
    nH,nW = shape[-2]//cut_size, shape[-1]//cut_size
    c = shape[0]    # channels
    img = np.empty((nH*nW*c,cut_size,cut_size),dtype=img_np.dtype)
    index = 0
    for i in range(nH):
        for j in range(nW):
            img[index*c:(index+1)*c,:,:] = img_np[:,i*cut_size:(i+1)*cut_size,j*cut_size:(j+1)*cut_size]
            index += 1 
            
    return img

def numpy2Tensor(img_np):
    """
    np.uint8[0,255] => torch.Tensor[0.0,1.0]
    """
    img_np = torch.from_numpy(img_np)
    return img_np.float().div(255)
 

def tensor2Numpy(img_tensor,normalize = True):
    """
    torch.Tensor[0.0,1.0] => np.uint8[0,255]
    """
    img_np = img_tensor.numpy()*255
    if normalize:
        img_np[img_np < 0.0] = 0
        img_np[img_np > 255.0] = 255
    return np.array(img_np,dtype = np.uint8)
 

class img2data(object):
    """
    transform images as numpy(dtype = np.uint8) into data storage in disk
    """
    def __init__(self,hr_dir, lr_dir = None,hr_size = 96,lr_size =24,down_scale = None, up_scale = None,img_num = 800):
        """
        hr_size: hr iamges cut to hr_size*hr_size 
        lr_size: lr iamges cut to lr_size*lr_size
        down_scale: if the lr images need to down scale, if lr_dir is None,
                    we need down sacle the hr image to the lr image
        up_scale:if we need to up the lr image to the same size of hr image, 
                    using up_scale,make lr_size = hr_size.
        """
        self.hr_size    = hr_size
        self.lr_size    = lr_size 
        self.down_scale = down_scale
        self.up_scale   = up_scale
        self.hr_paths   = [os.path.join(hr_dir, x) for x in os.listdir(hr_dir) if isImage(x)]
        self.hr_paths.sort()
        if lr_dir == None:  # downsample the hr iamges to lr images
            self.lr_paths = self.hr_paths
        else:
            self.lr_paths   = [os.path.join(lr_dir, x) for x in os.listdir(lr_dir) if isImage(x)]
            self.lr_paths.sort()
        assert len(self.hr_paths) == len(self.lr_paths),"hr_dir,lr_dir have the image num is not the same"
        # get the first img_num images
        if img_num < len(self.hr_paths):
            self.lr_paths = self.lr_paths[0:img_num]
            self.hr_paths = self.hr_paths[0:img_num]
            self.img_num    = img_num
        else:
            self.img_num = len(self.hr_paths)
        
        # save rgb image, 3 channel every image
        self.lr = np.array([],dtype = np.uint8).reshape(-1,self.lr_size,self.lr_size)
        self.hr = np.array([],dtype = np.uint8).reshape(-1,self.hr_size,self.hr_size)
        
        # save images' Y channel
        self.lrY = np.array([],dtype = np.uint8).reshape(-1,self.lr_size,self.lr_size)
        self.hrY = np.array([],dtype = np.uint8).reshape(-1,self.hr_size,self.hr_size)
        
        self.lrRGBY = np.array([],dtype = np.uint8).reshape(-1,self.lr_size,self.lr_size)
        self.hrRGBY = np.array([],dtype = np.uint8).reshape(-1,self.hr_size,self.hr_size)
    
    def loadImgRGB(self):
        for hr_path in self.hr_paths:
            imgs = cut2normal(loadImgRGB2Numpy(hr_path),cut_size = self.hr_size)
            self.hr = np.concatenate((self.hr,imgs),axis=0) # concat
        for lr_path in self.lr_paths:
            img = loadImgRGB2Numpy(lr_path, down_scale = self.down_scale, up_scale = self.up_scale)
            imgs = cut2normal(img,cut_size = self.lr_size)
            self.lr = np.concatenate((self.lr,imgs),axis=0) 
    
    def saveImgRGB(self,save_path):
        np.savez(save_path,lr = self.lr, hr = self.hr)
        
    def loadImgYChannel(self):
        """
        load images' Y channel
        """
        for hr_path in self.hr_paths:
            y = loadImgYCbCr2Numpy(hr_path)[0:1,:,:]
            ys = cut2normal(y, cut_size = self.hr_size)
            self.hrY = np.concatenate((self.hrY,ys),axis=0) # concat
        for lr_path in self.lr_paths:
            y = loadImgYCbCr2Numpy(lr_path, down_scale = self.down_scale, up_scale = self.up_scale)[0:1,:,:]
            ys = cut2normal(y, cut_size = self.lr_size)
            self.lrY = np.concatenate((self.lrY,ys),axis=0) 
    
    def saveImgYChannel(self,save_path):
        """
        save images' Y channel into disk
        """
        np.savez(save_path,lr = self.lrY, hr = self.hrY)
    
    def loadImgLrRGB_HrY(self):
        """
        load lr images' RGB channel
        load hr images' Y channel
        """
        # load lr rgb mode
        for lr_path in self.lr_paths:
            img = loadImgRGB2Numpy(lr_path, down_scale = self.down_scale, up_scale = self.up_scale)
            imgs = cut2normal(img, cut_size = self.lr_size)
            self.lr = np.concatenate((self.lr,imgs),axis=0) 
        # load hr y mode
        for hr_path in self.hr_paths:
            y = loadImgYCbCr2Numpy(hr_path)[0:1,:,:]
            ys = cut2normal(y, cut_size = self.hr_size)
            self.hrY = np.concatenate((self.hrY,ys),axis=0) # concat
    def saveImgLrRGB_HrY(self,save_path):
        """
        save lr images' RGB channel and hr images' Y channel,to disk
        """
        np.savez(save_path,lr = self.lr, hr = self.hrY)   
        
    
        
    def loadImgLrRGBY_HrRGBY(self):
        """
        load lr images' RGB channel
        load hr images' Y channel
        """
        # load lr rgb mode
        for lr_path in self.lr_paths:
            rgb = loadImgRGB2Numpy(lr_path, down_scale = self.down_scale, up_scale = self.up_scale)
            y = loadImgYCbCr2Numpy(lr_path, down_scale = self.down_scale, up_scale = self.up_scale)[0:1,:,:]
            rgby = np.concatenate((rgb,y),axis = 0) # concat at axis = 0
            imgs = cut2normal(rgby, cut_size = self.lr_size)
            self.lrRGBY = np.concatenate((self.lrRGBY,imgs),axis=0) 
        # load hr y mode
        for hr_path in self.hr_paths:
            rgb = loadImgRGB2Numpy(hr_path)
            y = loadImgYCbCr2Numpy(hr_path)[0:1,:,:]
            rgby = np.concatenate((rgb,y),axis = 0) # concat at axis = 0
            imgs = cut2normal(rgby, cut_size = self.hr_size)
            self.hrRGBY = np.concatenate((self.hrRGBY,imgs),axis=0) 
    
    def saveImgLrRGBY_HrRGBY(self,save_path):
        """
        save lr images' RGB channel and hr images' Y channel,to disk
        """
        np.savez(save_path,lr = self.lrRGBY, hr = self.hrRGBY)   
