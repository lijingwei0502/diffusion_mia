import scipy.io

# 加载meta.mat文件
meta_mat_file = '/data_server3/ljw/imagenet/ILSVRC2012_devkit_t12/data/meta.mat'
meta = scipy.io.loadmat(meta_mat_file)

# 打印meta['synsets']的结构
print(meta['synsets'])
