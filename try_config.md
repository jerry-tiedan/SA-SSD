<!--
 * @Description:
 * @Date: 2021-03-11 09:46:45
 * @LastEditors: Jerry Zhang
 * @LastEditTime: 2021-03-18 15:15:45
 * @FilePath: /SA-SSD/try_config.md
-->

cuda10.0+cudnn7.6.4+pytorch 1.1.0

pip install scikit-image scipy numba pillow==6.2.2 matplotlib fire tensorboardX protobuf opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.14.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install psutil -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flask flask_cors -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install shapely -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pyqt5 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pyqtgraph -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pyopengl -i https://pypi.tuna.tsinghua.edu.cn/simple

# 环境安装主要参考一下博客：

https://blog.csdn.net/qq_38316300/article/details/110161110?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control&dist_request_id=1328656.10532.16158789876387007&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control

# compile spconv

git clone https://github.com/traveller59/spconv --recursive
cd spconv
python setup.py bdist_wheel

# spconv 在安装过程中需要 cuda 和 cudnn 是安装好的状态

## mmcv 的安装，这个版本分为 mmcv 和 mmcv-full，mmcv-full 需要版本 pytorch 和 cuda 版本相应的支持，但是不支持 pytorch 1.1.0 版本，

## mmcv 内会有 parallel_test 不支持的问题，目前直接注释掉

export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
export LD_LIBRARY_PATH=/home/jerry/anaconda3/envs/pytorch_second/lib/python3.7/site-packages/spconv
export PYTHONPATH=$PYTHONPATH:/home/jerry/projects/SA-SSD

conda install numba

pip install mmcv-full==1.2.2 -f https://download.openmmlab.com/mmcv/dist/cu100/torch1.1.0/index.html
