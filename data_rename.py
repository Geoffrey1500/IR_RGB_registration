# 导入os包
import os

def file_rename(path, begin_num):
    """
    @param path: 文件夹路径
    @param begin_num: 起始命名数字
    """
    # 设定文件路径
    i = begin_num
    # 对目录下的文件进行遍历
    for file in os.listdir(path):
        # 判断是否是文件
        if os.path.isfile(os.path.join(path, file)) == True:
            # 设置新文件名
            new_name = file[6::]
            # 重命名
            os.rename(os.path.join(path, file), os.path.join(path, new_name))
            i += 1
    # 结束
    print("End")


if __name__ == '__main__':
    path = './triandata/Label/'
    file_rename(path, 1)
