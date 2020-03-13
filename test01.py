try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
import os

def GetAnnotBoxLoc(AnotPath):  # AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  # 打开文件，解析成一棵树型结构
    root = tree.getroot()  # 获取树型结构的根
    ObjectSet = root.findall("object")  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet = []  # 以目标类别为关键字，目标框为值组成的字典结构
    for Object in ObjectSet:
        BndBox = Object.find('bndbox')
        ObjName = int(Object.find('name').text)
        x1 = int(BndBox.find('xmin').text)  # -1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text)  # -1
        x2 = int(BndBox.find('xmax').text)  # -1
        y2 = int(BndBox.find('ymax').text)  # -1

        w = x2 - x1     #宽
        h = y2 - y1     #高
        cx = int(x1 + w/2)   #中心点x
        cy = int(y1 + h/2)   #中心点y

        BndBoxLoc = [ObjName,cx, cy, w, h]
        ObjBndBoxSet.append(BndBoxLoc)
    return ObjBndBoxSet

if __name__=="__main__":
    save_path = r'C:\Users\Administrator\Desktop\新建文件夹\label.txt'
    a = r"image/"
    path = r"C:\Users\Administrator\Desktop\新建文件夹\outputs"
    label = open(save_path,"w+")
    for name in os.listdir(path):
        class_name = a + str(name).split('.')[0] + '.jpg'
        file = os.path.join(path, name)
        out_box = GetAnnotBoxLoc(file)
        label.write(class_name + '  ')
        for box in out_box:
            label.write(str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ' + str(box[4]) + '  ')
        label.write('\n')


