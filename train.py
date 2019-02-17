from faster_rcnn.datasets.pascal_voc import Pascal_VOC

if __name__ == "__main__":
    dataset = Pascal_VOC("trainval", "2007")
    dataset.gt_roidb()
