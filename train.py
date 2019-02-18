from faster_rcnn.datasets.pascal_voc import Pascal_VOC
from faster_rcnn.roi_data_layer.roidb import prepare_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer

imdb = Pascal_VOC("trainval", "2007")
roidb = prepare_roidb(imdb)
data_layer = RoIDataLayer(roidb)



from IPython import embed; embed()