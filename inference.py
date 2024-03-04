from mmpretrain import ImageClassificationInferencer
inferencer = ImageClassificationInferencer(
        model='configs/vgg/custom_vgg.py',
        pretrained='work_dirs/custom_vgg/epoch_34.pth')
#inferencer(['demo/g1.png', 'demo/g.png'],rescale_factor=5.0, show_dir="./visualize/")
inferencer('demo/f', rescale_factor=5.0, show_dir="./visualize/f")

