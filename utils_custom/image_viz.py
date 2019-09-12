import glob
import cv2

data_dir = '../../data/argo/forecasting/train/rendered_image/'

img_dir = sorted(glob.glob(data_dir + '*.png'))
print(len(img_dir))

for dir in img_dir:
    name = dir.split('/')[-1][0:-4]
    img = cv2.imread(dir)
    cv2.imshow(name, img)
    while True:
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()

