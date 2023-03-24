from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()

    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image, crop=False, count=False)
            r_image.show()
            img = img.split('/')
            r_image.save("img/after_" + img[1])
