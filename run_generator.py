import argparse
import numpy as np
import PIL.Image

from training import misc

def create_from_images(checkpoint, image, mask, output):
    real = np.asarray(PIL.Image.open(image)).transpose([2, 0, 1])
    real = misc.adjust_dynamic_range(real, [0, 255], [-1, 1])
    mask = np.asarray(PIL.Image.open(mask).convert('1'))
    
    _, _, Gs = misc.load_pkl(checkpoint)
    latent = np.random.randn(1, *Gs.input_shape[1:])
    fake = Gs.run(latent, None, real[np.newaxis], mask[np.newaxis, np.newaxis])[0]
    fake = misc.adjust_dynamic_range(fake, [-1, 1], [0, 255])
    fake = fake.clip(0, 255).astype(np.uint8).transpose([1, 2, 0])
    fake = PIL.Image.fromarray(fake)
    fake.save(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', help='Network checkpoint path', required=True)
    parser.add_argument('-i', '--image', help='Original image path', required=True)
    parser.add_argument('-m', '--mask', help='Mask path', required=True)
    parser.add_argument('-o', '--output', help='Output (inpainted) image path', required=True)

    args = parser.parse_args()
    create_from_images(**vars(args))

if __name__ == "__main__":
    main()
