import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', required=True)
parser.add_argument('-d', '--data-dir', required=True)
parser.add_argument('-s', '--save-dir', default='images')
parser.add_argument('-w', '--window-size', type=int, default=512)
args = parser.parse_args()

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
import hashlib

import dnnlib

from training import misc, dataset


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.state = -1
        self.canvas = tk.Canvas(self, bg='gray', height=args.window_size, width=args.window_size)
        self.canvas.bind("<Button-1>", self.L_press)
        self.canvas.bind("<ButtonRelease-1>", self.L_release)
        self.canvas.bind("<B1-Motion>", self.L_move)
        self.canvas.bind("<Button-3>", self.R_press)
        self.canvas.bind("<ButtonRelease-3>", self.R_release)
        self.canvas.bind("<B3-Motion>", self.R_move)
        self.canvas.bind("<Key>", self.key_down)
        self.canvas.bind("<KeyRelease>", self.key_up)
        self.canvas.pack()

        self.canvas.focus_set()
        self.canvas_image = self.canvas.create_image(0, 0, anchor='nw')

        dnnlib.tflib.init_tf()
        self.dataset = dataset.load_dataset(tfrecord_dir=args.data_dir, verbose=True, shuffle_mb=0)
        
        self.networks = []
        self.truncations = []
        self.model_names = []
        for ckpt in args.checkpoint.split(','):
            if ':' in ckpt:
                ckpt, truncation = ckpt.split(':')
                truncation = float(truncation)
            else:
                truncation = None
           
            _, _, Gs = misc.load_pkl(ckpt)
            
            self.networks.append(Gs)
            self.truncations.append(truncation)
            self.model_names.append(os.path.basename(os.path.splitext(ckpt)[0]))
        
        self.key_list = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'][:len(self.networks)]
        self.image_id = -1
        
        self.new_image()
        self.display()
    
    def generate(self, idx=0):
        self.cur_idx = idx
        latent = np.random.randn(1, *self.networks[idx].input_shape[1:])
        real = misc.adjust_dynamic_range(self.real_image, [0, 255], [-1, 1])
        fake = self.networks[idx].run(latent, self.label, real, self.mask, truncation_psi=self.truncations[idx])
        self.fake_image = misc.adjust_dynamic_range(fake, [-1, 1], [0, 255]).clip(0, 255).astype(np.uint8)
    
    def new_image(self):
        self.image_id += 1
        self.save_count = 0
        self.real_image, self.label = self.dataset.get_minibatch_val_np(1)
        self.resolution = self.real_image.shape[-1]
        self.mask = np.ones((1, 1, self.resolution, self.resolution), np.uint8)
        self.mask_history = [self.mask]

    def display(self, state=0):
        if state != self.state:
            self.last_state = self.state
        self.state = state
        self.image = self.real_image if self.state == 1 else self.fake_image if self.state == 2 else self.real_image * self.mask
        self.image_for_display = np.transpose(self.image[0, :3], (1, 2, 0))
        self.image_for_display_resized = cv2.resize(self.image_for_display, (args.window_size, args.window_size))
        self.tkimage = ImageTk.PhotoImage(image=Image.fromarray(self.image_for_display_resized))
        self.canvas.itemconfig(self.canvas_image, image=self.tkimage)
    
    def save_image(self):
        folder_name = os.path.join(args.save_dir, '-'.join([os.path.basename(args.data_dir), str(self.image_id), hashlib.sha1(self.mask.tostring()).hexdigest()[:6]]))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            self.save_count = 0
            for img, name in [[self.real_image, 'real'], [self.real_image * self.mask, 'masked']]:
                cv2.imwrite(os.path.join(folder_name, name + '.jpg'), np.transpose(img[0, :3], (1, 2, 0))[..., ::-1])
        if self.state == 2:
            cv2.imwrite(os.path.join(folder_name, '-'.join([self.model_names[self.cur_idx], str(self.save_count)]) + '.jpg'), self.image_for_display[..., ::-1])
            self.save_count += 1
    
    def get_pos(self, event):
        return (int(event.x * self.resolution / args.window_size), int(event.y * self.resolution / args.window_size))
    
    def L_press(self, event):
        self.last_pos = self.get_pos(event)
    
    def L_move(self, event):
        a = self.last_pos
        b = self.get_pos(event)
        width = 30
        img = Image.fromarray(self.mask[0, 0])
        draw = ImageDraw.Draw(img)
        draw.line([a, b], fill=0, width=width)
        draw.ellipse((b[0] - width // 2, b[1] - width // 2, b[0] + width // 2, b[1] + width // 2), fill=0)
        self.mask = np.array(img)[np.newaxis, np.newaxis, ...]
        self.display()
        self.last_pos = b
    
    def L_release(self, event):
        self.L_move(event)
        self.mask_history.append(self.mask)
    
    def R_press(self, event):
        self.last_pos = self.get_pos(event)
    
    def R_move(self, event):
        a = self.last_pos
        b = self.get_pos(event)
        self.mask = self.mask_history[-1].copy()
        self.mask[0, 0, max(min(a[1], b[1]), 0): max(a[1], b[1]), max(min(a[0], b[0]), 0): max(a[0], b[0])] = 0
        self.display()
    
    def R_release(self, event):
        self.R_move(event)
        self.mask_history.append(self.mask)
    
    def key_down(self, event):
        if event.keysym == 'z':
            if len(self.mask_history) > 1:
                self.mask_history.pop()
                self.mask = self.mask_history[-1]
                self.display()
        elif event.keysym == 'space':
            self.generate()
            self.display(2)
        elif event.keysym in self.key_list:
            self.generate(self.key_list.index(event.keysym))
            self.display(2)
        elif event.keysym == 's':
            self.save_image()
        elif event.keysym == 'Return':
            self.new_image()
            self.display()
        elif event.keysym == '1':
            self.display(1)
        elif event.keysym == '2':
            self.display(0)
    
    def key_up(self, event):
        if event.keysym in ['1', '2']:
            self.display(self.last_state)

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()