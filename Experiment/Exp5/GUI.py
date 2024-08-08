import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image as Im
from PIL import ImageEnhance, ImageFilter, ImageTk
from skimage import img_as_float, img_as_ubyte, util, exposure
from skimage.morphology import erosion, dilation, opening, closing
from skimage.feature import canny
import cv2
import numpy as np
import threading

class Gui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('GUI')
        self.geometry('1360x650')
        self.close = False
        self.setup()

    def setup(self):
        # Create canvas
        canvas_1 = tk.Frame(self, height=650, width=500)
        canvas_1.pack(side=tk.LEFT)
        title_1 = tk.Label(canvas_1, text='Original Image', font=30)
        title_1.place(x=160, y=15, width=180, height=40)
        self.image_zone_1 = tk.Label(canvas_1, text='Please import the image.', font=20, relief='ridge')
        self.image_zone_1.place(x=10, y=60, width=480, height=480)

        canvas_2 = tk.Frame(self, height=650, width=500)
        canvas_2.pack(side=tk.LEFT)
        title_2 = tk.Label(canvas_2, text='Processed Image', font=30)
        title_2.place(x=160, y=15, width=180, height=40)
        self.image_zone_2 = tk.Label(canvas_2, relief='ridge')
        self.image_zone_2.place(x=10, y=60, width=480, height=480)

        # Create functional block
        functional = tk.Frame(self, height=650, width=360)
        functional.pack(side=tk.LEFT)
        title_3 = tk.Label(functional, text='Functional Block', font=30)
        title_3.place(x=90, y=15, width=180, height=40)

        # Buttons
        btn = tk.Button(functional, text='Import the image', fg='white', bg='blue', command=self.import_img)
        btn.place(y=600, x=100, width=150, height=40)

        btn0 = tk.Button(functional, text='Undo to origin', fg='white', bg='blue', command=self.show_origin)
        btn0.place(y=550, x=10, width=150, height=40)

        btn1 = tk.Button(functional, text='Clear import', fg='white', bg='blue', command=self.clear_img)
        btn1.place(y=550, x=200, width=150, height=40)

        btn2 = tk.Button(functional, text='Gaussian noise', bg='light blue', command=self.Gaussian_noise)
        btn2.place(y=70, x=10, width=150, height=40)

        btn3 = tk.Button(functional, text='S&P noise', bg='light blue', command=self.salt_and_pepper)
        btn3.place(y=70, x=200, width=150, height=40)

        btn4 = tk.Button(functional, text='Gaussian smooth', bg='light blue', command=self.Gaussian_blur)
        btn4.place(y=120, x=10, width=150, height=40)

        btn5 = tk.Button(functional, text='Median smooth', bg='light blue', command=self.median_blur)
        btn5.place(y=120, x=200, width=150, height=40)

        btn6 = tk.Button(functional, text='Histogram equalization', bg='white', command=self.hist_equal)
        btn6.place(y=170, x=10, width=150, height=40)

        btn7 = tk.Button(functional, text='Mirror flip', bg='white', command=self.mirror_flip)
        btn7.place(y=170, x=200, width=150, height=40)

        btn8 = tk.Button(functional, text='Grayscale', bg='white', command=self.grayscale)
        btn8.place(y=220, x=10, width=150, height=40)

        btn9 = tk.Button(functional, text='Sharpen', bg='white', command=self.sharpen)
        btn9.place(y=220, x=200, width=150, height=40)

        btn10 = tk.Button(functional, text='Erosion', bg='gray', command=self.mo_erosion)
        btn10.place(y=270, x=10, width=150, height=40)

        btn11 = tk.Button(functional, text='Dilation', bg='gray', command=self.mo_dilation)
        btn11.place(y=270, x=200, width=150, height=40)

        btn12 = tk.Button(functional, text='Opening', bg='gray', command=self.mo_opening)
        btn12.place(y=320, x=10, width=150, height=40)

        btn13 = tk.Button(functional, text='Closing', bg='gray', command=self.mo_closing)
        btn13.place(y=320, x=200, width=150, height=40)

        btn14 = tk.Button(functional, text='Edge detection', command=self.egde_detect)
        btn14.place(y=370, x=10, width=150, height=40)

        btn15 = tk.Button(functional, text='Homomorphic enhance', command=self.homomorphic)
        btn15.place(y=370, x=200, width=150, height=40)

        btn16 = tk.Button(functional, text='Face recognition', bg='red', command=self.facedetection)
        btn16.place(y=420, x=10, width=150, height=40)

        btn17 = tk.Button(functional, text='Close camera', bg='light green', command=self.close_win)
        btn17.place(y=420, x=200, width=150, height=40)

        lb1 = tk.Label(functional, text='Contrast', bg='light blue')
        lb1.place(y=480, x=10, width=70, height=30)
        lb1_0 = tk.Button(functional, text='+', width=2, command=self.contrast_add)
        lb1_0.place(y=480, x=90)
        lb1_1 = tk.Button(functional, text='-', width=2, command=self.contrast_sub)
        lb1_1.place(y=480, x=120)

        lb2 = tk.Label(functional, text='Brightness', bg='light blue')
        lb2.place(y=480, x=200, width=70, height=30)
        lb2_0 = tk.Button(functional, text='+', width=2, command=self.brightness_add)
        lb2_0.place(y=480, x=280)
        lb2_1 = tk.Button(functional, text='-', width=2, command=self.brightness_sub)
        lb2_1.place(y=480, x=310)

    # Get the image file address
    def getAddress(self):
        path = tk.StringVar()
        file_entry = tk.Entry(self, state='readonly', text=path)
        # File import window
        path.set(askopenfilename())
        addr = file_entry.get()
        return addr

    # Load the image from the address
    def open_img(self, address):
        img = Im.open(address).convert('RGBA')
        w, h = img.size
        w_, h_ = 480, 480
        f1 = 1.0 * w_ / w
        f2 = 1.0 * h_ / h
        factor = min([f1, f2])
        width = int((w * factor)//2 * 2)
        height = int((h * factor)//2 * 2)
        img_show = img.resize((width, height))
        return img_show

    # Import and show the original image
    def import_img(self):
        address = self.getAddress()
        self.original = self.open_img(address)
        self.img = self.original
        img_show = ImageTk.PhotoImage(self.img)
        self.image_zone_1.config(image=img_show)
        self.image_zone_1.image = img_show
        self.image_zone_2.image = None

    # Show the  image to canvas 2
    def show_img(self, img):
        self.img = img
        img_show = ImageTk.PhotoImage(img)
        self.image_zone_2.config(image=img_show)
        self.image_zone_2.image = img_show

    # Show the original image to canvas 2
    def show_origin(self):
        self.show_img(self.original)

    # Clear the images on both canvas
    def clear_img(self):
        self.img = None
        self.image_zone_1.image = None
        self.image_zone_2.image = None

    # Add Gaussian noise
    def Gaussian_noise(self):
        img = self.img
        img_float = img_as_float(img)
        img_noise = util.random_noise(img_float, mode = 'gaussian', mean=0, var=0.01)
        img_show = Im.fromarray(img_as_ubyte(img_noise))
        self.show_img(img_show)

    # Add salt and pepper noise
    def salt_and_pepper(self):
        img = self.img
        img_float = img_as_float(img)
        img_noise = util.random_noise(img_float, mode='s&p', amount=0.01)
        img_show = Im.fromarray(img_as_ubyte(img_noise))
        self.show_img(img_show)

    # Gaussian blur
    def Gaussian_blur(self):
        img = self.img
        img_show = img.filter(ImageFilter.GaussianBlur(2))
        self.show_img(img_show)

    # Median blur
    def median_blur(self):
        img = self.img
        img_show = img.filter(ImageFilter.MedianFilter(3))
        self.show_img(img_show)

    # Histogran equalization
    def hist_equal(self):
        img = self.img
        img_float = img_as_float(img)
        img_eq = exposure.equalize_hist(img_float)
        img_show = Im.fromarray(img_as_ubyte(img_eq))
        self.show_img(img_show)

    # Mirror flip
    def mirror_flip(self):
        img = self.img
        img_show = img.transpose(Im.FLIP_LEFT_RIGHT)
        self.show_img(img_show)

    # Convert to gray scale
    def grayscale(self):
        img = self.img
        img_gray = img.convert('L')
        # Fit the 4 channels
        img_show = img_gray.convert('RGBA')
        self.show_img(img_show)

    # Sharpening
    def sharpen(self):
        img = self.img
        img_show = img.filter(ImageFilter.SHARPEN)
        self.show_img(img_show)

    # Morphology erosion
    def mo_erosion(self):
        img = self.img
        img_float = img_as_float(img.convert('L'))
        img_ero = erosion(img_float)
        img_show = Im.fromarray(img_as_ubyte(img_ero))
        img_show = img_show.convert('RGBA')
        self.show_img(img_show)

    # Morphology dilation
    def mo_dilation(self):
        img = self.img
        img_float = img_as_float(img.convert('L'))
        img_dil = dilation(img_float)
        img_show = Im.fromarray(img_as_ubyte(img_dil))
        img_show = img_show.convert('RGBA')
        self.show_img(img_show)

    # Morphology opening
    def mo_opening(self):
        img = self.img
        img_float = img_as_float(img.convert('L'))
        img_op = opening(img_float)
        img_show = Im.fromarray(img_as_ubyte(img_op))
        img_show = img_show.convert('RGBA')
        self.show_img(img_show)

    # Morphology closing
    def mo_closing(self):
        img = self.img
        img_float = img_as_float(img.convert('L'))
        img_cl = closing(img_float)
        img_show = Im.fromarray(img_as_ubyte(img_cl))
        img_show = img_show.convert('RGBA')
        self.show_img(img_show)

    # Egde detection
    def egde_detect(self):
        img = self.img
        img_float = img_as_float(img.convert('RGB'))
        img_edge = img_float.copy()
        for i in range(3):
            ch = img_float[:, :, i]
            ch_edge = canny(ch, sigma=3)
            img_edge[:, :, i] = ch_edge
        img_show = Im.fromarray(img_as_ubyte(img_edge))
        img_show = img_show.convert('RGBA')
        self.show_img(img_show)

    # Homomorphic enhance
    def homomorphic(self, d0=10, rL=0.5, rH=2, c=4, h=2.0, l=0.5):
        img = self.img
        # Convert to 4 channels
        img = img_as_float(img.convert('RGB'))
        enhanced_img = img.copy()
        for i in range(3):
            component = img[:, :, i].copy()
            height, width = component.shape

            component = np.log(component+1)
            fft_ = np.fft.fft2(component)
            fft_shift = np.fft.fftshift(fft_)
            
            # Mesh two numpy arrays representing two axis variables
            height_center = height // 2
            width_center = width // 2
            u, v = np.meshgrid(np.arange(-width_center, width_center), np.arange(-height_center, height_center))
            # Distance from center
            dis2 = u**2 + v**2
            # Filter
            homomorphic_filter_fft_shift = (rH - rL) * (1 - np.exp(-c * (dis2 / d0**2))) + rL
            enhanced_fft_shift = homomorphic_filter_fft_shift * fft_shift
            # Post-process
            enhanced_fft_shift = (h - l) * enhanced_fft_shift + l
            
            enhanced = np.fft.ifft2(np.fft.ifftshift(enhanced_fft_shift))
            enhanced = np.exp(enhanced) - 1

            # Obtain enhanced image
            enhanced = np.clip(np.real(enhanced), 0, 1)
            enhanced_img[:, :, i] = enhanced

        # Convert to PIL format
        img_show = Im.fromarray(img_as_ubyte(enhanced_img))
        img_show = img_show.convert('RGBA')
        self.show_img(img_show)

    # Increase contrast
    def contrast_add(self):
        img = self.img
        img_show = ImageEnhance.Contrast(img).enhance(1.1)
        self.show_img(img_show)

    # Reduce contrast
    def contrast_sub(self):
        img = self.img
        img_show = ImageEnhance.Contrast(img).enhance(0.9)
        self.show_img(img_show)

    # Increase brightness
    def brightness_add(self):
        img = self.img
        img_show = ImageEnhance.Brightness(img).enhance(1.1)
        self.show_img(img_show)
        
    # Reduce brightness
    def brightness_sub(self):
        img = self.img
        img_show = ImageEnhance.Brightness(img).enhance(0.9)
        self.show_img(img_show)

    # Capture from laptop camera and detect faces
    def facedetection(self):
        def capture():
            capture = cv2.VideoCapture(0)
            faces = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_alt.xml')
            # Get each frame
            while True:
                ret, frame = capture.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect faces
                face_zone = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                # Add rectangle window
                for x, y, w, h in face_zone:
                    cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=[0, 255, 0], thickness=2)
       
                # Convert to PIL format
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Im.fromarray(cv2image)
                # Convert to PhotoImage to show correctly
                images = ImageTk.PhotoImage(img)
                # Show
                self.image_zone_2.config(image=images)
                self.image_zone_2.image = images
                # Close camera if ordering
                if self.close == True:
                    break
            # Close camera and flush image
            capture.release()
            cv2.destroyAllWindows()
            self.image_zone_2.image = None
            self.close = False

        # Flush current images and start capturing
        self.image_zone_1.image = None
        self.image_zone_2.image = None
        thread = threading.Thread(target=capture)
        thread.start()

    # Set the camera state to 'close'
    def close_win(self):
        self.close = True


if __name__ == '__main__':
    gui = Gui()
    gui.mainloop()