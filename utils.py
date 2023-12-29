from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)
from scipy.spatial import KDTree
from PIL import Image
import numpy as np
from abbr import COLORS
import matplotlib.pyplot as plt
from math import *
import re
from datetime import date
from abbr import state_dict
import cv2
import pytesseract
import imutils
import os
import shutil

def convert_rgb_to_names(rgb_tuple):   
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'closest match: {names[index]}'


class cached_property(object):
    """Decorator that creates converts a method with a single
    self argument into a property cached on the instance.
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, type):
        res = instance.__dict__[self.func.__name__] = self.func(instance)
        return res


class ColorThief(object):
    """Color thief main class."""
    def __init__(self, file):
        """Create one color thief for one image.

        :param file: A filename (string) or a file object. The file object
                     must implement `read()`, `seek()`, and `tell()` methods,
                     and be opened in binary mode.
        """
        self.image = Image.open(file)

    def get_color(self, quality=10):
        """Get the dominant color.

        :param quality: quality settings, 1 is the highest quality, the bigger
                        the number, the faster a color will be returned but
                        the greater the likelihood that it will not be the
                        visually most dominant color
        :return tuple: (r, g, b)
        """
        palette = self.get_palette(5, quality)
        return palette[0]

    def get_palette(self, color_count=10, quality=10):
        """Build a color palette.  We are using the median cut algorithm to
        cluster similar colors.

        :param color_count: the size of the palette, max number of colors
        :param quality: quality settings, 1 is the highest quality, the bigger
                        the number, the faster the palette generation, but the
                        greater the likelihood that colors will be missed.
        :return list: a list of tuple in the form (r, g, b)
        """
        image = self.image.convert('RGBA')
        width, height = image.size
        pixels = image.getdata()
        pixel_count = width * height
        valid_pixels = []
        for i in range(0, pixel_count, quality):
            r, g, b, a = pixels[i]
            # If pixel is mostly opaque and not white
            if a >= 125 and not(r==0 and g==0 and b==0):
              valid_pixels.append((r, g, b))
            
        # Send array to quantize function which clusters values
        # using median cut algorithm
        cmap = MMCQ.quantize(valid_pixels, color_count)
        return cmap.palette


class MMCQ(object):
    """Basic Python port of the MMCQ (modified median cut quantization)
    algorithm from the Leptonica library (http://www.leptonica.com/).
    """

    SIGBITS = 5
    RSHIFT = 8 - SIGBITS
    MAX_ITERATION = 1000
    FRACT_BY_POPULATIONS = 0.75

    @staticmethod
    def get_color_index(r, g, b):
        return (r << (2 * MMCQ.SIGBITS)) + (g << MMCQ.SIGBITS) + b

    @staticmethod
    def get_histo(pixels):
        """histo (1-d array, giving the number of pixels in each quantized
        region of color space)
        """
        histo = dict()
        for pixel in pixels:
            rval = pixel[0] >> MMCQ.RSHIFT
            gval = pixel[1] >> MMCQ.RSHIFT
            bval = pixel[2] >> MMCQ.RSHIFT
            index = MMCQ.get_color_index(rval, gval, bval)
            histo[index] = histo.setdefault(index, 0) + 1
        return histo

    @staticmethod
    def vbox_from_pixels(pixels, histo):
        rmin = 1000000
        rmax = 0
        gmin = 1000000
        gmax = 0
        bmin = 1000000
        bmax = 0
        for pixel in pixels:
            rval = pixel[0] >> MMCQ.RSHIFT
            gval = pixel[1] >> MMCQ.RSHIFT
            bval = pixel[2] >> MMCQ.RSHIFT
            rmin = min(rval, rmin)
            rmax = max(rval, rmax)
            gmin = min(gval, gmin)
            gmax = max(gval, gmax)
            bmin = min(bval, bmin)
            bmax = max(bval, bmax)
        return VBox(rmin, rmax, gmin, gmax, bmin, bmax, histo)

    @staticmethod
    def median_cut_apply(histo, vbox):
        if not vbox.count:
            return (None, None)

        rw = vbox.r2 - vbox.r1 + 1
        gw = vbox.g2 - vbox.g1 + 1
        bw = vbox.b2 - vbox.b1 + 1
        maxw = max([rw, gw, bw])
        # only one pixel, no split
        if vbox.count == 1:
            return (vbox.copy, None)
        # Find the partial sum arrays along the selected axis.
        total = 0
        sum_ = 0
        partialsum = {}
        lookaheadsum = {}
        do_cut_color = None
        if maxw == rw:
            do_cut_color = 'r'
            for i in range(vbox.r1, vbox.r2+1):
                sum_ = 0
                for j in range(vbox.g1, vbox.g2+1):
                    for k in range(vbox.b1, vbox.b2+1):
                        index = MMCQ.get_color_index(i, j, k)
                        sum_ += histo.get(index, 0)
                total += sum_
                partialsum[i] = total
        elif maxw == gw:
            do_cut_color = 'g'
            for i in range(vbox.g1, vbox.g2+1):
                sum_ = 0
                for j in range(vbox.r1, vbox.r2+1):
                    for k in range(vbox.b1, vbox.b2+1):
                        index = MMCQ.get_color_index(j, i, k)
                        sum_ += histo.get(index, 0)
                total += sum_
                partialsum[i] = total
        else:  # maxw == bw
            do_cut_color = 'b'
            for i in range(vbox.b1, vbox.b2+1):
                sum_ = 0
                for j in range(vbox.r1, vbox.r2+1):
                    for k in range(vbox.g1, vbox.g2+1):
                        index = MMCQ.get_color_index(j, k, i)
                        sum_ += histo.get(index, 0)
                total += sum_
                partialsum[i] = total
        for i, d in partialsum.items():
            lookaheadsum[i] = total - d

        # determine the cut planes
        dim1 = do_cut_color + '1'
        dim2 = do_cut_color + '2'
        dim1_val = getattr(vbox, dim1)
        dim2_val = getattr(vbox, dim2)
        for i in range(dim1_val, dim2_val+1):
            if partialsum[i] > (total / 2):
                vbox1 = vbox.copy
                vbox2 = vbox.copy
                left = i - dim1_val
                right = dim2_val - i
                if left <= right:
                    d2 = min([dim2_val - 1, int(i + right / 2)])
                else:
                    d2 = max([dim1_val, int(i - 1 - left / 2)])
                # avoid 0-count boxes
                while not partialsum.get(d2, False):
                    d2 += 1
                count2 = lookaheadsum.get(d2)
                while not count2 and partialsum.get(d2-1, False):
                    d2 -= 1
                    count2 = lookaheadsum.get(d2)
                # set dimensions
                setattr(vbox1, dim2, d2)
                setattr(vbox2, dim1, getattr(vbox1, dim2) + 1)
                return (vbox1, vbox2)
        return (None, None)

    @staticmethod
    def quantize(pixels, max_color):
        """Quantize.

        :param pixels: a list of pixel in the form (r, g, b)
        :param max_color: max number of colors
        """
        if not pixels:
            raise Exception('Empty pixels when quantize.')
        if max_color < 1 or max_color > 256:
            raise Exception('Wrong number of max colors when quantize.')

        histo = MMCQ.get_histo(pixels)

        # check that we aren't below maxcolors already
        if len(histo) <= max_color:
            # generate the new colors from the histo and return
            pass

        # get the beginning vbox from the colors
        vbox = MMCQ.vbox_from_pixels(pixels, histo)
        pq = PQueue(lambda x: x.count)
        pq.push(vbox)

        # inner function to do the iteration
        def iter_(lh, target):
            n_color = 1
            n_iter = 0
            while n_iter < MMCQ.MAX_ITERATION:
                vbox = lh.pop()
                if not vbox.count:  # just put it back
                    lh.push(vbox)
                    n_iter += 1
                    continue
                # do the cut
                vbox1, vbox2 = MMCQ.median_cut_apply(histo, vbox)
                if not vbox1:
                    raise Exception("vbox1 not defined; shouldn't happen!")
                lh.push(vbox1)
                if vbox2:  # vbox2 can be null
                    lh.push(vbox2)
                    n_color += 1
                if n_color >= target:
                    return
                if n_iter > MMCQ.MAX_ITERATION:
                    return
                n_iter += 1

        # first set of colors, sorted by population
        iter_(pq, MMCQ.FRACT_BY_POPULATIONS * max_color)

        # Re-sort by the product of pixel occupancy times the size in
        # color space.
        pq2 = PQueue(lambda x: x.count * x.volume)
        while pq.size():
            pq2.push(pq.pop())

        # next set - generate the median cuts using the (npix * vol) sorting.
        iter_(pq2, max_color - pq2.size())

        # calculate the actual colors
        cmap = CMap()
        while pq2.size():
            cmap.push(pq2.pop())
        return cmap


class VBox(object):
    """3d color space box"""
    def __init__(self, r1, r2, g1, g2, b1, b2, histo):
        self.r1 = r1
        self.r2 = r2
        self.g1 = g1
        self.g2 = g2
        self.b1 = b1
        self.b2 = b2
        self.histo = histo

    @cached_property
    def volume(self):
        sub_r = self.r2 - self.r1
        sub_g = self.g2 - self.g1
        sub_b = self.b2 - self.b1
        return (sub_r + 1) * (sub_g + 1) * (sub_b + 1)

    @property
    def copy(self):
        return VBox(self.r1, self.r2, self.g1, self.g2,
                    self.b1, self.b2, self.histo)

    @cached_property
    def avg(self):
        ntot = 0
        mult = 1 << (8 - MMCQ.SIGBITS)
        r_sum = 0
        g_sum = 0
        b_sum = 0
        for i in range(self.r1, self.r2 + 1):
            for j in range(self.g1, self.g2 + 1):
                for k in range(self.b1, self.b2 + 1):
                    histoindex = MMCQ.get_color_index(i, j, k)
                    hval = self.histo.get(histoindex, 0)
                    ntot += hval
                    r_sum += hval * (i + 0.5) * mult
                    g_sum += hval * (j + 0.5) * mult
                    b_sum += hval * (k + 0.5) * mult

        if ntot:
            r_avg = int(r_sum / ntot)
            g_avg = int(g_sum / ntot)
            b_avg = int(b_sum / ntot)
        else:
            r_avg = int(mult * (self.r1 + self.r2 + 1) / 2)
            g_avg = int(mult * (self.g1 + self.g2 + 1) / 2)
            b_avg = int(mult * (self.b1 + self.b2 + 1) / 2)

        return r_avg, g_avg, b_avg

    def contains(self, pixel):
        rval = pixel[0] >> MMCQ.RSHIFT
        gval = pixel[1] >> MMCQ.RSHIFT
        bval = pixel[2] >> MMCQ.RSHIFT
        return all([
            rval >= self.r1,
            rval <= self.r2,
            gval >= self.g1,
            gval <= self.g2,
            bval >= self.b1,
            bval <= self.b2,
        ])

    @cached_property
    def count(self):
        npix = 0
        for i in range(self.r1, self.r2 + 1):
            for j in range(self.g1, self.g2 + 1):
                for k in range(self.b1, self.b2 + 1):
                    index = MMCQ.get_color_index(i, j, k)
                    npix += self.histo.get(index, 0)
        return npix


class CMap(object):
    """Color map"""
    def __init__(self):
        self.vboxes = PQueue(lambda x: x['vbox'].count * x['vbox'].volume)

    @property
    def palette(self):
        return self.vboxes.map(lambda x: x['color'])

    def push(self, vbox):
        self.vboxes.push({
            'vbox': vbox,
            'color': vbox.avg,
        })

    def size(self):
        return self.vboxes.size()

    def nearest(self, color):
        d1 = None
        p_color = None
        for i in range(self.vboxes.size()):
            vbox = self.vboxes.peek(i)
            d2 = sqrt(
                pow(color[0] - vbox['color'][0], 2) +
                pow(color[1] - vbox['color'][1], 2) +
                pow(color[2] - vbox['color'][2], 2)
            )
            if d1 is None or d2 < d1:
                d1 = d2
                p_color = vbox['color']
        return p_color

    def map(self, color):
        for i in range(self.vboxes.size()):
            vbox = self.vboxes.peek(i)
            if vbox['vbox'].contains(color):
                return vbox['color']
        return self.nearest(color)


class PQueue(object):
    """Simple priority queue."""
    def __init__(self, sort_key):
        self.sort_key = sort_key
        self.contents = []
        self._sorted = False

    def sort(self):
        self.contents.sort(key=self.sort_key)
        self._sorted = True

    def push(self, o):
        self.contents.append(o)
        self._sorted = False

    def peek(self, index=None):
        if not self._sorted:
            self.sort()
        if index is None:
            index = len(self.contents) - 1
        return self.contents[index]

    def pop(self):
        if not self._sorted:
            self.sort()
        return self.contents.pop()

    def size(self):
        return len(self.contents)

    def map(self, f):
        return list(map(f, self.contents))
    

def process2(path,ch,id = 1):
  # img = plt.imread(path)
  # plt.imshow(img)
  # plt.show()
  color_thief = ColorThief(path)
  lst = color_thief.get_palette(quality=1,color_count=1)
  # lst = lst[:2]
  if id==1:
    for i in lst:
      if(ch==1):
        print(convert_rgb_to_names(i))
      elif(ch==2):
        print(closest_color(i))
      im = np.array(i).reshape(1,1,3)
      fig = plt.figure(figsize=(1,1))
      ax = fig.add_subplot(111)
      ax.axis("off")
      ax.imshow(im)
      plt.show()

  else:
    return [closest_color(i) for i in lst]
  

def closest_color(rgb):
    r, g, b = rgb
    color_diffs = []
    for color in COLORS.keys():
        cr, cg, cb = color
        color_diff = sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        color_diffs.append((color_diff, color))
    return COLORS[min(color_diffs)[1]]

def process3(path,id = 1):
  # img = plt.imread(path)
  # plt.imshow(img)
  # plt.show()
  color_thief = ColorThief(path)
  lst = color_thief.get_color(quality=1)
  if id==1:
    print(closest_color(lst))
    im = np.array(lst).reshape(1,1,3)
    fig = plt.figure(figsize=(1,1))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.imshow(im)
    plt.show()
    return lst
  else:
    return lst
  
def isValidLicenseNo(str):

    regex = ("^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
     
    p = re.compile(regex)
 
    if (str == None):
        return False
 
    if(re.search(p, str)):
        return True
    else:
        return False
    
def embassy_regex_checker(str):
  regex1 = ("UN")
  regex2 = ("CC")
  regex3 = ("CD")
     
  p1 = re.compile(regex1)
  p2 = re.compile(regex2)
  p3 = re.compile(regex3)

  if (str == None):
      return False

  if re.search(p1, str):
    return [True,"United Nations"]
  elif re.search(p2, str):
    return [True,"Consular Corps"]
  elif re.search(p3, str):
    return [True,"Diplomatic Corps"]
  else:
      return [False,""]
  
def army_lp_decoder(lp):
  if military_regex_checker(lp):
    year = lp[:2]
    vehicle_class = lp[2]
    serial_num = lp[3:9]
    check_code = lp[9]

    pre_date = date.today()
    present_year = pre_date.year
    temp_year1 = str(present_year)[:2]
    temp_year2 = str(present_year)[2:]
    
    if(int(year)<=int(temp_year2)):
      year = temp_year1+str(year)
    else:
      year = str(int(temp_year1)-1)+str(year)

    vehicle_class_dict = {
        "A":"Two-Wheel Vehicle",
        "B":"Car/Jeep",
        "C":"Light Motor Vehicle",
        "D":"Lorry Truck",
        "E":"Specialist Truck/Crane",
        "X":"Armour Vehicle"
    }
    
    if vehicle_class in vehicle_class_dict.keys():
      vehicle_class = vehicle_class_dict[vehicle_class]
    else:
      vehicle_class = "Cannot recognize"
    
    army_dict = {
        "year_of_procurement":year,
        "vehicle_class":vehicle_class,
        "serial_number":serial_num,
        "check_code":check_code
    }

    return army_dict
  

def lp_decoder(lp):

  if isValidLicenseNo(lp) and lp[:2] in state_dict.keys():
    n = len(lp)
    state = lp[:2]
    category = 0

    if state == "DL" or state == "RJ":
      if lp[3].isdigit():
        sequentialNo_district = lp[2:4]
      elif lp[3].isalpha():
        sequentialNo_district = "0"+lp[2]
        category = lp[3]
    else:
      sequentialNo_district = lp[2:4]
    
    rem = n-8
    rto_series = ""
    for i in range(4,4+rem):
      rto_series += lp[i]

    unique_num = lp[-4:]

    lp_dict = {
        "state":state_dict[state],
        "seq_no_distrct":sequentialNo_district,
        "unique_number":unique_num
    }
    if rto_series!="":
      lp_dict["rto_series"] = rto_series

    category_dict_DL = {
        "S":"Two Wheeler",
        "C":"Car/SUV",
        "E":"Electric",
        "P":"Passenger Vehicle",
        "R":"Three-Wheeled Rickshaw",
        "T":"Tourist Licensed Vehicle",
        "V":"Pick-up Truck/Van",
        "Y":"Hire Vehicle"
    }

    category_dict_RJ = {
        "P":"Passenger Vehicle",
        "C":"Car",
        "S":"Scooter",
        "G":"Goods Vehicle",
        "A":"Ambulance",
        "M":"Milk Van",
        "P":"Police"
    }

    if(category):
      if state == "DL":
        if category in category_dict_DL.keys():
          category = category_dict_DL[category]
        else:
          category = "Cannot recognize"
      elif state == "RJ":
        if category in category_dict_RJ.keys():
          category = category_dict_RJ[category]
        else:
          category = "Cannot recognize"
      
      lp_dict["category"] = category
    
    return lp_dict
  
def military_regex_checker(str):
    regex = ("^[0-9]{2}[A-Z]{1}[0-9]{6}[A-Z]{1}$")
     
    p = re.compile(regex)
 
    if (str == None):
        return False
 
    if(re.search(p, str)):
        return True
    else:
        return False
    
def process_tester21(source,rgb):

    flag_dict = {
        "army":0,
        "embassy":[0,""]
        }

    cntr = 1
    lst = list()
    image = cv2.imread(source)
    
    txt1 = pytesseract.image_to_string(image)
    txt2 = pytesseract.image_to_string(image, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7 --oem 3')
    txt3 = pytesseract.image_to_string(image, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    txt1 = re.sub('[\W_]+', '', txt1)
    txt2 = re.sub('[\W_]+', '', txt2)
    txt3 = re.sub('[\W_]+', '', txt3)
    lst.append(txt1)
    lst.append(txt2)
    lst.append(txt3)
    lst.extend([txt1[1:],txt2[1:],txt3[1:]])
    lst.extend([txt1[:10],txt2[:10],txt3[:10]])

    image = imutils.resize(image, width=300 )
    txt1 = pytesseract.image_to_string(image)
    txt2 = pytesseract.image_to_string(image, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7 --oem 3')
    txt3 = pytesseract.image_to_string(image, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    txt1 = re.sub('[\W_]+', '', txt1)
    txt2 = re.sub('[\W_]+', '', txt2)
    txt3 = re.sub('[\W_]+', '', txt3)
    lst.append(txt1)
    lst.append(txt2)
    lst.append(txt3)
    lst.extend([txt1[1:],txt2[1:],txt3[1:]])
    lst.extend([txt1[:10],txt2[:10],txt3[:10]])

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    txt1 = pytesseract.image_to_string(gray_image)
    txt2 = pytesseract.image_to_string(gray_image, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7 --oem 3')
    txt3 = pytesseract.image_to_string(gray_image, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    txt1 = re.sub('[\W_]+', '', txt1)
    txt2 = re.sub('[\W_]+', '', txt2)
    txt3 = re.sub('[\W_]+', '', txt3)
    lst.append(txt1)
    lst.append(txt2)
    lst.append(txt3)
    lst.extend([txt1[1:],txt2[1:],txt3[1:]])
    lst.extend([txt1[:10],txt2[:10],txt3[:10]])

    gray = cv2.resize(gray_image, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    txt1 = pytesseract.image_to_string(gray)
    txt2 = pytesseract.image_to_string(gray, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7 --oem 3')
    txt3 = pytesseract.image_to_string(gray, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    txt1 = re.sub('[\W_]+', '', txt1)
    txt2 = re.sub('[\W_]+', '', txt2)
    txt3 = re.sub('[\W_]+', '', txt3)
    lst.append(txt1)
    lst.append(txt2)
    lst.append(txt3)
    lst.extend([txt1[1:],txt2[1:],txt3[1:]])
    lst.extend([txt1[:10],txt2[:10],txt3[:10]])

    blur = cv2.medianBlur(gray, 5)
    txt1 = pytesseract.image_to_string(blur)
    txt2 = pytesseract.image_to_string(blur, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7 --oem 3')
    txt3 = pytesseract.image_to_string(blur, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    txt1 = re.sub('[\W_]+', '', txt1)
    txt2 = re.sub('[\W_]+', '', txt2)
    txt3 = re.sub('[\W_]+', '', txt3)
    lst.append(txt1)
    lst.append(txt2)
    lst.append(txt3)
    lst.extend([txt1[1:],txt2[1:],txt3[1:]])
    lst.extend([txt1[:10],txt2[:10],txt3[:10]])

    blur = cv2.bilateralFilter(gray,9,75,75)
    txt1 = pytesseract.image_to_string(blur)
    txt2 = pytesseract.image_to_string(blur, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7 --oem 3')
    txt3 = pytesseract.image_to_string(blur, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    txt1 = re.sub('[\W_]+', '', txt1)
    txt2 = re.sub('[\W_]+', '', txt2)
    txt3 = re.sub('[\W_]+', '', txt3)
    lst.append(txt1)
    lst.append(txt2)
    lst.append(txt3)
    lst.extend([txt1[1:],txt2[1:],txt3[1:]])
    lst.extend([txt1[:10],txt2[:10],txt3[:10]])

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    txt1 = pytesseract.image_to_string(blur)
    txt2 = pytesseract.image_to_string(blur, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7 --oem 3')
    txt3 = pytesseract.image_to_string(blur, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    txt1 = re.sub('[\W_]+', '', txt1)
    txt2 = re.sub('[\W_]+', '', txt2)
    txt3 = re.sub('[\W_]+', '', txt3)
    lst.append(txt1)
    lst.append(txt2)
    lst.append(txt3)
    lst.extend([txt1[1:],txt2[1:],txt3[1:]])
    lst.extend([txt1[:10],txt2[:10],txt3[:10]])

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 199, 5)
    # ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
  

    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
  
    if rgb == 0:
      roi = cv2.bitwise_not(dilation)
    else:
      roi = dilation
    
    txt1 = pytesseract.image_to_string(roi)
    txt2 = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7 --oem 3')
    txt3 = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    txt1 = re.sub('[\W_]+', '', txt1)
    txt2 = re.sub('[\W_]+', '', txt2)
    txt3 = re.sub('[\W_]+', '', txt3)
    
    lst.append(txt1)
    lst.append(txt2)
    lst.append(txt3)
    lst.extend([txt1[1:],txt2[1:],txt3[1:]])
    lst.extend([txt1[:10],txt2[:10],txt3[:10]])

    try:
      contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
      ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    im2 = gray.copy()
    # Normal Lp checker
    for j in lst:
      # print(j,end=" ")
      if(isValidLicenseNo(j) and j[:2] in state_dict.keys()):
        print("License Plate #: ",j)
        return [j,flag_dict]

    plate_num = ""
    
    cntr = len(plate_num)+1
    ctr = 0
    if not os.path. exists("/content/characters/"):
      !mkdir characters
    else:
      shutil.rmtree('/content/characters/')
      !mkdir characters

    for cnt in sorted_contours:
      x,y,w,h = cv2.boundingRect(cnt)
      height, width = im2.shape
      # if height of box is not tall enough relative to total height then skip
      if height / float(h) > 3.8: continue

      ratio = h / float(w)
      # # if height to width ratio is less than 1.5 skip
      if ratio<0.65 or ratio>4.28: continue

      # # if width is not wide enough relative to total width then skip
      # if width / float(w) > 15: continue

      area = h * w
      # if area is less than 100 pixels skip
      if area < 4000: continue

      # draw the rectangle
      rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
      
      # # grab character region of image
      roi = thresh[y-5:y+h+5, x-5:x+w+5]
      
      # # perfrom bitwise not to flip image to black text on white background
      if rgb == 0:
        roi = cv2.bitwise_not(roi)
        
      _roi = cv2.bilateralFilter(roi,9,75,75)
    
      filename = str(ctr)+'roi.jpg'
      cv2.imwrite("/content/characters/"+filename, _roi)
      ctr+=1

      try:
        text = pytesseract.image_to_string(_roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
        clean_text = re.sub('[\W_]+', '', text)

        if(clean_text == ""):
          _roi = cv2.medianBlur(roi, 5)
          clean_text = pytesseract.image_to_string(_roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10 --oem 3')
          clean_text = re.sub('\n', '',clean_text)
        if clean_text == "":
          clean_text = "$"
        
        plate_num += clean_text
      except: 
          text = None
  
    plate_num1 = plate_num[:2]
    plate_num_len = len(plate_num)
    cntr = 0
    ctr = 0
    plate_num = ""
    for cnt in sorted_contours:
      x,y,w,h = cv2.boundingRect(cnt)
      height, width = im2.shape
      # if height of box is not tall enough relative to total height then skip
      if height / float(h) > 3.8: continue

      ratio = h / float(w)
      # # if height to width ratio is less than 1.5 skip
      if ratio<0.65 or ratio>4.28: continue

      # # if width is not wide enough relative to total width then skip
      # if width / float(w) > 15: continue

      area = h * w
      # if area is less than 4000 pixels skip
      if area < 4000: continue

      # draw the rectangle
      rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
      
      # # grab character region of image
      roi = thresh[y-5:y+h+5, x-5:x+w+5]
      
      # # perfrom bitwise not to flip image to black text on white background
      if rgb == 0:
        roi = cv2.bitwise_not(roi)
            
      _roi = cv2.bilateralFilter(roi,9,75,75)
     

      filename = str(ctr)+'roi.jpg'
      cv2.imwrite("/content/characters/"+filename, _roi)
        
      ctr+=1
      cntr+=1

      if(plate_num_len not in [9,10] or (not plate_num1.isupper())):
          try:
              text = pytesseract.image_to_string(_roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
              clean_text = re.sub('[\W_]+', '', text)

              if(clean_text == ""):
                _roi = cv2.medianBlur(roi, 5)
                clean_text = pytesseract.image_to_string(_roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10 --oem 3')
                clean_text = re.sub('\n', '',clean_text)
              if clean_text == "":
                clean_text = "$"
             
              plate_num += clean_text
          except: 
              text = None
      else:
        try:
          # clean tesseract text by removing any unwanted blank spaces
          if(cntr==1):
            text = pytesseract.image_to_string(_roi, config='-c tessedit_char_whitelist=ABCDGHJKLMNOPRSTUW --psm 8 --oem 3')
          elif(cntr == 2):
            text = pytesseract.image_to_string(_roi, config='-c tessedit_char_whitelist=ABDGHJKLNPRSYZ --psm 8 --oem 3')
          elif(cntr==3):
            text = pytesseract.image_to_string(_roi, config='-c tessedit_char_whitelist=0123456789 --psm 8 --oem 3')
          elif(cntr>plate_num_len-4):
            text = pytesseract.image_to_string(_roi, config='-c tessedit_char_whitelist=0123456789 --psm 8 --oem 3')
          else:
            text = pytesseract.image_to_string(_roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
            
          clean_text = re.sub('[\W_]+', '', text)
          
          plate_num+=clean_text
          if(clean_text == ""):
            _roi = cv2.medianBlur(roi, 5)
            if(cntr==1):
              clean_text = pytesseract.image_to_string(_roi, config='-c tessedit_char_whitelist=ABCDGHJKLMNOPRSTUW --psm 10 --oem 3')
            elif(cntr == 2):
              clean_text = pytesseract.image_to_string(_roi, config='-c tessedit_char_whitelist=ABDGHJKLNPRSYZ --psm 10 --oem 3')
            elif(cntr==3):
              clean_text = pytesseract.image_to_string(_roi, config='-c tessedit_char_whitelist=0123456789 --psm 10 --oem 3')
            elif(cntr>plate_num_len-4):
              clean_text = pytesseract.image_to_string(_roi, config='-c tessedit_char_whitelist=0123456789 --psm 10 --oem 3')
            else:
              clean_text = pytesseract.image_to_string(_roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10 --oem 3')
              clean_text = re.sub('\n', '',clean_text)
          if clean_text == "":
            clean_text = "$"
            
            plate_num += clean_text
        except:
            text = None
        
    if plate_num != None:
        print("License Plate #: ", plate_num)
     
    return [plate_num,flag_dict]

def recog(path_dir):
  d_info = dict()
  roi = -1
  path = path_dir

  for i in os.listdir(path):
    id = i

    d_info[id]={
        "lp_color":"",
        "lp_number":"",
        "lp_class":""
    }

    img = cv2.imread(path+"/"+str(i))
    cv2.imshow("License Plate",img)

    # print("Best 3 colors")
    # process2(path+"/"+str(i),2,1)
    print("dominant color")
    process3(path+"/"+str(i),1)
    lst = process2(path+"/"+str(i),2,2)
    rgb = process3(path+"/"+str(i),2)
    dominant = closest_color(rgb)
    
    d_info[id]["lp_color"]=lst

    r,g,b = rgb
    # print(r,g,b)
    brightness = int(sqrt(0.241*(r**2)+0.691*(g**2)+0.068*(b**2)))
    # print(brightness)
    if brightness<120:
      roi = 1
    else:
      roi = 0

    if dominant == "Red":
      d_info[id]["lp_class"]="Unsold"
      roi = 1
    elif(set(["White","Black"]).issubset(lst)) or dominant == "White":
      d_info[id]["lp_class"]="Private"
      roi = 0
    elif(dominant == "Black" and (lst[1] in ["Yellow","Red"] or lst[2] in ["Yellow","Red"])):
      d_info[id]["lp_class"]="Rental Commercial"
      roi = 1
    elif ((lst[1] == "Black" and (lst[0] == "Yellow" or lst[0] == "Light Green")) or (lst[0] == "Black" and (lst[1] == "Yellow" or lst[1] == "Light Green"))):
      d_info[id]["lp_class"]="Commercial"
      roi = 0
    elif ("Dark Green" in lst and "White" in lst):
      d_info[id]["lp_class"]="Electric Private"
      roi = 1
    elif ("Dark Green" in lst and "Yellow" in lst):
      d_info[id]["lp_class"]="Electric Commercial"
      roi = 1
    else:
      d_info[id]["lp_class"]="Can't Identify"

    
    try:
      lst = process_tester21(path+"/"+str(i),roi)
      # print(lst)
      plate_num = lst[0]
      flag_dict = lst[1]

      d_info[id]["lp_number"] = plate_num

      if(flag_dict["army"] == 1):
        d_info[id]["lp_class"]="Indian Armed Forces"
      elif(flag_dict["embassy"][0] == 1):
        d_info[id]["lp_class"]=flag_dict["embassy"][1]

      print(d_info[id])

      if military_regex_checker(plate_num):
        print(army_lp_decoder(d_info[id]["lp_number"]))
      else:
        print(lp_decoder(d_info[id]["lp_number"]))

    except:
      pass