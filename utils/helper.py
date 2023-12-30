import cv2
import pytesseract
import imutils
import os
import shutil
from utils.lp_validation_rules import isValidLicenseNo, embassy_regex_checker, army_lp_decoder,lp_decoder,military_regex_checker
import re
from utils.abbr import state_dict,COLORS
import numpy as np
import matplotlib.pyplot as plt
from math import *
from utils.colorthief import ColorThief,convert_rgb_to_names

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
    if not os.path.exists("characters/"):
      os.mkdir("/characters/")
    else:
      shutil.rmtree('/characters/')
      os.mkdir("/characters/")

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
      cv2.imwrite("/characters/"+filename, _roi)
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
      cv2.imwrite("/characters/"+filename, _roi)
        
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