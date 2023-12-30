from datetime import date
import re
from utils.abbr import state_dict

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