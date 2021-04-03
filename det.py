import cv2
from os import walk
import os
import numpy as np
import time as tm
import logging
import math
import re
import pandas as pd
import matplotlib.pyplot as plt
import random



loc = os.path.abspath('')
src_file = loc+'/inputs/625_201708101101.mp4'
ret_array = []







class Vehicle():
    def __init__(self, number, position):
        self.number = number
        self.init_fr = 0
        self.frames_seen = 0
        self.counted = False
        self.direc = 0
        self.positions = [position]
    @property
    def last_position(self):
        return self.positions[-1]
    @property
    def last_position2(self):
        return self.positions[-2]

    def add_position(self, new_position):
        self.positions.append(new_position)
        self.init_fr = 0
        self.frames_seen += 1

    def draw(self, final_fr):
        for point in self.positions:
            cv2.circle(final_fr, point, 2, (255, 255, 255), 1)
            cv2.polylines(final_fr, [np.int32(self.positions)], False, (50,205,50), 1)


class Counter(object):
    def __init__(self, shape, divider):
        # self.log = logging.getLogger("vehicle_counter")

        self.height, self.width = shape
        self.divider = divider

        self.vehicles = []
        self.next_number = 0
        self.vehicle_count = 0
        self.vehicle_LHS = 0
        self.vehicle_RHS = 0
        self.max_unseen_frames = 10


    @staticmethod
    def get_vector(a, b):
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])

        distance = math.sqrt(dx**2 + dy**2)

        if dy > 0:
            angle = math.degrees(math.atan(-dx/dy))
        elif dy == 0:
            if dx < 0:
                angle = 90.0
            elif dx > 0:
                angle = -90.0
            else:
                angle = 0.0
        else:
            if dx < 0:
                angle = 180 - math.degrees(math.atan(dx/dy))
            elif dx > 0:
                angle = -180 - math.degrees(math.atan(dx/dy))
            else:
                angle = 180.0        

        return distance, angle, dx, dy 


    @staticmethod
    def is_valid(a, b):
        distance, angle, _, _ = a
        threshold_distance = 12.0
        return (distance <= threshold_distance)


    def update_vehicle(self, vehicle, matches):
        for i, match in enumerate(matches):
            _, centroid = match
            vector = self.get_vector(vehicle.last_position, centroid)
            if vehicle.frames_seen > 2:
                last_vector = self.get_vector(vehicle.last_position2, vehicle.last_position)
                Deviation = abs(last_vector[1]-vector[1])
            else:
                Deviation = 0

            if self.is_valid(vector, Deviation):    
                vehicle.add_position(centroid)
                vehicle.frames_seen += 1
                if vector[3] > 0:
                    vehicle.direc = 1
                elif vector[3] < 0:
                    vehicle.direc = -1
                return i     
        vehicle.init_fr += 1


    def update_count(self, matches, final_fr = None):
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches)
            if i is not None:
                del matches[i]                  #delete previous contours
        for match in matches:
            _, centroid = match
            new_vehicle = Vehicle(self.next_number, centroid)
            self.next_number += 1
            self.vehicles.append(new_vehicle)

        for vehicle in self.vehicles:
            if not vehicle.counted and (((vehicle.last_position[1] > self.divider) and (vehicle.direc == 1)) or
                                          ((vehicle.last_position[1] < self.divider) and (vehicle.direc == -1))) and (vehicle.frames_seen > 6):

                vehicle.counted = True
                if ((vehicle.last_position[1] > self.divider) and (vehicle.direc == 1) and (vehicle.last_position[0] >= (int(frame_w/2)-10))):
                    self.vehicle_RHS += 1
                    self.vehicle_count += 1
                elif ((vehicle.last_position[1] < self.divider) and (vehicle.direc == -1) and (vehicle.last_position[0] <= (int(frame_w/2)+10))):
                    self.vehicle_LHS += 1
                    self.vehicle_count += 1
        if final_fr is not None:
            for vehicle in self.vehicles:
                vehicle.draw(final_fr)
                
            # LHS
            # cv2.putText(final_fr, ("LH Lane: %02d" % self.vehicle_LHS), (12, 56)
            #     , cv2.FONT_HERSHEY_PLAIN, 1.2, (127,255, 255), 2)
            # RHS
            cv2.putText(final_fr, ("Right Side: %02d" % self.vehicle_RHS), (200, 80), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 2)







camera = re.match(r".*/(\d+)_.*", src_file)
camera = camera.group(1)

cap = cv2.VideoCapture(src_file)

bg_img = []
for (_, _, filenames) in walk(loc+"/backgrounds/"):
    bg_img.extend(filenames)
    break

if camera+"_bg.jpg" in bg_img:
    bg = loc+"/backgrounds/"+camera+"_bg.jpg"
    default_bg = cv2.imread(bg)
    default_bg = cv2.cvtColor(default_bg, cv2.COLOR_BGR2HSV)
    (_,avgSat,default_bg) = cv2.split(default_bg)
    avg = default_bg.copy().astype("float")
else:
    avg = None 
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_h)
print(frame_w)


mask = np.zeros((frame_h,frame_w), np.uint8)
mask[100:,182:] = 255
# mask[:,:] = 255
# mask[:100, :] = 0
# mask[230:, 160:190] = 0
# mask[170:230,170:190] = 0
# mask[140:170,176:190] = 0 
# mask[:,:] = 255
# mask[100:140,176:182] = 0
print(mask)
THRESHOLD_SENSITIVITY = 40
ret_array.append(THRESHOLD_SENSITIVITY)
CONTOUR_WIDTH = 21
CONTOUR_HEIGHT = 16#21
DEFAULT_AVERAGE_WEIGHT = 0.01
INITIAL_AVERAGE_WEIGHT = DEFAULT_AVERAGE_WEIGHT / 50
LINE_THICKNESS = 1
SMOOTH = max(2,int(round((CONTOUR_WIDTH**0.5)/2,0)))


blobs = []
car_counter = None 
frame_no = 0

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
total_cars = 0

start_time = tm.time()
ret, frame = cap.read()

black = np.zeros((frame_h,frame_w))

while ret:    
    ret, frame = cap.read()
    frame_no = frame_no + 1
    
    if ret and frame_no < total_frames:


# Shadow removal

        rgb_planes = cv2.split(frame)

        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes) 
        # cv2.imshow("result", result)  
        # cv2.imshow("result_norm",result_norm)  

# conversion to grayscale using bilateral filter

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow("HSV",frame)
        (_,_,grayFrame) = cv2.split(frame)
        # cv2.imshow("Value plane",grayFrame)
        grayFrame = cv2.bilateralFilter(grayFrame, 11, 21, 21)
        # cv2.imshow("bilateral filter",grayFrame)
        
# Foreground extraction
        
        differenceFrame = cv2.GaussianBlur(grayFrame, (5, 5), 0)
        # cv2.imshow("difference", differenceFrame)
        differenceFrame = cv2.absdiff(grayFrame, cv2.convertScaleAbs(avg))
        # cv2.imshow("absdiff", differenceFrame)
        differenceFrame = cv2.GaussianBlur(differenceFrame, (5, 5), 0)
        # cv2.imshow("difference", differenceFrame)

# Thresholding using otsu binarization

        retval, _ = cv2.threshold(differenceFrame, 0, 255,
                                  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret_array.append(retval)
        
        if frame_no < 10:
            ret2, thresholdImage = cv2.threshold(differenceFrame, 
                                                 int(np.mean(ret_array)*0.9),
                                                 255, cv2.THRESH_BINARY)
            # cv2.imshow("1st tthresh",thresholdImage)
        else:
            ret2, thresholdImage = cv2.threshold(differenceFrame, 
                                             int(np.mean(ret_array[-10:-1])*0.9),
                                             255, cv2.THRESH_BINARY)
            # cv2.imshow("1st tthresh",thresholdImage)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SMOOTH, SMOOTH))
        # Fill any small holes
        thresholdImage = cv2.morphologyEx(thresholdImage, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("small holes",thresholdImage)
        # Remove noise
        thresholdImage = cv2.morphologyEx(thresholdImage, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("remove noise",thresholdImage)
        # Dilate to merge adjacent blobs
        thresholdImage = cv2.dilate(thresholdImage, kernel, iterations = 2)
        # cv2.imshow("merge blobs",thresholdImage)
        # apply mask
        thresholdImage = cv2.bitwise_and(thresholdImage, thresholdImage, mask = mask)
        # cv2.imshow("masking", thresholdImage)
        threshout = cv2.cvtColor(thresholdImage, cv2.COLOR_GRAY2BGR)
        # cv2.imshow("threshout", threshout)

        contours, hierarchy = cv2.findContours(thresholdImage, 
                                                  cv2.RETR_EXTERNAL, 
                                                  cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            for (i, contour) in enumerate(contours):    
                (x, y, w, h) = cv2.boundingRect(contour)
                contour_valid = (w > CONTOUR_WIDTH) and (h > CONTOUR_HEIGHT)
                
                if not contour_valid:
                    continue
                
                center = (int(x + w/2), int(y + h/2))
                blobs.append(((x, y, w, h), center))
                    
        for (i, match) in enumerate(blobs):
            contour, centroid = match
            x, y, w, h = contour
            cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (50,205,50), 2)
            cv2.circle(frame, centroid, 2, (255, 0, 255), -1)
        
        if car_counter is None:
            print("Creating vehicle counter...")
            car_counter = Counter(frame.shape[:2], 2*frame.shape[0] / 3)
        car_counter.update_count(blobs, frame)
        current_count = car_counter.vehicle_RHS

        elapsed_time = tm.time()-start_time
        # print("-- %s seconds --" % round(elapsed_time,2))

        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        if current_count > total_cars:
            cv2.line(frame, (0, int(2*frame_h/3)),(frame_w, int(2*frame_h/3)),
                 (0,255,0), 2*LINE_THICKNESS)
        else:
            cv2.line(frame, (0, int(2*frame_h/3)),(frame_w, int(2*frame_h/3)),
             (0,0,255), LINE_THICKNESS)
        total_cars = current_count  

        cv2.line(frame, (0, 100),(frame_w, 100), (0,0,0), LINE_THICKNESS)
        
        cv2.imshow("preview", frame)
        
        if cv2.waitKey(1)==27:
            break
    else:
        break
print(car_counter.vehicle_RHS)

cv2.destroyAllWindows()
cap.release()


total = np.zeros(8)
time = np.zeros(8)
rate = np.array([20,12,8])
dept = np.array([60,54,43])
new_car = np.zeros(8)
old_car = np.zeros(8)
x = ["red" for i in range(8)]
light_stat = np.array(x)

def time_cal(total):
    for i in range(0,4):
        if total[i] > 140:
            time[i] = math.ceil(total[i]/25 )           
        elif total[i] > 70:
            time[i] = math.ceil(total[i]/20 )        
        else:
            time[i] = math.ceil(total[i]/15 )

# for i in range(0,4):
#     total[i] = int(input("Enter initial no of vehicle in " + str(i) + " light: "))
#     if(total[i]>200):
#         total[i] = 200
total[0] = random.randint(25,200)
print("Traffic in Road[0] is: " + str(total[0]))
total[1] = random.randint(25,200)
print("Traffic in Road[1] is: " + str(total[1]))
total[2] = random.randint(25,200)
print("Traffic in Road[2] is: " + str(total[2]))
total[3] = random.randint(25,200)
print("Traffic in Road[3] is: " + str(total[3]))

def is_heavy(i):
    if total[i] > 140:
        return "heavy traffic"
    elif total[i] > 70:
        return "medium traffic"
    else:
        return "low traffic"

time_cal(total)

old_car = total.copy()
active = time.argmax()
def time_dec(active):
    time[active] -= 1


def car_dec(active):
    if total[active] > 140:
        old_car[active] -= 25          
    elif total[active] > 70:
        old_car[active] -= 20        
    else:
        old_car[active] -= 15
    if old_car[active] < 0:
        old_car[active] = 0

dynamic = time[0:4].copy()

for i in range(0,4):                       #to be while for infinite loop
    print("\nRoad " + str(active) + " has " + is_heavy(active), end="\n")
    print("Light " + str(active) + " is green for " + str(time[active]) + "seconds", end="\n")
    for j in range(0,4):
        if(active!=j):
            print("Light " + str(j) + " is red", end="\n")
    while time[active]>0:
        car_dec(active)
        time_dec(active)
        print("\r" + "Time Left: "+ str(time[active]) + " Old Cars Left: "+ str(old_car[active]), end = "")
        tm.sleep(1)
    print("\nLight " + str(active) + " is red", end="\n")
    if active == 3:
        active = 0
    else:
        active += 1
old_car = 0
def insert_data_labels(bars):
	for bar in bars:
		bar_height = bar.get_height()
		ax.annotate('{0:.0f}'.format(bar.get_height()),
			xy=(bar.get_x() + bar.get_width() / 2, bar_height),
			xytext=(0, 3),
			textcoords='offset points',
			ha='center',
			va='bottom'
		)
static = [6,6,6,6]
roads = ["Road 0","Road 1","Road 2","Road 3"]
indx = np.arange(4)
bar = 0.25
fig, ax = plt.subplots()
st = ax.bar(indx-bar/2, static, bar, label = "Static timing")
dy = ax.bar(indx+bar/2, dynamic, bar, label = "Dynamic timing")
ax.set_xticks(indx + bar / 2)
ax.set_xticklabels(roads)
ax.set_ylabel('Time of green light')
ax.legend()

insert_data_labels(st)
insert_data_labels(dy)

plt.show()

remaining = []
for i in range(0,4):
    if total[i] > 150:
        remaining.append(total[i]%150)           
    elif total[i] > 120 and total[i]  < 140:
        remaining.append(total[i]%120)       
    else:
        remaining.append(0)
print("After one complete cycle\n")
indx = np.arange(4)
bar = 0.25
fig, ax = plt.subplots()
st = ax.bar(indx-bar/2, remaining, bar, label = "Static timing")
dy = ax.bar(indx+bar/2, old_car, bar, label = "Dynamic timing")
ax.set_title("Vehicles Remaining After 1st Cycle")
ax.set_xticks(indx + bar / 2)
ax.set_xticklabels(roads)
ax.set_ylabel('Number of cars left')
ax.legend()

insert_data_labels(st)
insert_data_labels(dy)

plt.show()


