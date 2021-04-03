import random
import numpy as np
import time as tm
import math
import matplotlib.pyplot as plt


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
total[0] = 80
print("Traffic in Road[0] is: " + str(total[0]))
total[1] = 121
print("Traffic in Road[1] is: " + str(total[1]))
total[2] = 45
print("Traffic in Road[2] is: " + str(total[2]))
total[3] = 169
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

dynamic = time[0:4].copy()

# for i in range(0,4):                       #to be while for infinite loop
#     print("\nRoad " + str(active) + " has " + is_heavy(active), end="\n")
#     print("Light " + str(active) + " is green for " + str(time[active]) + "seconds", end="\n")
#     for j in range(0,4):
#         if(active!=j):
#             print("Light " + str(j) + " is red", end="\n")
#     while time[active]>0:
#         car_dec(active)
#         time_dec(active)
#         print("\r" + "Time Left: "+ str(time[active]) + " Old Cars Left: "+ str(old_car[active]), end = "")
#         tm.sleep(1)
#     print("\nLight " + str(active) + " is red", end="\n")
#     if active == 3:
#         active = 0
#     else:
#         active += 1
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