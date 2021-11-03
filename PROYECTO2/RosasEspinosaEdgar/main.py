from Classes import *
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import math
import random
import tkinter as tk
from tkinter import *
from tkinter.ttk import Progressbar
from urllib.request import urlopen
import numpy as np


#from Classes import Drone

import os


#######
### Section 1; Energy Consumption Calculations
#######

def calculate_distance(loc1,loc2):
    """
    Currently calculating straight line distance between two points
    #eventually include the estimate for curvature of earth w/ Lat. & Lon. points.

    :param loc1: Tuple (x1, y1)
    :param loc2: Tuple (x2, y2)
    :return: dist: Float (rounded to two decimals)
    """

    x1, y1 = loc1
    x2, y2 = loc2

    return round(math.sqrt((x2-x1)**2 + (y2-y1)**2), 2)

def calculate_EC_steady1(drone, network, weather, d = None):
    """
    D'Andrea (2014)

    Simplification: Thrust = Weight = (mass * 9.81) / L/D Ratio

    ### w/o Wind

    :param drone: class obj
    :param network: class obj
    :param weather: class obj
    :param d: int (distance drone traveled, if d = None then distance to center (or waypoints????) is calculated)
    :return: (d, remainingCapacity[mAh]) : Tuple (int, float)
    """

    baseStation = network.BS
    nodes = network.nodes

    print("---------------------------------------------------")


    if d == None:
        d = calculate_distance(baseStation, nodes[0]) #Distance to center, Travel to waypoints???


    dist2 = calculate_distance(nodes[0], nodes[1])
    print("Drone travels ", dist2, "meters from ", str(nodes[0]), " to ", str(nodes[1]))

    dist3 = calculate_distance(nodes[1], baseStation)
    print("Drone travels ", dist3, "meters from ", str(nodes[1]), " to ", baseStation)

    totalDistance = d # Flight to center if d = none, else calculates distance
    print("Drone travels ", totalDistance, "meters ")


    thrust = round((drone.weight * 9.801) / drone.L_D, 2)
    energy = round((thrust * totalDistance)/3600, 2) ##In Joules, kgm^2/s^2 ->> Wh



    if drone.get_drone_name() == "EVO II Pro":
        remainingCapacity = (drone.battery - 1000*energy/ 11.1  ) #In mAh ###2s,##### 3S !
    else:
        remainingCapacity = (drone.battery - 1000 * energy / 7.4)

    ####Change this to calculate based on drone.cells param instead of if statement ^^^


    print(thrust, "Newtons of thrust needed to fly")
    print("Drone flying at  ", drone.speed, "m/s ")
    time = round(totalDistance/drone.speed/60,2)

    print(energy, "Wh lost and would take ", time," minutes cruising in flight.")
    print("drone battery remaining : ", round(remainingCapacity, 2) , "mAh")


    return(d, remainingCapacity)

def calculate_EC_steady1_DRAG(drone,network, d = None):
    """
    Uses (D'Andrea 2014) model as well as parasitic drag formula 1/2 * p * C_d * A * v^2


    :param drone: class obj
    :param network: class obj
    :param d: int (distance drone traveled, if d = None then waypoint travel distance is calculated
    """

    #print("calculating drag")
    #print("---------------------------------------------------")

    print(network.nodes)

    thrust = round( ((drone.weight * 9.801)/drone.L_D +.5*1.225*drone.c*drone.A*drone.speed*drone.speed), 2)

    if d == None:
        #d = calculate_distance(baseStation, nodes[0])
        temp_d = 0
        for i in range(len(network.nodes)):
            loc1 = network.nodes[i]

            if i == len(network.nodes) -1:  ##Last waypoint has to travel back to base station
                loc2 = network.nodes[0]
            else:
                loc2 = network.nodes[i+1] #Travel to next location

            temp_d += calculate_distance(loc1,loc2)

        d = temp_d


    energy = round((thrust * d) / 3600, 2)  ##In Joules, kgm^2/s^2 ->> Wh

    remainingCapacity = round(drone.battery - 1000 * energy / (drone.cells * 3.2))  # usable voltage range for standard lipo battery cell 3.2 - 4.2 v

    #print("---------------------------------")
    print(thrust, "Newtons of thrust needed to fly with drag")

    time = round(d / drone.speed / 60, 2)
    print(energy, "Wh lost and would take ", time, " minutes cruising in flight.")

    print("drone battery remaining : ", round(remainingCapacity, 2), "mAh")

    return(thrust, energy, time, remainingCapacity, d)

    #x = [i for i in range(totalDistance)]
    #y = [thrust * i for i in range(totalDistance)]

    #plt.plot(x,y)
    #plt.show()

def calculate_EC_steady2(drone, network, weather, d=None):
    """
    D'Andrea (2014)

    Simplification: Thrust = Weight = (mass * 9.81) / L/D Ratio * (1/omega)

    ### w/ wind

    :param drone: class obj
    :param network: class obj
    :param weather: class obj
    :param d: int (distance drone traveled, if d = None then max distance is calculated)

    :return: remaining Energy Capacity: (d, EC) Tuple
    """

    baseStation = network.BS
    # drone.position = Position(baseStation[0], baseStation[1])
    nodes = network.nodes

    print("---------------------------------------------------")

    if d == None:
        d = calculate_distance(baseStation, nodes[0])


    totalDistance = d  # * 2 # Flight there and back
    print("Drone travels ", totalDistance, "meters from ", baseStation, " to ", str(nodes[0]), "and back.")

    thrust = (1/1-(weather.wind_s/drone.speed)) * (drone.weight * 9.801) / drone.L_D

    energy = round((thrust * totalDistance) / 3600, 2)  ##In Joules, kgm^2/s^2 ->> Wh
    # energy = (drone.weight * drone.speed) / (370 * drone.L_D*.8)


    remainingCapacity = (drone.battery - 1000 * energy / (drone.cells * 3.2))

    print("---------------------------------")
    print(thrust, "Newtons of thrust needed to fly w/o wind")

    time = round(totalDistance / drone.speed / 60, 2)
    print(energy, "Wh lost and would take ", time, " minutes cruising in flight.")

    print("drone battery remaining : ", round(remainingCapacity, 2), "mAh")

    return (d, remainingCapacity)

def calculate_EC_hover(drone,network,weather, t = None):
    """

    Hasini et al. (2007)

    Simplifications: P = (m*g)**3/2 / (v * sqrt(2*n*rho*c)

    :param drone: class obj
    :param network: class obj
    :param weather: class obj
    :param t: int (time)
    :return: remaining Energy Capacity: (d, EC) Tuple
    """

    P = (drone.weight*9.801)**(3/2) * (1/(2*3.14*weather.p*.0508*.0508)**(1/2)) ###r = 2 in.
    E = 7.4 * drone.battery * 3600 / 1000

    print(P)
    if t == None:
        #Calculate max hover time given conditions

        t = round(E/P)
        print("Max time, ", t)

    energy = P*t

    remainingCapacity = (drone.battery - energy / 7.4)  # In mAh ###2s,
    return (t, remainingCapacity)

def calculate_EC_STAGES(drone,network):
    """
    #Breaking down EC into the 4 (or 3) dif. stages of flight
    #4 stages of flight: Takeoff, hover, steady, land
    + communications (FUTURE?)
    """
    return None


#######
### Section 2; Movement
#######

def test_collision(drone,network):
    """
    Function used to test the WSN boundaries and ensure drones are moving correctly
    """

    BS = network.BS
    x1 = BS[0]
    y1 = BS[1]

    print(x1, y1)

    drone.set_drone_position(Position(x1, y1))

    test_random_mov(drone, network)

def test_max_distance(drone, network):
    """
    Return the max distance a drone can travel under given conditions
    """
    X = []
    Y = []

    for d in range(100,10000):
        print(d)
        print(type(drone), print(type(network)))
        tempx, tempy = calculate_EC_steady1_DRAG(drone, network, d) ###Not using steady1 drag??????
        time = round((d/drone.speed)/60,2) #minute
        #time = round((x / drone.speed), 2)  #seconds

        safety_factor = 300 #mAH, or however much needed to return to B.S

        if tempy + safety_factor < 0:
            print("Max Distance: ", d)
            print("Max time: ", tempx)

            return(time, d)

        else:
            pass ##not at max capacity yet

        #X.append(int(tempx))
        #Y.append(int(tempy))

    #plt.plot(X,Y)
    #plt.show()

def travel(drones, network_index, weather, movement):
    """

    Run the simulation that will show the drones flight path in the WSN

    :param drone: class obj
    :param network: str
    :param weather: class obj
    :param movement: movement type
    :return: None
    """


    BS = network.BS
    x1 = BS[0]
    y1 = BS[1]

    #print(x1, y1)

      # room_type(2000, 1000, [(0,250),(1000,250),(1500,1750)] ) #2000, 1000, [(0,250),(1000,250),(1500,1750)]
    #print(drones)
    #for d in drones:
        #d.set_drone_position(Position(x1, y1))


    if movement == "1": ###Waypoint Travel
        print("here")
        test_model1(drones,network) ##only works with one drone for now

    if movement == "2": ###Random mov 1st (then random mov in wsn)
        test_random_mov(drones, network)


    if movement == "3":  ###hover?
        test_drone_movement(drones, network)
        ###Travel from BS to Center, use t.sleep(x) to stop program for x amount of seconds and invoke calculate hover method


#######
### Section 3; Bringing GUI Interface and methods together
#######


def GUI(defaultNetwork):
    """
    Opens up GUI given default network
    """
    #width = int(input("input GUI Width:"))
    #length = int(input("input GUI Length:"))

    screen1 = DialogBox()


    #anim = DroneVisualization(2, 500, 500, 3, defaultNetwork)
    #anim.done()

def run_simulation(num_drones, drone_type, speed, width, height, num_trials):
    """
    Runs num_trials trials of the simulation and returns the mean number of
    time-steps needed to clean the fraction min_coverage of the room.

    The simulation is run with num_robots robots of type robot_type, each
    with the input speed and capacity in a room of dimensions width x height
    with the dirt dirt_amount on each tile. Each trial is run in its own SimpleRoom
    with its own robots.

    num_drones:  int ( > 0)
    drone_type: class of robot to be instantiated (i.e. Cheap, Standard, Expensive)

    speed: float (> 0)
    width: int (> 0)
    height: int (> 0)

    num_trials: int ( > 0)

    """

    timeSteps = []
    totalTimeSteps = 0
    meanTimeSteps = 0

    for trial in range(num_trials):

        network = SimpleNetwork(width, height, 1, [(0,250), (500,500)])  # instantiating room
        drones = [StandardDrone("name", network, 2250, 2, 15, .5, .7, 1.1, .3) for x in range(num_drones)]  # list of robot objects

        BS = network.BS
        x1 = BS[0]
        y1 = BS[1]
        for drone in drones:
            drone.set_drone_position(Position(x1, y1))


        #anim = DroneVisualization()

        numSteps = 0

        for i in range(75): ###While Drones have enough energy
            ###Right now will be an INFINITE loop

            for drone in drones:  # each robot goes through one step
                drone.set_rand_direction()
                for i in range(5):
                    drone.update_position_noWind()

            numSteps += 1

        timeSteps.append(numSteps)  # adds the number of steps for each trial

    # calculating average number of timeSteps
    for t in timeSteps:
        totalTimeSteps += t
    meanTimeSteps = totalTimeSteps / num_trials

    return meanTimeSteps

def read_txt_files():
    x = open('Drones.txt.txt', 'r')
    y = open('Networks.txt.txt', 'r')

    x.readline()  ### Disregard the first two lines
    x.readline()

    global DRONES
    DRONES = []
    for line in x.readlines():
        temp = line.split()

        if len(temp) == 9:
            DRONES.append(
                StandardDrone(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], temp[6], temp[7], temp[8]))

    # print(drones)
    # for drone in drones:
    # print(drone.get_drone_speed())

    y.readline()  ### Disregard the first two lines
    y.readline()

    global NETWORKS
    NETWORKS = []

    for line in y.readlines():
        temp = line.split()

        if len(temp) == 4:
            NETWORKS.append(SimpleNetwork(temp[0], temp[1], temp[2], temp[3]))

        # print(networks[0].nodes)

    #print(NETWORKS)




if __name__ == '__main__':

    #defaultNetwork = SimpleNetwork(2000, 2000, 300, [(500,250),(1400,1100),(750,750),(1600,1300),(500,250)])
    #EDNetwork = SimpleNetwork(2000,2000,150,[(350,600),(900,1250),(1550,200),(400,1800),(550,1100),(2000,250),(750,450),(300,800),(1250,400),(350,600)])
    
    #EDNetwork = SimpleNetwork(2000,2000,150,[(350,600),(900,1250),(1550,200),(400,1800),(550,1100),(2000,250),(750,450),(300,800),(1250,400),(1850,1150),(150,1600),(350,600)])
    simpleNetwork = SimpleNetwork(2000, 2000, 500, [(500,250),(1400,1100),(750,750),(1600,1300),(500,250)])

    

   # droneFAST = StandardDrone("DJJ_Mini_2", simpleNetwork, 2250, 2, 60, .250, .75, .82, .016)

    #weather_noWind = Weather(16,25.6,(0,0),0)
    #weather_wind = Weather(16,25.6,(5,270),0)

    #GUI(simpleNetwork)

    #travel(droneFAST,simpleNetwork, weather_noWind, "2")
    #print("Done")


    #test_collision(droneFAST, simpleNetwork)

    
    #calculate_EC_steady1(drone1, simpleNetwork, weather_noWind)
    #calculate_EC_steady2(drone1, simpleNetwork, weather_wind)
    #calculate_EC_hover(drone1, simpleNetwork,weather_wind)
    #flag = calculate_EC_steady1_DRAG(drone1, simpleNetwork,d = 10000)

    #BS_dist = [1000, 3000, 5000, 7500, 10000, 15000, 20000]

    """
    
    for i in range(1,100000 +1):
        s = random.randint(13,68)
        drone1 = StandardDrone("Mavic_Air_2", EDNetwork, 3500, 10, s, 0.570, 0.75, .82, .0185)
        flag = calculate_EC_steady1_DRAG(drone1, EDNetwork)
        drone_tip = ['Mavic_Air_2', 'simpleNetwork', 3500, 10, s, 0.570, 0.75, 0.82, .0185]
       
        with open('Simulation24.txt', 'a+') as salida:
            salida.write('\n')
            salida.write(str(i))
            salida.write(',')
            for linea2 in drone_tip:
                salida.write(str(linea2))
                salida.write(',')
            
            for linea in flag:
                salida.write(str(linea))
                salida.write(',')
    
    

    """

    

    #drone2 = StandardDrone("DJJ_Mini_2", simpleNetwork, 2250, 3, 19, .250, .75, .82, .016)
    #print(calculate_EC_steady1_DRAG(drone2, simpleNetwork,d=1000))

    #drone3 = StandardDrone("DJJ_Mini_2", simpleNetwork1, 2250, 3, 12, .250, .75, .82, .016)
    #print(calculate_EC_steady1_DRAG(drone3, simpleNetwork1))

    #drone4 = StandardDrone("DJJ_Mini_2", simpleNetwork2, 2250, 3, 12, .250, .75, .82, .016)
    #print(calculate_EC_steady1_DRAG(drone4, simpleNetwork2))

    #drone5 = StandardDrone("DJJ_Mini_2", simpleNetwork3, 2250, 1, 12, .250, .75, .82, .016)
    # 87re3wqprint(calculate_EC_steady1_DRAG(drone5, simpleNetwork3))

    #drone1 =  StandardDrone("Mavic_DJI_Air_2s", simpleNetwork, 3500, 3, 10, .570, .75, .82, .185)
    #calculate_EC_steady1_DRAG(drone1, simpleNetwork,d = 20000)
    
    


    ###########
    ####TEST GRAPHS
    #############

    """

    weather_noWind = Weather(16, 25.6, (0, 0), 0)

    drone1 = StandardDrone("DJJ_Mini_2",simpleNetwork, 2250, 2, 15, .250, .75, .82, .016)
    drone2 = StandardDrone("DJJ_MavicAir_2", simpleNetwork, 3500, 2, 15, .570, .75, .82, .0195)
    drone3 = StandardDrone("EVO II Pro", simpleNetwork, 7100, 3, 15, 1.174, .75, .82, .195)

    drones = [drone1, drone2, drone3]
    #drones = [StandardDrone("DJJ_Mini_2", "0", 2250, 2, 70, .250, .75, .82, .016) for x in range(3)]

    for drone in drones:
        x = []
        y = []
        

        for speed in range(5, 25):
            drone.set_drone_speed(speed)
            valores = test_max_distance(drone, simpleNetwork)

            print(valores)
            
            for tempx,tempy in valores:
                x.append(tempx)
                print(x)
                y.append(tempy)

        plt.scatter(x,y)
    plt.legend()
    plt.ylabel("Max hover time (min)")
    plt.xlabel("Drone Type")

    plt.show()

    """

    """
    
    

    drones = [StandardDrone("DJJ_Mini_2","0", 2250, 1, 70, .250, .75, .82, .016) for x in range(10)]

    index = int(0)
    network = NETWORKS[index]

    tempx = network.BS[0]
    tempy = network.BS[1]

    for drone in drones:
        drone.set_drone_position(Position(tempx, tempy))

    travel(drones, network, weather_noWind, "1")
    
    """        

    ##Travel max distance, then get graphs on drones at dif speeds ... then with multiple drones ... then under dif weather conditions.