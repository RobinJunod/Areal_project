# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2014 Bitcraze AB
#
#  Crazyflie Nano Quadcopter Client
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Simple example that connects to the first Crazyflie found, logs the Stabilizer
and prints it to the console. After 10s the application disconnects and exits.
"""
import logging
import time
from threading import Timer
import numpy as np

import cflib.crtp  # noqa
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
from my_control_real import MyController
import random

uri = uri_helper.uri_from_env(default='radio://0/50/2M/E7E7E7E715')

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


class LoggingExample:
    """
    Simple logging example class that logs the Stabilizer from a supplied
    link uri and disconnects after 5s.
    """

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """

        self._cf = Crazyflie(rw_cache='./cache')

        # Connect some callbacks from the Crazyflie API
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print('Connecting to %s' % link_uri)

        # Try to connect to the Crazyflie
        self._cf.open_link(link_uri)

        # Variable used to keep main loop occupied until disconnect
        self.is_connected = True

        # Data dictionary
        self.data = {}

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        print('Connected to %s' % link_uri)

        # The definition of the logconfig can be made before connecting
        self._lg_stab = LogConfig(name='Stabilizer', period_in_ms=100)
        self._lg_stab.add_variable('stateEstimate.x', 'FP16')
        self._lg_stab.add_variable('stateEstimate.y', 'FP16')
        self._lg_stab.add_variable('stateEstimate.vx', 'FP16')
        self._lg_stab.add_variable('stateEstimate.vy', 'FP16')
        self._lg_stab.add_variable('stabilizer.yaw', 'FP16')
        self._lg_stab.add_variable('gyro.z', 'FP16')
        self._lg_stab.add_variable('range.front')
        self._lg_stab.add_variable('range.back')
        self._lg_stab.add_variable('range.left')
        self._lg_stab.add_variable('range.right')
        self._lg_stab.add_variable('range.zrange') # [mm]

        # The fetch-as argument can be set to FP16 to save space in the log packet
        # self._lg_stab.add_variable('pm.vbat', 'FP16')

        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
        try:
            self._cf.log.add_config(self._lg_stab)
            # This callback will receive the data
            self._lg_stab.data_received_cb.add_callback(self._stab_log_data)
            # This callback will be called on errors
            self._lg_stab.error_cb.add_callback(self._stab_log_error)
            # Start the logging
            self._lg_stab.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')

        # Start a timer to disconnect in 10s
        t = Timer(50, self._cf.close_link)
        t.start()

    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives"""
        #print(f'[{timestamp}][{logconf.name}]: ', end='')

        self.data['t'] = timestamp

        for name, value in data.items():
            #print(f'{name}: {value:3.3f} ', end='')
            self.data[name] = value
        #print()

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)
        self.is_connected = False

def transform_data(data) :
    # Data dictionary
    sensor_data = {'x_global' : data['stateEstimate.x']/1,
                'y_global' : data['stateEstimate.y']/1,
                'range_down' : data['range.zrange']/1000,
                'v_forward' : data['stateEstimate.vx']/1,
                'v_left' : data['stateEstimate.vy']/1,
                'yaw' : data['stabilizer.yaw']*np.pi/180,
                'yaw_rate' : data['gyro.z']*np.pi/180,
                'range_front' : data['range.front']/1000,
                'range_left' : data['range.left']/1000,
                'range_back' : data['range.back']/1000,
                'range_right' : data['range.right']/1000}

    return sensor_data

if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    my_controller = MyController()

    le = LoggingExample(uri)
    cf = le._cf

    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

    liftoff_flag = False

    # The Crazyflie lib doesn't contain anything to keep the application alive,
    # so this is where your application should do something. In our case we
    # are just waiting until we are disconnected.
    while le.is_connected:
        time.sleep(0.01)
        sensor_data = transform_data(le.data)

        control_commands = my_controller.step_control(sensor_data)

        cf.commander.send_hover_setpoint(control_commands[0], control_commands[1], control_commands[2]*180/np.pi, control_commands[3])

        # Read sensor data including []
        #sensor_data = drone.read_sensors()

        # Control commands with [v_forward, v_left, yaw_rate, altitude]
        # ---- Select only one of the following control methods ---- #
        #control_commands = drone.action_from_keyboard()
        #control_commands = my_controller.step_control(sensor_data)

        #cf.commander.send_hover_setpoint(control_commands[0], control_commands[1], control_commands[2], control_commands[3])

    cf.commander.send_stop_setpoint()
