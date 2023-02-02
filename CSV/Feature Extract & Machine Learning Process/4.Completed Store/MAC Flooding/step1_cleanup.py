#!/usr/bin/env python

import pandas as pd
import sys
from functools import reduce
import socket
import struct
import ipaddress
import macaddress

filename = sys.argv[1]
file1 = pd.read_csv(filename)
file1.head(10)
file1.isnull().sum
#print(file1.isnull().sum)
# step-1 to replace all null
update_file = file1.fillna(" ")
update_file.isnull().sum()
#print (update_file.isnull().sum()) 
update_file.to_csv('update_'+filename, index = False)
# step-2 to remove all rows with null value
update_file = file1.fillna(0)
#print (update_file.isnull().sum())
# step-3 to convert tcp.flag, ip.dst, ip.src to integer
# (unused)update_file['tcp.flags'] = update_file['tcp.flags'].apply(lambda x: int(str(x), 16))
update_file['ip.dst'] = update_file['ip.dst'].apply(lambda x: int(ipaddress.IPv4Address(x)))
update_file['ip.src'] = update_file['ip.src'].apply(lambda x: int(ipaddress.IPv4Address(x)))
update_file['eth.addr'] = update_file['eth.addr'].apply(lambda x: int(macaddress.EUI48(x)))
update_file['eth.dst'] = update_file['eth.dst'].apply(lambda x: int(macaddress.EUI48(x)))
update_file.to_csv('update_'+filename, index = False)