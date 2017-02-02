from __future__ import print_function
from __future__ import division

from navigation_map import NavigationMap

nmap_conf = {
    'width': 600,
    'height': 400,
    'patch_size': 110
}
nmap_save_path = './test1.pkl'
nmap_restore_path = './test1.pkl'

nmap = NavigationMap(nmap_conf)
nmap.edit()
print(nmap.dir_map)
nmap.visualize()
#nmap.save(nmap_save_path)
