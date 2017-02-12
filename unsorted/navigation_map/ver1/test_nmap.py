from __future__ import print_function
from __future__ import division

from navigation_map import NavigationMap

nmap_conf = {
    'width': 600,
    'height': 400,
    'patch_size': 10
}
nmap_save_path = './test1.pkl'
nmap_restore_path = './test1.pkl'

nmap = NavigationMap(nmap_conf)
nmap.restore(nmap_restore_path)
nmap.edit()
#nmap.visualize()

nmap.create_energy_map(verbose=True)
nmap.visualize_energy_map()
