from __future__ import print_function
from __future__ import division

from navigation_map import Navigation_Map_v0

nmap_conf = {
    'width': 100,
    'height': 100,
    'patch_size': 2
}
nmap_save_path = './test2.pkl'
nmap_restore_path = './test2.pkl'

nmap = Navigation_Map_v0(nmap_conf)
nmap.restore(nmap_restore_path)
#nmap.edit()
nmap.visualize()

nmap.create_energy_map(verbose=True)
nmap.visualize_energy_map()
#nmap.save(nmap_save_path)
nmap.save_bg_nmap('./no_man_v0_1')

