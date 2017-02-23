import numpy as np

from environment import No_Ped_v0

def test1():
    step_time = 0.8
    field_size = (300,300)
    nav_path = '~/Desktop/vslab/VisualNavigation/env/env_no_ped_v0/nav_data/no_ped_v0_1.pkl'
    max_eps_len = 100
    n_eps = 1000000
    fix_action = True
    use_default_start_dest = True

    env = No_Ped_v0(step_time, field_size, nav_path)
    act_space = env.action_space
    env.display()

    start = env._random_sample_pos()
    dest = env._random_sample_pos()
    params = {
        'start': start,
        'destination': dest
    }
    env.set_params(params)

    for ep in range(n_eps):
        if use_default_start_dest:
            env.reset()
        else:
            env.reset(random_start=False,random_dest=False)
        
        if fix_action:
            f_a = act_space.sample()
            print(f_a)

        non_disc_G = 0
        for i in range(max_eps_len):
            if fix_action:
                a = f_a
            else:
                a = act_space.sample()
            obs, r, t = env.step(a)
            non_disc_G += r

            if t:
                break
        print('Episode {}: non-discount_G = {}, eps_len = {}'.format(ep, non_disc_G, i))

