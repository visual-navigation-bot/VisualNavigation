from environment import No_Ped_v0

def test1():
    step_time = 0.2
    field_size = ?#TODO
    bg_path = ??#TODO
    nmap_path = ??#TODO

    env = No_Ped_v0(step_time, field_size, bg_path, nmap_path)
    env.display()

    for _ in range(50):
        env.reset()

        for _ in range(max_eps_len):
            obs, r, t = env.step()

            if t:
                break

