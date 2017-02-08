from environment import Example

def test1():
    step_time = 0.2
    field_size = tuple((400,600))
    x_start = 50

    env = Example(step_time, field_size, x_start)
    act_space = env.action_space
    obs_space = env.observation_space
    
    # check Example.__str__
    print(env)
    # check Example.action_space
    print(act_space)
    # check Example.observation_space
    print(obs_space)

    # check Example.display
    env.display()

    # check agent-env-interation function of Example
    env.reset()
    for i in range(7):
        obs, r, t = env.step(1.)
        print('obs: {}, r: {}, t: {}\n'.format(obs, r, t))
        if t:
            print('Episode terminated')
            break

    # check Example.set_params (optional for subclass)
    params = {'x': 3.}
    env.set_params(params)
    env.display()
    
