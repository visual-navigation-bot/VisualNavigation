import action_space.test_function as test_act_space
import env_example.test_function as test_env
import env_no_ped_v0.test_function as test_no_man_v0

'''
# test env_example
print('')
test_env.test1()
print('')
'''

'''
# test action_space
print('')
test_act_space.test_discrete()
test_act_space.test_continuous()
print('')
'''

# test environment no_man_v0
test_no_man_v0.test1()
