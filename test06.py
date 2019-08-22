from miniflow import *

x, y = Input(), Input()

f = Mul(x, y)

feed_dict = {x: 10, y: 5}

sorted_nodes = topological_sort(feed_dict)
output = forward_pass(f, sorted_nodes)

# NOTE: because topological_sort set the values for the `Input` nodes we could also access
# the value for x with x.value (same goes for y).
print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))

#
# from miniflow import *
#
# x, y, z = Input(), Input(), Input()
# inputs = [x, y, z]
#
# weight_x, weight_y, weight_z = Input(), Input(), Input()
# weights = [weight_x, weight_y, weight_z]
#
# bias = Input()
#
# f = Linear(inputs, weights, bias)
#
# feed_dict = {
#     x: 6,
#     y: 14,
#     z: 3,
#     weight_x: 0.5,
#     weight_y: 0.25,
#     weight_z: 1.4,
#     bias: 2
# }
#
# graph = topological_sort(feed_dict)
# output = forward_pass(f, graph)
#
print(output) # should be 12.7 with this example