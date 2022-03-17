# TODO burgern: Implement local MLP based 2D hilbert map here H_L.
# H_L: (x, z) -> P_occupied, where
# (x, z) € L := [x_0, x_1] x [z_0, z_1], where
# x_0, x_1, z_0 and z_1 are boundaries, which
# restrict H_L. Therefore, it holds
#
#                    { H(x, z), if (x, z) € L
# P_occupied(x, y) = {
#                    { 0, otherwise

#            x --------- X y
#                        |
#                        |
#                        | z
#
#   lidar coordinate system
