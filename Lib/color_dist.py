
import numpy as np

color_dist = {
              'forward_black_blind_path': {'Lower': np.array([0, 0, 0]), 'Upper': np.array([180, 200, 43])},
    	      'chin_black_path': {'Lower': np.array([113, 14, 21]), 'Upper': np.array([161, 186, 142])}
             }

# H:0~179
# S:0~255
# V:0~255

# np.array([120, 0, 0]), 'Upper': np.array([180, 255, 120])}

# (22, 99, 49)
# (151, 181, 38), (153, 59, 90)

# 竹子, (11, 104, 59)
# 黑道, (151, 181, 38), (11, 224, 41)
# 反光, (126, 22, 115)
# 边缘草地, (128, 46, 99)


