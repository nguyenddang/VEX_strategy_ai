import math 
def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi

