import numpy as np

#### Collision detection functions ####

def orientation(p1, p2, p3):
    '''Returns 1 if points p1 p2 p3 are listed in counterclockwise order.
    Returns -1 if points are listed in clockwise order.
    Returns 0 if points are colinear.
    '''

    ccw = (p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    return 1 if ccw > 0 else (-1 if ccw < 0 else 0)

def intersecting(p1, p2, p3, p4):
    '''Returns True if line segment p1p2 intersects line segment p3p4.
    '''

    return (orientation(p1,p3,p4) != orientation(p2,p3,p4) 
            and orientation(p1,p2,p3) != orientation(p1,p2,p4))

def get_intersection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray):
    '''If line segment p1p2 and p3p4 intersect, returns point of intersection. 
    If line segments are collinear and overlapping, 
    returns the endpoint of the second line segment that is closest to p1. 
    Else, returns None.
    The collinear and overlapping case assumes p1 is not between p3 and p4.
    '''

    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]

    d1 = p2 - p1
    d2 = p4 - p3

    dc = cross(d1, d2)
    pd = p3 - p1

    m1 = cross(pd, d2) / dc
    m2 = cross(pd, d1) / dc

    # Line segments are parallel
    if dc == 0:
        # Line segments are collinear
        if cross(pd, d1) == 0:
            t1 = np.dot(pd, d1) / (d1_sq := np.dot(d1, d1))
            t2 = t1 + (p := np.dot(d1, d2)) / d1_sq

            overlapping = (t2 <= 1 and 0 <= t1) if p < 0 else (t1 <= 1 and 0 <= t2)

            if overlapping:
                mag1 = np.sqrt(np.dot(p3 - p1, p3 - p1))
                mag2 = np.sqrt(np.dot(p4 - p1, p4 - p1))

                if mag1 < mag2:
                    return p3
                else:
                    return p4
            # Line segments are collinear but not ovelapping
            else:
                return None
        # Line segments are parallel but not collinear
        else:
            return None
    # Line segments are not parallel and intersect
    elif 0 <= m1 <= 1 and 0 <= m2 <= 1:
        return p1 + (m1 * d1)
    # Line segments are not parallel and do not intersect
    else:
        return None