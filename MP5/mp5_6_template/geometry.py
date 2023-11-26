# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy
import math


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
        
    """
    r=alien.get_width()

    if alien.is_circle()==False:
        myseg=(alien.get_head_and_tail()[0], alien.get_head_and_tail()[1])
        for wall in walls:
            w=((wall[0],wall[1]),(wall[2],wall[3]))
            if segment_distance(myseg, w)<=r:
                return True
    
    else:
        point=alien.get_centroid()
        for wall in walls:
            w=((wall[0],wall[1]),(wall[2],wall[3]))
            if point_segment_distance(point, w)<=r:
                return True
            
    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """

    r=alien.get_width()
    pos=alien.get_centroid()
    
    if alien.is_circle()==True:
        if pos[0]-r<=0 or r+pos[0]>=window[0]:
            return False
        elif pos[1]-r<=0 or r+pos[1]>=window[1]:
            return False
        
    if alien.get_head_and_tail()[0][0]<alien.get_head_and_tail()[1][0] or alien.get_head_and_tail()[0][1]>alien.get_head_and_tail()[1][1]:
        head=alien.get_head_and_tail()[0]
        tail=alien.get_head_and_tail()[1]
    
    else:
        head=alien.get_head_and_tail()[1]
        tail=alien.get_head_and_tail()[0]

    #vertical
    if head[1]>tail[1]:
        if pos[0]-r<=0 or r+pos[0]>=window[0]:
            return False
        elif head[1]+r>=window[1] or tail[1]-r<=0:
            return False
    #horizontal
    else:
        if tail[0]+r>=window[0] or head[0]-r<=0:
            return False
        elif pos[1]+r>=window[1] or pos[1]-r<=0:
            return False

    return True



def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
            clockwise -> positive sin
            counterclockwise -> negative sin
            pick start and end point and make it as vector
            and start at the start point and end point at the point and make as vector
            then get alpha between them
            So check the first sign of the sine alpha and check if all other three sign of the sine are equal
    """
    a=polygon[0]
    b=polygon[1]
    c=polygon[2]
    d=polygon[3]

    if (a[0]==b[0]==c[0]==d[0] or a[1]==b[1]==c[1]==d[1]):
        if (point_segment_distance(point, (a,b))==0 or point_segment_distance(point, (b,c))==0 or point_segment_distance(point, (c,d))==0 or point_segment_distance(point, (d,a))==0):
            return True
        else:
            return False
    if (point_segment_distance(point, (a,b))==0 or point_segment_distance(point, (b,c))==0 or point_segment_distance(point, (c,d))==0 or point_segment_distance(point, (d,a))==0):
        return True
    
    ab1=(b[0]-a[0])
    ab2=(b[1]-a[1])
    ab_l=dist(a,b)
    ap_l=dist(a,point)
    ap1=(point[0]-a[0])
    ap2=(point[1]-a[1])
    sin_ap=(ab1*ap2-ab2*ap1)/(ab_l*ap_l)

    bc1=(c[0]-b[0])
    bc2=(c[1]-b[1])
    bc_l=dist(b,c)
    bp_l=dist(b,point)
    bp1=(point[0]-b[0])
    bp2=(point[1]-b[1])
    sin_bp=(bc1*bp2-bc2*bp1)/(bc_l*bp_l)
    if sin_ap*sin_bp<0:
        # print(point, b, c)
        return False
    
    cd1=(d[0]-c[0])
    cd2=(d[1]-c[1])
    cd_l=dist(c,d)
    cp_l=dist(c,point)
    cp1=(point[0]-c[0])
    cp2=(point[0]-c[1])
    sin_cp=(cd1*cp2-cd2*cp1)/(cd_l*cp_l)
    if sin_ap*sin_cp<0:
        # print(point, c, d)
        return False
    
    da1=(a[0]-d[0])
    da2=(a[1]-d[1])
    da_l=dist(d,a)
    dp_l=dist(d,point)
    dp1=(point[0]-d[0])
    dp2=(point[1]-d[1])
    sin_dp=(da1*dp2-da2*dp1)/(da_l*dp_l)
    if sin_ap*sin_dp<0:
        # print(point, d, a)
        return False
    return True


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    center=alien.get_centroid()
    x_move=waypoint[0]-center[0]
    y_move=waypoint[1]-center[1]
    r=alien.get_width()
    if x_move==0 and y_move==0:
        return does_alien_touch_wall(alien, walls)
    path=(center, (center[0]+x_move, center[1]+y_move))

    if alien.is_circle()==True:
        for wall in walls:
            w=((wall[0],wall[1]),(wall[2],wall[3]))
            if(do_segments_intersect(path, w)):
                # print(w)
                return True
            if segment_distance(w, path)<=r:
                # print(w)
                return True

    else:
        if alien.get_head_and_tail()[0][0]<alien.get_head_and_tail()[1][0] or alien.get_head_and_tail()[0][1]>alien.get_head_and_tail()[1][1]:
            head=alien.get_head_and_tail()[0]
            tail=alien.get_head_and_tail()[1]
    
        else:
            head=alien.get_head_and_tail()[1]
            tail=alien.get_head_and_tail()[0]

        #vertical
        if head[1]>tail[1]: 
            polygon=((head[0],head[1]+r),(head[0]+x_move,head[1]+r+y_move), (tail[0]+x_move,tail[1]-r+y_move), (tail[0], tail[1]-r))
        #horizontal
        else:
            polygon=((head[0]-r, head[1]), (tail[0]+r, tail[1]), (tail[0]+r+x_move, tail[1]+y_move), (head[0]-r+x_move, head[1]+y_move))

        for wall in walls:
            w=((wall[0],wall[1]),(wall[2],wall[3]))
            if(do_segments_intersect(path, w)):
                # print(w)
                return True
            if is_point_in_polygon(w[0], polygon)==True or is_point_in_polygon(w[1], polygon)==True:
                # print(w)
                return True
            if segment_distance(w, (head, tail))<=r or segment_distance(w, ((head[0]+x_move,head[1]+y_move), (tail[0]+x_move,tail[1]+y_move)))<=r:
                return True
        # myseg=(head, tail)
        # newseg=((myseg[0][0]+x_move, myseg[0][1]+y_move), (myseg[1][0]+x_move, myseg[1][1]+y_move))
        # polygon=(myseg[0], newseg[0], myseg[1], newseg[1])

        # for wall in walls:
        #     w=((wall[0],wall[1]),(wall[2],wall[3]))
        #     if(do_segments_intersect(path, w)):
        #         # print(w)
        #         return True
        #     if is_point_in_polygon(w[0], polygon)==True or is_point_in_polygon(w[1], polygon)==True:
        #         # print(w)
        #         return True
        #     if point_segment_distance(w[0], (myseg[0],newseg[0]))<=r or point_segment_distance(w[0], (myseg[0],newseg[1]))<=r or point_segment_distance(w[0], (myseg[1],newseg[1]))<=r or point_segment_distance(w[0], (myseg[1],newseg[0]))<=r:
        #         # print(w)
        #         return True
        #     if point_segment_distance(w[1], (myseg[0],newseg[0]))<=r or point_segment_distance(w[1], (myseg[0],newseg[1]))<=r or point_segment_distance(w[1], (myseg[1],newseg[1]))<=r or point_segment_distance(w[1], (myseg[1],newseg[0]))<=r:
        #         return True

    return False

def dist(p1,p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    if dist(p,s[0])<dist(p,s[1]):
        a=(p[0]-s[0][0], p[1]-s[0][1])
        b=(s[1][0]-s[0][0], s[1][1]-s[0][1])
    else:
        a=(p[0]-s[1][0], p[1]-s[1][1])
        b=(s[0][0]-s[1][0], s[0][1]-s[1][1])
    a1=a[0]
    a2=a[1]
    b1=b[0]
    b2=b[1]
    a_length=dist((0,0),a)
    if a_length==0:
        return 0
    b_length=dist((0,0), b)
    # print(a_length,b_length)
    cosab=(a1*b1+a2*b2)/(a_length*b_length)
    if a_length*cosab<0:
        return min(dist(p,s[0]), dist(p,s[1]))
    else:
        return abs((a1*b2-a2*b1)/b_length)

#used https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/ as a source to implement intersect function
#q = point, s= segment
def onSegment(q, s): 
    p=s[0]
    r=s[1]
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
           (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))): 
        return True
    return False

def direction(p, q, r): 
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0]- p[0]) * (r[1] - q[1])) 
    if (val > 0): 
        # Clockwise 
        return 1
    elif (val < 0): 
        # Counterclockwise 
        return 2
    else: 
        # Collinear 
        return 0
    
def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    p1=s1[0]
    q1=s1[1]
    p2=s2[0]
    q2=s2[1]
    o1 = direction(p1, q1, p2) 
    o2 = direction(p1, q1, q2) 
    o3 = direction(p2, q2, p1) 
    o4 = direction(p2, q2, q1) 

    if ((o1 != o2) and (o3 != o4)): 
        return True

    if (o1 == 0) and onSegment( p2, (p1,q1)): 
        return True
  
    if (o2 == 0) and onSegment( q2, (p1,q1)): 
        return True
 
    if (o3 == 0) and onSegment( p1, (p2,q2)): 
        return True
  
    if (o4 == 0) and onSegment( q1, (p2,q2)): 
        return True
  
    return False
    # p1=s1[0]
    # p2=s1[1]
    # q1=s2[0]
    # q2=s2[1]
    
    # #vertical line
    # if (p2[0]-p1[0])==0:
    #     a_p=0
    #     b_p=-1
    #     c_p=p1[0]
    # #else
    # else:
    #     a_p=-1
    #     b_p=(p2[1]-p1[1])/(p2[0]-p1[0])
    #     c_p=p2[1]-(p2[1]-p1[1])/(p2[0]-p1[0])*p2[0]

    # #vertical line
    # if (q2[0]-q1[0])==0:
    #     a_q=0
    #     b_q=-1
    #     c_q=q1[0]
    # #else
    # else:
    #     a_q=-1
    #     b_q=(q2[1]-q1[1])/(q2[0]-q1[0])
    #     c_q=q2[1]-(q2[1]-q1[1])/(q2[0]-q1[0])*q2[0]

    # #parallel
    # if b_p==b_q:
    #     #colinear
    #     if a_p==a_q and c_p==c_q:
    #         if (p1[0]<q1[0] and q1[0]<=p2[0]) or (p1[0]<q1[0] and q1[0]<=p2[0]):
    #             return True
    #         elif (p1[0]<q2[0] and q2[0]<=p2[0]) or (p1[0]<q2[0] and q2[0]<=p2[0]):
    #             return True
    #     return False
    # #perpendicular
    # if a_p*a_q+b_p*b_q==0:
    #     if a_p==0:
    #         intersection=(c_p, c_q)
    #     elif a_q==0:
    #         intersection=(c_q, c_p)
    # else:
    #     intersection = ((a_p*c_q-a_q*c_p)/(b_p*a_q-b_q*a_p),(c_p*b_q-c_q*b_p)/(b_p*a_q-b_q*a_p))
    # if intersection[0]>p1[0] and intersection[0]<p2[0]:
    #     return True
    # return False

def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(s1, s2):
        return 0
    dist1=point_segment_distance(s1[0], s2)
    dist2=point_segment_distance(s1[1], s2)
    dist3=point_segment_distance(s2[0], s1)  
    dist4=point_segment_distance(s2[1], s1)
    return min(dist1, dist2, dist3, dist4)

if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
