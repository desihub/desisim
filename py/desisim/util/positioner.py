import desisim

def find_collision_type(pos_A, pos_B):
    """
    Checks for Type II and Type III collisions between positioners.
    
    Args:
        pos_A (object Positioner): object defining positioner A
        pos_B (object Positioner): object defining positioner B
    
    Returns:
       [True/False, True/True/False]: according 
            if TypeII, TypeIII are True/False, respectively
    """
    upper_A_poly  = shapeg.Polygon(pos_A.upper_pos)
    central_A_poly  = shapeg.Polygon(pos_A.central_pos)
    lower_A_poly  = shapeg.Polygon(pos_A.lower_pos)
    
    upper_B_poly  = shapeg.Polygon(pos_B.upper_pos)
    central_B_poly  = shapeg.Polygon(pos_B.central_pos)
    lower_B_poly  = shapeg.Polygon(pos_B.lower_pos)
    
    #Type II collision, Upper part of ferrule A with upper part of ferrule B
    collision_II = False
    if(upper_A_poly.intersects(upper_B_poly)):
        collision_II  = True
    
    #Type III collision, lower part of ferrule and central body
    collision_III = False
    if(lower_A_poly.intersects(central_B_poly)|lower_B_poly.intersects(central_A_poly)):
        collision_III = True
    return [collision_II, collision_III]

