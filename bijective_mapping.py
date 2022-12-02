#bijective mapping between multinomial opinion and dirichlet pdf

#ω_x = (b_x,u_x,a_x) be a multinomial opinion,and let DireX(p_x,r_x,a_x)be a Dirichlet PDF.

#input
# r: denote the number of observations of X
# R: denote a list that contains number of observation for each action
def translateB(r, R, w=4):
    return r / (w + sum(R))


#input
# R: denote a list that contains number of observation for each action
def translateU(R, w=4):
    return w / (w + sum(R))

#input
# w: non-informative prior weight
# b: belief of X
# u: uncertainty of X
def translateR(b, u, w=4):
    positive_infinity = float('inf')
    if (u != 0):
        return (w * b) / u
    if (u == 0):
        return b * positive_infinity


#input
# B: denote a list that contains belief for each action x
# u: uncertainty of X
def sanity_check(B, u):
    if (u != 0) and (u+sum(B) == 1):
        print("sanity_check1 passed")
    elif (u == 0) and (sum(B) == 1):
        print("sanity_check2 passed")
    else:
        print("not passed")




