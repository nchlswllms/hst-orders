'''
The higher Stasheff--Tamari posets

These are two poset structures on the set of triangulations of a cyclic polytope C(m,delta).
Here m is the number of vertices in the cyclic polytope and delta is its dimension.

The first higher Stasheff--Tamari poset was introduced by Kapranov and Voevodsky in their paper Combinatorial-geometric aspects of polycategory
theory.
The second higher Stasheff--Tamari poset was introduced by Edelman and Reiner in their paper The higher Stasheff--Tamari posets.

This module implements the combinatorial of the posets given by the author in arXiv:2007.12664.

The function hst1(m,delta) returns the first higher Stasheff--Tamari poset on the triangulations of C(m,delta).

The function hst2(m,delta) returns the second higher Stasheff--Tamari poset on the triangulations of C(m,delta).

Computing the second poset requires computing all the triangulations of C(m,delta), which effectively requires computing the first poset.
Therefore, if the first poset has already been computed as first=hst1(m,delta), it is quicker to compute the second poset as hst1_to_hst2(first,delta).
This is the point of the function hst1_to_hst2.

The implementation of the posets is different for delta odd and delta even, since the description in arXiv:2007.12664 is different in odd and even dimensions.
'''

# ****************************************************************************
#    Copyright (C) 2020 Nicholas Williams <nchlswllms@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#
#    This code is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    General Public License for more details.
#
#  The full text of the GPL is available at:
#
#                  https://www.gnu.org/licenses/
# ****************************************************************************

import numpy as np
import itertools
from sage.combinat.posets.posets import Poset

#####################################################################################################################
# MATERIAL USED IN BOTH EVEN AND ODD DIMENSIONS
#####################################################################################################################

def separated(tup):

    r"""
    Return whether tup contains any consecutive numbers.

    INPUT:

    - ``tup`` -- tuple; an ordered tuple of positive integers, representing a simplex in a triangulation.

    OUTPUT: truth value

    EXAMPLES:

        sage: separated((2,4,7))
        True
        sage: separated((1,2,4))
        False
        
    """    
    
    for i in range(len(tup)-1):
        if tup[i] + 1 == tup[i + 1]:
            return False
    
    return True


def intertwine(tup1,tup2):

    r"""
    Return whether tup1 intertwines tup2 in the sense of Oppermannn and Thomas ``Higher-dimensional cluster combinatorics and representation theory``.

    (a0, a1, ..., ad) intertwines (b0, b1, ..., bd) if and only if a0 < b0 < a1 < b1 < ... < ad < bd

    INPUT:

    - ``tup1`` -- tuple; a tuple of positive integers, representing a simplex in a triangulation.
    - ``tup2`` -- tuple; a tuple of positive integers, representing a simplex in a triangulation. It is assumed that this has the same number of elements as tup1.

    OUTPUT: truth value

    EXAMPLES:

        sage: intertwine((1,3,5),(2,4,7))
        True
        sage: intertwine((1,3,7),(2,4,7))
        False
        sage: intertwine((2,4,7),(1,3,5))
        False
        
    """   
    
    length = len(tup1)
    
    for i in range(len(tup1)-1):
        if tup1[i] >= tup2[i] or tup2[i] >= tup1[i + 1]:
            return False
    
    if tup1[len(tup1) - 1] >= tup2[len(tup1) - 1]:
            return False
    
    return True


#####################################################################################################################
# EVEN DIMENSIONS
#####################################################################################################################

r"""
Triangulations of C(m,2d) are given by lexicographically ordered tuples containing ordered (d+1)-tuples of numbers in [m].

Mathematically, this represents the set of internal d-simplices of the triangulation. Since these simplices do not intersect each other, the (d+1)-tuples from [m] must be pairwise non-intertwining.

"""


def lower(m, d):

    r"""
    Return the lower triangulation of C(m,2d).

    INPUT:

    - ``m`` -- integer; the number of vertices of the cyclic polytope.
    - ``d`` -- integer; the number such that 2d is the dimension of the cyclic polytope.

    OUTPUT: the tuple of tuples

    EXAMPLES:

        sage: lower(6,1)
        ((1, 3), (1, 4), (1, 5))
        sage: lower(8,2)
        ((1, 3, 5), (1, 3, 6), (1, 3, 7), (1, 4, 6), (1, 4, 7), (1, 5, 7))
        sage: lower(9,3)
        ((1, 3, 5, 7), (1, 3, 5, 8), (1, 3, 6, 8), (1, 4, 6, 8))
    """
    
    lower = list(filter(separated, itertools.combinations(range(3, m), d)))
    lower = [list(element) for element in lower]
        
    for element in lower:
        element.insert(0, 1)
    
    lower = [tuple(element) for element in lower]
    
    return tuple(lower)


def non_lower(m, d):

    r"""
    Return the list of internal d-simplices in C(m,2d) which do not occur in the lower triangulation.

    INPUT:

    - ``m`` -- integer; the number of vertices of the cyclic polytope.
    - ``d`` -- integer; the number such that 2d is the dimension of the cyclic polytope.

    OUTPUT: the tuple of tuples

    EXAMPLES:

        sage: non_lower(5,1)
        [(2, 4), (2, 5), (3, 5)]
        sage: non_lower(7,2)
        [(2, 4, 6), (2, 4, 7), (2, 5, 7), (3, 5, 7)]
    """
    
    return list(filter(separated, itertools.combinations(range(2, m + 1), d + 1)))


def inc_flips(triang,m,d):
    
    r"""
    Return a list of the triangulations of C(m,2d) which are increasing bistellar flips of triang.

    INPUT:

    - ``triang`` -- tuple; triangulation of the cyclic polytope C(m,2d) given as a tuple of integer (d+1)-tuples.
    - ``m`` -- integer; the number of vertices of the cyclic polytope.
    - ``d`` -- integer; the number such that 2d is the dimension of the cyclic polytope.

    OUTPUT: list of tuples

    EXAMPLES:

        sage: inc_flips(((1, 3), (1, 4), (1, 5)),6,1)
        [((1, 4), (1, 5), (2, 4)), ((1, 3), (1, 5), (3, 5)), ((1, 3), (1, 4), (4, 6))]
        sage: inc_flips(((1, 4), (1, 5), (2, 4)),6,1)
        [((1, 5), (2, 4), (2, 5)), ((1, 4), (2, 4), (4, 6))]
        sage: inc_flips(((2, 6), (3, 6), (4, 6)),6,1)
        []
        
    NOTE::
        
        The concept of an increasing bistellar flip here is replacing a (d+1)-tuple by one which it intertwines.
        Since nothing can intertwine a (d+1)-tuple in the lower triangulation, those can be ignored.
        We look for a (d+1)-tuple which a unique (d+1)-tuple of the triangulation intertwines, and which intertwines nothing in the triangulation.
        In the matrix this corresponds to a row which is all zeros apart from one `1`.
        If a (d+1)-tuple intertwines a (d+1)-tuple in the triangulation it can be completely discounted. This is marked with a `2`, and then the loop is broken.
        
    """
    
    arcs = list(set(non_lower(m, d)) - set(triang))
    record = np.zeros((len(arcs), len(triang)), int)
    
    for i in range(len(arcs)):
        for j in range(len(triang)):
            if intertwine(arcs[i], triang[j]):
                record[i,j] = 2
                break
            elif intertwine(triang[j], arcs[i]):
                record[i,j] = 1
    
    flips = []
    for i in range(len(arcs)):
        if sum(record[i]) == 1:
            flip = list(triang)
            mutable = np.where(record[i] == 1)
            flip[mutable[0][0]] = arcs[i]
            flip.sort()
            flips.append(tuple(flip))
    
    return flips


def even_dict_hst_1(m, d):
    
    r"""
    Return a dictionary with all triangulations of C(m,2d) as keys and with values as the list of increasing flips of the triangulation.

    INPUT:

    - ``m`` -- integer; the number of vertices of the cyclic polytope.
    - ``d`` -- integer; the number such that 2d is the dimension of the cyclic polytope.

    OUTPUT: dictionary
    
    EXAMPLES:

        sage: even_dict_hst_1(5,1)
        {((1, 3), (1, 4)): [((1, 4), (2, 4)), ((1, 3), (3, 5))],
         ((1, 4), (2, 4)): [((2, 4), (2, 5))],
         ((1, 3), (3, 5)): [((2, 5), (3, 5))],
         ((2, 4), (2, 5)): [((2, 5), (3, 5))],
         ((2, 5), (3, 5)): []}
        sage: even_dict_hst_1(7,2)
        {((1, 3, 5), (1, 3, 6), (1, 4, 6)): [((1, 3, 6), (1, 4, 6), (2, 4, 6)),
         ((1, 3, 5), (1, 3, 6), (3, 5, 7))],
         ((1, 3, 6), (1, 4, 6), (2, 4, 6)): [((1, 4, 6), (2, 4, 6), (2, 4, 7))],
         ((1, 3, 5), (1, 3, 6), (3, 5, 7)): [((1, 3, 5), (2, 5, 7), (3, 5, 7))],
         ((1, 4, 6), (2, 4, 6), (2, 4, 7)): [((2, 4, 6), (2, 4, 7), (2, 5, 7))],
         ((1, 3, 5), (2, 5, 7), (3, 5, 7)): [((2, 4, 7), (2, 5, 7), (3, 5, 7))],
         ((2, 4, 6), (2, 4, 7), (2, 5, 7)): [((2, 4, 7), (2, 5, 7), (3, 5, 7))],
         ((2, 4, 7), (2, 5, 7), (3, 5, 7)): []}
        
    NOTE::
        
        By a result of Rambau, every triangulation can be found by a sequence of increasing bistellar flips from the lower triangulation.
        Hence to find all triangulations we start at the lower triangulation and take all possible increasing bistellar flips.
        Whenever we look at the list of increasing bistellar flips of a triangulation, we record this in the dictionary.
        
    """
    
    hst_dict={}
    triangs = []
    triangs.append(lower(m, d))
    
    flipped_counter = 0
    while flipped_counter < len(triangs):
        flips = inc_flips(triangs[flipped_counter], m, d)
        hst_dict[triangs[flipped_counter]] = [flips[i] for i in range(len(flips))]
        for i in range(len(flips)):               
            if flips[i] not in set(triangs):
                triangs.append(flips[i])
        flipped_counter += 1
        
    return hst_dict

def even_hst_1(m,d):
    
    r"""
    Return the first higher Stasheff--Tamari poset on the set of triangulations of C(m,2d).

    INPUT:

    - ``m`` -- integer; the number of vertices of the cyclic polytope.
    - ``d`` -- integer; the number such that 2d is the dimension of the cyclic polytope.

    OUTPUT: sage poset object
    
    EXAMPLES:

        sage: even_hst_1(7,1)
        Finite poset containing 42 elements (use the .plot() method to plot)
        sage: even_hst_1(8,2)
        Finite poset containing 40 elements (use the .plot() method to plot)
        sage: even_hst_1(10,3)
        Finite poset containing 102 elements (use the .plot() method to plot)
        
    """
    
    return Poset(even_dict_hst_1(m, d), cover_relations=True)


def even_hst_2_rel(triang1, triang2):
    
    r"""
    Returns whether or not triang1 =< triang2 with respect to the second higher Stasheff--Tamari order.

    INPUT:

    - ``triang1`` -- tuple; triangulation of the cyclic polytope C(m,2d) given as a tuple of integer (d+1)-tuples.
    - ``triang2`` -- tuple; triangulation of the cyclic polytope C(m,2d) given as a tuple of integer (d+1)-tuples.

    OUTPUT: truth value

    EXAMPLES:

        sage: even_hst_2_rel(((1,3),(1,4),(1,5)),((1,4),(2,4),(1,5)))
        True
        sage: even_hst_2_rel(((2,6),(3,6),(4,6)),((1,4),(2,4),(1,5)))
        False
        
    NOTE::
        
        The interpretation of the second higher Stasheff--Tamari order is the one from arXiv:2007.12664.
        triang1 =< triang2 if no tuple from triang2 intertwines any tuple from triang1
        
    """
    
    for i in range(len(triang1)):
        for j in range(len(triang2)):
            if intertwine(triang2[j], triang1[i]):
                return False
            
    return True


def even_all_triangs(m, d):
    
    r"""
    Return a list of all triangulations of C(m,2d).
    
    INPUT:

    - ``m`` -- integer; the number of vertices of the cyclic polytope.
    - ``d`` -- integer; the number such that 2d is the dimension of the cyclic polytope.

    OUTPUT: list
    
    EXAMPLES:

        sage: even_all_triangs(6,1)
        [((1, 3), (1, 4), (1, 5)),
         ((1, 3), (1, 4), (4, 6)),
         ((1, 4), (1, 5), (2, 4)),
         ((1, 3), (1, 5), (3, 5)),
         ((1, 3), (3, 6), (4, 6)),
         ((1, 4), (2, 4), (4, 6)),
         ((1, 5), (2, 4), (2, 5)),
         ((1, 3), (3, 5), (3, 6)),
         ((1, 5), (2, 5), (3, 5)),
         ((2, 6), (3, 6), (4, 6)),
         ((2, 4), (2, 6), (4, 6)),
         ((2, 4), (2, 5), (2, 6)),
         ((2, 6), (3, 5), (3, 6)),
         ((2, 5), (2, 6), (3, 5))]
        sage: even_all_triangs(7,2)
        [((1, 3, 5), (1, 3, 6), (1, 4, 6)),
         ((1, 3, 6), (1, 4, 6), (2, 4, 6)),
         ((1, 3, 5), (1, 3, 6), (3, 5, 7)),
         ((1, 4, 6), (2, 4, 6), (2, 4, 7)),
         ((1, 3, 5), (2, 5, 7), (3, 5, 7)),
         ((2, 4, 6), (2, 4, 7), (2, 5, 7)),
         ((2, 4, 7), (2, 5, 7), (3, 5, 7))]
        
    NOTE::
        
        By a result of Rambau, every triangulation can be found by a sequence of increasing bistellar flips from the lower triangulation.
        Hence to find all triangulations we start at the lower triangulation and take all possible increasing bistellar flips.
        This function is used for computing the second poset. If the first poset has already been computed, it is quicker to use hst1_to_hst2.
        
    """
    
    triangs = []
    triangs.append(lower(m, d))
    
    flipped_counter = 0
    while flipped_counter < len(triangs):
        flips = inc_flips(triangs[flipped_counter], m, d)
        for i in range(len(flips)):               
            if flips[i] not in set(triangs):
                triangs.append(tuple(flips[i]))
        flipped_counter += 1
    
    return triangs


def even_hst_2(m, d):
    
    r"""
    Return the second higher Stasheff--Tamari poset on the set of triangulations of C(m,2d).

    INPUT:

    - ``m`` -- integer; the number of vertices of the cyclic polytope.
    - ``d`` -- integer; the number such that 2d is the dimension of the cyclic polytope.

    OUTPUT: sage poset object
    
    EXAMPLES:

        sage: even_hst_2(7,1)
        Finite poset containing 42 elements (use the .plot() method to plot)
        sage: even_hst_2(8,2)
        Finite poset containing 40 elements (use the .plot() method to plot)
        sage: even_hst_2(10,3)
        Finite poset containing 102 elements (use the .plot() method to plot)
        
    """
    
    return Poset((even_all_triangs(m, d), even_hst_2_rel))


#####################################################################################################################
# ODD DIMENSIONS
#####################################################################################################################

r"""
Triangulations of C(m,2d+1) are given by lexicographically ordered tuples containing ordered (d+1)-tuples of numbers in [m].

Mathematically, this represents the set of internal d-simplices of the triangulation. By arXiv:2007.12664, the set of (d+1)-tuples must be ``supporting`` and ``bridging``.

"""


def internal(m, d):
    
    r"""
    Return the (d+1)-tuples corresponding to internal d-simplices in C(m,2d+1).

    INPUT:

    - ``m`` -- integer; the number of vertices of the cyclic polytope.
    - ``d`` -- integer; the number such that 2d+1 is the dimension of the cyclic polytope.

    OUTPUT: list of tuples

    EXAMPLES:

        sage: internal(7,1)
        [(2, 4), (2, 5), (2, 6), (3, 5), (3, 6), (4, 6)]
        sage: internal(9,2)
        [(2, 4, 6),
         (2, 4, 7),
         (2, 4, 8),
         (2, 5, 7),
         (2, 5, 8),
         (2, 6, 8),
         (3, 5, 7),
         (3, 5, 8),
         (3, 6, 8),
         (4, 6, 8)]
 
    NOTE::
        
        The (d+1)-tuples corresponding to internal d-simplices of C(m,2d+1) are those with entries in [2,m-1] containing no consecutive numbers, by arXiv:2007.12664
        
    """   
    
    return list(filter(separated,itertools.combinations(range(2, m), d + 1)))


def poss_supp(tup):
    
    r"""
    Return the list of possible supporting tuples of tup.

    INPUT:

    - ``tup`` -- tuple;

    OUTPUT: list of tuples

    EXAMPLES:

        sage: poss_supp((1, 5))
        [(2,), (3,), (4,)]
        sage: poss_supp((2, 5, 9))
        [(3, 6), (3, 7), (3, 8), (4, 6), (4, 7), (4, 8)]
        
    """ 
    
    ranges = [range(tup[i] + 1, tup[i + 1]) for i in range(len(tup) - 1)]
    
    return list(itertools.product(*ranges))


def supp_need(tup, cand):
    
    r"""
    Return a list of the tuples which are required by the support property, given a tuple tup and a candidate supporting tuple cand.

    INPUT:

    - ``tup`` -- tuple; a (d+1)-tuple
    - ``cand`` -- tuple; a d-tuple which could be a supporting tuple for tup

    OUTPUT: list of tuples

    EXAMPLES:

        sage: supp_need((2, 6), (4,))
        [(2, 4), (4, 6)]
        sage: supp_need((2, 5, 9), (3, 7))
        [(2, 5, 7), (2, 7, 9), (3, 5, 7), (3, 5, 9), (3, 7, 9), (5, 7, 9)]
        
    """ 
    
    entries = sorted(tup + cand)
    initial = list(filter(separated, itertools.combinations(entries, len(tup)))) 
    initial.remove(tup)
    
    return initial


def test_supp(triang,tup):
    
    r"""
    Returns whether triang \cup \{tup\} possesses the support property.

    INPUT:

    - ``triang`` -- tuple; a tuple of tuples representing a triangulation
    - ``tup`` -- tuple; a (d+1)-tuple
    
    OUTPUT: truth value

    EXAMPLES:

        sage: test_supp(((2, 4),), (2, 5))
        True
        sage: test_supp((), (2, 4))
        True
        sage: test_supp(((2, 4),), (2, 6))
        False
        sage: test_supp(((2, 4, 6), (2, 4, 7)), (2, 5, 7))
        True        
        
    NOTE::
        
        A decreasing flip in this framework consists of adding a tuple to the triangulation.
        Since triangulations must possess the support property, the tuple can only be added to the triangulation if the result possesses the support property.
        
    """
    
    candidates = poss_supp(tup)
    for cand in candidates:
        cand_failure = False
        requirements = supp_need(tup,cand)
        for req in requirements:
            if not req in triang:
                cand_failure = True
                break
        if not cand_failure:
            return True
    
    return False


def bridging(tup1, tup2):
    
    r"""
    Return a list of the tuples which are required by the bridging property, given a pair of tuples.

    INPUT:

    - ``tup1`` -- tuple; a (d+1)-tuple
    - ``tup2`` -- tuple; a (d+1)-tuple

    OUTPUT: list of tuples

    EXAMPLES:

        sage: bridging((2, 4), (3, 5))
        [(2, 5)]
        sage: bridging((2, 5), (3, 5))
        []
        sage: bridging((2, 4, 6), (3, 5, 7))
        [(2, 5, 7), (2, 4, 7)]
        sage: bridging((2, 4, 7), (3, 5, 7))
        [(2, 5, 7)]
        
    """ 
    
    beginning = 0
    while tup1[beginning] == tup2[beginning]:
        beginning += 1
   
    finish = len(tup1)
    while tup1[finish - 1] == tup2[finish - 1]:
        finish -= 1
    
    if not intertwine(tup1[beginning: finish], tup2[beginning: finish]):
        return [] # If they're not, we return an empty list

    required = []
    counter = beginning + 1
    while counter < finish:
        required.append(tup1[0: counter] + tup2[counter: len(tup1)])
        counter += 1 
    
    return required


def test_bridge(triang, tup):
    
    r"""
    Returns whether triang \cup \{tup\} possesses the bridging property.

    INPUT:

    - ``triang`` -- tuple; a tuple of tuples representing a triangulation
    - ``tup`` -- tuple; a (d+1)-tuple
    
    OUTPUT: truth value

    EXAMPLES:

        sage: test_bridge(((2, 4),), (2, 5))
        True
        sage: test_bridge((), (2, 4))
        True
        sage: test_bridge(((2, 4), (2, 5)), (3, 5))
        True
        sage: test_bridge(((2, 4),), (3, 5))
        False
        sage: test_bridge(((2, 4, 6), (2, 4, 7)), (2, 5, 7))
        True
        
    NOTE::
        
        A decreasing flip in this framework consists of adding a tuple to the triangulation.
        Since triangulations must possess the bridging property, the tuple can only be added to the triangulation if the result possesses the bridging property.
        
    """
    
    for simp in triang:        
        for requirement in bridging(simp,tup):
            if requirement not in triang:
                return False        
        for requirement in bridging(tup,simp):
            if requirement not in triang:
                return False
            
    return True
    

def dec_flips(triang, m, d):
    
    r"""
    Return a list of the triangulations of C(m,2d+1) which are decreasing bistellar flips of triang.

    INPUT:

    - ``triang`` -- tuple; triangulation of the cyclic polytope C(m,2d+1) given as a tuple of integer (d+1)-tuples.
    - ``m`` -- integer; the number of vertices of the cyclic polytope.
    - ``d`` -- integer; the number such that 2d+1 is the dimension of the cyclic polytope.

    OUTPUT: list of tuples

    EXAMPLES:

        sage: dec_flips((), 6, 1)
        [((2, 4),), ((3, 5),)]
        sage: dec_flips((), 8, 2)
        [((2, 4, 6),), ((3, 5, 7),)]
        sage: dec_flips(((2, 4, 6),), 9, 2)
        [((2, 4, 6), (2, 4, 7)), ((2, 4, 6), (4, 6, 8))]
        sage: dec_flips(((3, 5),), 7, 1)
        [((3, 5), (3, 6)), ((2, 5), (3, 5))]
        
    NOTE::
    
        It is easier to compute decreasing flips than increasing flips in odd dimensions, since one only has to check the supporting and bridging properties with respect to the tuple one is adding.
        Hence we test which d-simplices which are not in the triangulation can be added, according to whether the supporting and bridging properties are satisfied.
        
    """
    
    flips = []    
    all_simps = internal(m, d)
    tups = list(set(all_simps) - set(triang))
    for tup in tups:
        if test_supp(triang, tup) and test_bridge(triang, tup):
            flip = list(triang)
            flip.append(tup)
            flip.sort()
            flips.append(tuple(flip))
    
    return flips


def odd_dict_hst_1(m,d):
    
    r"""
    Return a dictionary with all triangulations of C(m,2d+1) as keys and with values as the list of increasing flips of the triangulation.

    INPUT:

    - ``m`` -- integer; the number of vertices of the cyclic polytope.
    - ``d`` -- integer; the number such that 2d+1 is the dimension of the cyclic polytope.

    OUTPUT: dictionary
    
    EXAMPLES:

        sage: odd_dict_hst_1(6, 1)
        {(): [((2, 4),), ((3, 5),)],
         ((2, 4),): [((2, 4), (2, 5))],
         ((3, 5),): [((2, 5), (3, 5))],
         ((2, 4), (2, 5)): [((2, 4), (2, 5), (3, 5))],
         ((2, 5), (3, 5)): [((2, 4), (2, 5), (3, 5))],
         ((2, 4), (2, 5), (3, 5)): []}
        sage: odd_dict_hst_1(8, 2)
        {(): [((2, 4, 6),), ((3, 5, 7),)],
         ((2, 4, 6),): [((2, 4, 6), (2, 4, 7))],
         ((3, 5, 7),): [((2, 5, 7), (3, 5, 7))],
         ((2, 4, 6), (2, 4, 7)): [((2, 4, 6), (2, 4, 7), (2, 5, 7))],
         ((2, 5, 7), (3, 5, 7)): [((2, 4, 7), (2, 5, 7), (3, 5, 7))],
         ((2, 4, 6),
          (2, 4, 7),
          (2, 5, 7)): [((2, 4, 6), (2, 4, 7), (2, 5, 7), (3, 5, 7))],
         ((2, 4, 7),
          (2, 5, 7),
          (3, 5, 7)): [((2, 4, 6), (2, 4, 7), (2, 5, 7), (3, 5, 7))],
         ((2, 4, 6), (2, 4, 7), (2, 5, 7), (3, 5, 7)): []}
        
    NOTE::
        
        By a result of Rambau, every triangulation can be found by a sequence of decreasing bistellar flips from the upper triangulation.
        Hence to find all triangulations we start at the upper triangulation and take all possible decreasing bistellar flips.
        Whenever we look at the list of decreasing bistellar flips of a triangulation, we record this in the dictionary.
        
    """
    
    hst_dict = {}
    triangs = []
    triangs.append(()) # The upper triangulation corresponds to the empty tuple
    
    flipped_counter = 0
    while flipped_counter < len(triangs):
        flips = dec_flips(triangs[flipped_counter], m, d)
        hst_dict[triangs[flipped_counter]] = [flips[i] for i in range(len(flips))]
        for i in range(len(flips)): 
            if flips[i] not in triangs:
                triangs.append(flips[i])     
        flipped_counter += 1
        
    return hst_dict


def odd_hst_1(m,d):
    
    r"""
    Return the first higher Stasheff--Tamari poset on the set of triangulations of C(m,2d+1).

    INPUT:

    - ``m`` -- integer; the number of vertices of the cyclic polytope.
    - ``d`` -- integer; the number such that 2d+1 is the dimension of the cyclic polytope.

    OUTPUT: sage poset object
    
    EXAMPLES:

        sage: odd_hst_1(8, 1)
        Finite poset containing 138 elements (use the .plot() method to plot)
        sage: odd_hst_1(9, 2)
        Finite poset containing 67 elements (use the .plot() method to plot)
        sage: odd_hst_1(11, 3)
        Finite poset containing 165 elements (use the .plot() method to plot)
    
    NOTE:
    
        Since we have been computing decreasing flips rather than increasing flips, we apply .dual() before returning the poset.
    
    """
    
    return Poset(odd_dict_hst_1(m, d), cover_relations=True).dual()


def odd_all_triangs(m, d):
    
    r"""
    Return a dictionary with all triangulations of C(m,2d+1) as keys and with values as the list of increasing flips of the triangulation.

    INPUT:

    - ``m`` -- integer; the number of vertices of the cyclic polytope.
    - ``d`` -- integer; the number such that 2d+1 is the dimension of the cyclic polytope.

    OUTPUT: list of tuples
    
    EXAMPLES:

        sage: odd_all_triangs(6, 1)
        [(),
         ((2, 4),),
         ((3, 5),),
         ((2, 4), (2, 5)),
         ((2, 5), (3, 5)),
         ((2, 4), (2, 5), (3, 5))]
        sage: odd_all_triangs(8, 2)
        [(),
         ((2, 4, 6),),
         ((3, 5, 7),),
         ((2, 4, 6), (2, 4, 7)),
         ((2, 5, 7), (3, 5, 7)),
         ((2, 4, 6), (2, 4, 7), (2, 5, 7)),
         ((2, 4, 7), (2, 5, 7), (3, 5, 7)),
         ((2, 4, 6), (2, 4, 7), (2, 5, 7), (3, 5, 7))]
        
    NOTE::
        
        By a result of Rambau, every triangulation can be found by a sequence of decreasing bistellar flips from the upper triangulation.
        Hence to find all triangulations we start at the upper triangulation and take all possible decreasing bistellar flips.
        This function is used for computing the second poset. If the first poset has already been computed, it is quicker to use hst1_to_hst2.
        
    """
    
    triangs = []
    triangs.append(())
    
    flipped_counter = 0
    while flipped_counter < len(triangs):
        flips = dec_flips(triangs[flipped_counter], m, d)
        for i in range(len(flips)):
            if flips[i] not in set(triangs):
                triangs.append(flips[i])
        flipped_counter += 1
    
    return triangs


def contains(triang1,triang2):
    
    r"""
    Returns whether or not triang1 is contains in triang2.

    INPUT:

    - ``triang1`` -- tuple; triangulation of the cyclic polytope C(m,2d+1) given as a tuple of integer (d+1)-tuples.
    - ``triang2`` -- tuple; triangulation of the cyclic polytope C(m,2d+1) given as a tuple of integer (d+1)-tuples.

    OUTPUT: truth value

    EXAMPLES:

        sage: contains(((2, 4), (2, 5)), ((2, 4),))
        True
        sage: contains(((2, 4), (2, 5)), ((3, 5),))
        False
        
    NOTE::
        
        This is used as a function which gives the second higher Stasheff--Tamari order in odd dimensions
        The interpretation of the second higher Stasheff--Tamari order is the one from arXiv:2007.12664.
        triang1 =< triang2 with respect to the second order if triang1 contains triang2.
        
    """
    
    return set(triang2).issubset(set(triang1))


def odd_hst_2(m,d):
    
    r"""
    Return the second higher Stasheff--Tamari poset on the set of triangulations of C(m,2d+1).

    INPUT:

    - ``m`` -- integer; the number of vertices of the cyclic polytope.
    - ``d`` -- integer; the number such that 2d+1 is the dimension of the cyclic polytope.

    OUTPUT: sage poset object
    
    EXAMPLES:

        sage: odd_hst_2(8, 1)
        Finite poset containing 138 elements (use the .plot() method to plot)
        sage: odd_hst_2(9, 2)
        Finite poset containing 67 elements (use the .plot() method to plot)
        sage: odd_hst_2(11, 3)
        Finite poset containing 165 elements (use the .plot() method to plot)
        
    """
    
    return Poset((odd_all_triangs(m, d), contains))


#####################################################################################################################
# BOTH PARITIES COMBINED
#####################################################################################################################


def hst1(m, delta):
    
    r"""
    Return the first higher Stasheff--Tamari poset on the set of triangulations of C(m,delta).

    INPUT:

    - ``m`` -- integer; the number of vertices of the cyclic polytope.
    - ``delta`` -- integer; the dimension of the cyclic polytope.

    OUTPUT: sage poset object
    
    EXAMPLES:

        sage: hst1(8, 1)
        Finite poset containing 64 elements (use the .plot() method to plot)
        sage: hst1(8, 2)
        Finite poset containing 132 elements (use the .plot() method to plot)
        sage: hst1(9, 3)
        Finite poset containing 972 elements (use the .plot() method to plot)
    
    """    
    
    d = delta // 2
    if delta % 2 == 0:
        return even_hst_1(m,d)
    else:
        return odd_hst_1(m,d)
    
    
def hst2(m,delta):
    
    r"""
    Return the second higher Stasheff--Tamari poset on the set of triangulations of C(m,delta).

    INPUT:

    - ``m`` -- integer; the number of vertices of the cyclic polytope.
    - ``delta`` -- integer; the dimension of the cyclic polytope.

    OUTPUT: sage poset object
    
    EXAMPLES:

        sage: hst2(8, 1)
        Finite poset containing 64 elements (use the .plot() method to plot)
        sage: hst2(8, 2)
        Finite poset containing 132 elements (use the .plot() method to plot)
        sage: hst2(9, 3)
        Finite poset containing 972 elements (use the .plot() method to plot)
    
    """ 
    
    d = delta // 2
    if delta % 2 == 0:
        return even_hst_2(m, d)
    else:
        return odd_hst_2(m, d)
    

def hst1_to_hst2(first, delta):
    
    r"""
    Converts the first higher Stasheff--Tamari poset into the second higher Stasheff--Tamari order.

    INPUT:

    - ``first`` -- poset; an instance of the first higher Stasheff--Tamari order.
    - ``delta`` -- integer; the dimension of the cyclic polytope.

    OUTPUT: sage poset object
    
    EXAMPLES:

        sage: hst1_to_hst2(hst1(8, 1), 1)
        Finite poset containing 64 elements (use the .plot() method to plot)
        sage: hst1_to_hst2(hst1(8, 2), 2)
        Finite poset containing 132 elements (use the .plot() method to plot)
        sage: hst1_to_hst2(hst1(9, 3), 3)
        Finite poset containing 972 elements (use the .plot() method to plot)
        
    NOTE::
    
        The point of this function is that if the first poset has already been computed, 
        it is much quicker to compute the second poset by applying the second order to the objects of the first order, 
        rather than repeating the work of computing all the triangulations.
        ``first`` could just be the set of triangulations, but it is assumed that this function will be used when the first poset has already been computed.
        ``delta`` just tells us whether we are in odd or even dimensions here.
    
    """ 
    
    d = delta // 2
    if delta % 2 == 0:
        return Poset((first, even_hst_2_rel))
    else:
        return Poset((first, contains))
