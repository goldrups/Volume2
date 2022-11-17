# test_specs.py
"""Python Essentials: Unit Testing.
Sam Goldrup
MATH 321
23 September 2021
"""

import specs #the code we test is in this file
import pytest #we need this module to test it from the command line


def test_add():
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8

def test_divide():
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.

def test_smallest_factor():
    """test the function on even, odd and prime numbers on smallest_factor() in specs.py"""
    assert specs.smallest_factor(1) == 1, "one"
    assert specs.smallest_factor(2) == 2, "even"
    assert specs.smallest_factor(3) == 3, "odd"
    assert specs.smallest_factor(4) == 2, "even"
    assert specs.smallest_factor(5) == 5, "prime"
    assert specs.smallest_factor(6) == 2, "even"
    assert specs.smallest_factor(7) == 7, "prime"
    assert specs.smallest_factor(8) == 2, "even"
    assert specs.smallest_factor(9) == 3, "odd"
    assert specs.smallest_factor(10) == 2, "even"

# Problem 2: write a unit test for specs.month_length().
def test_month_length():
    """test every month and both februaries on month_length() in specs.py"""
    assert specs.month_length("September") == 30
    assert specs.month_length("April") == 30
    assert specs.month_length("June") == 30
    assert specs.month_length("November") == 30
    assert specs.month_length("January") == 31
    assert specs.month_length("March") == 31
    assert specs.month_length("May") == 31
    assert specs.month_length("July") == 31
    assert specs.month_length("August") == 31
    assert specs.month_length("October") == 31
    assert specs.month_length("December") == 31
    assert specs.month_length("February", leap_year = True) == 29
    assert specs.month_length("February", leap_year = False) == 28
    assert specs.month_length("poop") == None

# Problem 3: write a unit test for specs.operate().
def test_operate():
    """tests operate() from specs.property
    tests four basic operations
    tests if errors are raised:
    ZeroDivisionError if second arguement is 0
    ValueError if third argument is not one of +,-,*,/
    TypeError if the oper passed into the argument isn't a string
    """
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.operate(4,0,"/")
    assert excinfo.value.args[0] == "division by zero is undefined"
    with pytest.raises(ValueError) as excinfo:
        specs.operate(6,3,"%")
    assert excinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'"
    with pytest.raises(TypeError) as excinfo:
        specs.operate(4,3,6)
    assert excinfo.value.args[0] == "oper must be a string"
    assert specs.operate(4,2,"+") == 6
    assert specs.operate(4,2,"-") == 2
    assert specs.operate(4,5,"*") == 20
    assert specs.operate(10,2,"/") == 5

# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    """Creates fraction objects"""
    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3

def test_fraction_init(set_up_fractions):
    """
    Uses the fraction objects from set_up_fractions
    Tests the vaalues of numerators/denominators
    Tests different errors:
    ZeroDivisionError if denominator is entered as 0
    TypeError if either numerator or denominator is entered as a non-int type
    """
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42) #30/42 reduces to 5/7
    assert frac.numer == 5
    assert frac.denom == 7
    with pytest.raises(ZeroDivisionError) as excinfo:
        frac_4_0 = specs.Fraction(4,0)
    assert excinfo.value.args[0] == "denominator cannot be zero"
    with pytest.raises(TypeError) as excinfo:
        frac_string_denom = specs.Fraction(4,"0")
    assert excinfo.value.args[0] == "numerator and denominator must be integers"
    with pytest.raises(TypeError) as excinfo:
        frac_string_numer = specs.Fraction("4",1)
    assert excinfo.value.args[0] == "numerator and denominator must be integers"

def test_fraction_str(set_up_fractions):
    "tests the string method of the class"
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"
    frac = specs.Fraction(4,1)
    assert str(frac) == "4"

def test_fraction_float(set_up_fractions):
    """tests float method"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.


def test_fraction_eq(set_up_fractions):
    """tests equality method"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)
    assert frac_1_2 == float(1/2)

def test_fraction_add(set_up_fractions):
    """tests add method"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    frac = specs.Fraction(5,6)
    assert (frac_1_3 + frac_1_2) == frac

def test_fraction_sub(set_up_fractions):
    """tests subtraction method"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    frac = specs.Fraction(-1,6)
    assert (frac_1_3 - frac_1_2) == frac

def test_fraction_mul(set_up_fractions):
    """tests multiplication method"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    frac = specs.Fraction(1,6)
    assert (frac_1_3 * frac_1_2) == frac

def test_fraction_truediv(set_up_fractions):
    """tests division method, while checking if a ZeroDivisionError is raised
    when necessary"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    frac = specs.Fraction(2,3)
    assert (frac_1_3 / frac_1_2) == frac
    with pytest.raises(ZeroDivisionError) as excinfo:
        frac_1_3 / specs.Fraction(0,3)
    assert excinfo.value.args[0] == "cannot divide by zero"


# Problem 5: Write test cases for Set.
def test_count_sets():
    """
    first digit (number): 0 for 1, 2 for 3
    second digit (color): 0 for red, 1 for purple, 2 for green
    third dig (shape): 0 for oval, 1 for squiggle, 2 for diamond
    fourth dig (shade): 0 for clear, 1 for lines, 2 for solid

    In this test function, we test that sets are correctly counted
    We also test that the four different ValueErrors are raised:
    non-unique cards, cards with more than 4 digits, cards with digits not in
    integers modulo 3, and a hand with more or less than 12 cards
    """
    cards_list =   ["2220", "2200", "0002", "2121",
                    "1022", "0111", "1101", "2022",
                    "0200", "0012", "1102", "2222"]
    assert specs.count_sets(cards_list) == 6
    cards_list =   ["2220", "2200", "0002", "21212",
                    "1022", "0111", "1101", "2022",
                    "0200", "0012", "1102", "1102"]
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(cards_list)
    assert excinfo.value.args[0] == "the cards are not all unique"
    cards_list =   ["2220", "2200", "0002", "21212",
                    "1022", "0111", "1101", "2022",
                    "0200", "0012", "1102", "2222"]
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(cards_list)
    assert excinfo.value.args[0] == "one or more cards does not have exactly 4 digits"
    cards_list =   ["2220", "2200", "0002", "2123",
                    "1022", "0111", "1101", "2022",
                    "0200", "0012", "1102", "2222"]
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(cards_list)
    assert excinfo.value.args[0] == "one or more cards has a character other than 0,1, or 2"
    cards_list =   ["2220", "2200", "0002", "2123",
                    "1022", "0111", "1101", "2022",
                    "0200", "0012", "1102", "2222", "1111"]
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(cards_list)
    assert excinfo.value.args[0] == "there are not exactly 12 cards"


def test_is_set():
    """Tests the is_set() function from specs.py"""
    assert specs.is_set("1010", "1111", "2222") == False
    assert specs.is_set("0000", "1111", "2222") == True
    assert specs.is_set("0000","1110","2220") == True
    assert specs.is_set("0020","1120","2220") == True
    assert specs.is_set("0120","1120","2120") == True
    assert specs.is_set("2120","2120","2120") == True
