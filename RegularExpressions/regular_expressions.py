# regular_expressions.py
"""Volume 3: Regular Expressions.
Samuel Goldrup
MATH 323
17 Febuary 2022
"""

import re #the cute regular expressions thingy

# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    pattern = re.compile("python")
    return pattern

# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    pattern = re.compile(r"\^\{@\}\(\?\)\[%\]\{\.\}\(\*\)\[_\]\{&\}\$")
    return pattern

# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    pattern = re.compile(r"^(Book|Mattress|Grocery) (store|supplier)$") #allow for two words where first and second word may vary
    return pattern

# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #tests

    # indentifier = re.compile(r"^[_a-zA-Z][\w]*$")
    # for test in ["Mouse", "compile", "_123456789", "__x__", "while", "_"]:
    #   print(test + ":", bool(indentifier.match(test)))
    
    # for test in ["3rats", "err*r", "sq(x)", "sleep()", " x"]:
    #     print(test + ":", bool(indentifier.match(test)))

    parameter = re.compile(r"^([_a-zA-Z])\w*\s*(=\s*)?([\d\.]*)?(\'\w*\')?$") #yeah this is so goated

    #more tests

    # for test in ["max=4.2", "string= ''", "num_guesses"]:
    #        print(test + ":", bool(parameter.match(test)))

    # for test in ["300", "is_4=(value==4)", "pattern = r'^one|two fish$'"]:
    #        print(test + ":", bool(parameter.match(test)))

    return parameter



# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """
    colon_buddies = re.compile(r"(^*\s(if|for|elif|else|while|try|except|finally|def|class|with)[^\n]*)", re.MULTILINE) #get all possible blocks
    new_code = colon_buddies.sub(r"\1:",code) #make the sub

    return new_code

# Problem 6
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """
    output_dict = {}
    name_pattern = re.compile(r"[a-zA-Z]+ +(?:[A-Z]\. [a-zA-Z]+|[a-zA-Z]+)")
    birthday_pattern = re.compile(r"([\d]{1,2})/([\d]{1,2})/([\d]{4}|[\d]{2})")
    phone_pattern =  re.compile(r"\d?-?\(?([\d]{3})\)?-?([\d]{3}-[\d]{4})")
    email_pattern = re.compile(r"[\._\w]+@[\._\w]+")

    with open(filename) as cutefile:
        lines = cutefile.readlines()

    for line in lines: #precept upon precept
        name = name_pattern.findall(line)
        bday = birthday_pattern.findall(line)
        digits = phone_pattern.findall(line)
        email = email_pattern.findall(line)

        if len(bday) != 0: #list is nonempty
            if len(bday[0][0]) == 1: #double index to get inside to tuple within the list
                m = "0"+bday[0][0]
            else:
                m = bday[0][0]
            if len(bday[0][1]) == 1:
                d = "0"+bday[0][1]
            else:
                d = bday[0][1]
            if len(bday[0][2]) == 2:
                y = "20"+bday[0][2]
            else:
                y = bday[0][2]

            date = m + "/" + d + "/" + y #format properly
        
        else:
            date = None

        if len(digits) == 1:
            da_number = "(" + digits[0][0] + ")" + digits[0][1] #format properly
        else:
            da_number = None
        
        if len(email) == 1:
            yahooligans = email[0]
        else:
            yahooligans = None

                
        output_dict[name[0]] = {"birthday": date, "email": yahooligans, "phone": da_number} #this is what each key will be like

    return output_dict