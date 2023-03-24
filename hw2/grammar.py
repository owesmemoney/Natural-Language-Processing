"""
COMS W4705 - Natural Language Processing - Summer 2022 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""
import sys
from collections import defaultdict
from math import fsum
import math

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        The function checks two things: the right format and the sum of probability equal to 1
        check if the grammar is in CNF; distinguish between terminals and non-terminals
        """
        # TODO, Part 1
        #collect the set of nonterminals from left-hand-sides 
        #and nonterminals start with uppercase letters
        for key in grammar.lhs_to_rules.keys():
            item_set = grammar.lhs_to_rules[key]
            for item in item_set:
                lhs_set = item[1] 
                for x in lhs_set:
                    if x.upper() == x:
                        return ('Grammar is a valid PCFG')
                    else:
                        return ('error message')
            p = [item_set[i][2] for i in range(len(item_set))]
            pp = math.fsum(p)
            if math.isclose(pp,1)==False:
                return ('Grammar is a valid PCFG')
            else:
                return ("error message")



if __name__ == "__main__":
    with open('atis3.pcfg','r') as grammar_file:
        grammar = Pcfg(grammar_file)
        #print(grammar.startsymbol)
        #print(grammar.lhs_to_rules['PP'])
        #print(grammar.rhs_to_rules[('ABOUT','NP')])
        #print(grammar.rhs_to_rules[('NP','VP')])
        print(grammar.verify_grammar())
        
