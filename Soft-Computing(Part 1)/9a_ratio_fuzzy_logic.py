from fuzzywuzzy import fuzz
from fuzzywuzzy import process

s1 = "I love fuzzysforfuzzys"
s2 = "I am loving fuzzysforfuzzys"

# FuzzyWuzzy ratios
print("FuzzyWuzzy Ratio:", fuzz.ratio(s1, s2))
print("FuzzyWuzzyPartialRatio: ", fuzz.partial_ratio(s1, s2))
print("FuzzyWuzzyTokenSortRatio: ", fuzz.token_sort_ratio(s1, s2))
print("FuzzyWuzzyTokenSetRatio: ", fuzz.token_set_ratio(s1, s2))
print("FuzzyWuzzyWRatio: ", fuzz.WRatio(s1, s2), '\n\n')

# FuzzyWuzzy process library
query = 'fuzzys for fuzzys'
choices = ['fuzzy for fuzzy', 'fuzzy fuzzy', 'g. for fuzzys']

# List of ratios
print("List of ratios: ")
print(process.extract(query, choices), '\n')

# Best among the above list
print("Best among the above list:", process.extractOne(query, choices))
