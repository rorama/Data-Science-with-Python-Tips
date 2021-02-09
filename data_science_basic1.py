#!/usr/bin/env python
# coding: utf-8

#change 1 made here

# # <font color=blue><b>Lesson 1: Data Structures</b></font>
# 
# Data structures in Python: 
# - Integer, Float, String, Tuple (immutable - cannot change once assigned)
# - List, Dictionary, Set (mutable)

# ## <font color=blue>1.1: Integer, Float, String</font>

# In[209]:


#Immutable - cannot change once assigned


# ## <font color=blue>1.2: Lists</font>

# In[147]:



"""
Lists (mutable)
Probably the most fundamental data structure in Python is the list. A list is simply an
ordered collection. (It is similar to what in other languages might be called an array, but
with some added functionality.)
"""

integer_list = [1, 2, 3]
heterogeneous_list = ["string", 0.1, True]
list_of_lists = [ integer_list, heterogeneous_list, [] ]
list_length = len(integer_list) # equals 3
list_sum = sum(integer_list) # equals 6



# ## <font color=blue>1.3: Tuples</font>

# In[148]:


"""
Tuples (immutable)
Tuples are lists’ immutable cousins. Pretty much anything you can do to a list that doesn’t
involve modifying it, you can do to a tuple. You specify a tuple by using parentheses (or
nothing) instead of square brackets: e.g. (3,2,4,2,6)
"""

my_list = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4
my_list[1] = 3 # my_list is now [1, 3]
try:
    my_tuple[1] = 3
except TypeError:
    print ("cannot modify a tuple")


# ## <font color=blue>1.4: Dictionaries</font>

# In[149]:


"""
Dictionaries (mutable)
Another fundamental data structure is a dictionary, which associates values with keys and
allows you to quickly retrieve the value corresponding to a given key:
"""

empty_dict = {} # Pythonic
empty_dict2 = dict() # less Pythonic
grades = { "Joel" : 80, "Tim" : 95 } # dictionary literal
#You can look up the value for a key using square brackets:
joels_grade = grades["Joel"] # equals 80


# ## <font color=blue>1.5: Sets</font>

# In[150]:


#Sets (mutable)
#Another data structure is set, which represents a collection of distinct elements:
s = set()
s.add(1) # s is now { 1 }
s.add(2) # s is now { 1, 2 }
s.add(2) # s is still { 1, 2 }
x = len(s) # equals 2
y = 2 in s # equals True
z = 3 in s # equals False

"""
We’ll use sets for two main reasons. The first is that in is a very fast operation on sets. If
we have a large collection of items that we want to use for a membership test, a set is more
appropriate than a list:
"""

stopwords_list = ["a","an","at"] + ["hundreds_of_other_words"] + ["yet", "you"]
"zip" in stopwords_list # False, but have to check every element
stopwords_set = set(stopwords_list)
"zip" in stopwords_set # very fast to check

#The second reason is to find the distinct items in a collection:
item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list) # 6
item_set = set(item_list) # {1, 2, 3}
num_distinct_items = len(item_set) # 3
distinct_item_list = list(item_set) # [1, 2, 3]
#We’ll use sets much less frequently than dicts and lists.


# # <font color=blue><b>Lesson 2: Some applications - users and friendships</b></font>

# In[151]:


users = [
{ "id": 0, "name": "Hero" },
{ "id": 1, "name": "Dunn" },
{ "id": 2, "name": "Sue" },
{ "id": 3, "name": "Chi" },
{ "id": 4, "name": "Thor" },
{ "id": 5, "name": "Clive" },
{ "id": 6, "name": "Hicks" },
{ "id": 7, "name": "Devin" },
{ "id": 8, "name": "Kate" },
{ "id": 9, "name": "Klein" }
]

users


# In[152]:


friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
(4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

friendships


# In[153]:


for user in users:
    user["friends"] = []
    
users    


# In[211]:


for i, j in friendships:
    # this works because users[i] is the user whose id is i
    users[i]["friends"].append(users[j]) # add i as a friend of j
    users[j]["friends"].append(users[i]) # add j as a friend of i
    
#users    


# In[155]:


def number_of_friends(user):
    """how many friends does _user_ have?"""
    return len(user["friends"]) # length of friend_ids list

total_connections = sum(number_of_friends(user)
                        for user in users)

total_connections

number_of_friends(users[1])
users[1]['name']


# In[156]:


from __future__ import division # integer division is lame
num_users = len(users) # length of the users list
avg_connections = total_connections / num_users # 2.4
avg_connections


# In[212]:


# create a list (user_id, number_of_friends)
num_friends_by_id = [(number_of_friends(user), user["id"]) for user in users] #list-comprehension, concise way to create new lists


sorted(num_friends_by_id
           , key = lambda num_friends: num_friends, reverse=True) #lambda is not accepting bracketed inputs (>1 input)
           


# # <font color=blue><b>Lesson 3: Lambda, Counter, Unpack lists, Sort, Enumerate</b></font>

# In[158]:


for x in range(3):
    print (x, "is less than 3")
    
#dont assign lambdas to variables like below. instead use a function def or pass parameters immediately at the end  
full_name = lambda first, last: f'Full name: {first.upper()} {last.title()}' #lambda is an anonymus function. This one takes 2 parameters (first, last) and returns one
full_name('guido', 'van rossum') #'Full name: GUIDO Van Rossum'

#pass parameters immediately at the end - notice parenthesis on both
(lambda first, last: f'Full name: {first.upper()} {last.title()}') ('aaaa', 'bbbb')
(lambda x, y: x + y)(2, 3)


# In[159]:


#Python Counter takes in input a list, tuple, dictionary, string, which are all iterable objects, and it will give you output that will have the count of each element.

from collections import Counter
lang_list = ['Python', 'C++', 'C', 'Java', 'Python', 'C', 'Python', 'C++', 'Python', 'C']
C = Counter(lang_list)
print("The occurrance of Python is: " ,C['Python'])

aa=[1,2,3,4,4,4]
Counter(aa) #Counter({1: 1, 2: 1, 3: 1, 4: 3})
Counter(aa)[4] #3


# In[160]:


#unpack lists
x, y = [1, 2] # now x is 1, y is 2
_, y = [1, 2] # now y == 2, didn't care about the first element, ignore it

ids = ['id1', 'id2', 'id30', 'id3', 'id22', 'id100'] # Lexicographic sort #['id1', 'id100', 'id2', 'id22', 'id3', 'id30']
sorted_ids = sorted(ids, key=lambda x: int(x[2:])) # Integer sort #['id1', 'id2', 'id3', 'id22', 'id30', 'id100']


# In[161]:


# sort the list by absolute value from largest to smallest
x = sorted([-4,1,-2,3], key=abs, reverse=True) # is [-4,3,-2,1]

# sort the words and counts from highest count to lowest
aa= ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'aa', 'bb'] #'abcdefgh abc' #[1,2,3,4,4,4]
word_counts = Counter(aa)
#wc = sorted(word_counts.items(), key=lambda (word, count): count, reverse=True)

#A Counter instance has a most_common method that is frequently useful:
# print the 10 most common words and their counts
for word, count in word_counts.most_common(5):
    print (word, count)


# In[162]:


x = [4,1,2,3]
y = sorted(x) # is [1,2,3,4], x is unchanged
x.sort() #x is changed

# sort the list by absolute value from largest to smallest
x = sorted([-4,1,-2,3], key=abs, reverse=True) # is [-4,3,-2,1]
x


# In[217]:

#change 2 made here

#enumerate
#Not infrequently, you’ll want to iterate over a list and use both its elements and their indexes:
#The Pythonic solution is enumerate, which produces tuples (index, element):
documents = ['aa', 'bb', 'cc']
for i, document in enumerate(documents):
    print(i, document)
    
#if you want to use only index:
for i, _ in enumerate(documents):
    print(i)

