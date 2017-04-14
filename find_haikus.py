import nltk
from nltk.corpus import cmudict

import curses
from curses.ascii import isdigit
import re

import itertools

def print_haikus(haikus):
    for haiku in haikus:
        print_haiku(haiku)

def print_haiku(haiku):
    print '\n--'
    for _, v in haiku.iteritems():
        for word in v:
            print word,
        print ''
    print '--\n'

def is_haiku(text):
    text = text.lower()
    words = nltk.wordpunct_tokenize(re.sub('[^a-zA-Z_ ]', '', text))

    d = cmudict.dict()
    syl_count_total = 0
    haiku = dict()
    line = []
    for word in words:
        if word in d.keys():
            syl_count_total += [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
            line.append(word)
        else:
            print word, 'not in cmudict'
            return None

        if syl_count_total == 5:
            haiku[0] = line
            line = []

        if syl_count_total == 12:
            if len(haiku) < 1:
                break
            haiku[1] = line
            line = []

        if syl_count_total == 17:
            if len(haiku) < 2:
                break
            haiku[2] = line
            #print_haiku(haiku)
            return haiku

        if syl_count_total > 17:
            break
    return None


def next_sequence(text):
    split = text.split()
    words = split[1:]
    next_sentence = ''
    for word in words:
        next_sentence += ' %s' % word
    return next_sentence

def stop_search(text):
    if text == '':
        return True
    return False


def find_all_haikus(text, haikus):
    if stop_search(text):
        return haikus

    haiku = is_haiku(text)
    if haiku is not None:
        haikus.append(haiku)
    return find_all_haikus(next_sequence(text), haikus)


def get_stress(word):
    d = cmudict.dict()
    word = word.lower()
    if word not in d.keys():
        print word, "not in cmudict"
        return None

    possibilities = []
    for pronounce in d[word]:
        sub_meter = []
        for syl in pronounce:
            if syl[-1].isdigit():
                sub_meter.append(int(syl[-1]))
        possibilities.append(sub_meter)
    possibilities.sort()
    possibilities = list(i for i,_ in itertools.groupby(possibilities))
    return possibilities


def print_stress_pattern(text):
    text = text.lower()
    print text
    words = nltk.wordpunct_tokenize(re.sub('[^a-zA-Z_ ]', '', text))

    pattern = []
    for word in words:
        stress = get_stress(word)
        if stress is None:
            continue
        if len(stress) > 1:
            pattern.append(stress)
        else:
            pattern.append(stress[0])
 
    print pattern
    

def fit_pattern(pattern, context, word):
    print pattern, context, word

    pattern = str(pattern)
    context = str(context)
    word_stress = get_stress(word)
    print 'orig pattern = ', pattern

    for possible_stress in word_stress:
        stress = ''
        for syl in possible_stress:
            stress = stress+str(syl)
        print 'stress = ', stress
        new_stress = context+stress
        print 'context + stress = ', new_stress
        if new_stress == pattern[:len(new_stress)]:
            return True, new_stress
    return False, new_stress

    

if __name__ == '__main__':
    '''
    is_haiku('Where do we go from here? What is the meaning of life? Lets try longer more difficult vocabulary words as;doifj;asdofij')
    is_haiku('For this I should stay? To hear some shmeggegge kvetch about his lawsuit?')
    is_haiku('Let go of all the hate. Life is too short to spend your time filled with hate.')
    is_haiku('Difficult difficult difficult difficult difficult difficult')
    is_haiku('Difficult difficult difficult no difficult difficult no')

    #test = find_all_haikus('Where do we go from here? What is the meaning of life? Lets try longer more difficult vocabulary words as;doifj;asdofij', [])
    #print_haikus(test)
    print_stress_pattern('Where do we go from here? What is the meaning of life? Lets try longer more difficult vocabulary words as;doifj;asdofij')
    print get_stress('Where')
    print get_stress('do')
    print get_stress('we')
    print get_stress('from')
    print get_stress('of')
    print get_stress('is')
    print get_stress('difficult')
    '''

    #print_stress_pattern('Where do we go from here? What is the meaning of life? Lets try longer more difficult vocabulary words as;doifj;asdofij')
    print fit_pattern(10101010, 1010, 'of')
    print fit_pattern(10101010, 1010, 'beauty')
    print fit_pattern(10101010, 1010, 'beautiful')
    print fit_pattern(10101010, 1010, 'vocabulary')
