import nltk
from nltk.corpus import cmudict


def test_rhyme(first, second):
    first = first.lower()
    second = second.lower()

    d = cmudict.dict()

    if first not in d:
        print 'first word ', first, ' is not in dict'
        return None

    if second not in d:
        print 'second word ', second, ' is not in dict'
        return None

    first = d[first][0]
    second = d[second][0]

    results = {}
    for i in first:
        results[i] = second.count(i)
    rhyme_level = 0
    for k, v in results.iteritems():
        rhyme_level += v

    return rhyme_level


def find_rhymes(inp, lvl):
    entries = nltk.corpus.cmudict.entries()
    syllables = [(word, syl) for word, syl in entries if word == inp]
    rhymes = []
    for (word, syllable) in syllables:
        rhymes += [word for word, pron in entries if pron[-lvl:] == syllable[-lvl:]]
    return set(rhymes)


if __name__ == "__main__":
    print test_rhyme("word", "heard")
    print test_rhyme("test", "word")
    print test_rhyme("duplicate", "replicate")
    print test_rhyme("this should fail", "check that inputs are words")

    print find_rhymes("duplicate", 2)
