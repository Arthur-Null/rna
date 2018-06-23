from wordseg import WordSegment
doc = u'AAATTTAAATTTAAATTTCCCGGGGGGCCCAAACCC'
ws = WordSegment(doc,max_word_len=3,min_aggregation=1,min_entropy=0.5)
print(ws.segSentence(doc,method=0))
