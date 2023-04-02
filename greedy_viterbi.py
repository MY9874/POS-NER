import json
import numpy as np

# config

# threshold of unk token
threshold = 2
# main directory path
mainPath = "/Users/libin/Downloads/cs544_HW2_MoyuLi/"
# path for inputs
trainPath = mainPath + "train"
devPath = mainPath + "dev"
testPath = mainPath + "test"
# path for outputs
vocabPath = mainPath + "vocab.txt"
jsonPath = mainPath + "hmm.json"
greedyOutPath = mainPath + "greedy.out"
viterbiOutPath = mainPath + "viterbi.out"

# end of config


# helper function to output hmm.json
def getJsonDict(theDict):
    retDict = {}
    for key in theDict.keys():
        theKey = str(key[0]) + ', ' + str(key[1])
        retDict[theKey] = theDict[key]
    return retDict


# output the .out files
def outputTest(prediction, outPath):
    file = open(testPath)
    lines = file.read().splitlines()
    file.close()
    outFile = open(outPath, 'w')

    predIndex = 0
    for line in lines:
        parts = line.split('\t')
        if len(parts) < 2:
            outFile.writelines('\n')
            continue
        index, word = parts[0], parts[1]
        newLine = str(index) + '\t' + str(word) + '\t' + str(prediction[predIndex]) + '\n'
        predIndex += 1
        outFile.writelines((newLine))
    outFile.close()


# greedy
def greedy(filePath, wordSet, tagSet, transition, emission, isTest):
    file = open(filePath)
    #file = open("/Users/libin/Desktop/usc/cs544/HW2/data/dev")
    lines = file.read().splitlines()
    file.close()
    label = []
    prediction = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        if isTest:
            index, word = parts[0], parts[1]
        else:
            index, word, trueTag = parts[0], parts[1], parts[2]
            label.append(trueTag)
        if word not in wordSet:
            word = '<unk>'
        maxVal = -float('inf')
        result = None
        if index == '1':
            for tag in tagSet:
                if ('START', tag) in transition and (tag, word) in emission:
                    product = transition[('START', tag)] * emission[(tag, word)]
                    if product > maxVal:
                        maxVal = product
                        result = tag
        else:
            error = 1
            for tag in tagSet:
                if (prev, tag) in transition and (tag, word) in emission:
                    error = 0
                    product = transition[(prev, tag)] * emission[(tag, word)]
                    if product > maxVal:
                        maxVal = product
                        result = tag
            if error == 1:
                newTag = None
                newVal = 0
                for tag in tagSet:
                    if (tag, word) in emission:
                        if emission[(tag, word)] > newVal:
                            error = 0
                            newVal = emission[(tag, word)]
                            newTag = tag
                if newTag != None:
                    result = newTag
            if error == 1:
                print("too bad!")

        prediction.append(result)
        prev = result

    if isTest:
        outputTest(prediction, greedyOutPath)
    else:
        count = 0
        for i in range(len(prediction)):
            if prediction[i] == label[i]:
                count += 1

        accuracy = count / len(prediction)
        percentage = accuracy * 100
        print("Accuracy of Greedy algorithm on dev set is " + str(format(percentage, '.3f')) + "%, in decimal is " + str(accuracy))


# Viterbi
def viterbi(filePath, wordSet, tagSet, transition, emission, isTest):
    file = open(filePath)
    lines = file.read().splitlines()
    file.close()
    label = []
    prediction = []


    for line in lines:
        parts = line.split('\t')
        if len(parts) < 2:
            maxVal = -float('inf')
            for key in prevDict.keys():
                if prevDict[key][1] >= maxVal:
                    maxTag = key
                    maxVal = prevDict[key][1]
            path = prevDict[maxTag][0]
            path.reverse()
            prediction.extend(path)
            continue
        if isTest:
            index, word = parts[0], parts[1]
        else:
            index, word, trueTag = parts[0], parts[1], parts[2]
            label.append(trueTag)
        if word not in wordSet:
            word = '<unk>'

        if index == '1':
            curDict = {}
            for tag in tagSet:
                if ('START', tag) in transition and (tag, word) in emission:
                    m1 = np.log(transition[('START', tag)])
                    m2 = np.log(emission[(tag, word)])
                    curDict[tag] = ([tag], m1 + m2)
        else:
            curDict = {}
            error = 1
            for tag in tagSet:
                if (tag, word) not in emission:
                    continue
                temp = {}
                for prevTag in prevDict.keys():
                    if (prevTag, tag) in transition:
                        error = 0
                        m1 = np.log(transition[(prevTag, tag)])
                        m2 = np.log(emission[(tag, word)])
                        temp[prevTag] = m1 + m2 + prevDict[prevTag][1]
                maxVal = -float('inf')
                maxTag = None
                for key in temp.keys():
                    if temp[key] >= maxVal:
                        maxVal = temp[key]
                        maxTag = key
                if maxTag == None:
                    continue
                #print(maxTag)
                #print(prevDict[maxTag][0])
                #print(tag)
                thePath = [tag]
                thePath.extend(prevDict[maxTag][0])
                curDict[tag] = (thePath, maxVal)
            if error == 1:
                maxVal = -float('inf')
                for rawTag in tagSet:
                    if (rawTag, word) in emission:
                        error = 0
                        if emission[(rawTag, word)] > maxVal:
                            maxVal = emission[(rawTag, word)]
                            maxTag = rawTag
                maxVal = -float('inf')
                maxKey = None
                for key in prevDict.keys():
                    if prevDict[key][1] > maxVal:
                        maxVal = prevDict[key][1]
                        maxKey = key
                newPath = [maxTag]
                newPath.extend(prevDict[maxKey][0])
                curDict[maxTag] = (newPath, prevDict[maxKey][1])

            if error == 1:
                print("too bad!!!")
        prevDict = {}
        prevDict = curDict.copy()
        curDict = {}


    maxVal = -float('inf')
    for key in prevDict.keys():
        if prevDict[key][1] >= maxVal:
            maxTag = key
            maxVal = prevDict[key][1]
    path = prevDict[maxTag][0]
    path.reverse()
    prediction.extend(path)


    if isTest:
        outputTest(prediction, viterbiOutPath)
    else:
        count = 0
        for i in range(len(prediction)):
            if prediction[i] == label[i]:
                count += 1

        accuracy = count / len(prediction)
        percentage = accuracy * 100
        print("Accuracy of Viterbi algorithm on dev set is " + str(format(percentage, '.3f')) + "%, in decimal is " + str(accuracy))


# Task 1
file = open(trainPath)
lines = file.read().splitlines()
file.close()
words = []
tags = []
for line in lines:
    parts = line.split('\t')
    if len(parts) > 1:
        words.append(parts[1])
        tags.append(parts[2])

#print(len(words))
#print(len(tags))
freqDict = {}
for word in words:
    if word not in freqDict:
        freqDict[word] = 1
    else:
        freqDict[word] += 1
unkCount = 0
wordType = []
counts = []
for key in freqDict.keys():
    wordType.append(key)
    counts.append(freqDict[key])
for i in counts:
    if i < threshold:
        unkCount += i
theList = list(zip(wordType, counts))

theList.sort(key = (lambda x:x[1]), reverse = True)
stopIndex = 0
for i in range(len(theList)):
    if theList[i][1] < threshold:
        stopIndex = i
        break

final = [('<unk>', unkCount)]
final.extend(theList[:stopIndex])
#final.extend(theList)

vocab = open(vocabPath, 'w')
for i in range(len(final)):
    newline = final[i][0] + '\t' + str(i) + '\t' + str(final[i][1]) + '\n'
    vocab.writelines(newline)
print("The total size of the vocabulary is " + str(len(final)))
print("The total occurrences of <unk> token is " + str(unkCount))
vocab.close()


# Task 2
wordSet = set()
file = open(vocabPath)
lines = file.read().splitlines()
file.close()
for line in lines:
    parts = line.split('\t')
    if parts[1] == '0':
        continue
    wordSet.add(parts[0])


file = open(trainPath)
lines = file.read().splitlines()
file.close()

sCount = {}
ssCount = {}
sxCount = {}

sCount['START'] = 0
tagSet = set()

for line in lines:
    parts = line.split('\t')
    if len(parts) < 2:
        continue

    index, word, tag = parts[0], parts[1], parts[2]
    if tag not in tagSet:
        tagSet.add(tag)

    if tag not in sCount:
        sCount[tag] = 1
    else:
        sCount[tag] += 1

    if word in wordSet:
        sxPair = (tag, word)
    else:
        sxPair = (tag, '<unk>')
    if sxPair not in sxCount:
        sxCount[sxPair] = 1
    else:
        sxCount[sxPair] += 1

    if index == '1':
        sCount['START'] += 1
        ssPair = ('START', tag)
        if ssPair not in ssCount:
            ssCount[ssPair] = 1
        else:
            ssCount[ssPair] += 1
    else:
        ssPair = (prev, tag)
        if ssPair not in ssCount:
            ssCount[ssPair] = 1
        else:
            ssCount[ssPair] += 1
    prev = tag

transition = {}
emission = {}

for key in ssCount.keys():
    transition[key] = ssCount[key] / sCount[key[0]]

for key in sxCount.keys():
    if key[1] in wordSet:
        emission[key] = sxCount[key] / sCount[key[0]]
    else:
        emission[(key[0], '<unk>')] = sxCount[(key[0], '<unk>')] / sCount[key[0]]
print("There are " + str(len(transition)) + " transition parameters.")
print("There are " + str(len(emission)) + " emission parameters.")


outputTrans = getJsonDict(transition)
outputEmi = getJsonDict(emission)
with open(jsonPath, 'w') as fp:
    json.dump({'transition': outputTrans, 'emission': outputEmi}, fp)

greedy(devPath, wordSet, tagSet, transition, emission, False)
viterbi(devPath, wordSet, tagSet, transition, emission, False)



greedy(testPath, wordSet, tagSet, transition, emission, True)
print("Finished generating greedy.out!")
viterbi(testPath, wordSet, tagSet, transition, emission, True)
print("Finished generating viterbi.out!")

