import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# config
modelPath = 'blstm2.pt'
devOutPath = 'dev2.out'

mainPath = 'data/'
trainPath = mainPath + 'train'
devPath = mainPath + 'dev'
testPath = mainPath + 'test'
glovePath = 'glove.6B.100d'
# config end

wordPadding = '<wordpad>'
unkToken = '<unk>'
tagPadding = '<tagPad>'



def readTrain(trainPath):
    train_file = open(trainPath)
    lines = train_file.read().splitlines()
    train_file.close()
    words = []
    tags = []
    curWords = []
    curTags = []
    allWords = []
    allTags = []
    masks = []
    curMasks = []
    for line in lines:
        parts = line.split(' ')
        if len(parts) > 1:
            theWord = parts[1]
            curWords.append(theWord.lower())
            curTags.append(parts[2])
            allWords.append(theWord.lower())
            allTags.append(parts[2])
            if theWord.lower() == theWord:
                curMasks.append(0)
            elif theWord.upper() == theWord:
                curMasks.append(2)
            else:
                curMasks.append(1)
        else:
            words.append(curWords)
            tags.append(curTags)
            masks.append(curMasks)
            curMasks = []
            curWords = []
            curTags = []
    if len(curWords) != 0:
        words.append(curWords)
        tags.append(curTags)
        masks.append(curMasks)

    maxLen = 0
    total = 0
    for item in words:
        maxLen = max(maxLen, len(item))
        total += len(item)

    wordCount = {}
    for word in allWords:
        if word not in wordCount:
            wordCount[word] = 1
        else:
            wordCount[word] += 1

    wordToIdx = {}
    index = 2
    for word in allWords:
        if wordCount[word] > 1:
            if word not in wordToIdx:
                wordToIdx[word] = index
                index += 1

    wordToIdx[wordPadding] = 0



    wordToIdx[unkToken] = 1
    unkIndex = index

    tagToIdx = {}
    tagIndex = 1
    for tag in allTags:
        if tag not in tagToIdx:
            tagToIdx[tag] = tagIndex
            tagIndex += 1

    tagToIdx[tagPadding] = 0

    idxToTag = {}
    for key, value in tagToIdx.items():
        if value not in idxToTag:
            idxToTag[value] = key
        else:
            print("bad!")

    trainWordList = []
    trainTagList = []
    trainMask = []
    for seq in words:
        newWordSeq = []
        for word in seq:
            if word in wordToIdx:
                newWordSeq.append(wordToIdx[word])
            else:
                newWordSeq.append(wordToIdx[unkToken])
        if len(newWordSeq) < maxLen:
            newWordSeq.extend([wordToIdx[wordPadding] for _ in range(maxLen - len(newWordSeq))])
        trainWordList.append(newWordSeq)

    for seq in tags:
        newTagSeq = []
        for tag in seq:
            newTagSeq.append(tagToIdx[tag])
        if len(newTagSeq) < maxLen:
            newTagSeq.extend([tagToIdx[tagPadding] for _ in range(maxLen - len(newTagSeq))])
        trainTagList.append(newTagSeq)

    for seq in masks:
        newMaskSeq = []
        for mask in seq:
            newMaskSeq.append(mask)
        if len(newMaskSeq) < maxLen:
            newMaskSeq.extend(-1 for _ in range(maxLen - len(newMaskSeq)))
        trainMask.append(newMaskSeq)

    return trainWordList, trainTagList, trainMask, wordToIdx, tagToIdx, idxToTag

def readDev(devPath, wordToIdx, tagToIdx):
    dev_file = open(devPath)
    lines = dev_file.read().splitlines()
    dev_file.close()
    devWords = []
    devTags = []
    curWords = []
    curTags = []
    devAllWords = []
    devAllTags = []
    devMasks = []
    curMasks = []
    for line in lines:
        parts = line.split(' ')
        if len(parts) > 1:
            word = parts[1]
            curWords.append(parts[1].lower())
            curTags.append(parts[2])
            devAllWords.append(parts[1].lower())
            devAllTags.append(parts[2])
            if word.lower() == word:
                curMasks.append(0)
            elif word.upper() == word:
                curMasks.append(2)
            else:
                curMasks.append(1)
        else:
            devWords.append(curWords)
            devTags.append(curTags)
            devMasks.append(curMasks)
            curWords = []
            curTags = []
            curMasks = []
    if len(curWords) != 0:
        devWords.append(curWords)
        devTags.append(curTags)
        devMasks.append(curMasks)

    devMaxLen = 0
    devTotal = 0
    for item in devWords:
        devMaxLen = max(devMaxLen, len(item))
        devTotal += len(item)

    devWordList = []
    devTagList = []
    devMaskPadded = []

    for seq in devWords:
        newWordSeq = []
        for word in seq:
            if word in wordToIdx:
                newWordSeq.append(wordToIdx[word])
            else:
                newWordSeq.append(wordToIdx[unkToken])
        if len(newWordSeq) < devMaxLen:
            newWordSeq.extend([wordToIdx[wordPadding] for _ in range(devMaxLen - len(newWordSeq))])
        devWordList.append(newWordSeq)

    for seq in devTags:
        newTagSeq = []
        for tag in seq:
            newTagSeq.append(tagToIdx[tag])
        if len(newTagSeq) < devMaxLen:
            newTagSeq.extend([tagToIdx[tagPadding] for _ in range(devMaxLen - len(newTagSeq))])
        devTagList.append(newTagSeq)

    for seq in devMasks:
        newMaskSeq = []
        for mask in seq:
            newMaskSeq.append(mask)
        if len(newMaskSeq) < devMaxLen:
            newMaskSeq.extend(-1 for _ in range(devMaxLen - len(newMaskSeq)))
        devMaskPadded.append(newMaskSeq)

    return devWordList, devTagList, devMaskPadded


class NERDataset(Dataset):
    def __init__(self, texts, tags, masks):
        self.texts = texts
        self.tags = tags
        self.masks = masks

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tag = self.tags[idx]
        mask = self.masks[idx]

        return torch.tensor(text), torch.tensor(tag), torch.tensor(mask)




def outPutFile(devPred, epochNum, idxToTag):
    #outPath = "try2/devEpoch" + str(epochNum + 1) +".out"
    outPath = devOutPath
    devPath = mainPath + "dev"
    outFile = open(outPath, 'w')
    devFile = open(devPath, 'r')
    lines = devFile.read().splitlines()
    devFile.close()
    seqNum = 0
    wordNum = 0
    for line in lines:
        parts = line.split(' ')
        if len(parts) > 1:
            newLine = str(parts[0]) + ' ' + str(parts[1]) + ' ' + str(parts[2]) + ' ' + str(idxToTag[int(devPred[seqNum][wordNum])]) + '\n'
            outFile.writelines(newLine)
            wordNum += 1
        else:
            newLine = '\n'
            outFile.writelines(newLine)
            wordNum = 0
            seqNum += 1
    outFile.close()


class TestDataset(Dataset):
    def __init__(self, texts,  masks):
        self.texts = texts
        self.masks = masks

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        mask = self.masks[idx]

        return torch.tensor(text), torch.tensor(mask)




def outputTest(modelPath, testPath, outPath):
    trainWordList, trainTagList, trainMask, wordToIdx, tagToIdx, idxToTag = readTrain(trainPath)
    weightMatrix, idxToWord = getEmbebMat(wordToIdx, glovePath)

    test_file = open(testPath)
    lines = test_file.read().splitlines()
    test_file.close()
    testWords = []
    curWords = []
    testAllWords = []
    testMasks = []
    curMasks = []
    for line in lines:
        parts = line.split(' ')
        if len(parts) > 1:
            word = parts[1]
            curWords.append(parts[1].lower())
            testAllWords.append(parts[1].lower())
            if word.lower() == word:
                curMasks.append(0)
            elif word.upper() == word:
                curMasks.append(2)
            else:
                curMasks.append(1)
        else:
            testWords.append(curWords)
            testMasks.append(curMasks)
            curWords = []
            curMasks = []
    if len(curWords) != 0:
        testWords.append(curWords)
        testMasks.append(curMasks)

    testMaxLen = 0
    testTotal = 0
    for item in testWords:
        testMaxLen = max(testMaxLen, len(item))
        testTotal += len(item)

    testWordList = []
    testMaskPadded = []

    for seq in testWords:
        newWordSeq = []
        for word in seq:
            if word in wordToIdx:
                newWordSeq.append(wordToIdx[word])
            else:
                newWordSeq.append(wordToIdx[unkToken])
        if len(newWordSeq) < testMaxLen:
            newWordSeq.extend([wordToIdx[wordPadding] for _ in range(testMaxLen - len(newWordSeq))])
        testWordList.append(newWordSeq)

    for seq in testMasks:
        newMaskSeq = []
        for mask in seq:
            newMaskSeq.append(mask)
        if len(newMaskSeq) < testMaxLen:
            newMaskSeq.extend(-1 for _ in range(testMaxLen - len(newMaskSeq)))
        testMaskPadded.append(newMaskSeq)

    test_dataset = TestDataset(testWordList, testMaskPadded)
    test_loader = DataLoader(test_dataset, shuffle=False)



    vocab_size = len(wordToIdx)
    embedding_dim = 101
    hidden_dim = 256
    output_dim = 128
    final_dim = 10
    pad_idx = wordToIdx[wordPadding]
    drop_out_prob = 0.33

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = GLSTMNER(vocab_size, embedding_dim, hidden_dim, output_dim, final_dim, pad_idx, drop_out_prob,
                         weightMatrix, idxToWord)
        model.load_state_dict(torch.load(modelPath))
        model.to(device)
    else:
        device = torch.device("cpu")
        model = GLSTMNER(vocab_size, embedding_dim, hidden_dim, output_dim, final_dim, pad_idx, drop_out_prob,
                         weightMatrix, idxToWord)
        model.load_state_dict(torch.load(modelPath, map_location=device))

    model.eval()
    testPred = []
    for data, maskForTest in test_loader:
        data = data.to(device)

        output = model(data, maskForTest, device)
        newOut = output.view(-1, 10)

        predList = []
        for i in range(len(newOut)):
            _, pred = torch.max(newOut[i], -1)
            predList.append(pred)

        testPred.append(predList)


    outFile = open(outPath, 'w')
    testFile = open(testPath, 'r')
    lines = testFile.read().splitlines()
    testFile.close()
    seqNum = 0
    wordNum = 0
    for line in lines:
        parts = line.split(' ')
        if len(parts) > 1:
            newLine = str(parts[0]) + ' ' + str(parts[1]) + ' ' + str(idxToTag[int(testPred[seqNum][wordNum])]) + '\n'
            outFile.writelines(newLine)
            wordNum += 1
        else:
            newLine = '\n'
            outFile.writelines(newLine)
            wordNum = 0
            seqNum += 1
    outFile.close()



def getEmbebMat(wordToIdx, glovePath):
    filePath = glovePath
    gloveEmbedding = {}
    gloveFile = open(filePath, 'r', encoding='utf-8')
    lines = gloveFile.read().splitlines()
    gloveFile.close()
    for line in lines:
        parts = line.split(' ')
        if len(parts) > 1:
            word = parts[0]
            vector = torch.FloatTensor([float(val) for val in parts[1:]])
            gloveEmbedding[word] = vector



    vocabSize = len(wordToIdx)
    vecDim = 100
    weightMatrix = torch.zeros(vocabSize, vecDim)
    idxToWord = {}

    for key, value in wordToIdx.items():
        idxToWord[int(value)] = key
        lowerKey = key.lower()
        if lowerKey in gloveEmbedding:
            weightMatrix[int(value)] = gloveEmbedding[lowerKey]

    return weightMatrix, idxToWord





class GLSTMNER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, final_dim, pad_idx, drop_out_prob, weightMatrix, idxToWord):
        super().__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding = nn.Embedding.from_pretrained(weightMatrix, freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=drop_out_prob)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.elu = nn.ELU()
        self.output_fc = nn.Linear(output_dim, final_dim)

    def forward(self, data, mask, device):

        embedded = self.embedding(data)
        #boolean mask

        tensorMask = torch.Tensor(mask).to(device)

        newEmbedded = torch.cat((embedded, tensorMask.unsqueeze(2)), dim=-1)
        newEmbedded = newEmbedded.to(device)


        outputs, _ = self.lstm(newEmbedded)
        outputs = self.dropout(outputs)

        outputs = self.fc(outputs)
        activated = self.elu(outputs)
        predictions = self.output_fc(activated)

        return predictions



def train_dev():
    trainWordList, trainTagList, trainMask, wordToIdx, tagToIdx, idxToTag = readTrain(trainPath)
    devWordList, devTagList, devMaskPadded = readDev(devPath, wordToIdx, tagToIdx)
    weightMatrix, idxToWord = getEmbebMat(wordToIdx, glovePath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the dataset
    train_dataset = NERDataset(trainWordList, trainTagList, trainMask)
    valid_dataset = NERDataset(devWordList, devTagList, devMaskPadded)

    # Create the DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, shuffle=False)




    # Hyperparameters
    vocab_size = len(wordToIdx)
    embedding_dim = 101
    hidden_dim = 256
    output_dim = 128
    final_dim = 10
    pad_idx = wordToIdx[wordPadding]
    drop_out_prob = 0.33

    # Model instantiation
    model = GLSTMNER(vocab_size, embedding_dim, hidden_dim, output_dim, final_dim, pad_idx, drop_out_prob, weightMatrix, idxToWord)


    model = model.to(device)
    weights = torch.FloatTensor([0.0,   1.2,  0.4,  1.0,    1.2,   3.0,   1.0,    2.0,   1.5,    1.2]).to(device)
    #                         <tagPad> B-ORG   O   B-MISC  B-PER  I-PER   B-LOC  I-ORG  I-MISC  I-LOC}
    # use CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=tagToIdx[tagPadding])
    #criterion = nn.CrossEntropyLoss(weight=weights)

    # use Adam GD and learning rate = 0.0005
    optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9)


    n_epochs = 50

    for epoch in range(n_epochs):
        train_loss = 0.0

        model.train()

        for data, target, maskForTrain in train_loader:

            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model(data, maskForTrain, device)

            newOut = output.view(-1, 10)
            newTar = target.view(-1)

            loss = criterion(newOut, newTar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)


        train_loss = train_loss/len(train_loader.dataset)

        if epoch > 45:
            model.eval()
        #if True:
            count = 0
            total = 0
            devPred = []
            for data, target, maskForDev in valid_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data, maskForDev, device)
                newOut = output.view(-1, 10)
                newTar = target.view(-1)
                predList = []
                for i in range(len(newOut)):
                    _, pred = torch.max(newOut[i], -1)
                    predList.append(pred)
                    label = newTar[i]
                    if  label != tagToIdx[tagPadding] and label != tagToIdx['O']:
                    #if  label != tagToIdx[tagPadding]:
                        total += 1
                        if pred == label:
                            count += 1
                devPred.append(predList)
            outPutFile(devPred, epoch, idxToTag)
            torch.save(model.state_dict(), 'task2Model/epoch' + str(epoch + 1) + 'model.pt')
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss), 'The accuracy is ',count/total)
            print("count: ", count)
            print("total: ", total)
        else:
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))


def load_model(modelPath):
    trainWordList, trainTagList, trainMask, wordToIdx, tagToIdx, idxToTag = readTrain(trainPath)
    devWordList, devTagList, devMaskPadded = readDev(devPath, wordToIdx, tagToIdx)
    weightMatrix, idxToWord = getEmbebMat(wordToIdx, glovePath)

    # Instantiate the dataset
    valid_dataset = NERDataset(devWordList, devTagList, devMaskPadded)
    # Create the DataLoader
    valid_loader = DataLoader(valid_dataset, shuffle=False)

    # Hyperparameters
    vocab_size = len(wordToIdx)
    embedding_dim = 101
    hidden_dim = 256
    output_dim = 128
    final_dim = 10
    pad_idx = wordToIdx[wordPadding]
    drop_out_prob = 0.33

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = GLSTMNER(vocab_size, embedding_dim, hidden_dim, output_dim, final_dim, pad_idx, drop_out_prob,
                         weightMatrix, idxToWord)
        model.load_state_dict(torch.load(modelPath))
        model.to(device)
    else:
        device = torch.device("cpu")
        model = GLSTMNER(vocab_size, embedding_dim, hidden_dim, output_dim, final_dim, pad_idx, drop_out_prob,
                         weightMatrix, idxToWord)
        model.load_state_dict(torch.load(modelPath, map_location=device))

    model.eval()
    devPred = []
    for data, target, maskForDev in valid_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data, maskForDev, device)
        newOut = output.view(-1, 10)
        newTar = target.view(-1)
        predList = []
        for i in range(len(newOut)):
            _, pred = torch.max(newOut[i], -1)
            predList.append(pred)

        devPred.append(predList)
    outPutFile(devPred, 50, idxToTag)
    print("dev2.out has been generated successfully!")




#train_dev()
load_model(modelPath)



#outputTest(modelPath, testPath, 'test2.out')


