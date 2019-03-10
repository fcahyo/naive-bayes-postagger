import nltk

def readFile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    return text

def wordWithTag(text):
    return [nltk.tag.str2tuple(i) for i in text.split()]

text = readFile('tagged_corpus.txt')
x = wordWithTag(text)

def posterior(wordSearch, tagSearch):
    #Count specified word on specified tag
    fd = nltk.FreqDist(x)
    wordTagFreq = 0
    for (word,freq) in fd.most_common():
        if word[0] == wordSearch and word[1] == tagSearch:
            wordTagFreq = freq    
    
    #Count unique term |V|
    uniqueWord = []    
    for (word, tag) in x:
        if word not in uniqueWord:
            uniqueWord.append(word)
        
    #Count total tag and tag frequency
    td = nltk.FreqDist([tag for (word, tag) in x])    
    tagFreq = 0
    countTag = 0
    for (tag, freq) in td.most_common():
        countTag += freq
        if tag == tagSearch:
            tagFreq = freq
    
    #Calculate prior probability
    prior = tagFreq / countTag
    
    #Calculate likelihood using laplace smoothing
    likelihood = (wordTagFreq + 1) / (tagFreq + len(uniqueWord))
    
    return (prior, likelihood, prior * likelihood)

test = ["Menjaga kelestarian budaya sangat penting",
        "Berkomunikasi dengan seseorang yang memiliki latar belakang budaya yang berbeda dapat menjadi kebahagiaan dan tantangan tersendiri",
        "Selama budaya tidak bertentangan dengan dalil , maka itu dapat dijadikan hukum",
        "Memberi makan kepada patung merupakan sebuah kebudayaan yang harus dimusnahkan",
        """Mukidi menyukai baju bertuliskan " love Indonesia " """]

def naiveBayes(testData):
    tags = []
    td = nltk.FreqDist([tag for (word, tag) in x])   
    for tag, freq in td.most_common():
        tags.append(tag)
            
    result = []
    for word in testData.lower().split(" "):           
        max = (tags[0], 0)
        for tag in tags:        
            post = posterior(word, tag)[2]
            if post > max[1]:            
                max = (tag, post)
#            print(word, tag)
#            print("Prior      : ", posterior(word, tag)[0])
#            print("Likelihood : ", posterior(word, tag)[1])
#            print("Posterior  : ", posterior(word, tag)[2])

        result.append(word + "/" + str(max[0]))
    
    return result

def main():
    for i, v in enumerate(test):
    	print(naiveBayes(test[i]))

if __name__ == "__main__":
    main()