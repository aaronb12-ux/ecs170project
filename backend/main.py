from skipwords import skipWords

def readFile(file_name):

    try:
        with open(file_name, 'r') as file1:
            data1 = file1.read()
        return data1
    except FileNotFoundError as e:
        return {"error": str(e)}
    

def getMatchingWords(resume, jobdescription):
    matching = []

    for item in resume:
        if item in jobdescription:
            if item in skipWords:
                continue
            else:
                matching.append(item)
    
    return matching

def showMatchingWords(words):
    for word in words:
        print("Both the resume and job share:", word)

def main():

    resume = readFile('resume.txt').split() #read .txt files
    jobdescription = readFile('jobdescription.txt').split()

    resume = [item.lower().strip('.,:\n') for item in resume] #lower case everything and strip punctuation
    jobdescription = [item.lower().strip('.,:\n') for item in jobdescription] 

    matchingWords = getMatchingWords(resume, jobdescription)

    showMatchingWords(matchingWords)

    
if __name__ == "__main__":
    main()



