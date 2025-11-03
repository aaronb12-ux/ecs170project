#read in two files 

def read_file(file_name):
    try:
        with open(file_name, 'r') as file1:
            data1 = file1.read()
        return data1
    except FileNotFoundError as e:
        return {"error": str(e)}

def main():

    resume = read_file('resue.txt')
    jobdescription = read_file('jobdescription.txt')

    print("Resume Content:")
    print(resume)
    print("\nJob Description Content:")
    print(jobdescription)


if __name__ == "__main__":
    main()



#How to tie sentiment analysis into:
