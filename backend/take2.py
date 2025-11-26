
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

def preprocess_text(text):
    """
    Normalize the text: convert to lowercase.
    """
    return text.lower()

def compute_average_embeddings(sentences):
    """
    Compute the average embeddings for a list of sentences.
    """
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)

        # Use the encoder's last hidden state
        last_hidden_state = outputs.encoder_last_hidden_state
        sentence_embedding = last_hidden_state.mean(dim=1).numpy()  # Average the embeddings
        embeddings.append(sentence_embedding)

    # Stack and compute the overall average embedding
    return np.mean(np.vstack(embeddings), axis=0)

def compute_similarity(resume_text, job_description_text):
    resume_text = preprocess_text(resume_text)
    job_description_text = preprocess_text(job_description_text)

    # Tokenizing sentences
    resume_sentences = [sentence.strip() for sentence in resume_text.split('.') if sentence.strip()]
    job_description_sentences = [sentence.strip() for sentence in job_description_text.split('.') if sentence.strip()]

    # Compute average embeddings for the resume and job description
    resume_embedding = compute_average_embeddings(resume_sentences)
    job_embedding = compute_average_embeddings(job_description_sentences)

    # Ensure both embeddings are 1D before calculating cosine similarity
    similarity_score = cosine_similarity(resume_embedding.reshape(1, -1), job_embedding.reshape(1, -1))[0][0]

    print(f"Overall similarity score: {similarity_score:.4f}\n")

    # Check sentence similarities for areas of high overlap
    print("Areas of high overlap:")
    
    for resume_sentence in resume_sentences:
        resume_vec = compute_average_embeddings([resume_sentence])
        
        for job_sentence in job_description_sentences:
            job_vec = compute_average_embeddings([job_sentence])
            individual_similarity = cosine_similarity(resume_vec.reshape(1, -1), job_vec.reshape(1, -1))[0][0]

            if individual_similarity > 0.9:  # Set a threshold for high overlap
                print(f"Resume: '{resume_sentence}' <-> Job Description: '{job_sentence}' (Similarity: {individual_similarity:.4f})")

    return similarity_score


if __name__ == "__main__":
    # Example usage
    resume = '''
    UC Davis Programming Languages Research Group February 2024 – Present
Programming Languages Researcher Davis, CA
• Extending the theory of regular expressions to incorporate string and character variables, integrating Brzozowski &
Antimirov derivatives for advanced pattern matching.
• Constructing an SMT parser and solver that performs satisfiability checks and string matching functions on regular
expressions with uninterpreted string variables in Rust to compete with and improve upon Z3 and cvc5.
Amazon June 2025 – September 2025
Software Development Engineer Intern
• Designed and implemented a dynamic data testing framework to enhance automation for the Alexa Music team
• Integrated DynamoDB with Amazon Bedrock to build a scalable internal service that enables QA engineers to eﬀiciently
utilize production data and broaden test coverage
Panasonic Avionics June 2024 – August 2024 Software Engineering Intern Irvine, CA
• Engineered automation scripts utilizing Ansible & Python to streamline upgrade processes for Cisco network devices
• Automated upgrade of network devices at LAX airport, improving network speeds by 20% for over 75 million annual fliers
• Configured Ansible Automation Platform to enable network engineers with varying levels of programming experience to
easily run automation scripts, streamlining their workflow and reducing manual tasks
California Department of Motor Vehicles June 2023 – June 2024 Software Engineering Intern Sacramento, CA
• Developed application components and resolved critical bugs for high-traﬀic applications serving over 100,000 drivers.
• Designed and implemented a parser in Java for JSON and XML dependency files, extracting comprehensive dependency trees
and classpaths, resulting in an 80% reduction in deployment time.
• Created and optimized Selenium automation scripts to streamline the testing process of 12 applications during the
organization’s transition to AWS servers, achieving a 60% reduction in testing time.
• Collaborated with other government agencies to retrieve and analyze critical information from the database using SQL,
ensuring accurate and timely data delivery while maintaining data privacy and security standards
Projects
SafeDrive AI | Python, opencv, Raspberry Pi, Arduino
• Developed an economical driver safety system utilizing a Raspberry Pi, integrating lane detection and driver awareness monitoring using OpenCV.
• Implemented a pedestrian alert system and connected the detection software with Arduino to control alert mechanisms such as lights and speakers, enhancing driver safety.
Pill Pusher | Python, Raspberry Pi, Flask, Twilio
• Created an automated pill dispenser system for elderly, leveraging a Flask server for scheduling and Twilio for reminders. • Engineered a medication dispensing circuit with a Raspberry Pi, improving medication adherence.
• Awarded Best Healthcare Hack and Best Interdisciplinary Hack at UC Davis’s HackDavis 2023 for project
BrainFun | Dafny, Python
• A formally verified compiler for the Brainf*ck programming language in Dafny
• Compiles BF code into an Intermediate Representation while ensuring functional correctness using First-Order Logic
Skills
Languages: Java, JavaScript/TypeScript, Python, Rust, Dafny, Kotlin, C, C++, Golang, HTML5, CSS3 Frameworks: React, Node.js, Next.js, Express, MongoDB, SQL, Flask, Spring Boot, Ansible, Antlr4 Other: LaTeX, Git, REST APIs, XML, Postman, Android Studio, Microcontrollers'''


    job_description = '''
    RELOCATION ASSISTANCE: Relocation assistance may be available

CLEARANCE TYPE: Top Secret

TRAVEL: Yes, 10% of the Time

Description

At Northrop Grumman, our employees have incredible opportunities to work on revolutionary systems that impact people's lives around the world today, and for generations to come. Our pioneering and inventive spirit has enabled us to be at the forefront of many technological advancements in our nation's history - from the first flight across the Atlantic Ocean, to stealth bombers, to landing on the moon. We look for people who have bold new ideas, courage and a pioneering spirit to join forces to invent the future, and have fun along the way. Our culture thrives on intellectual curiosity, cognitive diversity and bringing your whole self to work — and we have an insatiable drive to do what others think is impossible. Our employees are not only part of history, they're making history.

Northrop Grumman Mission Systems is a trusted provider of mission-enabling solutions for global security. Our Engineering and Sciences (E&S) organization pushes the boundaries of innovation, redefines engineering capabilities, and drives advances in various sciences. Our team is chartered with providing the skills, innovative technologies to develop, design, produce and sustain optimized product lines across the sector while providing a decisive advantage to the warfighter. Come be a part of our mission!


We are looking for you to join our team as a Principal / Sr. Principal Software Engineer based out of Woodland Hills, CA. As a Principal / Sr. Principal Software Engineer at Northrop Grumman, you will have a challenging and rewarding opportunity to be a part of our Enterprise-wide digital transformation. Through the use of Model-based Engineering, DevSecOps and Agile practices we continue to evolve how we deliver critical national defense products and capabilities for the warfighter. Our success is grounded in our ability to embrace change, move quickly and continuously drive innovation. The successful candidate will be collaborative, open, transparent, and team-oriented with a focus on team empowerment & shared responsibility, flexibility, continuous learning, and a culture of innovation.


For this role, responsibilities include but are not limited to:

Provide technical leadership for junior software engineers
Develop software utilizing C/C++ to modernize and productionize a research codebase with modern C++ features
Develop software infrastructure to support CI/CD, software metrics collection, and MLOps
Implement software-systems, applications, and architectures that leverage techniques to support achieving increased modularity, scalability, and reliability, while also maintaining precision, accuracy, and speed to meet performance requirements
Ensure industry software engineering best practices and standards are applied and maintained
Work closely with Software Leads and Architects to understand program intent, system capabilities, and output requirements


This requisition may be filled at a higher grade based on qualifications listed below


This requisition may be filled at either a Principal Level or a Sr. Principal Level.


Basic Qualifications for a Principal Software Engineer (T03)

Bachelor's degree in a STEM discipline with 5+ years of relative experience; Master's degree in a STEM discipline with 3+ years of relative experience; PhD + 1 year of relative experience
Active Top Secret security clearance
Ability to meet customer-specific security screening requirements within a timeframe set forth by management
Willingness and ability to work onsite full-time
Experience working in C/C++
Familiarity with modern C++ standards and features
Demonstrated ability to analyze system requirements to derive software design and performance requirements
Proven ability to design and code new software, as well as modify existing software to add new features
Ability to debug existing software and correct defects
Experience with open software/system architecture solutions
Effective communication and interpersonal skills, with the ability to collaborate effectively with diverse stakeholders
Experience with developing and maintaining CI/CD pipelines
Experience with Git-based or other software configuration management tools


Basic Qualifications for a Sr. Principal Software Engineer (T04)

Bachelor's degree in a STEM discipline with 8+ years of relative experience; Master's degree in a STEM discipline with 6+ years of relative experience; PhD + 4 years of relative experience
Active Top Secret security clearance
Ability to meet customer-specific security screening requirements within a timeframe set forth by management
Willingness and ability to work onsite full-time
Experience working in C/C++
Familiarity with modern C++ standards and features
Demonstrated ability to analyze system requirements to derive software design and performance requirements
Proven ability to design and code new software or modify existing software to add new features
Ability to debug existing software and correct defects
Experience with open software/system architecture solutions
Effective communication and interpersonal skills, with the ability to collaborate effectively with diverse stakeholders
Experience with Git-based or other software configuration management tools


Preferred Qualifications

Experience leading the performance of tasks on schedule, at cost and achieving specified requirements
Experience with modern C++ standards and features (e.g., C++ 17 onwards)
Experience with containers (Docker, Kubernetes)
Experience with Linux operating systems
Experience with CUDA and GPUs
Experience with high performance numerical/scientific computing, parallel computing
Experience with developing for SWaP-constrained environments
Experience with developing and maintaining CI/CD pipelines
Familiarity with signal-processing algorithms
Familiarity with Agile lifecycle process including Scrum and DevSecOps

Primary Level Salary Range: $110,300.00 - $165,500.00

Secondary Level Salary Range: $137,400.00 - $206,000.00

The above salary range represents a general guideline; however, Northrop Grumman considers a number of factors when determining base salary offers such as the scope and responsibilities of the position and the candidate's experience, education, skills and current market conditions.

Depending on the position, employees may be eligible for overtime, shift differential, and a discretionary bonus in addition to base pay. Annual bonuses are designed to reward individual contributions as well as allow employees to share in company results. Employees in Vice President or Director positions may be eligible for Long Term Incentives. In addition, Northrop Grumman provides a variety of benefits including health insurance coverage, life and disability insurance, savings plan, Company paid holidays and paid time off (PTO) for vacation and/or personal business.

The application period for the job is estimated to be 20 days from the job posting date. However, this timeline may be shortened or extended depending on business needs and the availability of qualified candidates.

Northrop Grumman is an Equal Opportunity Employer, making decisions without regard to race, color, religion, creed, sex, sexual orientation, gender identity, marital status, national origin, age, veteran status, disability, or any other protected class. For our complete EEO and pay transparency statement, please visit http://www.northropgrumman.com/EEO. U.S. Citizenship is required for all positions with a government clearance and certain other restricted positions.
    '''
    similarity = compute_similarity(resume, job_description)
    print(f"Similarity Score: {similarity:.4f}")
