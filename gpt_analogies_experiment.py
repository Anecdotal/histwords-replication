import csv
import openai
import time 

openai.api_key = ''

def read_csv(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def write_csv(file_path, data):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def create_prompt(template, example):
    return template.format(example=example)

def call_gpt4(prompt):
    response = openai.Completion.create(
        engine="gpt-4-turbo-2024-04-09",  # Make sure to use the correct engine name for GPT-4
        prompt=prompt,
        max_tokens=20
    )
    return response.choices[0].text.strip()

def main():
    input_csvs = ['testset_1.csv', 'testset_2.csv', 'szymanski_analogies.csv']

    input_csv = input_csvs[0]
    output_csv = 'gpt_responses_yao_analogies_1.csv' 

    template_prompt = \
        "This is a test of temporal analogies. You're trying to find the words \
        in one year whose meaning is closest to the meaning of a different word in another year.\
        Please respond with a comma-separated list of 10 most similar words and nothing else. \
        The list should be ranked so that the first is the most similar, and the 10th is the \
        10th most similar. Do not explain or say anything beyond the words, \
        but please be as precise as possible. Now, the target word is {word1} in {year1}. \
        Please output the 10 words whose meaning in {year2} is most similar to {word1} in {year1}."
    
    analogies = read_csv(input_csv)
    results = []
    
    for (w1, w2, y1, y2) in analogies:

        prompt = template_prompt.format(word1=w1, year1=y1, year2=y2)

        response = call_gpt4(prompt)

        results.append([w1, y1, y2, w2, response])

        time.sleep(5)  # Pause for 5 seconds before the next API call
    
    write_csv(output_csv, results)

if __name__ == "__main__":
    main()
