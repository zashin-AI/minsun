from transformers import pipeline
generator = pipeline("text-generation", model='EleutherAI/gpt-neo-2.7B')
prompt = "What is the meaning of life?"

res = generator(prompt, max_length=20, do_sample=True, temperature=0.9 )

res
print(res[0]['generated_text'])

# gpt로 생성한 텍스트를 저장한다.
with open('gpttest.txt', 'w') as f :
    f.writelines(res[0]['generated_text'])