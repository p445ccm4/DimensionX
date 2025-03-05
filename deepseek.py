from openai import OpenAI

class DeepSeek():
    def __init__(self):
        self.client = OpenAI(api_key="sk-f14f2d1f72f54c348f3fd325ca0e2ba0", base_url="https://api.deepseek.com")

    def gen_prompt(self, input: str):
        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", 
                 "content": "You are making prompt to a text-to-image generation model to generate interior design images. Always just reply with a string of English prompt with no quotation marks. Describe the details based on the user's request in less than 70 words."},
                {"role": "user", "content": input},
            ],
            stream=False
        )

        return response.choices[0].message.content