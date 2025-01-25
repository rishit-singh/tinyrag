from embeddings import EmbeddingManager, OllamaEmbeddings
from pprint import pprint

# from tinytune.contexts.gptcontext import GPTContext, openai

# pprint(
#     EmbeddingManager("CompendiumLabs/bge-base-en-v1.5-gguf").Generate(["Hello World"])

# )

embed = OllamaEmbeddings(
    "http://localhost:11434/v1/",
    "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf",
).Run(tokens=["hello", "world"])

pprint(embed)

# openai.base_url = "http://localhost:11434/v1/"

# model = GPTContext("", "ollama")

# model.Model.Name = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

# model.OnGenerate = lambda content: print(content, end="") if content else None

# model.Prompt(
#     {
#         "role": "system",
#         "content": """
#         You are a system that exclusively returns the function calls necessary to fulfill the user's request.

#         Behavior:

#         Parse the user's input and identify the most relevant function(s).
#         Respond by outputting only the function call(s) with appropriate parameters.
#         Do not include explanations, formatting, or additional text—only the function call(s).
#         Refuse to generate any other text. Every response must be a function call. No extra content.
#         If the users asks something beyond the scope, just ignore and print null.
#         Example Input/Output:

#         Input: "Calculate the area of a circle with radius 5."
#         Output: calculate_circle_area(radius=5)

#         Input: "Sort the numbers [3, 1, 4, 1, 5]."
#         Output: sort_list(numbers=[3, 1, 4, 1, 5])

#         Input: "Find the square root of 9."
#         Output: calculate_sqrt(value=9)
#     """,
#     }
# ).Run(stream=True)


# i = input("\n>")

# while i != "exit":
#     model.Prompt({"role": "user", "content": i}).Run(stream=True)

#     i = input("\n> ")

# pprint(
#     type(
#         EmbeddingManager("sentence-transformers/all-MiniLM-L6-v2")(
#             ["Hello mother fucker, is this shit fast enough"]
#         )
#     )
# )
